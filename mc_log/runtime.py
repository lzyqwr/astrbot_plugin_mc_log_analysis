from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import re
import threading
import time
import traceback
import uuid
from pathlib import Path

from astrbot.api import logger
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from .models import ArchiveSessionState


MC_RUN_ID_CTX = contextvars.ContextVar("mc_run_id", default="")


class RedactingLogFilter(logging.Filter):
    def __init__(self, redactor):
        super().__init__()
        self._redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            clean = self._redactor(msg)
            if clean != msg:
                record.msg = clean
                record.args = ()
        except Exception:
            return True
        return True


class RunIdLogFilter(logging.Filter):
    def __init__(self, run_id: str):
        super().__init__()
        self._run_id = str(run_id or "")

    def filter(self, record: logging.LogRecord) -> bool:
        return str(getattr(record, "mc_run_id", "") or "") == self._run_id


def build_mc_run_record_factory(prev_factory):
    def _factory(*args, **kwargs):
        record = prev_factory(*args, **kwargs)
        if not hasattr(record, "mc_run_id"):
            record.mc_run_id = MC_RUN_ID_CTX.get()
        elif not getattr(record, "mc_run_id", None):
            record.mc_run_id = MC_RUN_ID_CTX.get()
        return record

    return _factory


class PluginRuntime:
    def __init__(self):
        data_path = Path(get_astrbot_data_path())
        self.temp_root = data_path / "temp"
        self.debug_log_path = data_path / "debug.log"
        self.debug_log_dir = data_path / "debug_runs"
        self._debug_handler_map: dict[str, logging.Handler] = {}
        self._debug_path_map: dict[str, Path] = {}
        self._debug_handler_lock = asyncio.Lock()
        self._last_debug_log_path: Path | None = None
        self._prev_record_factory = None
        self._rate_limit_map: dict[str, float] = {}
        self._rate_limit_lock = asyncio.Lock()
        self._global_sema_size = 2
        self._global_sema = asyncio.Semaphore(2)
        self._io_sema_size = 1
        self._io_sema = asyncio.Semaphore(1)
        self._active_jobs = 0
        self._active_jobs_lock = asyncio.Lock()
        self._archive_sessions: dict[str, ArchiveSessionState] = {}

    def install_record_factory(self):
        if self._prev_record_factory is not None:
            return
        prev = logging.getLogRecordFactory()
        self._prev_record_factory = prev
        logging.setLogRecordFactory(build_mc_run_record_factory(prev))

    def uninstall_record_factory(self):
        prev = self._prev_record_factory
        if prev is None:
            return
        logging.setLogRecordFactory(prev)
        self._prev_record_factory = None

    def bind_run_id(self, run_id: str):
        return MC_RUN_ID_CTX.set(run_id)

    def reset_run_id(self, token):
        MC_RUN_ID_CTX.reset(token)

    def time_left(self, deadline: float | None) -> float:
        if deadline is None:
            return float("inf")
        return deadline - time.monotonic()

    def sync_global_sema(self, cfg):
        size = max(1, int(cfg.get("max_concurrent_jobs", 2)))
        if size == self._global_sema_size:
            return
        if self._active_jobs > 0:
            return
        self._global_sema_size = size
        self._global_sema = asyncio.Semaphore(size)

    def sync_io_sema(self, cfg):
        size = max(1, int(cfg.get("max_concurrent_io", 1)))
        if size == self._io_sema_size:
            return
        self._io_sema_size = size
        self._io_sema = asyncio.Semaphore(size)

    def ensure_not_timed_out(self, deadline: float | None, run_id: str = "", stage: str = ""):
        left = self.time_left(deadline)
        if left <= 0:
            stage_info = f", stage={stage}" if stage else ""
            logger.warning(f"[mc_log][{run_id}] 已达到全局超时时间{stage_info}")
            raise TimeoutError("global timeout reached")

    def check_deadline_or_cancel(
        self,
        deadline: float | None,
        cancel_event: threading.Event | None,
        stage: str,
    ):
        if cancel_event and cancel_event.is_set():
            raise TimeoutError(f"io canceled: {stage}")
        if deadline is not None and time.monotonic() > deadline:
            if cancel_event:
                cancel_event.set()
            raise TimeoutError(f"io deadline exceeded: {stage}")

    async def run_io_in_thread(self, call_name: str, func, *args, deadline: float | None = None):
        cancel_event = threading.Event()
        async with self._io_sema:
            self.ensure_not_timed_out(deadline, stage=f"{call_name}_before")
            task = asyncio.to_thread(func, *args, deadline, cancel_event)
            timeout = None if deadline is None else max(0.1, self.time_left(deadline))
            try:
                if timeout is None:
                    return await task
                return await asyncio.wait_for(task, timeout=timeout)
            except asyncio.TimeoutError as exc:
                cancel_event.set()
                raise TimeoutError(f"{call_name} timeout") from exc

    async def check_rate_limit(self, event, cfg) -> bool:
        limit_sec = float(cfg.get("rate_limit_user_sec", 0))
        if limit_sec <= 0:
            return True
        user_id = str(event.get_sender_id() or "")
        if not user_id:
            return True
        now = time.monotonic()
        async with self._rate_limit_lock:
            last = self._rate_limit_map.get(user_id, 0.0)
            if now - last < limit_sec:
                return False
            self._rate_limit_map[user_id] = now
        return True

    async def acquire_global_slot(self, cfg, run_id: str = "") -> bool:
        wait_sec = float(cfg.get("queue_wait_sec", 0))
        if wait_sec <= 0:
            await self._global_sema.acquire()
            async with self._active_jobs_lock:
                self._active_jobs += 1
                logger.info(
                    f"[mc_log][{run_id}] 获取全局并发槽位: active_jobs={self._active_jobs}, max={self._global_sema_size}"
                )
            return True
        try:
            await asyncio.wait_for(self._global_sema.acquire(), timeout=wait_sec)
        except asyncio.TimeoutError:
            logger.warning(
                f"[mc_log][{run_id}] 等待全局并发槽位超时: "
                f"wait_sec={wait_sec}, active_jobs={self._active_jobs}, max={self._global_sema_size}"
            )
            return False
        async with self._active_jobs_lock:
            self._active_jobs += 1
            logger.info(f"[mc_log][{run_id}] 获取全局并发槽位: active_jobs={self._active_jobs}, max={self._global_sema_size}")
        return True

    async def release_global_slot(self, run_id: str = ""):
        released = False
        try:
            self._global_sema.release()
            released = True
        except Exception as exc:
            logger.warning(f"[mc_log][{run_id}] 释放全局并发槽位异常: {exc}")
        async with self._active_jobs_lock:
            self._active_jobs = max(0, self._active_jobs - 1)
            logger.info(
                f"[mc_log][{run_id}] 释放全局并发槽位: released={released}, "
                f"active_jobs={self._active_jobs}, max={self._global_sema_size}"
            )

    async def start_debug_capture(self, run_id: str, event, file_name: str, is_archive: bool, cfg, sanitizer):
        self.debug_log_dir.mkdir(parents=True, exist_ok=True)
        run_path = self.debug_log_dir / f"mc_log_debug_{run_id}.log"
        handler = logging.FileHandler(run_path, mode="w", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handler.addFilter(RunIdLogFilter(run_id))
        handler.addFilter(RedactingLogFilter(sanitizer))
        async with self._debug_handler_lock:
            logger.addHandler(handler)
            self._debug_handler_map[run_id] = handler
            self._debug_path_map[run_id] = run_path
        logger.info(
            f"[mc_log][{run_id}] 已开始写入单次调试日志: path={run_path}, sender={event.get_sender_id()}, "
            f"message_id={getattr(event.message_obj, 'message_id', '')}, file={file_name}, is_archive={is_archive}"
        )
        logger.info(f"[mc_log][{run_id}] 运行配置快照(脱敏): {sanitizer(json.dumps(cfg.to_dict(), ensure_ascii=False))}")
        logger.info(
            f"[mc_log][{run_id}] 事件快照: platform={event.get_platform_name()}, sender_name={event.get_sender_name()}, "
            f"sender_id={event.get_sender_id()}, group_id={event.get_group_id()}, session_id={event.get_session_id()}, "
            "message_preview=<REDACTED>"
        )

    async def stop_debug_capture(self, run_id: str):
        async with self._debug_handler_lock:
            handler = self._debug_handler_map.pop(run_id, None)
            path = self._debug_path_map.pop(run_id, None)
        if not handler:
            return
        try:
            handler.flush()
        except Exception:
            pass
        try:
            logger.removeHandler(handler)
        except Exception:
            pass
        try:
            handler.close()
        except Exception:
            pass
        if path and path.exists():
            self._last_debug_log_path = path

    async def sync_latest_debug_copy(self, copier):
        path = self._last_debug_log_path
        if path and path.exists():
            try:
                await copier(path, self.debug_log_path)
            except Exception:
                pass

    async def stop_all_debug_captures(self):
        async with self._debug_handler_lock:
            run_ids = list(self._debug_handler_map.keys())
        for run_id in run_ids:
            await self.stop_debug_capture(run_id)

    def resolve_latest_debug_log_path(self) -> Path | None:
        if self._last_debug_log_path and self._last_debug_log_path.exists():
            return self._last_debug_log_path
        if self.debug_log_path.exists():
            return self.debug_log_path
        if not self.debug_log_dir.exists():
            return None
        files = sorted(
            [path for path in self.debug_log_dir.glob("mc_log_debug_*.log") if path.is_file()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return files[0] if files else None

    def event_context_key(self, event) -> str:
        msg_id = str(getattr(event.message_obj, "message_id", "") or "").strip()
        return msg_id if msg_id else f"obj_{id(event.message_obj)}"

    def set_active_archive_file_map(self, event, archive_file_map: dict[str, Path]):
        key = self.event_context_key(event)
        if not key:
            return
        session = self._archive_sessions.get(key, ArchiveSessionState())
        session.file_map = dict(archive_file_map or {})
        session.tool_calls = 0
        self._archive_sessions[key] = session

    def set_active_primary_source_name(self, event, source_name: str):
        key = self.event_context_key(event)
        if not key:
            return
        session = self._archive_sessions.get(key, ArchiveSessionState())
        session.primary_source_name = str(source_name or "").strip()
        self._archive_sessions[key] = session

    def get_active_archive_file_map(self, event) -> dict[str, Path]:
        key = self.event_context_key(event)
        if not key:
            return {}
        session = self._archive_sessions.get(key)
        return dict(session.file_map) if session else {}

    def is_primary_source_file(self, event, archive_key: str) -> bool:
        key = self.event_context_key(event)
        if not key:
            return False
        session = self._archive_sessions.get(key)
        if not session or not session.primary_source_name:
            return False
        requested_name = Path(str(archive_key or "")).name.lower().strip()
        return bool(requested_name and requested_name == session.primary_source_name.lower().strip())

    def consume_archive_tool_call(self, event, cfg) -> tuple[bool, str]:
        key = self.event_context_key(event)
        if not key:
            return False, "会话上下文不可用，无法读取归档文件。"
        limit = max(0, int(cfg.get("read_archive_file_limit", 1)))
        if limit <= 0:
            return False, "read_archive_file 已被配置禁用。"
        session = self._archive_sessions.get(key, ArchiveSessionState())
        if session.tool_calls >= limit:
            return False, f"read_archive_file 调用次数已达上限（{limit}）。"
        session.tool_calls += 1
        self._archive_sessions[key] = session
        logger.info(f"[mc_log][tool] read_archive_file 调用计数: used={session.tool_calls}/{limit}")
        return True, ""

    def clear_active_archive_file_map(self, event):
        key = self.event_context_key(event)
        if not key:
            return
        self._archive_sessions.pop(key, None)

    def build_work_dir(self, event) -> Path:
        msg_id = str(getattr(event.message_obj, "message_id", "") or f"mid_{int(time.time())}")
        msg_id = re.sub(r"[^\w\-\.]", "_", msg_id)
        base = self.temp_root / msg_id
        if base.exists():
            return self.temp_root / f"{msg_id}_{uuid.uuid4().hex[:6]}"
        return base

    def build_run_id(self, event) -> str:
        msg_id = str(getattr(event.message_obj, "message_id", "") or "mid")
        msg_id = re.sub(r"[^\w\-\.]", "_", msg_id)
        return f"{msg_id}-{uuid.uuid4().hex[:6]}"

    def cleanup_stale_temp_dirs(self, hours: int, remover):
        now = time.time()
        ttl = hours * 3600
        if not self.temp_root.exists():
            return
        for child in self.temp_root.iterdir():
            if not child.is_dir():
                continue
            marker = child / ".mc_log_analysis"
            if not marker.exists():
                continue
            try:
                mtime = child.stat().st_mtime
                if now - mtime > ttl:
                    remover(child, None, None)
            except Exception:
                continue

    def log_remove_dir_failure(self):
        logger.warning("[mc_log] 删除临时目录失败:\n" + traceback.format_exc())
