from __future__ import annotations

import asyncio
import shutil
import time
import zipfile
from pathlib import Path

from astrbot.api import logger
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from ..models import RunReport


class ReportStore:
    """跟踪最近的运行，并在请求时将分析工件打包到 plugin_data 中。"""

    def __init__(self, config_manager, runtime):
        self.config_manager = config_manager
        self.runtime = runtime
        data_path = Path(get_astrbot_data_path())
        self._pending_dir = data_path / "report_pending"
        self._output_dir = data_path / "plugin_data" / "mc_log_reports"
        self._runs: dict[str, list[RunReport]] = {}
        self._lock = asyncio.Lock()

    def _cfg(self):
        return self.config_manager.get()

    def enabled(self) -> bool:
        return bool(self._cfg().get("report_enabled", True))

    def _history_cap(self) -> int:
        return max(1, int(self._cfg().get("report_history_size", 3)))

    def resolve_debug_log_path(self, run_id: str) -> Path:
        return self.runtime.debug_log_dir / f"mc_log_debug_{run_id}.log"

    async def save_input_file(self, local_file: Path, run_id: str) -> Path | None:
        if not local_file or not local_file.exists():
            return None
        try:
            target_dir = self._pending_dir / run_id
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / local_file.name
            await self.runtime.run_io_in_thread(
                "copy_report_input", self._copy_file_blocking, local_file, target
            )
            return target
        except Exception as exc:
            logger.warning(f"[mc_log][report] 保存输入文件失败: {exc}")
            return None

    def _copy_file_blocking(self, src: Path, dst: Path, deadline, cancel_event):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

    async def register_run(self, report: RunReport):
        async with self._lock:
            history = self._runs.setdefault(report.session_id, [])
            history.insert(0, report)
            del history[self._history_cap():]

    async def find_recent(self, session_id: str) -> RunReport | None:
        async with self._lock:
            history = self._runs.get(session_id)
            return history[0] if history else None

    async def package_report(self, report: RunReport) -> Path | None:
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            safe_run = "".join(
                c if (c.isalnum() or c in "-_") else "_"
                for c in (report.run_id or "run")
            )[:64]
            zip_path = self._output_dir / f"report_{safe_run}_{stamp}.zip"
            await self.runtime.run_io_in_thread(
                "package_report", self._package_report_blocking, zip_path, report
            )
            if zip_path.exists():
                logger.info(
                    f"[mc_log][report] 已打包上报数据: path={zip_path}, "
                    f"size={zip_path.stat().st_size}"
                )
                return zip_path
            return None
        except Exception as exc:
            logger.error(f"[mc_log][report] 打包上报数据失败: {exc}", exc_info=True)
            return None

    def _package_report_blocking(
        self, zip_path: Path, report: RunReport, deadline, cancel_event
    ):
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("analysis.md", report.analysis_md or "")
            if report.input_file_path and Path(report.input_file_path).exists():
                zf.write(
                    report.input_file_path, "input/" + Path(report.input_file_path).name
                )
            if report.debug_log_path and Path(report.debug_log_path).exists():
                zf.write(report.debug_log_path, "debug.log")
            meta = (
                f"run_id: {report.run_id}\n"
                f"session_id: {report.session_id}\n"
                f"sender_id: {report.sender_id}\n"
                f"timestamp: {report.timestamp}\n"
                f"source_name: {report.source_name}\n"
            )
            zf.writestr("meta.txt", meta)

    def cleanup_stale_pending(self, hours: int):
        if not self._pending_dir.exists():
            return
        now = time.time()
        ttl = hours * 3600
        for child in self._pending_dir.iterdir():
            if not child.is_dir():
                continue
            try:
                mtime = child.stat().st_mtime
                if now - mtime > ttl:
                    shutil.rmtree(child, ignore_errors=True)
            except Exception:
                continue
