from __future__ import annotations

import asyncio
import base64
import gzip
import html
import io
import json
import os
import re
import shutil
import textwrap
import time
import traceback
import uuid
import zipfile
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
from urllib.parse import quote_plus

import aiohttp
import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.event.filter import EventMessageType, event_message_type
from astrbot.api.star import Context, Star, register
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


ARCHIVE_EXTS = {".zip", ".gz"}
TEXT_EXTS = {".txt", ".log"}
ARCHIVE_NAME_KEYS = ("错误报告", "日志", "log")
TEXT_NAME_KEYS = ("crash", "hs_err", "latest", "debug", "fcl", "pcl", "游戏崩溃", "日志", "log")

DEFAULT_CONFIG = {
    "chunk_size": 100000,
    "global_timeout_sec": 300,
    "map_select_provider": "",
    "analyze_select_provider": "",
    "render_mode": "html_to_image",
    "image_width": 640,
    "full_read_char_limit": 140000,
    "total_char_limit": 140000,
    "max_tool_calls": 6,
    "tool_timeout_sec": 120,
    "tool_retry_limit": 1,
    "map_llm_timeout_sec": 120,
    "analyze_llm_timeout_sec": 240,
    "max_map_chunks": 10,
    "map_timeout_break_threshold": 2,
    "skip_final_analyze_on_map_timeout": True,
    "html_render_timeout_sec": 30,
    "file_download_timeout_sec": 30,
    "messages": {
        "accepted_notice": "已接收文件，正在分析，请稍候。",
        "download_failed": "日志文件下载失败，请稍后重试。",
        "no_extractable_content": "未在文件中识别到可分析的日志内容。",
        "analyze_failed_logged": "日志分析失败，已记录日志，请联系管理员检查检查。",
        "analyze_failed_retry": "日志分析失败，请联系管理员检查。",
        "prompt_missing": "日志分析模板缺失，请联系管理员检查 assets 目录。",
        "provider_not_configured": "请先在插件配置中填写 map_select_provider 与 analyze_select_provider。",
        "global_timeout": "日志分析超时，请稍后重试。",
        "html_render_fallback_notice": "[提示] HTML 渲染失败，已降级发送原始文本。",
        "text_render_fallback_notice": "[提示] 渲染不可用，已降级为纯文本发送。",
        "forward_sender_name": "MC日志分析",
        "summary_template": "分析完成，耗时 {elapsed:.2f} 秒\n文件: {source_name}\n策略: {strategy}",
    },
}

MAX_ARCHIVE_FILE_COUNT = 160
MAX_ARCHIVE_SINGLE_FILE_BYTES = 12 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 32 * 1024 * 1024
MAX_GZ_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_SECTION_LINES = 400
MR_SKIP = "__MR_SKIP__"

PROMPT_FILES = {
    "map_system": "map_system.txt",
    "map_user": "map_user.txt",
    "analyze_system": "analyze_system.txt",
    "analyze_user": "analyze_user.txt",
}

ERROR_LINE_RE = re.compile(
    r"(ERROR|WARN|Exception|Caused by|FATAL|Failed|Stacktrace|Problematic frame|siginfo)",
    re.IGNORECASE,
)
TIMESTAMP_LINE_RE = re.compile(r"^\s*(\[\d{2}:\d{2}:\d{2}\]|\d{4}-\d{2}-\d{2}|\[\d{2}[/:]\d{2}[/:]\d{2})")


@register(
    "astrbot_plugin_mc_log_analysis",
    "lzyqwr",
    "Minecraft 日志分析插件",
    "1.0.0",
    "https://github.com/lzyqwr/astrbot_plugin_mc_log_analysis",
)
class LogAnalyzer(Star):
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self._raw_config = config
        self.cfg = self._load_config()
        self.temp_root = Path(get_astrbot_data_path()) / "temp"
        self.assets_dir = Path(__file__).resolve().parent / "assets"
        self.html_template_path = self.assets_dir / "html_to_image.html.j2"
        self.prompt_dir = self.assets_dir
        self.prompts: dict[str, str] = {}
        self.prompts_ready = False
        self._tool_registered = False

    async def initialize(self):
        self.cfg = self._load_config()
        self._load_prompts()
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self._register_tools()
        self._cleanup_stale_temp_dirs(hours=24)
        logger.info("[mc_log] 插件已初始化")

    async def terminate(self):
        logger.info("[mc_log] 插件已停止")

    @event_message_type(EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        self.cfg = self._load_config()
        selected = self._pick_target_file(event)
        if not selected:
            return
        started = time.monotonic()
        deadline = started + float(self.cfg["global_timeout_sec"])
        if not self.prompts_ready:
            self._load_prompts()
        if not self.prompts_ready:
            logger.error("[mc_log] 提示词模板未就绪，已终止本次分析")
            result = event.plain_result(self._msg("prompt_missing"))
            result.stop_event()
            yield result
            return

        file_comp, is_archive = selected
        run_id = self._build_run_id(event)
        map_provider_id, analyze_provider_id = self._configured_provider_ids()
        if not map_provider_id or not analyze_provider_id:
            logger.error(
                f"[mc_log][{run_id}] 未配置模型提供商: "
                f"map_select_provider={map_provider_id!r}, analyze_select_provider={analyze_provider_id!r}"
            )
            result = event.plain_result(self._msg("provider_not_configured"))
            result.stop_event()
            yield result
            return
        if not self.context.get_provider_by_id(map_provider_id) or not self.context.get_provider_by_id(
            analyze_provider_id
        ):
            logger.error(
                f"[mc_log][{run_id}] 配置的模型提供商不存在: "
                f"map={map_provider_id}, analyze={analyze_provider_id}"
            )
            result = event.plain_result(self._msg("provider_not_configured"))
            result.stop_event()
            yield result
            return

        accepted_notice = self._msg("accepted_notice")
        if accepted_notice:
            yield event.plain_result(accepted_notice)
        logger.info(
            f"[mc_log][{run_id}] 开始分析: "
            f"message_id={getattr(event.message_obj, 'message_id', '')}, "
            f"name={self._detect_file_name(event, file_comp)}, is_archive={is_archive}, "
            f"map_provider={map_provider_id}, analyze_provider={analyze_provider_id}, "
            f"global_timeout_sec={self.cfg['global_timeout_sec']}"
        )
        logger.info(
            f"[mc_log] 文件命中规则: name={self._detect_file_name(event, file_comp)}, is_archive={is_archive}"
        )
        work_dir = self._build_work_dir(event)
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / ".mc_log_analysis").write_text("1", encoding="utf-8")

        try:
            self._ensure_not_timed_out(deadline, run_id=run_id, stage="before_download")
            t_download = time.monotonic()
            local_file = await self._download_to_workdir(file_comp, work_dir, deadline=deadline)
            logger.info(
                f"[mc_log][{run_id}] 阶段下载: "
                f"{time.monotonic() - t_download:.2f}s, ok={bool(local_file and local_file.exists())}"
            )
            if not local_file or not local_file.exists():
                logger.warning(f"[mc_log][{run_id}] 下载阶段失败，未获取到本地文件")
                result = event.plain_result(self._msg("download_failed"))
                result.stop_event()
                yield result
                return

            t_extract = time.monotonic()
            extracted, source_name, strategy, skip_final_analyze = await self._extract_content(
                local_file=local_file,
                is_archive=is_archive,
                work_dir=work_dir,
                event=event,
                map_provider_id=map_provider_id,
                run_id=run_id,
                deadline=deadline,
            )
            logger.info(
                f"[mc_log][{run_id}] 阶段提取: "
                f"{time.monotonic() - t_extract:.2f}s, strategy={strategy}, "
                f"chars={len(extracted)}, skip_final={skip_final_analyze}"
            )
            if not extracted.strip():
                logger.warning(f"[mc_log][{run_id}] 提取结果为空，无法继续分析")
                result = event.plain_result(self._msg("no_extractable_content"))
                result.stop_event()
                yield result
                return

            extracted = self._apply_total_budget(extracted, self.cfg["total_char_limit"])
            extracted_for_llm = self._redact_text(extracted)

            t_analyze = time.monotonic()
            report_md = await self._analyze_with_llm(
                event=event,
                source_name=source_name,
                strategy=strategy,
                content=extracted_for_llm,
                analyze_provider_id=analyze_provider_id,
                skip_due_to_map_timeouts=skip_final_analyze,
                run_id=run_id,
                deadline=deadline,
            )
            logger.info(
                f"[mc_log][{run_id}] 阶段分析: "
                f"{time.monotonic() - t_analyze:.2f}s, ok={bool(report_md)}"
            )
            if not report_md:
                logger.warning(f"[mc_log][{run_id}] 最终分析未返回内容")
                result = event.plain_result(self._msg("analyze_failed_logged"))
                result.stop_event()
                yield result
                return
            report_md = self._redact_text(report_md)

            t_render = time.monotonic()
            render_mode, render_payload = await self._render_report(
                report_md,
                run_id=run_id,
                deadline=deadline,
            )
            logger.info(
                f"[mc_log][{run_id}] 阶段渲染: "
                f"{time.monotonic() - t_render:.2f}s, mode={render_mode}"
            )
            elapsed = time.monotonic() - started
            logger.info(f"[mc_log][{run_id}] 分析完成，总耗时: {elapsed:.2f}s")

            response = self._build_forward_response(
                event=event,
                source_name=source_name,
                strategy=strategy,
                elapsed=elapsed,
                render_mode=render_mode,
                render_payload=render_payload,
                report_md=report_md,
            )
            response.stop_event()
            yield response
        except TimeoutError as exc:
            logger.warning(f"[mc_log][{run_id}] 全局超时: {exc}")
            result = event.plain_result(self._msg("global_timeout"))
            result.stop_event()
            yield result
        except Exception as exc:
            logger.error(f"[mc_log][{run_id}] 处理流程异常: {exc}", exc_info=True)
            result = event.plain_result(self._msg("analyze_failed_retry"))
            result.stop_event()
            yield result
        finally:
            self._safe_remove_dir(work_dir)

    def _load_config(self) -> dict:
        cfg = dict(DEFAULT_CONFIG)
        for key in cfg:
            cfg[key] = self._read_conf_value(key, cfg[key])
        cfg["chunk_size"] = max(500, int(cfg["chunk_size"]))
        cfg["global_timeout_sec"] = max(10, int(cfg.get("global_timeout_sec", 300)))
        cfg["map_select_provider"] = str(cfg.get("map_select_provider", "") or "").strip()
        cfg["analyze_select_provider"] = str(cfg.get("analyze_select_provider", "") or "").strip()
        cfg["image_width"] = max(320, int(cfg.get("image_width", 640)))
        cfg["full_read_char_limit"] = max(10000, int(cfg["full_read_char_limit"]))
        cfg["total_char_limit"] = max(3000, int(cfg["total_char_limit"]))
        cfg["max_tool_calls"] = max(1, int(cfg["max_tool_calls"]))
        cfg["tool_timeout_sec"] = max(2, int(cfg["tool_timeout_sec"]))
        cfg["tool_retry_limit"] = max(0, int(cfg["tool_retry_limit"]))
        cfg["map_llm_timeout_sec"] = max(3, int(cfg["map_llm_timeout_sec"]))
        cfg["analyze_llm_timeout_sec"] = max(5, int(cfg["analyze_llm_timeout_sec"]))
        cfg["max_map_chunks"] = max(1, int(cfg["max_map_chunks"]))
        cfg["map_timeout_break_threshold"] = max(1, int(cfg["map_timeout_break_threshold"]))
        cfg["skip_final_analyze_on_map_timeout"] = bool(cfg["skip_final_analyze_on_map_timeout"])
        cfg["html_render_timeout_sec"] = max(5, int(cfg["html_render_timeout_sec"]))
        cfg["file_download_timeout_sec"] = max(5, int(cfg["file_download_timeout_sec"]))
        raw_render_mode = str(cfg["render_mode"]).lower().strip()
        if raw_render_mode in {"html", "html_to_image"}:
            cfg["render_mode"] = "html_to_image"
        elif raw_render_mode in {"text", "text_to_image"}:
            cfg["render_mode"] = "text_to_image"
        else:
            cfg["render_mode"] = "html_to_image"
        cfg["messages"] = self._normalize_messages_config(cfg.get("messages"))
        return cfg

    def _time_left(self, deadline: float | None) -> float:
        if deadline is None:
            return float("inf")
        return deadline - time.monotonic()

    def _ensure_not_timed_out(self, deadline: float | None, run_id: str = "", stage: str = ""):
        left = self._time_left(deadline)
        if left <= 0:
            stage_info = f", stage={stage}" if stage else ""
            logger.warning(f"[mc_log][{run_id}] 已达到全局超时时间{stage_info}")
            raise TimeoutError("global timeout reached")

    def _read_conf_value(self, key: str, default):
        """
        优先读取 AstrBot 注入的插件配置对象（来自 _conf_schema.json 对应的配置文件）。
        读取失败时回退到默认值。
        """
        raw = self._raw_config
        if raw is None:
            return default
        try:
            if hasattr(raw, "get"):
                value = raw.get(key, default)
            else:
                value = raw[key]
            return default if value is None else value
        except Exception:
            return default

    def _normalize_messages_config(self, raw_messages) -> dict:
        defaults = DEFAULT_CONFIG["messages"]
        out = dict(defaults)
        if isinstance(raw_messages, dict):
            for key, value in raw_messages.items():
                if key in out and value is not None:
                    out[key] = str(value)
        return out

    def _msg(self, key: str) -> str:
        messages = self.cfg.get("messages", {})
        if isinstance(messages, dict):
            value = messages.get(key)
            if isinstance(value, str) and value:
                return value
        return str(DEFAULT_CONFIG["messages"].get(key, ""))

    def _configured_provider_ids(self) -> tuple[str, str]:
        map_provider = str(self.cfg.get("map_select_provider", "") or "").strip()
        analyze_provider = str(self.cfg.get("analyze_select_provider", "") or "").strip()
        return map_provider, analyze_provider

    def _load_prompts(self):
        self.prompts = {}
        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        missing = []
        for key, filename in PROMPT_FILES.items():
            content = self._read_prompt_file(filename)
            if not content:
                missing.append(filename)
            self.prompts[key] = content
        self.prompts_ready = len(missing) == 0
        if missing:
            logger.error(
                f"[mc_log] 提示词文件缺失或为空: {missing}; 目录: {self.prompt_dir}"
            )

    def _read_prompt_file(self, filename: str) -> str:
        path = self.prompt_dir / filename
        try:
            if not path.exists():
                return ""
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
            logger.warning(f"[mc_log] 提示词文件为空: {path}")
        except Exception as exc:
            logger.warning(f"[mc_log] 读取提示词文件失败 {path}: {exc}")
        return ""

    def _get_prompt(self, key: str) -> str:
        return self.prompts.get(key, "")

    def _render_prompt(self, template: str, values: dict[str, str]) -> str:
        out = template
        for k, v in values.items():
            out = out.replace(f"{{{{{k}}}}}", str(v))
        return out

    def _register_tools(self):
        if self._tool_registered:
            return
        search_mcmod_tool = FunctionTool(
            name="search_mcmod",
            description="在 mcmod.cn 中搜索 Minecraft 模组信息、依赖、冲突和兼容线索。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要搜索的模组名、模组ID或关键词。",
                    }
                },
                "required": ["query"],
            },
            handler=self._tool_search_mcmod,
        )
        search_wiki_tool = FunctionTool(
            name="search_minecraft_wiki",
            description="在 Minecraft Wiki 中搜索错误代码、机制说明和实体条目。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要搜索的错误代码、机制词或实体名称。",
                    }
                },
                "required": ["query"],
            },
            handler=self._tool_search_minecraft_wiki,
        )
        self.context.add_llm_tools(search_mcmod_tool, search_wiki_tool)
        self._tool_registered = True

    async def _tool_search_mcmod(self, event: AstrMessageEvent, query: str) -> str:
        normalized = self._normalize_tool_query(query)
        if not normalized:
            return "query 无效，请提供简短关键词。"

        url = f"https://search.mcmod.cn/s?key={quote_plus(normalized)}&filter=0&page=1"
        tries = self.cfg["tool_retry_limit"] + 1
        last_error = ""
        started = time.monotonic()
        logger.info(f"[mc_log][tool] 调用 search_mcmod: query={normalized}, tries={tries}")
        for i in range(tries):
            try:
                text = await self._http_get_text(url, timeout=self.cfg["tool_timeout_sec"])
                entries = self._parse_mcmod_search_html(text)
                if not entries:
                    logger.info(
                        f"[mc_log][tool] search_mcmod 无结果: query={normalized}, "
                        f"elapsed={time.monotonic() - started:.2f}s"
                    )
                    return f"mcmod 未找到 `{normalized}` 的结果。"
                lines = [f"mcmod 搜索 `{normalized}`："]
                for idx, (title, link) in enumerate(entries[:5], start=1):
                    lines.append(f"{idx}. {title} - {link}")
                logger.info(
                    f"[mc_log][tool] search_mcmod 成功: query={normalized}, results={len(entries)}, "
                    f"elapsed={time.monotonic() - started:.2f}s"
                )
                return "\n".join(lines)
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    f"[mc_log][tool] search_mcmod 第{i + 1}/{tries}次失败: query={normalized}, err={exc}"
                )
                await asyncio.sleep(0.2)
        logger.error(
            f"[mc_log][tool] search_mcmod 最终失败: query={normalized}, err={last_error}, "
            f"elapsed={time.monotonic() - started:.2f}s"
        )
        return f"mcmod 查询失败：{last_error}"

    async def _tool_search_minecraft_wiki(self, event: AstrMessageEvent, query: str) -> str:
        normalized = self._normalize_tool_query(query)
        if not normalized:
            return "query 无效，请提供简短关键词。"

        url = (
            "https://minecraft.wiki/api.php"
            f"?action=query&list=search&srsearch={quote_plus(normalized)}&utf8=1&format=json&srlimit=5"
        )
        tries = self.cfg["tool_retry_limit"] + 1
        last_error = ""
        started = time.monotonic()
        logger.info(f"[mc_log][tool] 调用 search_minecraft_wiki: query={normalized}, tries={tries}")
        for i in range(tries):
            try:
                text = await self._http_get_text(url, timeout=self.cfg["tool_timeout_sec"])
                payload = json.loads(text)
                rows = payload.get("query", {}).get("search", [])
                if not rows:
                    logger.info(
                        f"[mc_log][tool] search_minecraft_wiki 无结果: query={normalized}, "
                        f"elapsed={time.monotonic() - started:.2f}s"
                    )
                    return f"Minecraft Wiki 未找到 `{normalized}` 的结果。"
                lines = [f"Minecraft Wiki 搜索 `{normalized}`："]
                for idx, item in enumerate(rows[:5], start=1):
                    title = str(item.get("title", "")).strip()
                    snippet = re.sub(r"<[^>]+>", "", str(item.get("snippet", ""))).strip()
                    page_url = f"https://minecraft.wiki/w/{quote_plus(title.replace(' ', '_'))}"
                    lines.append(f"{idx}. {title} - {page_url}")
                    if snippet:
                        lines.append(f"   摘要: {snippet}")
                logger.info(
                    f"[mc_log][tool] search_minecraft_wiki 成功: query={normalized}, results={len(rows)}, "
                    f"elapsed={time.monotonic() - started:.2f}s"
                )
                return "\n".join(lines)
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    f"[mc_log][tool] search_minecraft_wiki 第{i + 1}/{tries}次失败: "
                    f"query={normalized}, err={exc}"
                )
                await asyncio.sleep(0.2)
        logger.error(
            f"[mc_log][tool] search_minecraft_wiki 最终失败: query={normalized}, err={last_error}, "
            f"elapsed={time.monotonic() - started:.2f}s"
        )
        return f"Minecraft Wiki 查询失败：{last_error}"

    async def _http_get_text(self, url: str, timeout: int) -> str:
        headers = {
            "User-Agent": "AstrBot-MC-Log-Analyzer/1.0",
            "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
        }
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(trust_env=True, timeout=client_timeout) as session:
            async with session.get(url, headers=headers) as resp:
                resp.raise_for_status()
                return await resp.text(errors="replace")

    def _parse_mcmod_search_html(self, html_text: str) -> list[tuple[str, str]]:
        matches = re.findall(
            r'<a[^>]*href="(https?://[^"]*mcmod\.cn[^"]*)"[^>]*>(.*?)</a>',
            html_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        results: list[tuple[str, str]] = []
        seen = set()
        for link, title_html in matches:
            if link in seen:
                continue
            if any(skip in link for skip in ("/s?", "/s/", "search.mcmod.cn")):
                continue
            title = re.sub(r"<[^>]+>", "", title_html)
            title = html.unescape(title).strip()
            if not title:
                continue
            seen.add(link)
            results.append((title, link))
        return results

    def _normalize_tool_query(self, query: str) -> str:
        q = str(query or "").replace("\n", " ").replace("\r", " ").strip()
        q = re.sub(r"\s+", " ", q)
        if len(q) > 120:
            q = q[:120]
        q = re.sub(r"[^\w\-\.\+\#:\u4e00-\u9fff ]", " ", q)
        q = re.sub(r"\s+", " ", q).strip()
        return q

    def _pick_target_file(self, event: AstrMessageEvent):
        files = [comp for comp in event.get_messages() if isinstance(comp, Comp.File)]
        if not files:
            return None

        for file_comp in files:
            name = self._detect_file_name(event, file_comp).lower()
            ext = Path(name).suffix.lower()
            if ext in ARCHIVE_EXTS and any(k in name for k in ARCHIVE_NAME_KEYS):
                return file_comp, True
            if ext in TEXT_EXTS and any(k in name for k in TEXT_NAME_KEYS):
                return file_comp, False
        logger.debug(
            f"[mc_log] 收到文件消息但未命中规则: "
            f"{[self._detect_file_name(event, f) for f in files]}"
        )
        return None

    def _detect_file_name(self, event: AstrMessageEvent, file_comp: Comp.File) -> str:
        name = str(getattr(file_comp, "name", "") or "").strip()
        if name:
            return name

        file_field = str(getattr(file_comp, "file_", "") or "").strip()
        if file_field:
            base = Path(file_field).name
            if base:
                return base

        url = str(getattr(file_comp, "url", "") or "").strip()
        if url:
            parsed = urlparse(url)
            base = Path(parsed.path).name
            if base:
                return base

        try:
            raw = getattr(event.message_obj, "raw_message", None)
            if raw and isinstance(raw, dict):
                segs = raw.get("message", [])
                if isinstance(segs, list):
                    for seg in segs:
                        if isinstance(seg, dict) and seg.get("type") == "file":
                            data = seg.get("data", {})
                            if isinstance(data, dict):
                                raw_name = str(
                                    data.get("file_name")
                                    or data.get("name")
                                    or data.get("file")
                                    or ""
                                ).strip()
                                if raw_name:
                                    return raw_name
        except Exception:
            pass
        return ""

    def _build_work_dir(self, event: AstrMessageEvent) -> Path:
        msg_id = str(getattr(event.message_obj, "message_id", "") or f"mid_{int(time.time())}")
        msg_id = re.sub(r"[^\w\-\.]", "_", msg_id)
        base = self.temp_root / msg_id
        if base.exists():
            return self.temp_root / f"{msg_id}_{uuid.uuid4().hex[:6]}"
        return base

    def _build_run_id(self, event: AstrMessageEvent) -> str:
        msg_id = str(getattr(event.message_obj, "message_id", "") or "mid")
        msg_id = re.sub(r"[^\w\-\.]", "_", msg_id)
        return f"{msg_id}-{uuid.uuid4().hex[:6]}"

    async def _download_to_workdir(
        self,
        file_comp: Comp.File,
        work_dir: Path,
        deadline: float | None = None,
    ) -> Path | None:
        self._ensure_not_timed_out(deadline, stage="download_prepare")
        src = await asyncio.wait_for(
            file_comp.get_file(allow_return_url=True),
            timeout=max(0.1, self._time_left(deadline)),
        )
        if not src:
            logger.warning("[mc_log] 获取文件源失败：file_comp.get_file 返回空")
            return None

        detected_name = self._detect_file_name_dummy(file_comp)
        safe_name = re.sub(r"[^\w\-.]", "_", detected_name or "upload.log")
        dst = work_dir / safe_name

        if str(src).startswith("http://") or str(src).startswith("https://"):
            try:
                timeout_sec = min(
                    float(self.cfg["file_download_timeout_sec"]),
                    max(0.1, self._time_left(deadline)),
                )
                logger.info(
                    f"[mc_log] 准备下载远程文件: url={src}, dst={dst.name}, timeout={timeout_sec:.2f}s"
                )
                await self._download_url_to_path(
                    url=str(src),
                    dst=dst,
                    timeout_sec=timeout_sec,
                )
                return dst if dst.exists() else None
            except Exception as exc:
                if self._time_left(deadline) <= 0:
                    raise TimeoutError("global timeout reached during download") from exc
                logger.error(f"[mc_log] 远程文件下载失败: {exc}", exc_info=True)
                return None

        self._ensure_not_timed_out(deadline, stage="download_copy")
        src_path = Path(src)
        if not src_path.exists():
            logger.warning(f"[mc_log] 本地文件源不存在: {src_path}")
            return None
        if not safe_name or safe_name == "upload.log":
            safe_name = re.sub(r"[^\w\-.]", "_", src_path.name or "upload.log")
            dst = work_dir / safe_name
        shutil.copy2(src_path, dst)
        logger.info(f"[mc_log] 已复制本地文件到工作目录: src={src_path.name}, dst={dst.name}")
        return dst

    def _detect_file_name_dummy(self, file_comp: Comp.File) -> str:
        name = str(getattr(file_comp, "name", "") or "").strip()
        if name:
            return name
        file_field = str(getattr(file_comp, "file_", "") or "").strip()
        if file_field:
            base = Path(file_field).name
            if base:
                return base
        url = str(getattr(file_comp, "url", "") or "").strip()
        if url:
            parsed = urlparse(url)
            base = Path(parsed.path).name
            if base:
                return base
        return ""

    async def _download_url_to_path(self, url: str, dst: Path, timeout_sec: float):
        headers = {"User-Agent": "AstrBot-MC-Log-Analyzer/1.0"}
        timeout = aiohttp.ClientTimeout(total=timeout_sec)
        async with aiohttp.ClientSession(trust_env=True, timeout=timeout) as session:
            async with session.get(url, headers=headers) as resp:
                resp.raise_for_status()
                dst.parent.mkdir(parents=True, exist_ok=True)
                with open(dst, "wb") as f:
                    async for chunk in resp.content.iter_chunked(64 * 1024):
                        if chunk:
                            f.write(chunk)

    async def _extract_content(
        self,
        local_file: Path,
        is_archive: bool,
        work_dir: Path,
        event: AstrMessageEvent,
        map_provider_id: str,
        run_id: str = "",
        deadline: float | None = None,
    ) -> tuple[str, str, str, bool]:
        self._ensure_not_timed_out(deadline, run_id=run_id, stage="extract_start")
        logger.info(
            f"[mc_log][{run_id}] 开始提取内容: file={local_file.name}, is_archive={is_archive}"
        )
        if is_archive:
            return await self._extract_from_archive(
                local_file,
                work_dir,
                event,
                map_provider_id=map_provider_id,
                run_id=run_id,
                deadline=deadline,
            )
        name_lower = local_file.name.lower()
        strategy = self._strategy_from_text_name(name_lower)
        logger.info(
            f"[mc_log][{run_id}] 文本文件策略判定: file={local_file.name}, strategy={strategy}"
        )
        if strategy == "A":
            kind = "hs_err" if "hs_err" in name_lower else "crash"
            content = self._strategy_a_extract(local_file, kind)
            return content, local_file.name, strategy, False
        elif strategy == "B":
            content = self._strategy_b_extract(local_file)
            return content, local_file.name, strategy, False
        else:
            content, skip_final_analyze = await self._strategy_c_extract(
                local_file,
                event,
                map_provider_id=map_provider_id,
                run_id=run_id,
                deadline=deadline,
            )
            return content, local_file.name, strategy, skip_final_analyze

    async def _extract_from_archive(
        self,
        archive_path: Path,
        work_dir: Path,
        event: AstrMessageEvent,
        map_provider_id: str,
        run_id: str = "",
        deadline: float | None = None,
    ) -> tuple[str, str, str, bool]:
        self._ensure_not_timed_out(deadline, run_id=run_id, stage="extract_archive_start")
        ext = archive_path.suffix.lower()
        extracted_paths: list[Path] = []
        logger.info(f"[mc_log][{run_id}] 开始解压归档文件: file={archive_path.name}, ext={ext}")

        if ext == ".zip":
            extracted_paths = self._safe_extract_zip(archive_path, work_dir / "unzipped")
        elif ext == ".gz":
            out_path = self._safe_extract_gz(archive_path, work_dir / "ungz")
            extracted_paths = [out_path] if out_path else []

        if not extracted_paths:
            raise RuntimeError("archive has no extractable file")
        logger.info(f"[mc_log][{run_id}] 解压完成: extracted_count={len(extracted_paths)}")

        selected = self._pick_priority_file(extracted_paths)
        if not selected:
            raise RuntimeError("archive has no matching log file")
        logger.info(f"[mc_log][{run_id}] 归档内选中文件: {selected.name}")

        lower = selected.name.lower()
        strategy = self._strategy_from_text_name(lower)
        logger.info(
            f"[mc_log][{run_id}] 归档文件策略判定: file={selected.name}, strategy={strategy}"
        )
        if strategy == "A":
            kind = "hs_err" if "hs_err" in lower else "crash"
            content = self._strategy_a_extract(selected, kind)
            return content, selected.name, strategy, False
        if strategy == "B":
            content = self._strategy_b_extract(selected)
            return content, selected.name, strategy, False

        content, skip_final_analyze = await self._strategy_c_extract(
            selected,
            event,
            map_provider_id=map_provider_id,
            run_id=run_id,
            deadline=deadline,
        )
        return content, selected.name, strategy, skip_final_analyze

    def _safe_extract_zip(self, zip_path: Path, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        extracted: list[Path] = []
        total_bytes = 0

        with zipfile.ZipFile(zip_path, "r") as zf:
            infos = zf.infolist()
            if len(infos) > MAX_ARCHIVE_FILE_COUNT:
                raise RuntimeError("archive entry count exceeded")

            base = out_dir.resolve()
            for info in infos:
                if info.is_dir():
                    continue
                if self._zipinfo_is_link(info):
                    continue
                if info.file_size > MAX_ARCHIVE_SINGLE_FILE_BYTES:
                    continue
                total_bytes += info.file_size
                if total_bytes > MAX_ARCHIVE_TOTAL_BYTES:
                    raise RuntimeError("archive total size exceeded")

                safe_rel = Path(info.filename)
                target = (out_dir / safe_rel).resolve()
                if not str(target).startswith(str(base)):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)

                written = 0
                with zf.open(info, "r") as src, open(target, "wb") as dst:
                    while True:
                        chunk = src.read(64 * 1024)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > MAX_ARCHIVE_SINGLE_FILE_BYTES:
                            raise RuntimeError("archive single file extracted size exceeded")
                        dst.write(chunk)
                extracted.append(target)
        return extracted

    def _zipinfo_is_link(self, info: zipfile.ZipInfo) -> bool:
        mode = (info.external_attr >> 16) & 0xF000
        return mode in {0xA000, 0x6000, 0x2000, 0x1000, 0xC000}

    def _safe_extract_gz(self, gz_path: Path, out_dir: Path) -> Path | None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = gz_path.stem or f"unzipped_{uuid.uuid4().hex[:6]}.log"
        target = out_dir / re.sub(r"[^\w\-.]", "_", out_name)
        size = 0
        with gzip.open(gz_path, "rb") as src, open(target, "wb") as dst:
            while True:
                chunk = src.read(64 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_GZ_OUTPUT_BYTES:
                    raise RuntimeError("gz extracted size exceeded")
                dst.write(chunk)
        return target if target.exists() else None

    def _pick_priority_file(self, files: Iterable[Path]) -> Path | None:
        priority = [
            ("hs_err", "A"),
            ("crash", "A"),
            ("游戏崩溃", "C"),
            ("debug", "B"),
            ("日志", "B"),
            ("log", "B"),
            ("latest", "C"),
            ("fcl", "C"),
            ("pcl", "C"),
        ]
        files_list = list(files)
        lowered = [(p, p.name.lower()) for p in files_list]
        for key, _ in priority:
            for path, name in lowered:
                if key in name:
                    return path
        logger.debug(
            f"[mc_log] 归档内未找到优先关键词文件: {[p.name for p in files_list]}"
        )
        return None

    def _strategy_from_text_name(self, name_lower: str) -> str:
        if "hs_err" in name_lower or "crash" in name_lower:
            return "A"
        if "游戏崩溃" in name_lower:
            return "C"
        if "debug" in name_lower or "日志" in name_lower or "log" in name_lower:
            return "B"
        if "latest" in name_lower or "fcl" in name_lower or "pcl" in name_lower:
            return "C"
        return "C"

    def _read_text_with_fallback(self, path: Path) -> str:
        raw = path.read_bytes()
        for encoding in ("utf-8-sig", "utf-16", "gbk"):
            try:
                return raw.decode(encoding)
            except Exception:
                continue
        return raw.decode("utf-8", errors="replace")

    def _strategy_a_extract(self, path: Path, kind: str) -> str:
        content = self._read_text_with_fallback(path)
        if len(content) <= self.cfg["full_read_char_limit"]:
            return self._apply_total_budget(content, self.cfg["total_char_limit"])

        key_sections = self._extract_strategy_a_key_sections(content, kind)
        merged = "\n\n".join(key_sections).strip()
        if not merged:
            merged = self._build_error_focused_text(content)
        return self._apply_total_budget(merged, self.cfg["total_char_limit"])

    def _extract_strategy_a_key_sections(self, content: str, kind: str) -> list[str]:
        lines = content.splitlines()
        patterns_hs_err = [
            ("fatal error", re.compile(r"fatal error|A fatal error has been detected", re.I)),
            ("problematic frame", re.compile(r"Problematic frame|siginfo", re.I)),
            ("java/native frames", re.compile(r"Java frames|Native frames", re.I)),
            ("vm arguments", re.compile(r"VM Arguments|Command Line", re.I)),
            ("system info", re.compile(r"OS:|CPU:|Memory:|Host:", re.I)),
        ]
        patterns_crash = [
            ("header/description", re.compile(r"Description:|---- Minecraft Crash Report ----", re.I)),
            ("stacktrace", re.compile(r"Stacktrace:|Exception Details:", re.I)),
            ("caused by chain", re.compile(r"Caused by:", re.I)),
            ("suspected mods", re.compile(r"Suspected Mods|Mod File|Mod List", re.I)),
            ("system details", re.compile(r"System Details|-- System Details --", re.I)),
        ]
        patterns = patterns_hs_err if kind == "hs_err" else patterns_crash

        sections = []
        for label, pat in patterns:
            idx = self._first_match_index(lines, pat)
            if idx is None:
                continue
            block = lines[idx : idx + MAX_SECTION_LINES]
            if not block:
                continue
            sections.append(f"[{label}]\n" + "\n".join(block))
        return sections

    def _first_match_index(self, lines: list[str], pattern: re.Pattern) -> int | None:
        for i, line in enumerate(lines):
            if pattern.search(line):
                return i
        return None

    def _strategy_b_extract(self, path: Path) -> str:
        content = self._read_text_with_fallback(path)
        lines = content.splitlines()
        if len(lines) <= 1000:
            return self._apply_total_budget(content, self.cfg["total_char_limit"])

        selected = set(range(0, 500))
        selected.update(range(max(0, len(lines) - 500), len(lines)))
        selected.update(self._collect_error_window_indexes(lines, window=8, max_hits=300))
        merged = self._compose_lines_with_gaps(lines, sorted(selected))
        return self._apply_total_budget(merged, self.cfg["total_char_limit"])

    async def _strategy_c_extract(
        self,
        path: Path,
        event: AstrMessageEvent,
        map_provider_id: str,
        run_id: str = "",
        deadline: float | None = None,
    ) -> tuple[str, bool]:
        self._ensure_not_timed_out(deadline, run_id=run_id, stage="strategy_c_start")
        content = self._read_text_with_fallback(path)
        chunks = self._smart_chunk_text(content, self.cfg["chunk_size"])
        if not chunks:
            return "", False

        selected = self._select_chunks_for_map(chunks, self.cfg["max_map_chunks"])
        if len(chunks) > len(selected):
            logger.info(
                f"[mc_log][{run_id}] C策略分块截断: total={len(chunks)}, selected={len(selected)}"
            )

        mapped_results = []
        llm_degrade_count = 0
        threshold = self.cfg["map_timeout_break_threshold"]
        hit_degrade_threshold = False
        for idx, chunk in selected:
            self._ensure_not_timed_out(deadline, run_id=run_id, stage=f"map_chunk_{idx}")
            summary, llm_degraded = await self._map_chunk(
                event,
                chunk,
                idx,
                len(chunks),
                map_provider_id,
                run_id=run_id,
                deadline=deadline,
            )
            if llm_degraded:
                llm_degrade_count += 1
                if llm_degrade_count >= threshold:
                    hit_degrade_threshold = True
                    logger.warning(
                        f"[mc_log][{run_id}] Map阶段降级达到阈值: "
                        f"count={llm_degrade_count}, threshold={threshold}; "
                        "后续分块将停止处理并走降级路径"
                    )
                    break
            if summary and summary.strip() != MR_SKIP:
                mapped_results.append(summary.strip())

        skip_final_analyze = (
            hit_degrade_threshold and self.cfg["skip_final_analyze_on_map_timeout"]
        )
        if not mapped_results:
            logger.warning(f"[mc_log][{run_id}] Map阶段无有效结果，使用启发式降级文本")
            heuristic = self._build_error_focused_text(content)
            return self._apply_total_budget(heuristic, self.cfg["total_char_limit"]), skip_final_analyze

        reduced = self._reduce_map_results(mapped_results)
        return self._apply_total_budget(reduced, self.cfg["total_char_limit"]), skip_final_analyze

    def _select_chunks_for_map(
        self, chunks: list[str], max_chunks: int
    ) -> list[tuple[int, str]]:
        indexed = list(enumerate(chunks, start=1))
        if len(indexed) <= max_chunks:
            return indexed

        scored: list[tuple[int, int, str]] = []
        for idx, chunk in indexed:
            score = len(ERROR_LINE_RE.findall(chunk))
            if "Exception" in chunk or "Caused by" in chunk:
                score += 3
            if "ERROR" in chunk:
                score += 2
            scored.append((score, idx, chunk))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        top = scored[:max_chunks]
        top.sort(key=lambda x: x[1])
        return [(idx, chunk) for _, idx, chunk in top]

    def _smart_chunk_text(self, content: str, chunk_size: int) -> list[str]:
        chunks: list[str] = []
        n = len(content)
        pos = 0
        while pos < n:
            end = min(pos + chunk_size, n)
            if end < n:
                nl = content.rfind("\n", pos, end)
                if nl > pos:
                    end = nl + 1
            if end <= pos:
                end = min(pos + chunk_size, n)

            chunk = content[pos:end]
            if re.search(r"(Exception|Caused by)", chunk, re.IGNORECASE) and end < n:
                extended_end = end
                while extended_end < n:
                    next_nl = content.find("\n", extended_end)
                    if next_nl == -1:
                        extended_end = n
                        break
                    line = content[extended_end:next_nl]
                    if not line.strip() or TIMESTAMP_LINE_RE.search(line):
                        break
                    extended_end = next_nl + 1
                    if extended_end - pos > int(chunk_size * 1.6):
                        break
                chunk = content[pos:extended_end]
                end = extended_end

            chunks.append(chunk)
            pos = end
        return chunks

    async def _map_chunk(
        self,
        event: AstrMessageEvent,
        chunk: str,
        idx: int,
        total: int,
        map_provider_id: str,
        run_id: str = "",
        deadline: float | None = None,
    ) -> tuple[str, bool]:
        if not ERROR_LINE_RE.search(chunk):
            return MR_SKIP, False
        map_system = self._get_prompt("map_system")
        map_user_tpl = self._get_prompt("map_user")
        if not map_system or not map_user_tpl:
            return self._heuristic_map(chunk), False

        prompt = self._render_prompt(
            map_user_tpl,
            {
                "idx": str(idx),
                "total": str(total),
                "mr_skip": MR_SKIP,
                "chunk": chunk,
            },
        )
        try:
            timeout_sec = min(
                float(self.cfg["map_llm_timeout_sec"]),
                max(0.1, self._time_left(deadline)),
            )
            llm_resp = await asyncio.wait_for(
                self.context.llm_generate(
                    chat_provider_id=map_provider_id,
                    system_prompt=map_system,
                    prompt=prompt,
                ),
                timeout=timeout_sec,
            )
            summary = (llm_resp.completion_text or "").strip()
            if not summary:
                return MR_SKIP, False
            return summary, False
        except asyncio.TimeoutError:
            logger.warning(
                f"[mc_log][{run_id}] Map分块LLM超时: idx={idx}/{total}, timeout={self.cfg['map_llm_timeout_sec']}s"
            )
            if self._time_left(deadline) <= 0:
                raise TimeoutError("global timeout reached during map")
            return self._heuristic_map(chunk), True
        except Exception as exc:
            logger.warning(f"[mc_log][{run_id}] Map分块LLM失败: idx={idx}/{total}, err={exc}")
            return self._heuristic_map(chunk), True

    def _heuristic_map(self, chunk: str) -> str:
        lines = chunk.splitlines()
        idxs = self._collect_error_window_indexes(lines, window=4, max_hits=80)
        if not idxs:
            return MR_SKIP
        return self._compose_lines_with_gaps(lines, idxs)

    def _reduce_map_results(self, mapped: list[str]) -> str:
        deduped = []
        seen = set()
        for item in mapped:
            sig = re.sub(r"\s+", " ", item.lower()).strip()
            sig = re.sub(r"0x[0-9a-f]+", "0x*", sig)
            sig = sig[:280]
            if sig in seen:
                continue
            seen.add(sig)
            deduped.append(item)
        return "\n\n".join(deduped)

    def _collect_error_window_indexes(
        self, lines: list[str], window: int, max_hits: int
    ) -> list[int]:
        hits = []
        for i, line in enumerate(lines):
            if ERROR_LINE_RE.search(line):
                hits.append(i)
                if len(hits) >= max_hits:
                    break
        selected = set()
        for i in hits:
            lo = max(0, i - window)
            hi = min(len(lines), i + window + 1)
            selected.update(range(lo, hi))
        return sorted(selected)

    def _compose_lines_with_gaps(self, lines: list[str], indexes: list[int]) -> str:
        if not indexes:
            return ""
        result = []
        prev = -2
        for idx in indexes:
            if idx < 0 or idx >= len(lines):
                continue
            if idx != prev + 1:
                result.append("...[中间内容已省略]...")
            result.append(lines[idx])
            prev = idx
        return "\n".join(result).strip()

    def _apply_total_budget(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        focused = self._build_error_focused_text(text)
        if len(focused) <= limit:
            return focused
        return focused[:limit]

    def _build_error_focused_text(self, text: str) -> str:
        lines = text.splitlines()
        if not lines:
            return ""

        selected = set()
        selected.update(range(0, min(len(lines), 180)))
        selected.update(range(max(0, len(lines) - 180), len(lines)))
        selected.update(self._collect_error_window_indexes(lines, window=8, max_hits=400))

        ts_count = 0
        for i, line in enumerate(lines):
            if TIMESTAMP_LINE_RE.search(line):
                selected.add(i)
                ts_count += 1
                if ts_count >= 240:
                    break

        merged = self._compose_lines_with_gaps(lines, sorted(selected))
        return merged or "\n".join(lines[:500])

    async def _analyze_with_llm(
        self,
        event: AstrMessageEvent,
        source_name: str,
        strategy: str,
        content: str,
        analyze_provider_id: str,
        skip_due_to_map_timeouts: bool = False,
        run_id: str = "",
        deadline: float | None = None,
    ) -> str | None:
        if skip_due_to_map_timeouts:
            logger.warning(
                f"[mc_log][{run_id}] 因Map阶段降级达到阈值，跳过最终LLM分析"
            )
            return None

        toolset = self._build_toolset(["search_mcmod", "search_minecraft_wiki"])
        system_prompt = self._get_prompt("analyze_system")
        analyze_user_tpl = self._get_prompt("analyze_user")
        if not system_prompt or not analyze_user_tpl:
            logger.error("[mc_log] 最终分析失败：分析提示词缺失")
            return None
        user_prompt = self._render_prompt(
            analyze_user_tpl,
            {
                "source_name": source_name,
                "strategy": strategy,
                "content": content,
            },
        )
        try:
            timeout_sec = min(
                float(self.cfg["analyze_llm_timeout_sec"]),
                max(0.1, self._time_left(deadline)),
            )
            llm_resp = await asyncio.wait_for(
                self.context.tool_loop_agent(
                    event=event,
                    chat_provider_id=analyze_provider_id,
                    system_prompt=system_prompt,
                    prompt=user_prompt,
                    tools=toolset,
                    max_steps=self.cfg["max_tool_calls"],
                    tool_call_timeout=self.cfg["tool_timeout_sec"],
                ),
                timeout=timeout_sec,
            )
            text = (llm_resp.completion_text or "").strip()
            if text:
                return text
            logger.error("[mc_log] 最终分析失败：LLM返回空内容")
        except asyncio.TimeoutError:
            logger.error(
                f"[mc_log][{run_id}] 最终分析LLM超时: {self.cfg['analyze_llm_timeout_sec']}s"
            )
            if self._time_left(deadline) <= 0:
                raise TimeoutError("global timeout reached during analyze")
        except Exception as exc:
            logger.error(f"[mc_log][{run_id}] 最终分析LLM异常: {exc}", exc_info=True)
        return None

    def _build_toolset(self, names: list[str]) -> ToolSet | None:
        manager = self.context.get_llm_tool_manager()
        toolset = ToolSet()
        for name in names:
            tool = manager.get_func(name)
            if tool:
                toolset.add_tool(tool)
        return toolset if len(toolset) > 0 else None

    async def _render_report(
        self,
        markdown_text: str,
        run_id: str = "",
        deadline: float | None = None,
    ) -> tuple[str, str]:
        mode = self.cfg["render_mode"]
        if mode == "html_to_image":
            try:
                timeout_sec = min(
                    float(self.cfg["html_render_timeout_sec"]),
                    max(0.1, self._time_left(deadline)),
                )
                url = await asyncio.wait_for(
                    self._render_markdown_html(markdown_text),
                    timeout=timeout_sec,
                )
                if url:
                    return "image_url", url
            except asyncio.TimeoutError:
                logger.warning(
                    f"[mc_log][{run_id}] HTML渲染超时: {self.cfg['html_render_timeout_sec']}s"
                )
                if self._time_left(deadline) <= 0:
                    raise TimeoutError("global timeout reached during render")
            except Exception as exc:
                logger.warning(f"[mc_log][{run_id}] HTML渲染失败: {exc}")
            fallback = markdown_text + "\n\n" + self._msg("html_render_fallback_notice")
            return "text", fallback

        if mode == "text_to_image":
            try:
                b64 = self._render_text_image_base64(markdown_text)
                return "image_b64", b64
            except Exception as exc:
                logger.warning(f"[mc_log][{run_id}] 文本转图片渲染失败: {exc}")
                fallback = markdown_text + "\n\n" + self._msg("text_render_fallback_notice")
                return "text", fallback

        logger.warning(f"[mc_log][{run_id}] 未知渲染模式: {mode}，回退到HTML渲染")
        try:
            url = await asyncio.wait_for(
                self._render_markdown_html(markdown_text),
                timeout=self.cfg["html_render_timeout_sec"],
            )
            if url:
                return "image_url", url
        except Exception as exc:
            logger.warning(f"[mc_log][{run_id}] 未知模式回退后 HTML渲染仍失败: {exc}")
        fallback = markdown_text + "\n\n" + self._msg("html_render_fallback_notice")
        return "text", fallback

    async def _render_markdown_html(self, markdown_text: str) -> str:
        if not self.html_template_path.exists():
            raise FileNotFoundError(f"html template not found: {self.html_template_path}")
        template = self.html_template_path.read_text(encoding="utf-8")
        time_str = time.strftime("%Y/%m/%d %H:%M", time.localtime())
        image_width = int(self.cfg.get("image_width", 640))
        url = await self.html_render(
            template,
            {
                "message": markdown_text,
                "content": markdown_text,
                "time": time_str,
                "image_width": image_width,
            },
            options={
                "type": "png",
                "quality": None,
                "omit_background": True,
                "full_page": True,
                "viewport_width": image_width,
                "animations": "disabled",
                "caret": "hide",
                "scale": "css",
            },
        )
        return str(url)

    def _render_text_image_base64(self, text: str) -> str:
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow not available")

        lines = []
        for raw_line in text.splitlines():
            wrapped = textwrap.wrap(raw_line, width=50) or [""]
            lines.extend(wrapped)
        if not lines:
            lines = [""]

        font = self._load_font()
        line_height = 30
        width = int(self.cfg.get("image_width", 640))
        margin = 36
        height = margin * 2 + line_height * len(lines) + 24

        image = Image.new("RGB", (width, height), color=(248, 251, 248))
        draw = ImageDraw.Draw(image)
        y = margin
        for line in lines:
            draw.text((margin, y), line, fill=(28, 40, 32), font=font)
            y += line_height

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _load_font(self):
        if not PIL_AVAILABLE:
            return None
        for candidate in (
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ):
            if os.path.exists(candidate):
                try:
                    return ImageFont.truetype(candidate, 22)
                except Exception:
                    continue
        return ImageFont.load_default()

    def _build_forward_response(
        self,
        event: AstrMessageEvent,
        source_name: str,
        strategy: str,
        elapsed: float,
        render_mode: str,
        render_payload: str,
        report_md: str,
    ):
        sender_name = self._msg("forward_sender_name")
        sender_uin = str(event.get_self_id() or "0")
        summary_template = self._msg("summary_template")
        try:
            summary = summary_template.format(
                elapsed=elapsed,
                source_name=source_name,
                strategy=strategy,
            )
        except Exception:
            summary = DEFAULT_CONFIG["messages"]["summary_template"].format(
                elapsed=elapsed,
                source_name=source_name,
                strategy=strategy,
            )
        node1 = Comp.Node(content=[Comp.Plain(summary)], name=sender_name, uin=sender_uin)

        if render_mode == "image_url":
            core_content = [Comp.Image.fromURL(render_payload)]
        elif render_mode == "image_b64":
            core_content = [Comp.Image.fromBase64(render_payload)]
        else:
            core_content = [Comp.Plain(render_payload)]
        node2 = Comp.Node(content=core_content, name=sender_name, uin=sender_uin)

        nodes = [node1, node2]

        return event.chain_result([Comp.Nodes(nodes=nodes)])

    def _redact_text(self, text: str) -> str:
        if not text:
            return text
        out = text
        patterns = [
            (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP]"),
            (re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"), "[EMAIL]"),
            (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.I), "[UUID]"),
            (re.compile(r"\b(access[_-]?token|api[_-]?key|secret|password)\s*[:=]\s*[^\s]+", re.I), r"\1=[REDACTED]"),
            (re.compile(r"[A-Za-z]:\\[^\s:\"'<>|]+"), "[PATH]"),
            (re.compile(r"/(?:home|root|Users|var|etc)/[^\s:\"'<>|]+"), "[PATH]"),
        ]
        for pat, rep in patterns:
            out = pat.sub(rep, out)
        return out

    def _cleanup_stale_temp_dirs(self, hours: int):
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
                    self._safe_remove_dir(child)
            except Exception:
                continue

    def _safe_remove_dir(self, path: Path):
        try:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        except Exception:
            logger.warning("[mc_log] 删除临时目录失败:\n" + traceback.format_exc())
