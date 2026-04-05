from __future__ import annotations

import asyncio
import json
import re
import time
from urllib.parse import quote_plus

from astrbot.api import logger
from astrbot.core.agent.tool import FunctionTool, ToolSet

from ..config import MAX_ARCHIVE_TOOL_CHARS
from ..domain.detection import looks_like_multiple_paths, resolve_archive_file_request


SEARCH_SOURCE_SPECS = (
    ("github_issues", "GitHub Issues"),
    ("minecraft_forum", "Minecraft Forum"),
    ("modrinth", "Modrinth"),
)
SEARCH_SOURCE_PRIORITY = {key: idx for idx, (key, _) in enumerate(SEARCH_SOURCE_SPECS)}


class ToolRegistry:
    def __init__(self, context, config_manager, runtime, file_adapter, http_adapter, privacy_service):
        self.context = context
        self.config_manager = config_manager
        self.runtime = runtime
        self.file_adapter = file_adapter
        self.http_adapter = http_adapter
        self.privacy_service = privacy_service
        self._tool_registered = False

    def _cfg(self):
        return self.config_manager.get()

    def register_tools(self):
        if self._tool_registered:
            return
        search_sites_tool = FunctionTool(
            name="search_mc_sites",
            description="聚合搜索 GitHub Issues、Minecraft Forum 和 Modrinth 的公开结果，用于补充模组兼容、已知问题和社区排障线索。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要搜索的模组名、模组ID、报错关键词或版本号。"}
                },
                "required": ["query"],
            },
            handler=self.tool_search_mc_sites,
        )
        read_archive_file_tool = FunctionTool(
            name="read_archive_file",
            description="读取当前待分析压缩包里的指定文本文件内容。请优先使用完整路径,仅在当前日志不足以提供充足信息时使用,严禁在日志信息充足时使用。",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "压缩包内文件路径，可从可用文件列表中选择。"},
                    "max_chars": {"type": "integer", "description": "返回内容字符上限，建议 5000-35000。"},
                },
                "required": ["file_path"],
            },
            handler=self.tool_read_archive_file,
        )
        self.context.add_llm_tools(search_sites_tool, read_archive_file_tool)
        self._tool_registered = True
        logger.info("[mc_log] 工具已注册: search_mc_sites, read_archive_file")

    def build_toolset(self, names: list[str]) -> ToolSet | None:
        manager = self.context.get_llm_tool_manager()
        toolset = ToolSet()
        for name in names:
            tool = manager.get_func(name)
            if tool:
                toolset.add_tool(tool)
        return toolset if len(toolset) > 0 else None

    async def tool_search_mc_sites(self, event, query: str) -> str:
        normalized = self.http_adapter.normalize_tool_query(query)
        if not normalized:
            return "query 无效，请提供简短关键词。"

        results_list = await asyncio.gather(
            self.search_github_issues(normalized),
            self.search_minecraft_forum(normalized),
            self.search_modrinth(normalized),
        )
        results = {item["source_key"]: item for item in results_list}
        self.dedupe_search_results(results)
        formatted = self.format_search_results(normalized, results)
        return self.http_adapter.tool_response_sanitize(formatted, "mc_sites", reference_only=True)

    async def search_github_issues(self, query: str) -> dict:
        search_query = f"{query} is:issue"
        url = f"https://api.github.com/search/issues?q={quote_plus(search_query)}&sort=updated&order=desc&per_page=5"
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        return await self._run_site_search(
            source_key="github_issues",
            query=query,
            fetcher=lambda: self._fetch_github_issues(url, headers),
        )

    async def _fetch_github_issues(self, url: str, headers: dict[str, str]) -> list[dict]:
        payload = await self.http_adapter.http_get_json(url, timeout=self._cfg()["tool_timeout_sec"], headers=headers)
        return self.http_adapter.parse_github_issue_search_payload(payload)[:5]

    async def search_minecraft_forum(self, query: str) -> dict:
        encoded = quote_plus(query)
        urls = [
            f"https://www.minecraftforum.net/search?search={encoded}",
            f"https://www.minecraftforum.net/search?search={encoded}&search-thread-title=1",
        ]
        return await self._run_site_search(
            source_key="minecraft_forum",
            query=query,
            fetcher=lambda: self._fetch_minecraft_forum(urls),
        )

    async def _fetch_minecraft_forum(self, urls: list[str]) -> list[dict]:
        for url in urls:
            text = await self.http_adapter.http_get_text(url, timeout=self._cfg()["tool_timeout_sec"])
            items = self.http_adapter.parse_minecraftforum_search_html(text)
            if items:
                return items[:5]
        return []

    async def search_modrinth(self, query: str) -> dict:
        facets = json.dumps([["project_type:mod"]], separators=(",", ":"))
        url = (
            "https://api.modrinth.com/v2/search"
            f"?query={quote_plus(query)}&limit=5&index=relevance&facets={quote_plus(facets)}"
        )
        return await self._run_site_search(
            source_key="modrinth",
            query=query,
            fetcher=lambda: self._fetch_modrinth(url),
        )

    async def _fetch_modrinth(self, url: str) -> list[dict]:
        payload = await self.http_adapter.http_get_json(url, timeout=self._cfg()["tool_timeout_sec"])
        return self.http_adapter.parse_modrinth_search_payload(payload)[:5]

    async def _run_site_search(self, source_key: str, query: str, fetcher) -> dict:
        label = self.source_label(source_key)
        tries = self._cfg()["tool_retry_limit"] + 1
        last_error = ""
        started = time.monotonic()
        logger.info(f"[mc_log][tool] 调用 {source_key}: query={query}, tries={tries}")
        for i in range(tries):
            try:
                items = await fetcher()
                status = "ok" if items else "empty"
                logger.info(
                    f"[mc_log][tool] {source_key} 完成: query={query}, status={status}, raw_results={len(items)}, "
                    f"elapsed={time.monotonic() - started:.2f}s"
                )
                return {
                    "source_key": source_key,
                    "label": label,
                    "status": status,
                    "items": items,
                    "raw_count": len(items),
                    "kept_items": [],
                    "kept_count": 0,
                    "deduped_out": 0,
                    "error": "",
                }
            except Exception as exc:
                last_error = str(exc)
                logger.warning(f"[mc_log][tool] {source_key} 第{i + 1}/{tries}次失败: query={query}, err={exc}")
                if i < tries - 1:
                    await asyncio.sleep(0.2)
        logger.error(
            f"[mc_log][tool] {source_key} 最终失败: query={query}, err={last_error}, "
            f"elapsed={time.monotonic() - started:.2f}s"
        )
        return {
            "source_key": source_key,
            "label": label,
            "status": "failed",
            "items": [],
            "raw_count": 0,
            "kept_items": [],
            "kept_count": 0,
            "deduped_out": 0,
            "error": last_error,
        }

    def source_label(self, source_key: str) -> str:
        for key, label in SEARCH_SOURCE_SPECS:
            if key == source_key:
                return label
        return source_key

    def dedupe_search_results(self, results: dict[str, dict]) -> None:
        seen_urls: dict[str, dict] = {}
        seen_titles: dict[str, dict] = {}
        for source_key, _ in SEARCH_SOURCE_SPECS:
            result = results.get(source_key) or {}
            kept_items: list[dict] = []
            deduped_out = 0
            for raw_item in result.get("items", []):
                item = dict(raw_item)
                item["source_key"] = source_key
                item["source_label"] = result.get("label", self.source_label(source_key))
                item["also_seen_in"] = []
                winner = None
                url_key = item.get("url_key") or ""
                title_key = item.get("title_key") or ""
                if url_key and url_key in seen_urls:
                    winner = seen_urls[url_key]
                elif title_key and title_key in seen_titles:
                    winner = seen_titles[title_key]
                if winner is not None:
                    deduped_out += 1
                    label = item["source_label"]
                    if label != winner.get("source_label") and label not in winner["also_seen_in"]:
                        winner["also_seen_in"].append(label)
                    continue
                kept_items.append(item)
                if url_key:
                    seen_urls[url_key] = item
                if title_key:
                    seen_titles[title_key] = item
            result["kept_items"] = kept_items
            result["kept_count"] = len(kept_items)
            result["deduped_out"] = deduped_out

    def format_search_results(self, query: str, results: dict[str, dict]) -> str:
        raw_total = sum(int(result.get("raw_count", 0)) for result in results.values())
        kept_total = sum(int(result.get("kept_count", 0)) for result in results.values())
        lines = [f"聚合搜索 `{query}`："]
        lines.append(f"【去重统计】原始结果 {raw_total} 条，去重后 {kept_total} 条，移除重复 {max(0, raw_total - kept_total)} 条。")
        for source_key, label in SEARCH_SOURCE_SPECS:
            result = results.get(source_key) or {
                "label": label,
                "status": "failed",
                "raw_count": 0,
                "kept_count": 0,
                "kept_items": [],
                "error": "tool execution missing",
            }
            status = result.get("status")
            if status == "failed":
                lines.append(f"【{label}】失败：{result.get('error') or '未知错误'}")
                continue
            if status == "empty":
                lines.append(f"【{label}】无结果")
                continue
            lines.append(f"【{label}】成功：原始 {result.get('raw_count', 0)} 条，去重后 {result.get('kept_count', 0)} 条")
            kept_items = result.get("kept_items", [])
            if not kept_items:
                lines.append("该站点结果已被更高优先级来源去重。")
                continue
            for idx, item in enumerate(kept_items, start=1):
                lines.extend(self.format_search_item(idx, item))
        return "\n".join(lines)

    def format_search_item(self, idx: int, item: dict) -> list[str]:
        lines = []
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        if item.get("source_key") == "github_issues":
            repo = str(item.get("repo") or "").strip()
            number = int(item.get("number") or 0)
            state = str(item.get("state") or "").strip() or "unknown"
            prefix = f"{repo}#{number} [{state}] " if repo and number else ""
            line = f"{idx}. {prefix}{title} - {url}"
        else:
            line = f"{idx}. {title} - {url}"
        also_seen = item.get("also_seen_in", [])
        if also_seen:
            line += f" (also seen in: {', '.join(also_seen)})"
        lines.append(line)
        summary = str(item.get("summary") or "").strip()
        if summary:
            lines.append(f"摘要: {summary}")
        meta = item.get("meta") or {}
        meta_parts = []
        loaders = meta.get("loaders") or []
        versions = meta.get("versions") or []
        if loaders:
            meta_parts.append("loaders=" + ", ".join(loaders))
        if versions:
            meta_parts.append("versions=" + ", ".join(versions))
        if meta_parts:
            lines.append("附加: " + "; ".join(meta_parts))
        return lines

    async def tool_read_archive_file(self, event, file_path: str, max_chars: int = MAX_ARCHIVE_TOOL_CHARS) -> str:
        logger.info(f"[mc_log][tool] read_archive_file 请求: file_path={file_path}, max_chars={max_chars}")
        file_map = self.runtime.get_active_archive_file_map(event)
        if not file_map:
            return "当前任务没有可读取的压缩包文件。"

        normalized = str(file_path or "").replace("\\", "/").strip().strip("/")
        normalized = re.sub(r"/+", "/", normalized)
        if not normalized:
            return "file_path 无效，请提供可用文件列表中的路径。"
        if looks_like_multiple_paths(normalized):
            return "单次调用只能索要一个文件，请仅提供一个 file_path。"

        consumed, consume_msg = self.runtime.consume_archive_tool_call(event, self._cfg())
        if not consumed:
            return consume_msg

        resolved_key, resolved_path, err = resolve_archive_file_request(file_map, normalized)
        if not resolved_path:
            return err
        if self.runtime.is_primary_source_file(event, resolved_key):
            return f"`{resolved_key}` 已作为当前主分析日志输入，无需重复读取。仅在当前日志无法提供有用信息时再读取其他文件。"

        try:
            raw = await self.file_adapter.read_bytes(resolved_path)
        except Exception as exc:
            logger.warning(f"[mc_log][tool] 读取归档文件失败: path={resolved_path}, err={exc}")
            return f"读取 `{resolved_key}` 失败：{exc}"

        if b"\x00" in raw[:4096]:
            return f"`{resolved_key}` 看起来是二进制文件，无法直接作为文本分析。"

        limit = max(1000, min(int(max_chars or MAX_ARCHIVE_TOOL_CHARS), 80000))
        text = self.privacy_service.guard_for_llm(await self.file_adapter.read_text_with_fallback(resolved_path))
        if len(text) > limit:
            text = text[:limit] + "\n...[内容已截断]..."
        logger.info(
            f"[mc_log][tool] read_archive_file 命中: file={resolved_key}, raw_bytes={len(raw)}, returned_chars={len(text)}"
        )
        return self.http_adapter.tool_response_sanitize(
            f"归档文件 `{resolved_key}` 内容:\n{text}",
            "read_archive_file",
            reference_only=False,
        )

    def build_available_files_prompt_block(self, files: list[str]) -> str:
        limit = max(0, int(self._cfg().get("read_archive_file_limit", 1)))
        if not files:
            return "【可用归档文件列表】\n当前任务无可读取的压缩包文件；不要调用 `read_archive_file`。"
        clipped = files[:120]
        lines = [f"- {name}" for name in clipped]
        if len(files) > len(clipped):
            lines.append(f"- ...(其余 {len(files) - len(clipped)} 个文件已省略)")
        return (
            "【可用归档文件列表】\n"
            f"如需补充证据，可调用工具 `read_archive_file(file_path, max_chars)`；本次最多可调用 {limit} 次，且单次只能读取 1 个文件。\n"
            "仅当当前已提供日志内容不足以支撑定位结论时才调用该工具。若当前日志已能定位问题，请不要调用工具。\n"
            "禁止读取当前主分析日志文件（避免重复读取）。\n"
            + "\n".join(lines)
        )
