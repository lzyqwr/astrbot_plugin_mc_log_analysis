from __future__ import annotations

import sys
import time
from pathlib import Path

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import EventMessageType, event_message_type
from astrbot.api.star import Context, Star, register

if __package__:
    from .mc_log.adapters import FileAdapter, HttpToolAdapter, RenderingAdapter
    from .mc_log.config import ConfigManager
    from .mc_log.domain import ExtractionDomain, MetricsService, PrivacyService
    from .mc_log.prompts import PromptManager
    from .mc_log.runtime import PluginRuntime
    from .mc_log.services import AnalysisService, Coordinator, ReportStore, ResultCache, TokenStatsService, ToolRegistry
else:
    plugin_root = Path(__file__).resolve().parent
    if str(plugin_root) not in sys.path:
        sys.path.insert(0, str(plugin_root))

    from mc_log.adapters import FileAdapter, HttpToolAdapter, RenderingAdapter
    from mc_log.config import ConfigManager
    from mc_log.domain import ExtractionDomain, MetricsService, PrivacyService
    from mc_log.prompts import PromptManager
    from mc_log.runtime import PluginRuntime
    from mc_log.services import AnalysisService, Coordinator, ReportStore, ResultCache, TokenStatsService, ToolRegistry


@register(
    "astrbot_plugin_mc_log_analysis",
    "lzyqwr",
    "Minecraft 日志分析插件",
    "2.5.0",
    "https://github.com/lzyqwr/astrbot_plugin_mc_log_analysis",
)
class LogAnalyzer(Star):
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self._raw_config = config
        self.assets_dir = Path(__file__).resolve().parent / "assets"
        self.html_template_path = self.assets_dir / "html_to_image.html.j2"
        self.prompt_dir = self.assets_dir

        self.config_manager = ConfigManager(config)
        self.runtime = PluginRuntime()
        self.privacy_service = PrivacyService()
        self.metrics_service = MetricsService()
        self.extraction_domain = ExtractionDomain(self.config_manager.get)
        self.prompt_manager = PromptManager(self.prompt_dir)
        self.file_adapter = FileAdapter(self.config_manager, self.runtime)
        self.http_adapter = HttpToolAdapter(self.config_manager, self.privacy_service)
        self.rendering_adapter = RenderingAdapter(
            self.config_manager,
            self.runtime,
            self.html_template_path,
            self.html_render,
        )
        self.result_cache = ResultCache(self.config_manager)
        self.token_stats = TokenStatsService()
        self.tool_registry = ToolRegistry(
            self.context,
            self.config_manager,
            self.runtime,
            self.file_adapter,
            self.http_adapter,
            self.privacy_service,
            self.extraction_domain,
            self.result_cache,
        )
        self.analysis_service = AnalysisService(
            self.context,
            self.config_manager,
            self.runtime,
            self.prompt_manager,
            self.tool_registry,
            self.metrics_service,
            self.token_stats,
        )
        self.report_store = ReportStore(self.config_manager, self.runtime)
        self.coordinator = Coordinator(
            self.context,
            self.config_manager,
            self.runtime,
            self.prompt_manager,
            self.privacy_service,
            self.metrics_service,
            self.extraction_domain,
            self.file_adapter,
            self.rendering_adapter,
            self.analysis_service,
            self.tool_registry,
            self.report_store,
            self.result_cache,
        )
        self._sync_aliases()

    def _sync_aliases(self):
        self.cfg = self.config_manager.get().values
        self.temp_root = self.runtime.temp_root
        self.debug_log_path = self.runtime.debug_log_path
        self.debug_log_dir = self.runtime.debug_log_dir
        self.prompts = self.prompt_manager.prompts
        self.prompts_ready = self.prompt_manager.prompts_ready

    async def initialize(self):
        self.config_manager.reload()
        self._sync_aliases()
        self.prompt_manager.load_prompts()
        self._sync_aliases()
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.debug_log_dir.mkdir(parents=True, exist_ok=True)
        self.tool_registry.register_tools()
        self.runtime.install_record_factory()
        self.runtime.cleanup_stale_temp_dirs(hours=24, remover=self.file_adapter.safe_remove_dir_blocking)
        self.report_store.cleanup_stale_pending(hours=24)
        await self.result_cache.cleanup_expired()
        logger.info("[mc_log] 插件已初始化")

    @filter.on_llm_request()
    async def on_llm_request(self, event, req):
        """框架 LLM 请求前注入：若用户 10 分钟内调用过分析，提示 LLM 主动调用 get_cached_analysis。"""
        if self.result_cache is None:
            return
        uid = str(event.get_sender_id() or "").strip()
        if not uid:
            return
        try:
            recent = await self.result_cache.has_recent(uid, 600.0)
        except Exception as exc:
            logger.warning(f"[mc_log] 检查最近分析缓存失败: {exc}")
            return
        if not recent:
            return
        hint = "[系统状态] 该用户在过去10分钟内调用过崩溃日志分析。若对话涉及该结果,请主动调用 get_cached_analysis 获取,不要凭空作答。"
        if getattr(req, "prompt", None):
            req.prompt = f"{req.prompt}\n\n{hint}"
        else:
            req.prompt = hint
        logger.info(f"[mc_log] 已为 uid={uid} 注入分析缓存提示")

    @filter.on_llm_response()
    async def on_llm_response(self, event, response):
        """框架 LLM 响应后：记录 token 用量。"""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        try:
            input_tokens = int(getattr(usage, "input", 0) or 0)
            output_tokens = int(getattr(usage, "output", 0) or 0)
        except Exception:
            return
        if input_tokens == 0 and output_tokens == 0:
            return
        uid = str(event.get_sender_id() or "")
        session_id = str(getattr(event, "unified_msg_origin", "") or getattr(event, "session_id", "") or "")
        try:
            await self.token_stats.record(uid, session_id, input_tokens, output_tokens, "framework")
        except Exception as exc:
            logger.warning(f"[mc_log] 记录 token 用量失败: {exc}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("token_stats")
    async def show_token_stats(self, event: AstrMessageEvent):
        stats = await self.token_stats.get_stats()
        sender_name = "Token统计"
        sender_uin = str(event.get_self_id() or "0")
        nodes = []

        summary_lines = ["===== Token 用量统计 ====="]
        for label, key in [
            ("总使用量", "total"),
            ("本月使用量", "month"),
            ("昨日使用量", "yesterday"),
            ("近6小时使用量", "six_hour"),
        ]:
            d = stats[key]
            summary_lines.append(f"{label}: 输入 {d['input']} / 输出 {d['output']} / 合计 {d['sum']}")
        summary_lines.append(f"\n记录总数: {stats['record_count']}")
        nodes.append(Comp.Node(content=[Comp.Plain("\n".join(summary_lines))], name=sender_name, uin=sender_uin))

        per_user = stats["per_user"]
        if per_user:
            user_lines = ["--- 各用户使用量 ---"]
            for uid, d in list(per_user.items())[:20]:
                user_lines.append(f"  {uid}: 输入 {d['input']} / 输出 {d['output']} / 合计 {d['sum']}")
            nodes.append(Comp.Node(content=[Comp.Plain("\n".join(user_lines))], name=sender_name, uin=sender_uin))

        per_session = stats["per_session"]
        if per_session:
            session_lines = ["--- 各会话使用量 ---"]
            for sid, d in list(per_session.items())[:20]:
                session_lines.append(f"  {sid}: 输入 {d['input']} / 输出 {d['output']} / 合计 {d['sum']}")
            nodes.append(Comp.Node(content=[Comp.Plain("\n".join(session_lines))], name=sender_name, uin=sender_uin))

        result = event.chain_result([Comp.Nodes(nodes=nodes)])
        result.stop_event()
        yield result

    async def terminate(self):
        await self.runtime.stop_all_debug_captures()
        self.runtime.uninstall_record_factory()
        logger.info("[mc_log] 插件已停止")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("debug")
    async def send_debug_log(self, event: AstrMessageEvent):
        path = self.runtime.resolve_latest_debug_log_path()
        if not path or not path.exists():
            result = event.plain_result("暂无调试日志，请先触发一次日志分析。")
            result.stop_event()
            yield result
            return
        if path.stat().st_size <= 0:
            result = event.plain_result("调试日志为空，请先触发一次日志分析。")
            result.stop_event()
            yield result
            return

        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(path.stat().st_mtime))
        result = event.chain_result(
            [
                Comp.Plain("这是最近一次触发的单次调试日志。"),
                Comp.File(name=f"mc_log_debug_{stamp}.log", file=str(path)),
            ]
        )
        result.stop_event()
        yield result

    @filter.command("上报")
    async def report_run(self, event: AstrMessageEvent):
        self.config_manager.reload()
        self._sync_aliases()
        if not self.report_store.enabled():
            result = event.plain_result(self.config_manager.msg("report_disabled"))
            result.stop_event()
            yield result
            return
        report = await self.report_store.find_recent(
            str(event.get_session_id() or "")
        )
        if not report:
            result = event.plain_result(self.config_manager.msg("report_no_run"))
            result.stop_event()
            yield result
            return
        zip_path = await self.report_store.package_report(report)
        if not zip_path or not zip_path.exists():
            result = event.plain_result(
                self.config_manager.msg("report_packaged_failed")
            )
            result.stop_event()
            yield result
            return
        notice = self.config_manager.msg("report_packaged")
        summary = (
            f"{notice}\n运行ID: {report.run_id}\n文件: {report.source_name}\n"
            f"时间: {report.timestamp}\n存档: {zip_path}"
        )
        result = event.chain_result(
            [
                Comp.Plain(summary),
                Comp.File(name=zip_path.name, file=str(zip_path)),
            ]
        )
        result.stop_event()
        yield result

    @event_message_type(EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        self.config_manager.reload()
        self._sync_aliases()
        async for result in self.coordinator.handle_message(event):
            yield result
        self._sync_aliases()
