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
    from .mc_log.services import AnalysisService, Coordinator, ToolRegistry
else:
    plugin_root = Path(__file__).resolve().parent
    if str(plugin_root) not in sys.path:
        sys.path.insert(0, str(plugin_root))

    from mc_log.adapters import FileAdapter, HttpToolAdapter, RenderingAdapter
    from mc_log.config import ConfigManager
    from mc_log.domain import ExtractionDomain, MetricsService, PrivacyService
    from mc_log.prompts import PromptManager
    from mc_log.runtime import PluginRuntime
    from mc_log.services import AnalysisService, Coordinator, ToolRegistry


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
        self.tool_registry = ToolRegistry(
            self.context,
            self.config_manager,
            self.runtime,
            self.file_adapter,
            self.http_adapter,
            self.privacy_service,
        )
        self.analysis_service = AnalysisService(
            self.context,
            self.config_manager,
            self.runtime,
            self.prompt_manager,
            self.tool_registry,
            self.metrics_service,
        )
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
        logger.info("[mc_log] 插件已初始化")

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

    @event_message_type(EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        self.config_manager.reload()
        self._sync_aliases()
        async for result in self.coordinator.handle_message(event):
            yield result
        self._sync_aliases()
