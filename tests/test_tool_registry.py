from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from tests.support import DATA_ROOT, FakeContext, FakeEvent, install_astrbot_stubs

install_astrbot_stubs()

from mc_log.adapters.files import FileAdapter
from mc_log.adapters.http_tools import HttpToolAdapter
from mc_log.config import ConfigManager
from mc_log.domain.metrics import MetricsService
from mc_log.domain.privacy import PrivacyService
from mc_log.runtime import PluginRuntime
from mc_log.services.analysis import AnalysisService
from mc_log.services.tool_registry import ToolRegistry


class DummyPromptManager:
    def __init__(self):
        self.prompts = {
            "analyze_system": "system",
            "analyze_user": "user {{content}}",
        }

    def get_prompt(self, key: str) -> str:
        return self.prompts.get(key, "")

    def render_prompt(self, template: str, values: dict[str, str]) -> str:
        out = template
        for key, value in values.items():
            out = out.replace(f"{{{{{key}}}}}", str(value))
        return out


class ToolRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.case_root = DATA_ROOT / "tool_registry"
        shutil.rmtree(self.case_root, ignore_errors=True)
        self.case_root.mkdir(parents=True, exist_ok=True)
        self.config_manager = ConfigManager({"read_archive_file_limit": 1, "tool_snippet_chars": 5000})
        self.runtime = PluginRuntime()
        self.privacy = PrivacyService()
        self.file_adapter = FileAdapter(self.config_manager, self.runtime)
        self.http_adapter = HttpToolAdapter(self.config_manager, self.privacy)
        self.context = FakeContext()
        self.registry = ToolRegistry(
            self.context,
            self.config_manager,
            self.runtime,
            self.file_adapter,
            self.http_adapter,
            self.privacy,
        )
        self.event = FakeEvent(message_id="tool-mid")

    async def test_registers_unified_search_tool(self):
        self.registry.register_tools()
        self.assertEqual(set(self.context.tools.keys()), {"search_mc_sites", "read_archive_file"})

    async def test_analysis_uses_unified_search_tool(self):
        captured = {}
        self.registry.register_tools()

        async def fake_tool_loop_agent(**kwargs):
            captured["tools"] = [tool.name for tool in kwargs.get("tools").tools]
            return SimpleNamespace(completion_text="分析结果")

        self.context.tool_loop_agent = fake_tool_loop_agent
        service = AnalysisService(
            self.context,
            self.config_manager,
            self.runtime,
            DummyPromptManager(),
            self.registry,
            MetricsService(),
        )

        result = await service.analyze_with_llm(
            event=self.event,
            source_name="latest.log",
            strategy="C",
            content="ERROR test",
            available_archive_files=[],
            analyze_provider_id="provider-a",
        )

        self.assertEqual(result, "分析结果")
        self.assertEqual(captured["tools"], ["search_mc_sites", "read_archive_file"])

    async def test_search_mc_sites_success_and_dedup(self):
        github_payload = {
            "items": [
                {
                    "html_url": "https://github.com/example/mod/issues/12",
                    "title": "Sodium crash on launch",
                    "body": "Game crashes during launch when Fabric API is missing.",
                    "repository_url": "https://api.github.com/repos/example/mod",
                    "number": 12,
                    "state": "open",
                },
                {
                    "html_url": "https://github.com/example/mod/pull/99",
                    "title": "This PR should be filtered",
                    "body": "pull request body",
                    "repository_url": "https://api.github.com/repos/example/mod",
                    "number": 99,
                    "state": "open",
                    "pull_request": {"url": "https://api.github.com/repos/example/mod/pulls/99"},
                },
            ]
        }
        modrinth_payload = {
            "hits": [
                {
                    "title": "Sodium crash on launch",
                    "slug": "sodium-crash-on-launch",
                    "project_type": "mod",
                    "description": "Duplicate title from Modrinth should be deduped.",
                    "display_categories": ["fabric"],
                    "versions": ["1.20.1", "1.20.2"],
                }
            ]
        }
        forum_html = """
        <html><body>
        <a href="/forums/support/java-edition-support/12345-sodium-crash-on-launch">Sodium crash on launch</a>
        </body></html>
        """

        async def fake_get_json(url, timeout, headers=None):
            if "api.github.com" in url:
                return github_payload
            if "api.modrinth.com" in url:
                return modrinth_payload
            raise AssertionError(f"unexpected url: {url}")

        self.http_adapter.http_get_json = AsyncMock(side_effect=fake_get_json)
        self.http_adapter.http_get_text = AsyncMock(return_value=forum_html)

        result = await self.registry.tool_search_mc_sites(self.event, "sodium crash")

        self.assertIn("[tool:mc_sites]", result)
        self.assertIn("【去重统计】原始结果 3 条，去重后 1 条，移除重复 2 条。", result)
        self.assertIn("【GitHub Issues】成功：原始 1 条，去重后 1 条", result)
        self.assertIn("example/mod#12 [open] Sodium crash on launch", result)
        self.assertIn("also seen in: Minecraft Forum, Modrinth", result)
        self.assertIn("【Minecraft Forum】成功：原始 1 条，去重后 0 条", result)
        self.assertIn("该站点结果已被更高优先级来源去重。", result)
        self.assertIn("【Modrinth】成功：原始 1 条，去重后 0 条", result)
        self.assertNotIn("This PR should be filtered", result)

    async def test_search_mc_sites_handles_failure_and_empty(self):
        modrinth_payload = {
            "hits": [
                {
                    "title": "FerriteCore fix",
                    "slug": "ferritecore-fix",
                    "project_type": "mod",
                    "description": "Modrinth result survives when others fail.",
                    "display_categories": ["forge"],
                    "versions": ["1.20.1"],
                }
            ]
        }

        async def fake_get_json(url, timeout, headers=None):
            if "api.github.com" in url:
                raise RuntimeError("github offline")
            if "api.modrinth.com" in url:
                return modrinth_payload
            raise AssertionError(f"unexpected url: {url}")

        self.http_adapter.http_get_json = AsyncMock(side_effect=fake_get_json)
        self.http_adapter.http_get_text = AsyncMock(return_value="<html><body>no threads</body></html>")

        result = await self.registry.tool_search_mc_sites(self.event, "ferritecore")

        self.assertIn("【GitHub Issues】失败：github offline", result)
        self.assertIn("【Minecraft Forum】无结果", result)
        self.assertIn("【Modrinth】成功：原始 1 条，去重后 1 条", result)
        self.assertIn("FerriteCore fix", result)

    async def test_search_mc_sites_rejects_empty_query(self):
        result = await self.registry.tool_search_mc_sites(self.event, " \n ")
        self.assertEqual(result, "query 无效，请提供简短关键词。")

    async def test_read_archive_file_without_context(self):
        result = await self.registry.tool_read_archive_file(self.event, "latest.log")
        self.assertIn("没有可读取的压缩包文件", result)

    async def test_read_archive_file_primary_source_and_limit(self):
        path = self.case_root / "latest.log"
        path.write_text("hello", encoding="utf-8")
        self.runtime.set_active_archive_file_map(self.event, {"logs/latest.log": path})
        self.runtime.set_active_primary_source_name(self.event, "latest.log")
        result = await self.registry.tool_read_archive_file(self.event, "logs/latest.log")
        self.assertIn("无需重复读取", result)
        second = await self.registry.tool_read_archive_file(self.event, "logs/latest.log")
        self.assertIn("调用次数已达上限", second)

    async def test_read_archive_file_binary(self):
        path = self.case_root / "bin.dat"
        path.write_bytes(b"\x00\x01\x02abc")
        self.runtime.set_active_archive_file_map(self.event, {"bin.dat": path})
        result = await self.registry.tool_read_archive_file(self.event, "bin.dat")
        self.assertIn("二进制文件", result)

    async def test_read_archive_file_gb18030_text_is_readable(self):
        path = self.case_root / "latest.log"
        text = "[00:10:02] [main/ERROR]: 75服务端启动异常\nCaused by: 缺少依赖"
        path.write_bytes(text.encode("gb18030"))
        self.runtime.set_active_archive_file_map(self.event, {"logs/latest.log": path})

        result = await self.registry.tool_read_archive_file(self.event, "logs/latest.log")

        self.assertIn("75服务端启动异常", result)
        self.assertIn("缺少依赖", result)
        self.assertNotIn("二进制文件", result)

    async def test_search_tool_docs_updated(self):
        analyze_system = Path("assets/analyze_system.txt").read_text(encoding="utf-8")
        readme = Path("README.md").read_text(encoding="utf-8")

        self.assertIn("search_mc_sites", analyze_system)
        self.assertNotIn("search_mcmod", analyze_system)
        self.assertNotIn("search_minecraft_wiki", analyze_system)
        self.assertIn("search_mc_sites", readme)
        self.assertNotIn("search_mcmod", readme)
        self.assertNotIn("search_minecraft_wiki", readme)


if __name__ == "__main__":
    unittest.main()
