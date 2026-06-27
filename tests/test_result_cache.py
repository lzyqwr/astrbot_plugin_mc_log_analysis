from __future__ import annotations

import unittest
from unittest.mock import patch

from tests.support import FakeContext, FakeEvent, install_astrbot_stubs

install_astrbot_stubs()

from mc_log.config import ConfigManager
from mc_log.domain.extraction import ExtractionDomain
from mc_log.domain.privacy import PrivacyService
from mc_log.runtime import PluginRuntime
from mc_log.adapters.files import FileAdapter
from mc_log.adapters.http_tools import HttpToolAdapter
from mc_log.services.result_cache import ResultCache
from mc_log.services.tool_registry import ToolRegistry


class ResultCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_store_and_get_all(self):
        cache = ResultCache(ConfigManager({}))
        await cache.store("123456", "分析结果A")
        await cache.store("123456", "分析结果B")
        entries = await cache.get_all("123456")
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["text"], "分析结果B")
        self.assertEqual(entries[1]["text"], "分析结果A")

    async def test_get_all_unknown_uid_returns_empty(self):
        cache = ResultCache(ConfigManager({}))
        self.assertEqual(await cache.get_all("999"), [])

    async def test_store_rejects_empty_uid_or_text(self):
        cache = ResultCache(ConfigManager({}))
        await cache.store("", "text")
        await cache.store("123456", "")
        self.assertEqual(await cache.get_all("123456"), [])

    async def test_expired_entries_filtered_on_get(self):
        cache = ResultCache(ConfigManager({"result_cache_ttl_sec": 60}))
        with patch("mc_log.services.result_cache.time.monotonic") as mock_time:
            mock_time.return_value = 1000.0
            await cache.store("123456", "旧结果")
            mock_time.return_value = 1000.0 + 61.0
            entries = await cache.get_all("123456")
        self.assertEqual(entries, [])

    async def test_non_expired_entries_kept(self):
        cache = ResultCache(ConfigManager({"result_cache_ttl_sec": 1800}))
        with patch("mc_log.services.result_cache.time.monotonic") as mock_time:
            mock_time.return_value = 1000.0
            await cache.store("123456", "结果A")
            mock_time.return_value = 1000.0 + 100.0
            entries = await cache.get_all("123456")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["text"], "结果A")

    async def test_max_per_uid_cap(self):
        cache = ResultCache(ConfigManager({"result_cache_max_per_uid": 2}))
        await cache.store("123456", "结果A")
        await cache.store("123456", "结果B")
        await cache.store("123456", "结果C")
        entries = await cache.get_all("123456")
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["text"], "结果C")
        self.assertEqual(entries[1]["text"], "结果B")

    async def test_cleanup_expired_removes_all_stale(self):
        cache = ResultCache(ConfigManager({"result_cache_ttl_sec": 60}))
        with patch("mc_log.services.result_cache.time.monotonic") as mock_time:
            mock_time.return_value = 1000.0
            await cache.store("111", "old1")
            await cache.store("222", "old2")
            mock_time.return_value = 1000.0 + 61.0
            await cache.store("333", "new")
            await cache.cleanup_expired()
            self.assertEqual(await cache.get_all("111"), [])
            self.assertEqual(await cache.get_all("222"), [])
            self.assertEqual(len(await cache.get_all("333")), 1)

    async def test_uids_are_isolated(self):
        cache = ResultCache(ConfigManager({}))
        await cache.store("111", "用户一结果")
        await cache.store("222", "用户二结果")
        self.assertEqual(len(await cache.get_all("111")), 1)
        self.assertEqual(len(await cache.get_all("222")), 1)
        self.assertEqual((await cache.get_all("111"))[0]["text"], "用户一结果")


class GetCachedAnalysisToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config_manager = ConfigManager({})
        self.runtime = PluginRuntime()
        self.privacy = PrivacyService()
        self.extraction_domain = ExtractionDomain(self.config_manager.get)
        self.file_adapter = FileAdapter(self.config_manager, self.runtime)
        self.http_adapter = HttpToolAdapter(self.config_manager, self.privacy)
        self.context = FakeContext()
        self.result_cache = ResultCache(self.config_manager)
        self.registry = ToolRegistry(
            self.context,
            self.config_manager,
            self.runtime,
            self.file_adapter,
            self.http_adapter,
            self.privacy,
            self.extraction_domain,
            self.result_cache,
        )
        self.event = FakeEvent(message_id="cache-mid")

    async def test_tool_registered_via_add_llm_tools(self):
        self.registry.register_tools()
        self.assertIn("get_cached_analysis", self.context.tools)

    async def test_tool_not_in_analyze_toolset(self):
        self.registry.register_tools()
        toolset = self.registry.build_toolset(
            ["search_mc_sites", "read_archive_file", "check_mod_fix_status"]
        )
        names = [tool.name for tool in toolset.tools]
        self.assertNotIn("get_cached_analysis", names)
        self.assertIn("search_mc_sites", names)

    async def test_handler_returns_cached_results(self):
        await self.result_cache.store("123456", "核心问题：缺少 Fabric API")
        result = await self.registry.tool_get_cached_analysis(self.event, "123456")
        self.assertIn("1 条缓存的日志分析结果", result)
        self.assertIn("核心问题：缺少 Fabric API", result)

    async def test_handler_returns_empty_message_for_unknown_uid(self):
        result = await self.registry.tool_get_cached_analysis(self.event, "999999")
        self.assertIn("暂无日志分析结果", result)

    async def test_handler_rejects_empty_uid(self):
        result = await self.registry.tool_get_cached_analysis(self.event, "  ")
        self.assertIn("uid 无效", result)

    async def test_handler_without_result_cache(self):
        registry = ToolRegistry(
            self.context,
            self.config_manager,
            self.runtime,
            self.file_adapter,
            self.http_adapter,
            self.privacy,
            self.extraction_domain,
            None,
        )
        result = await registry.tool_get_cached_analysis(self.event, "123456")
        self.assertIn("缓存未启用", result)


if __name__ == "__main__":
    unittest.main()
