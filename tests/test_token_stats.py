from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.support import DATA_ROOT, FakeContext, FakeEvent, install_astrbot_stubs

install_astrbot_stubs()

from main import LogAnalyzer
from mc_log.services.token_stats import TokenStatsService

_STATS_FILE = DATA_ROOT / "plugin_data" / "mc_log_token_stats.json"


def _clear_stats_file():
    if _STATS_FILE.exists():
        _STATS_FILE.unlink()


class TokenUsage:
    def __init__(self, input_other=0, input_cached=0, output=0):
        self.input_other = input_other
        self.input_cached = input_cached
        self.output = output

    @property
    def input(self):
        return self.input_other + self.input_cached

    @property
    def total(self):
        return self.input + self.output


class FakeLLMResponse:
    def __init__(self, usage=None):
        self.usage = usage


class TokenStatsServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _clear_stats_file()

    async def test_record_and_get_stats(self):
        svc = TokenStatsService()
        await svc.record("111", "s1", 100, 50, "framework")
        await svc.record("222", "s2", 200, 80, "analysis")
        stats = await svc.get_stats()
        self.assertEqual(stats["total"]["input"], 300)
        self.assertEqual(stats["total"]["output"], 130)
        self.assertEqual(stats["total"]["sum"], 430)
        self.assertEqual(stats["record_count"], 2)
        self.assertIn("111", stats["per_user"])
        self.assertEqual(stats["per_user"]["111"]["sum"], 150)
        self.assertIn("s2", stats["per_session"])
        self.assertEqual(stats["per_session"]["s2"]["sum"], 280)

    async def test_six_hour_window(self):
        svc = TokenStatsService()
        now = time.time()
        svc._records = [
            {"ts": now - 100, "uid": "1", "session_id": "s", "input": 10, "output": 5, "source": "framework"},
            {"ts": now - 7 * 3600, "uid": "2", "session_id": "s", "input": 20, "output": 10, "source": "framework"},
        ]
        stats = await svc.get_stats()
        self.assertEqual(stats["six_hour"]["sum"], 15)
        self.assertEqual(stats["total"]["sum"], 45)

    async def test_persistence_across_reload(self):
        svc1 = TokenStatsService()
        await svc1.record("999", "s9", 500, 200, "framework")
        svc2 = TokenStatsService()
        stats = await svc2.get_stats()
        self.assertEqual(stats["total"]["sum"], 700)
        self.assertIn("999", stats["per_user"])

    async def test_old_records_pruned(self):
        svc = TokenStatsService()
        old_ts = time.time() - 100 * 86400
        svc._records = [
            {"ts": old_ts, "uid": "old", "session_id": "s", "input": 999, "output": 999, "source": "x"},
        ]
        await svc.record("new", "s", 10, 5, "framework")
        stats = await svc.get_stats()
        self.assertNotIn("old", stats["per_user"])
        self.assertEqual(stats["record_count"], 1)

    async def test_empty_stats(self):
        svc = TokenStatsService()
        stats = await svc.get_stats()
        self.assertEqual(stats["total"]["sum"], 0)
        self.assertEqual(stats["record_count"], 0)
        self.assertEqual(stats["per_user"], {})

    async def test_yesterday_window(self):
        svc = TokenStatsService()
        now = time.time()
        yesterday = now - ((int(now) % 86400) + 3600)
        svc._records = [
            {"ts": yesterday, "uid": "1", "session_id": "s", "input": 100, "output": 50, "source": "x"},
            {"ts": now - 100, "uid": "2", "session_id": "s", "input": 10, "output": 5, "source": "x"},
        ]
        stats = await svc.get_stats()
        self.assertEqual(stats["yesterday"]["sum"], 150)


class OnLlmResponseHookTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _clear_stats_file()

    async def asyncSetUp(self):
        self.plugin = LogAnalyzer(FakeContext(), {})

    async def test_records_framework_token_usage(self):
        event = FakeEvent(sender_id="123456", unified_msg_origin="aiocqhttp:GroupMessage:1")
        resp = FakeLLMResponse(usage=TokenUsage(input_other=300, output=120))
        await self.plugin.on_llm_response(event, resp)
        stats = await self.plugin.token_stats.get_stats()
        self.assertEqual(stats["total"]["input"], 300)
        self.assertEqual(stats["total"]["output"], 120)

    async def test_skips_when_usage_none(self):
        event = FakeEvent(sender_id="123456")
        resp = FakeLLMResponse(usage=None)
        await self.plugin.on_llm_response(event, resp)
        stats = await self.plugin.token_stats.get_stats()
        self.assertEqual(stats["total"]["sum"], 0)

    async def test_skips_zero_tokens(self):
        event = FakeEvent(sender_id="123456")
        resp = FakeLLMResponse(usage=TokenUsage(0, 0, 0))
        await self.plugin.on_llm_response(event, resp)
        stats = await self.plugin.token_stats.get_stats()
        self.assertEqual(stats["total"]["sum"], 0)

    async def test_records_with_correct_session_id(self):
        event = FakeEvent(sender_id="111", unified_msg_origin="aiocqhttp:GroupMessage:42")
        resp = FakeLLMResponse(usage=TokenUsage(input_other=100, output=50))
        await self.plugin.on_llm_response(event, resp)
        stats = await self.plugin.token_stats.get_stats()
        self.assertIn("aiocqhttp:GroupMessage:42", stats["per_session"])


class TokenStatsCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _clear_stats_file()

    async def asyncSetUp(self):
        self.plugin = LogAnalyzer(FakeContext(), {})

    async def test_command_outputs_stats(self):
        await self.plugin.token_stats.record("111", "s1", 100, 50, "framework")
        event = FakeEvent(sender_id="admin")
        results = []
        async for r in self.plugin.show_token_stats(event):
            results.append(r)
        self.assertEqual(len(results), 1)
        nodes = results[0].payload[0].nodes
        summary_text = nodes[0].content[0].text
        self.assertIn("Token 用量统计", summary_text)
        self.assertIn("总使用量", summary_text)
        self.assertIn("100", summary_text)
        self.assertIn("50", summary_text)

    async def test_command_shows_per_user(self):
        await self.plugin.token_stats.record("userA", "s1", 200, 100, "framework")
        event = FakeEvent(sender_id="admin")
        results = []
        async for r in self.plugin.show_token_stats(event):
            results.append(r)
        nodes = results[0].payload[0].nodes
        user_node_text = nodes[1].content[0].text
        self.assertIn("userA", user_node_text)
        self.assertIn("300", user_node_text)

    async def test_command_empty_stats(self):
        event = FakeEvent(sender_id="admin")
        results = []
        async for r in self.plugin.show_token_stats(event):
            results.append(r)
        nodes = results[0].payload[0].nodes
        summary_text = nodes[0].content[0].text
        self.assertIn("Token 用量统计", summary_text)
        self.assertIn("0", summary_text)
