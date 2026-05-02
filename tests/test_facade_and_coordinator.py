from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from tests.support import DATA_ROOT, FakeContext, FakeEvent, FakeFileComponent, install_astrbot_stubs

install_astrbot_stubs()

from main import LogAnalyzer


class FacadeAndCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.case_root = DATA_ROOT / "facade"
        shutil.rmtree(self.case_root, ignore_errors=True)
        self.case_root.mkdir(parents=True, exist_ok=True)

    async def collect_results(self, agen):
        return [item async for item in agen]

    async def test_send_debug_log(self):
        plugin = LogAnalyzer(FakeContext(), {})
        plugin.runtime.debug_log_path.write_text("debug log", encoding="utf-8")
        event = FakeEvent()
        results = await self.collect_results(plugin.send_debug_log(event))
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].payload)

    async def test_on_message_provider_not_configured(self):
        context = FakeContext(provider_ids={"provider-a"})
        plugin = LogAnalyzer(context, {})
        plugin.prompt_manager.prompts = {
            "analyze_system": "z",
            "analyze_user": "k",
        }
        plugin.prompt_manager.prompts_ready = True
        event = FakeEvent(messages=[FakeFileComponent(name="latest.log", source=str(self.case_root / "latest.log"))])
        results = await self.collect_results(plugin.on_message(event))
        self.assertEqual(len(results), 1)
        self.assertIn("analyze_select_provider", results[0].payload)
        self.assertNotIn("map_select_provider", results[0].payload)

    async def test_on_message_runs_with_analyze_provider_only(self):
        log_path = self.case_root / "latest.log"
        log_path.write_text(
            "\n".join(
                [
                    "[00:10:02] [main/INFO]: Minecraft Version: 1.20.1",
                    "[00:10:03] [main/INFO]: Fabric Loader 0.15.11",
                    "[00:10:05] [main/ERROR]: java.lang.RuntimeException: Boot failed",
                    "[00:10:05] [main/ERROR]: Caused by: java.lang.ClassNotFoundException: com.example.MissingClass",
                    "[00:10:05] [main/ERROR]: at com.example.Loader.load(Loader.java:42)",
                ]
            ),
            encoding="utf-8",
        )
        context = FakeContext(provider_ids={"provider-a"})
        plugin = LogAnalyzer(context, {"analyze_select_provider": "provider-a"})
        plugin.prompt_manager.prompts = {
            "analyze_system": "z",
            "analyze_user": "k {{content}}",
        }
        plugin.prompt_manager.prompts_ready = True
        event = FakeEvent(messages=[FakeFileComponent(name="latest.log", source=str(log_path))])

        results = await self.collect_results(plugin.on_message(event))

        self.assertGreaterEqual(len(results), 2)
        self.assertIn("正在分析", results[0].payload)
        self.assertTrue(results[-1].payload)

    async def test_html_render_downloads_url_image_for_forward_message(self):
        log_path = self.case_root / "latest-local-image.log"
        log_path.write_text(
            "\n".join(
                [
                    "[00:10:02] [main/INFO]: Minecraft Version: 1.20.1",
                    "[00:10:05] [main/ERROR]: java.lang.RuntimeException: Boot failed",
                ]
            ),
            encoding="utf-8",
        )
        context = FakeContext(provider_ids={"provider-a"})
        plugin = LogAnalyzer(context, {"analyze_select_provider": "provider-a"})
        plugin.prompt_manager.prompts = {
            "analyze_system": "z",
            "analyze_user": "k {{content}}",
        }
        plugin.prompt_manager.prompts_ready = True
        image_path = self.case_root / "downloaded-render.png"
        image_path.write_bytes(b"png")

        async def fake_html_render(template, values, options=None):
            return "https://example.com/rendered.png"

        async def fake_download_rendered_image(url):
            self.assertEqual(url, "https://example.com/rendered.png")
            return str(image_path)

        plugin.rendering_adapter.html_render_func = fake_html_render
        plugin.rendering_adapter.download_rendered_image = fake_download_rendered_image
        event = FakeEvent(messages=[FakeFileComponent(name="latest.log", source=str(log_path))])

        results = await self.collect_results(plugin.on_message(event))

        forwarded_nodes = results[-1].payload[0].nodes
        image = forwarded_nodes[1].content[0]
        actual_path = Path(getattr(image, "value", None) or getattr(image, "path", ""))
        self.assertTrue(actual_path.exists())
        self.assertEqual(actual_path.read_bytes(), b"png")

    async def test_plain_text_render_mode_sends_plain_result(self):
        log_path = self.case_root / "latest-plain-text.log"
        log_path.write_text(
            "\n".join(
                [
                    "[00:10:02] [main/INFO]: Minecraft Version: 1.20.1",
                    "[00:10:05] [main/ERROR]: java.lang.RuntimeException: Boot failed",
                ]
            ),
            encoding="utf-8",
        )
        context = FakeContext(provider_ids={"provider-a"})
        context.next_llm_text = "纯文本分析结果"
        plugin = LogAnalyzer(context, {"analyze_select_provider": "provider-a", "render_mode": "plain_text"})
        plugin.prompt_manager.prompts = {
            "analyze_system": "z",
            "analyze_user": "k {{content}}",
        }
        plugin.prompt_manager.prompts_ready = True
        event = FakeEvent(messages=[FakeFileComponent(name="latest.log", source=str(log_path))])

        results = await self.collect_results(plugin.on_message(event))

        self.assertIsInstance(results[-1].payload, str)
        self.assertIn("分析完成", results[-1].payload)
        self.assertIn("纯文本分析结果", results[-1].payload)

    async def test_on_message_runs_for_whitelisted_session(self):
        log_path = self.case_root / "latest-whitelist.log"
        log_path.write_text(
            "\n".join(
                [
                    "[00:10:02] [main/INFO]: Minecraft Version: 1.20.1",
                    "[00:10:03] [main/ERROR]: java.lang.RuntimeException: Boot failed",
                    "[00:10:03] [main/ERROR]: Caused by: java.lang.IllegalStateException: Missing dependency",
                ]
            ),
            encoding="utf-8",
        )
        context = FakeContext(provider_ids={"provider-a"})
        plugin = LogAnalyzer(
            context,
            {"analyze_select_provider": "provider-a", "session_whitelist": ["session-1", "session-2"]},
        )
        plugin.prompt_manager.prompts = {
            "analyze_system": "z",
            "analyze_user": "k {{content}}",
        }
        plugin.prompt_manager.prompts_ready = True
        event = FakeEvent(messages=[FakeFileComponent(name="latest.log", source=str(log_path))], session_id="session-2")

        results = await self.collect_results(plugin.on_message(event))

        self.assertGreaterEqual(len(results), 2)
        self.assertIn("正在分析", results[0].payload)
        self.assertTrue(results[-1].payload)

    async def test_on_message_ignores_non_whitelisted_session(self):
        plugin = LogAnalyzer(FakeContext(), {"session_whitelist": ["session-allow"]})
        plugin.prompt_manager.prompts_ready = False
        event = FakeEvent(
            messages=[FakeFileComponent(name="latest.log", source=str(self.case_root / "latest.log"))],
            session_id="session-blocked",
        )

        results = await self.collect_results(plugin.on_message(event))

        self.assertEqual(results, [])

    async def test_on_message_prompt_missing(self):
        plugin = LogAnalyzer(FakeContext(), {})
        plugin.prompt_manager.prompts_ready = False
        plugin.prompt_manager.load_prompts = lambda: setattr(plugin.prompt_manager, "prompts_ready", False)
        event = FakeEvent(messages=[FakeFileComponent(name="latest.log", source=str(self.case_root / "latest.log"))])
        results = await self.collect_results(plugin.on_message(event))
        self.assertEqual(len(results), 1)
        self.assertIn("模板缺失", results[0].payload)


if __name__ == "__main__":
    unittest.main()
