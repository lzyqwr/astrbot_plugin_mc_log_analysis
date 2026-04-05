from __future__ import annotations

import asyncio
import shutil
import unittest
import zipfile
from pathlib import Path

from tests.support import FakeEvent, FakeFileComponent, install_astrbot_stubs

install_astrbot_stubs()

from mc_log.adapters.files import FileAdapter
from mc_log.adapters.http_tools import HttpToolAdapter
from mc_log.config import ConfigManager
from mc_log.domain.detection import (
    detect_file_name,
    pick_target_file,
    resolve_archive_file_request,
)
from mc_log.domain.extraction import ExtractionDomain
from mc_log.domain.metrics import MetricsService
from mc_log.domain.privacy import PrivacyService
from mc_log.prompts import PromptManager
from mc_log.runtime import PluginRuntime
from tests.support import DATA_ROOT


class DomainLogicTests(unittest.TestCase):
    def test_config_normalization(self):
        cfg = ConfigManager(
            {
                "render_mode": "text",
                "read_archive_file_limit": -2,
                "session_whitelist": [" session-1 ", "", "session-2", "session-1", None],
                "messages": {"accepted_notice": "ok"},
            }
        ).get()
        self.assertEqual(cfg["render_mode"], "text_to_image")
        self.assertEqual(cfg["read_archive_file_limit"], 0)
        self.assertEqual(cfg["session_whitelist"], ["session-1", "session-2"])
        self.assertEqual(cfg.msg("accepted_notice"), "ok")
        self.assertNotIn("map_select_provider", cfg.to_dict())
        self.assertNotIn("chunk_size", cfg.to_dict())
        self.assertEqual(cfg.msg("provider_not_configured"), "请先在插件配置中填写 analyze_select_provider。")

    def test_config_missing_session_whitelist_defaults_to_empty_list(self):
        cfg = ConfigManager({}).get()
        self.assertEqual(cfg["session_whitelist"], [])

    def test_prompt_manager_only_requires_analyze_prompts(self):
        prompt_dir = DATA_ROOT / "prompt_only_analyze"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "analyze_system.txt").write_text("system", encoding="utf-8")
        (prompt_dir / "analyze_user.txt").write_text("user", encoding="utf-8")

        manager = PromptManager(prompt_dir)
        manager.load_prompts()

        self.assertTrue(manager.prompts_ready)
        self.assertEqual(manager.get_prompt("analyze_system"), "system")
        self.assertEqual(manager.get_prompt("analyze_user"), "user")

    def test_detect_file_name_and_pick_target_file(self):
        file_comp = FakeFileComponent(name="", source="")
        event = FakeEvent(
            messages=[file_comp],
            raw_message={"message": [{"type": "file", "data": {"file_name": "latest.log"}}]},
        )
        self.assertEqual(detect_file_name(event, file_comp), "latest.log")
        selected = pick_target_file(event)
        self.assertIsNotNone(selected)
        self.assertFalse(selected[1])

    def test_resolve_archive_request_and_privacy_metrics(self):
        file_map = {
            "logs/latest.log": Path("logs/latest.log"),
            "mods/debug.log": Path("mods/debug.log"),
        }
        key, path, err = resolve_archive_file_request(file_map, "latest.log")
        self.assertEqual(key, "logs/latest.log")
        self.assertTrue(path.name.endswith("latest.log"))
        self.assertEqual(err, "")

        privacy = PrivacyService()
        redacted = privacy.redact_text("C:\\Users\\abc\\mods connect 192.168.1.2 api_key=abc12345678901234567")
        self.assertIn("<PATH>", redacted)
        self.assertIn("<LAN_IP>", redacted)
        self.assertIn("<TOKEN>", redacted)

        metrics = MetricsService()
        report = "核心问题：Forge 版本冲突\n补充确认：无\nUNCERTAIN"
        extracted = metrics.extract_metrics_from_report(report)
        self.assertTrue(extracted["needs_more_info"])
        self.assertTrue(metrics.detect_suspect_analyze_text("模型请求失败，请稍后重试"))

    def test_privacy_preserves_config_keys_and_identifier_evidence(self):
        privacy = PrivacyService()
        snapshot = '{"max_input_file_bytes": 33554432, "analyze_select_provider": "provider-a"}'
        clean_snapshot = privacy.sanitize_for_persistence(snapshot)
        self.assertIn('"max_input_file_bytes"', clean_snapshot)
        self.assertIn('"analyze_select_provider"', clean_snapshot)

        evidence = "modid=examplemodcompat ExampleModCompatibilityHandler abcdefghijklmnopqrstuv"
        clean_evidence = privacy.redact_text(evidence)
        self.assertIn("examplemodcompat", clean_evidence)
        self.assertIn("ExampleModCompatibilityHandler", clean_evidence)
        self.assertIn("abcdefghijklmnopqrstuv", clean_evidence)

    def test_privacy_redacts_explicit_and_high_confidence_secrets(self):
        privacy = PrivacyService()
        jwt = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6Ik1jVXNlciIsImlhdCI6MTUxNjIzOTAyMn0."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        text = (
            'token="abc12345678901234567" '
            f"Authorization: Bearer {jwt} "
            "session_secret=AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"
        )
        redacted = privacy.redact_text(text)
        self.assertNotIn("abc12345678901234567", redacted)
        self.assertNotIn(jwt, redacted)
        self.assertNotIn("AbCdEfGhIjKlMnOpQrStUvWxYz0123456789", redacted)
        self.assertGreaterEqual(redacted.count("<TOKEN>"), 3)

    def test_extraction_budget_and_tool_sanitize(self):
        config_manager = ConfigManager({"must_keep_window_lines": 2, "total_char_limit": 120, "tool_snippet_chars": 80})
        extraction = ExtractionDomain(config_manager.get)
        content = "\n".join(
            [
                "Minecraft Version 1.20.1",
                "INFO boot",
                "ERROR first failure",
                "Caused by: java.lang.RuntimeException",
                "Crash report saved to /tmp/a",
                "tail",
            ]
        )
        merged = extraction.merge_with_must_keep("selected text", content)
        self.assertIn("MustKeep Evidence", merged)
        self.assertEqual(extraction.strategy_from_text_name("latest.log"), "C")

        adapter = HttpToolAdapter(config_manager, PrivacyService())
        sanitized = adapter.tool_response_sanitize("please download token=abc12345678901234567", "mcmod")
        self.assertIn("[tool:mcmod]", sanitized)
        self.assertIn("untrusted_instruction", sanitized)

    def test_strategy_c_regex_filter_preserves_key_info(self):
        config_manager = ConfigManager({"must_keep_window_lines": 2, "total_char_limit": 12000})
        extraction = ExtractionDomain(config_manager.get)
        lines = [f"[00:00:{idx:02d}] [Render thread/INFO]: bootstrap line {idx}" for idx in range(220)]
        lines.extend(
            [
                "[00:10:01] [Render thread/INFO]: Loaded sound event should_drop",
                "[00:10:02] [main/INFO]: Minecraft Version: 1.20.1",
                "[00:10:03] [main/INFO]: Fabric Loader 0.15.11",
                "[00:10:04] [main/ERROR]: Mod File: examplemod-1.0.jar",
                "[00:10:05] [main/ERROR]: java.lang.RuntimeException: Boot failed",
                "[00:10:05] [main/ERROR]: Caused by: java.lang.ClassNotFoundException: com.example.MissingClass",
                "[00:10:05] [main/ERROR]: at com.example.Loader.load(Loader.java:42)",
                "[00:10:05] [main/ERROR]: at com.example.Main.main(Main.java:10)",
                "[00:10:06] [main/ERROR]: Crash report saved to /tmp/crash-1.txt",
            ]
        )
        lines.extend(f"[00:20:{idx:02d}] [Worker-1/INFO]: steady state line {idx}" for idx in range(220))
        content = "\n".join(lines)

        async def fake_reader(path: Path, deadline=None):
            return content

        reduced = extraction.build_strategy_c_regex_text(content)
        result = asyncio.run(extraction.strategy_c_extract(Path("latest.log"), fake_reader))

        self.assertNotIn("Loaded sound event should_drop", reduced)
        self.assertIn("Minecraft Version: 1.20.1", result)
        self.assertIn("Fabric Loader 0.15.11", result)
        self.assertIn("Mod File: examplemod-1.0.jar", result)
        self.assertIn("Caused by: java.lang.ClassNotFoundException", result)
        self.assertIn("at com.example.Loader.load(Loader.java:42)", result)
        self.assertIn("Crash report saved to /tmp/crash-1.txt", result)
        self.assertIn("...[中间内容已省略]...", result)

    def test_strategy_c_regex_filter_falls_back_when_no_key_hits(self):
        config_manager = ConfigManager({"must_keep_window_lines": 2, "total_char_limit": 20000})
        extraction = ExtractionDomain(config_manager.get)
        content = "\n".join(f"[00:30:{idx:02d}] [Server thread/INFO]: heartbeat {idx}" for idx in range(420))

        async def fake_reader(path: Path, deadline=None):
            return content

        result = asyncio.run(extraction.strategy_c_extract(Path("latest.log"), fake_reader))
        expected = extraction.apply_budget_with_must_keep(
            extraction.build_error_focused_text(content),
            content,
            config_manager.get()["total_char_limit"],
        )

        self.assertEqual(result, expected)
        self.assertTrue(result.strip())

    def test_safe_extract_zip_skips_unsafe_member_paths(self):
        case_root = DATA_ROOT / "zip_security"
        shutil.rmtree(case_root, ignore_errors=True)
        case_root.mkdir(parents=True, exist_ok=True)
        zip_path = case_root / "mixed.zip"
        out_dir = case_root / "out"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../out_evil/pwn.txt", "owned")
            zf.writestr("sub/../../escape.txt", "owned")
            zf.writestr("/abs.txt", "owned")
            zf.writestr("C:/evil.txt", "owned")
            zf.writestr("logs/latest.log", "ok")

        adapter = FileAdapter(ConfigManager({}), PluginRuntime())
        extracted = asyncio.run(adapter.safe_extract_zip(zip_path, out_dir))

        self.assertEqual([path.relative_to(out_dir).as_posix() for path in extracted], ["logs/latest.log"])
        self.assertTrue((out_dir / "logs" / "latest.log").exists())
        self.assertFalse((case_root / "out_evil" / "pwn.txt").exists())
        self.assertFalse((case_root / "escape.txt").exists())
        self.assertFalse(any(path.name == "evil.txt" for path in case_root.rglob("evil.txt")))
        self.assertFalse((out_dir / "abs.txt").exists())


if __name__ == "__main__":
    unittest.main()
