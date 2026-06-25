from __future__ import annotations

import shutil
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from tests.support import DATA_ROOT, FakeContext, FakeEvent, install_astrbot_stubs

install_astrbot_stubs()

from mc_log.adapters.files import FileAdapter
from mc_log.adapters.http_tools import HttpToolAdapter
from mc_log.config import ConfigManager
from mc_log.domain.extraction import ExtractionDomain
from mc_log.domain.metrics import MetricsService
from mc_log.domain.privacy import PrivacyService
from mc_log.runtime import PluginRuntime
from mc_log.services.analysis import AnalysisService, NETWORK_SEARCH_NOTICE
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
        self.extraction_domain = ExtractionDomain(self.config_manager.get)
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
            self.extraction_domain,
        )
        self.event = FakeEvent(message_id="tool-mid")

    async def test_registers_unified_search_tool(self):
        self.registry.register_tools()
        self.assertEqual(set(self.context.tools.keys()), {"search_mc_sites", "read_archive_file", "check_mod_fix_status"})

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
        self.assertEqual(captured["tools"], ["search_mc_sites", "read_archive_file", "check_mod_fix_status"])

    async def test_analysis_trims_model_added_sections(self):
        self.registry.register_tools()
        self.context.next_llm_text = "\n".join(
            [
                "好的，我来分析。",
                "",
                "### 核心问题：缺少 Fabric API 导致启动失败",
                "",
                "关键证据：",
                "- E1: ClassNotFoundException: net.fabricmc.fabric.api.event.Event",
                "- E2: Mod foo requires fabric-api",
                "- E3: Loading Minecraft 1.20.1",
                "- E4: Fabric Loader 0.15.11",
                "- E5: 这条应该被裁剪",
                "",
                "解决步骤：",
                "1. 安装匹配版本 Fabric API；依据：E1/E2；可逆：先备份 mods [风险：低] [端：Both]",
                "2. 核对 MC/Loader/模组版本；依据：E3/E4；可逆：记录原版本 [风险：低] [端：Both]",
                "3. 检查客户端和服务端 mods 是否一致；依据：E2；可逆：备份列表 [风险：低] [端：Both]",
                "4. 临时移出 foo 验证；依据：E2；可逆：移回原目录 [风险：中] [端：Both]",
                "5. 这一步应该被裁剪",
                "",
                "补充确认：",
                "- 无",
                "",
                "免责声明：",
                "以上只是模型猜测。",
                "更多建议：",
                "可以随便删除模组。",
                "请注意，以下部分信息来源于网络搜索，仅供参考，请谨慎判断其准确性。",
            ]
        )
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

        self.assertTrue(result.startswith("核心问题："))
        self.assertIn("缺少 Fabric API", result)
        self.assertIn(NETWORK_SEARCH_NOTICE, result)
        self.assertNotIn("好的，我来分析", result)
        self.assertNotIn("E5", result)
        self.assertNotIn("5. 这一步", result)
        self.assertNotIn("免责声明", result)
        self.assertNotIn("更多建议", result)

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

    async def test_read_archive_file_wrong_name_does_not_consume_limit(self):
        path = self.case_root / "latest.log"
        path.write_text("[00:10:02] [main/ERROR]: java.lang.RuntimeException: Boot failed", encoding="utf-8")
        self.runtime.set_active_archive_file_map(self.event, {"logs/latest.log": path})

        missing = await self.registry.tool_read_archive_file(self.event, "missing.log")
        result = await self.registry.tool_read_archive_file(self.event, "logs/latest.log")

        self.assertIn("未找到 `missing.log`", missing)
        self.assertIn("Boot failed", result)
        self.assertNotIn("调用次数已达上限", result)

    async def test_read_archive_file_uses_normal_extraction_strategy(self):
        path = self.case_root / "latest.log"
        lines = ["[00:10:02] [main/INFO]: Minecraft Version: 1.20.1"]
        lines.extend(f"[00:10:{index % 60:02d}] [main/INFO]: bootstrap line {index}" for index in range(1, 220))
        lines.append("[00:13:40] [main/INFO]: CHAT_NOISE_SHOULD_DROP")
        lines.extend(f"[00:14:{index % 60:02d}] [main/INFO]: middle line {index}" for index in range(221, 390))
        lines.extend(
            [
                "[00:20:00] [main/ERROR]: java.lang.RuntimeException: Boot failed",
                "Caused by: java.lang.ClassNotFoundException: com.example.MissingClass",
                "at com.example.Loader.load(Loader.java:42)",
            ]
        )
        path.write_text("\n".join(lines), encoding="utf-8")
        self.runtime.set_active_archive_file_map(self.event, {"logs/latest.log": path})

        result = await self.registry.tool_read_archive_file(self.event, "logs/latest.log", max_chars=35000)

        self.assertIn("提取策略：C", result)
        self.assertIn("Boot failed", result)
        self.assertIn("ClassNotFoundException", result)
        self.assertNotIn("CHAT_NOISE_SHOULD_DROP", result)

    async def test_search_tool_docs_updated(self):
        analyze_system = Path("assets/analyze_system.txt").read_text(encoding="utf-8")
        analyze_user = Path("assets/analyze_user.txt").read_text(encoding="utf-8")
        readme = Path("README.md").read_text(encoding="utf-8")

        self.assertIn("search_mc_sites", analyze_system)
        self.assertIn("check_mod_fix_status", analyze_system)
        self.assertNotIn("search_mcmod", analyze_system)
        self.assertNotIn("search_minecraft_wiki", analyze_system)
        self.assertIn("优先给出“核对并调整相关配置文件", analyze_system)
        self.assertIn("不得因为存在新版就建议更新", analyze_system)
        self.assertIn("禁止把“删除/卸载模组”作为前置建议", analyze_system)
        self.assertIn("优先给“检查并调整配置文件", analyze_user)
        self.assertIn("不能因为有新版就建议更新", analyze_user)
        self.assertIn("“删除/卸载模组”只能作为最后的隔离验证手段", analyze_user)
        self.assertIn("search_mc_sites", readme)
        self.assertIn("check_mod_fix_status", readme)
        self.assertNotIn("search_mcmod", readme)
        self.assertNotIn("search_minecraft_wiki", readme)

    async def test_check_mod_fix_status_likely_fix_from_modrinth_changelog(self):
        self.context.provider_ids.add("provider-a")
        self.config_manager.set_raw_config(
            {"analyze_select_provider": "provider-a", "tool_timeout_sec": 30, "tool_snippet_chars": 5000}
        )
        self.config_manager.reload()
        project = {
            "id": "project-sodium",
            "slug": "sodium",
            "title": "Sodium",
            "loaders": ["fabric"],
            "game_versions": ["1.21.8"],
            "source_url": "https://github.com/CaffeineMC/sodium",
        }
        versions = [
            {
                "id": "v-new",
                "project_id": "project-sodium",
                "version_number": "0.6.14",
                "name": "Sodium 0.6.14",
                "date_published": "2026-06-01T00:00:00Z",
                "loaders": ["fabric"],
                "game_versions": ["1.21.8"],
                "changelog": "Fixed a render thread crash in Sodium when starting Minecraft 1.21.8.",
                "files": [{"filename": "sodium-fabric-0.6.14+mc1.21.8.jar"}],
            },
            {
                "id": "v-current",
                "project_id": "project-sodium",
                "version_number": "0.6.13",
                "name": "Sodium 0.6.13",
                "date_published": "2026-05-01T00:00:00Z",
                "loaders": ["fabric"],
                "game_versions": ["1.21.8"],
                "changelog": "Current release.",
                "files": [{"filename": "sodium-fabric-0.6.13+mc1.21.8.jar"}],
            },
        ]

        async def fake_get_json(url, timeout, headers=None):
            if "api.modrinth.com/v2/project/project-sodium/version" in url:
                return versions
            if "api.modrinth.com/v2/project/sodium" in url:
                return project
            if "api.github.com/repos/" in url:
                return []
            raise AssertionError(f"unexpected url: {url}")

        async def fake_llm_generate(**kwargs):
            return SimpleNamespace(
                completion_text=json.dumps(
                    {
                        "fix_status": "fixed_likely",
                        "update_recommendation": "recommend_update",
                        "evidence": [
                            {
                                "evidence_id": "M1",
                                "quote": "Fixed a render thread crash in Sodium",
                                "why_related": "同为 render thread 与 Sodium 启动崩溃。",
                            }
                        ],
                        "reasoning": "新版 changelog 明确提到相同模块和崩溃类型。",
                    },
                    ensure_ascii=False,
                )
            )

        self.http_adapter.http_get_json = AsyncMock(side_effect=fake_get_json)
        self.context.llm_generate = fake_llm_generate

        raw = await self.registry.tool_check_mod_fix_status(
            self.event,
            problem="启动时崩溃，日志中出现 render thread 和 sodium 相关栈",
            mc_version="1.21.8",
            loader="fabric",
            mods=[
                {
                    "name": "Sodium",
                    "version": "0.6.13",
                    "mod_id": "sodium",
                    "filename": "sodium-fabric-0.6.13+mc1.21.8.jar",
                }
            ],
        )
        payload = json.loads(raw)

        self.assertEqual(payload["results"][0]["fix_status"], "fixed_likely")
        self.assertEqual(payload["results"][0]["update_recommendation"], "recommend_update")
        self.assertEqual(payload["results"][0]["evidence"][0]["evidence_id"], "M1")

    async def test_check_mod_fix_status_caps_generic_bugfix_to_possible(self):
        self.context.provider_ids.add("provider-a")
        self.config_manager.set_raw_config({"analyze_select_provider": "provider-a", "tool_timeout_sec": 30})
        self.config_manager.reload()
        project = {
            "id": "project-example",
            "slug": "examplemod",
            "title": "ExampleMod",
            "loaders": ["fabric"],
            "game_versions": ["1.21.8"],
        }
        versions = [
            {
                "id": "v-new",
                "project_id": "project-example",
                "version_number": "2.0.1",
                "date_published": "2026-06-01T00:00:00Z",
                "loaders": ["fabric"],
                "game_versions": ["1.21.8"],
                "changelog": "Bug fixes and compatibility improvements.",
            },
            {
                "id": "v-current",
                "project_id": "project-example",
                "version_number": "2.0.0",
                "date_published": "2026-05-01T00:00:00Z",
                "loaders": ["fabric"],
                "game_versions": ["1.21.8"],
                "changelog": "Current release.",
            },
        ]

        async def fake_get_json(url, timeout, headers=None):
            if "api.modrinth.com/v2/project/project-example/version" in url:
                return versions
            if "api.modrinth.com/v2/project/examplemod" in url:
                return project
            return []

        async def fake_llm_generate(**kwargs):
            return SimpleNamespace(
                completion_text=json.dumps(
                    {
                        "fix_status": "fixed_likely",
                        "update_recommendation": "recommend_update",
                        "evidence": [{"evidence_id": "M1", "quote": "Bug fixes", "why_related": "提到 bug fixes。"}],
                        "reasoning": "模型误判为明确修复。",
                    },
                    ensure_ascii=False,
                )
            )

        self.http_adapter.http_get_json = AsyncMock(side_effect=fake_get_json)
        self.context.llm_generate = fake_llm_generate

        raw = await self.registry.tool_check_mod_fix_status(
            self.event,
            "启动崩溃",
            "1.21.8",
            "fabric",
            [{"name": "ExampleMod", "version": "2.0.0", "mod_id": "examplemod"}],
        )
        payload = json.loads(raw)

        self.assertEqual(payload["results"][0]["fix_status"], "fixed_possible")
        self.assertEqual(payload["results"][0]["update_recommendation"], "consider_update")
        self.assertIn("generic_changelog_capped", payload["results"][0]["warnings"])

    async def test_check_mod_fix_status_not_found_without_relevant_evidence(self):
        self.context.provider_ids.add("provider-a")
        self.config_manager.set_raw_config({"analyze_select_provider": "provider-a", "tool_timeout_sec": 30})
        self.config_manager.reload()
        project = {
            "id": "project-example",
            "slug": "examplemod",
            "title": "ExampleMod",
            "loaders": ["fabric"],
            "game_versions": ["1.21.8"],
        }
        versions = [
            {
                "id": "v-new",
                "project_id": "project-example",
                "version_number": "2.0.1",
                "date_published": "2026-06-01T00:00:00Z",
                "loaders": ["fabric"],
                "game_versions": ["1.21.8"],
                "changelog": "Added a new config screen.",
            },
            {
                "id": "v-current",
                "project_id": "project-example",
                "version_number": "2.0.0",
                "date_published": "2026-05-01T00:00:00Z",
                "loaders": ["fabric"],
                "game_versions": ["1.21.8"],
                "changelog": "Current release.",
            },
        ]

        async def fake_get_json(url, timeout, headers=None):
            if "api.modrinth.com/v2/project/project-example/version" in url:
                return versions
            if "api.modrinth.com/v2/project/examplemod" in url:
                return project
            return []

        async def fake_llm_generate(**kwargs):
            return SimpleNamespace(
                completion_text=json.dumps(
                    {
                        "fix_status": "not_found",
                        "update_recommendation": "do_not_update",
                        "evidence": [],
                        "reasoning": "更新日志只提到配置界面，与启动崩溃无关。",
                    },
                    ensure_ascii=False,
                )
            )

        self.http_adapter.http_get_json = AsyncMock(side_effect=fake_get_json)
        self.context.llm_generate = fake_llm_generate

        raw = await self.registry.tool_check_mod_fix_status(
            self.event,
            "启动崩溃",
            "1.21.8",
            "fabric",
            [{"name": "ExampleMod", "version": "2.0.0", "mod_id": "examplemod"}],
        )
        payload = json.loads(raw)

        self.assertEqual(payload["results"][0]["fix_status"], "not_found")
        self.assertEqual(payload["results"][0]["update_recommendation"], "do_not_update")

    async def test_check_mod_fix_status_accepts_github_release_asset_correlation(self):
        self.context.provider_ids.add("provider-a")
        self.config_manager.set_raw_config({"analyze_select_provider": "provider-a", "tool_timeout_sec": 30})
        self.config_manager.reload()
        project = {
            "id": "project-example",
            "slug": "examplemod",
            "title": "ExampleMod",
            "loaders": ["fabric"],
            "game_versions": ["1.21.8"],
            "source_url": "https://github.com/example/examplemod",
        }
        versions = [
            {
                "id": "v-new",
                "project_id": "project-example",
                "version_number": "2.0.1+mc1.21.8",
                "date_published": "2026-06-01T00:00:00Z",
                "loaders": ["fabric"],
                "game_versions": ["1.21.8"],
                "changelog": "",
            },
            {
                "id": "v-current",
                "project_id": "project-example",
                "version_number": "2.0.0",
                "date_published": "2026-05-01T00:00:00Z",
                "loaders": ["fabric"],
                "game_versions": ["1.21.8"],
                "changelog": "Current release.",
            },
        ]
        github_releases = [
            {
                "id": 10,
                "tag_name": "release-2026-06",
                "name": "June maintenance",
                "published_at": "2026-06-02T00:00:00Z",
                "html_url": "https://github.com/example/examplemod/releases/tag/release-2026-06",
                "body": "Fixed a render thread crash during startup.",
                "assets": [{"name": "examplemod-fabric-2.0.1+mc1.21.8.jar"}],
            }
        ]

        async def fake_get_json(url, timeout, headers=None):
            if "api.modrinth.com/v2/project/project-example/version" in url:
                return versions
            if "api.modrinth.com/v2/project/examplemod" in url:
                return project
            if "api.github.com/repos/example/examplemod/releases" in url:
                return github_releases
            return []

        async def fake_llm_generate(**kwargs):
            prompt = kwargs["prompt"]
            self.assertIn("G1", prompt)
            self.assertIn("Fixed a render thread crash", prompt)
            return SimpleNamespace(
                completion_text=json.dumps(
                    {
                        "fix_status": "fixed_possible",
                        "update_recommendation": "consider_update",
                        "evidence": [
                            {
                                "evidence_id": "G1",
                                "quote": "Fixed a render thread crash",
                                "why_related": "GitHub release asset 关联到兼容版本。",
                            }
                        ],
                        "reasoning": "GitHub release 与新版文件名关联，且 changelog 提到相似崩溃。",
                    },
                    ensure_ascii=False,
                )
            )

        self.http_adapter.http_get_json = AsyncMock(side_effect=fake_get_json)
        self.context.llm_generate = fake_llm_generate

        raw = await self.registry.tool_check_mod_fix_status(
            self.event,
            "启动时 render thread 崩溃",
            "1.21.8",
            "fabric",
            [{"name": "ExampleMod", "version": "2.0.0", "mod_id": "examplemod"}],
        )
        payload = json.loads(raw)

        self.assertEqual(payload["results"][0]["fix_status"], "fixed_possible")
        self.assertEqual(payload["results"][0]["evidence"][0]["source"], "github_releases")


if __name__ == "__main__":
    unittest.main()
