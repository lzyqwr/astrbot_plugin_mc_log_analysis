from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from astrbot.api import logger
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from ..domain.detection import detect_file_name, pick_target_file
from ..models import AnalysisResult, BudgetExceeded, ExtractionResult, RenderResult, RunContext


class Coordinator:
    def __init__(
        self,
        context,
        config_manager,
        runtime,
        prompt_manager,
        privacy_service,
        metrics_service,
        extraction_domain,
        file_adapter,
        rendering_adapter,
        analysis_service,
        tool_registry,
    ):
        self.context = context
        self.config_manager = config_manager
        self.runtime = runtime
        self.prompt_manager = prompt_manager
        self.privacy_service = privacy_service
        self.metrics_service = metrics_service
        self.extraction_domain = extraction_domain
        self.file_adapter = file_adapter
        self.rendering_adapter = rendering_adapter
        self.analysis_service = analysis_service
        self.tool_registry = tool_registry
        self._metrics_lock = asyncio.Lock()

    def _cfg(self):
        return self.config_manager.get()

    def _msg(self, key: str) -> str:
        return self.config_manager.msg(key)

    def configured_analyze_provider_id(self) -> str:
        return str(self._cfg().get("analyze_select_provider", "") or "").strip()

    def is_session_whitelisted(self, event, cfg) -> bool:
        whitelist = cfg.get("session_whitelist", [])
        if not whitelist:
            return True
        session_id = str(event.get_session_id() or "").strip()
        return session_id in whitelist

    async def write_metrics(self, data: dict):
        if not self._cfg().get("metrics_enabled", True):
            return
        if not data:
            return
        try:
            path = Path(get_astrbot_data_path()) / str(self._cfg().get("metrics_path", "audit_metrics.jsonl"))
            line = self.privacy_service.sanitize_for_persistence(json.dumps(data, ensure_ascii=False))
            async with self._metrics_lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as file_obj:
                    file_obj.write(line + "\n")
        except Exception as exc:
            logger.warning(f"[mc_log] 写入指标失败: {exc}")

    async def extract_content(
        self,
        local_file: Path,
        is_archive: bool,
        work_dir: Path,
        run_id: str = "",
        deadline: float | None = None,
    ) -> ExtractionResult:
        self.runtime.ensure_not_timed_out(deadline, run_id=run_id, stage="extract_start")
        logger.info(f"[mc_log][{run_id}] 开始提取内容: file={local_file.name}, is_archive={is_archive}")
        if is_archive:
            return await self.extract_from_archive(local_file, work_dir, run_id=run_id, deadline=deadline)
        content, strategy = await self.extract_selected_path(local_file, run_id=run_id, deadline=deadline)
        return ExtractionResult(content=content, source_name=local_file.name, strategy=strategy)

    async def extract_from_archive(
        self,
        archive_path: Path,
        work_dir: Path,
        run_id: str = "",
        deadline: float | None = None,
    ) -> ExtractionResult:
        self.runtime.ensure_not_timed_out(deadline, run_id=run_id, stage="extract_archive_start")
        ext = archive_path.suffix.lower()
        extracted_paths: list[Path] = []
        extract_root = work_dir / ("unzipped" if ext == ".zip" else "ungz")
        logger.info(f"[mc_log][{run_id}] 开始解压归档文件: file={archive_path.name}, ext={ext}")
        if ext == ".zip":
            extracted_paths = await self.file_adapter.safe_extract_zip(archive_path, extract_root, deadline=deadline)
        elif ext == ".gz":
            out_path = await self.file_adapter.safe_extract_gz(archive_path, extract_root, deadline=deadline)
            extracted_paths = [out_path] if out_path else []
        if not extracted_paths:
            raise RuntimeError("archive has no extractable file")
        logger.info(f"[mc_log][{run_id}] 解压完成: extracted_count={len(extracted_paths)}")
        preview_files = [f"{path.name}({path.stat().st_size}B)" for path in extracted_paths[:80] if path.exists()]
        logger.info(f"[mc_log][{run_id}] 解压文件列表(前80): {preview_files}")
        archive_file_map = self.extraction_domain.build_archive_file_map(extracted_paths, extract_root)
        selected = self.extraction_domain.pick_priority_file(extracted_paths)
        if not selected:
            raise RuntimeError("archive has no matching log file")
        logger.info(f"[mc_log][{run_id}] 归档内选中文件: {selected.name}")
        content, strategy = await self.extract_selected_path(selected, run_id=run_id, deadline=deadline)
        return ExtractionResult(
            content=content,
            source_name=selected.name,
            strategy=strategy,
            archive_file_map=archive_file_map,
        )

    async def extract_selected_path(self, path: Path, run_id: str = "", deadline: float | None = None) -> tuple[str, str]:
        name_lower = path.name.lower()
        strategy = self.extraction_domain.strategy_from_text_name(name_lower)
        logger.info(f"[mc_log][{run_id}] 文本文件策略判定: file={path.name}, strategy={strategy}")
        if strategy == "A":
            kind = "hs_err" if "hs_err" in name_lower else "crash"
            content = await self.extraction_domain.strategy_a_extract(
                path,
                kind,
                self.file_adapter.read_text_with_fallback,
                deadline=deadline,
            )
            return content, strategy
        if strategy == "B":
            content = await self.extraction_domain.strategy_b_extract(path, self.file_adapter.read_text_with_fallback, deadline=deadline)
            return content, strategy

        content = await self.extraction_domain.strategy_c_extract(
            path,
            self.file_adapter.read_text_with_fallback,
            run_id=run_id,
            deadline=deadline,
        )
        return content, strategy

    async def handle_message(self, event):
        self.config_manager.reload()
        cfg = self._cfg()
        self.runtime.sync_global_sema(cfg)
        self.runtime.sync_io_sema(cfg)
        selected = pick_target_file(event)
        if not selected:
            return

        if not self.is_session_whitelisted(event, cfg):
            logger.info(f"[mc_log] 会话未命中白名单，已忽略: session_id={event.get_session_id()!r}")
            return

        file_comp, is_archive = selected
        if not await self.runtime.check_rate_limit(event, cfg):
            result = event.plain_result(self._msg("rate_limited"))
            result.stop_event()
            yield result
            return

        acquired = await self.runtime.acquire_global_slot(cfg, run_id="pre")
        if not acquired:
            result = event.plain_result(self._msg("queue_busy"))
            result.stop_event()
            yield result
            return

        run_id = self.runtime.build_run_id(event)
        run_token = self.runtime.bind_run_id(run_id)
        terminal_result = None
        terminal_extra_result = None
        slot_released = False

        async def release_slot_once(reason: str):
            nonlocal slot_released
            if slot_released:
                logger.info(f"[mc_log][{run_id}] 跳过重复释放全局并发槽位: reason={reason}")
                return
            slot_released = True
            await self.runtime.release_global_slot(run_id=run_id)

        async def watchdog_release():
            try:
                delay = float(cfg["global_timeout_sec"]) + 5.0
                await asyncio.sleep(max(0.0, delay))
                logger.warning(f"[mc_log][{run_id}] 任务疑似卡住，watchdog 触发释放槽位")
                await release_slot_once("watchdog")
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning(f"[mc_log][{run_id}] watchdog 异常: {exc}")

        watchdog_task = asyncio.create_task(watchdog_release())
        try:
            await self.runtime.start_debug_capture(
                run_id,
                event,
                detect_file_name(event, file_comp),
                is_archive,
                cfg,
                self.privacy_service.sanitize_for_persistence,
            )
            started = time.monotonic()
            deadline = started + float(cfg["global_timeout_sec"])
            run_ctx = RunContext(event=event, file_comp=file_comp, is_archive=is_archive, run_id=run_id, started_at=started, deadline=deadline)
            work_dir: Path | None = None
            metrics_data = {
                "diag_version": str(cfg.get("diag_version", "")),
                "claim_type": "",
                "guard_flags": [],
                "needs_more_info": False,
                "root_cause_stability_key": "",
                "resolved_feedback": "no_feedback",
                "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            }
            try:
                if not self.prompt_manager.prompts_ready:
                    self.prompt_manager.load_prompts()
                if not self.prompt_manager.prompts_ready:
                    logger.error("[mc_log] 提示词模板未就绪，已终止本次分析")
                    result = event.plain_result(self._msg("prompt_missing"))
                    result.stop_event()
                    terminal_result = result
                else:
                    analyze_provider_id = self.configured_analyze_provider_id()
                    if not analyze_provider_id:
                        logger.error(f"[mc_log][{run_id}] 未配置模型提供商: analyze_select_provider={analyze_provider_id!r}")
                        result = event.plain_result(self._msg("provider_not_configured"))
                        result.stop_event()
                        terminal_result = result
                    elif not self.context.get_provider_by_id(analyze_provider_id):
                        logger.error(f"[mc_log][{run_id}] 配置的模型提供商不存在: analyze={analyze_provider_id}")
                        result = event.plain_result(self._msg("provider_not_configured"))
                        result.stop_event()
                        terminal_result = result
                    else:
                        accepted_notice = self._msg("accepted_notice")
                        if accepted_notice:
                            yield event.plain_result(accepted_notice)
                        logger.info(
                            f"[mc_log][{run_id}] 开始分析: message_id={getattr(event.message_obj, 'message_id', '')}, "
                            f"name={detect_file_name(event, file_comp)}, is_archive={is_archive}, "
                            f"analyze_provider={analyze_provider_id}, "
                            f"global_timeout_sec={cfg['global_timeout_sec']}"
                        )
                        logger.info(f"[mc_log] 文件命中规则: name={detect_file_name(event, file_comp)}, is_archive={is_archive}")
                        work_dir = self.runtime.build_work_dir(event)
                        run_ctx.work_dir = work_dir
                        work_dir.mkdir(parents=True, exist_ok=True)
                        (work_dir / ".mc_log_analysis").write_text("1", encoding="utf-8")

                        self.runtime.ensure_not_timed_out(deadline, run_id=run_id, stage="before_download")
                        t_download = time.monotonic()
                        local_file = await self.file_adapter.download_to_workdir(file_comp, work_dir, deadline=deadline)
                        logger.info(
                            f"[mc_log][{run_id}] 阶段下载: {time.monotonic() - t_download:.2f}s, "
                            f"ok={bool(local_file and local_file.exists())}"
                        )
                        if not local_file or not local_file.exists():
                            logger.warning(f"[mc_log][{run_id}] 下载阶段失败，未获取到本地文件")
                            result = event.plain_result(self._msg("download_failed"))
                            result.stop_event()
                            terminal_result = result
                        else:
                            t_extract = time.monotonic()
                            extraction_result = await self.extract_content(
                                local_file=local_file,
                                is_archive=is_archive,
                                work_dir=work_dir,
                                run_id=run_id,
                                deadline=deadline,
                            )
                            logger.info(
                                f"[mc_log][{run_id}] 阶段提取: {time.monotonic() - t_extract:.2f}s, "
                                f"strategy={extraction_result.strategy}, chars={len(extraction_result.content)}"
                            )
                            if not extraction_result.content.strip():
                                logger.warning(f"[mc_log][{run_id}] 提取结果为空，无法继续分析")
                                result = event.plain_result(self._msg("no_extractable_content"))
                                result.stop_event()
                                terminal_result = result
                            else:
                                extracted = self.extraction_domain.apply_total_budget(extraction_result.content, cfg["total_char_limit"])
                                extracted_for_llm = self.privacy_service.guard_for_llm(extracted)
                                self.runtime.set_active_archive_file_map(event, extraction_result.archive_file_map)
                                self.runtime.set_active_primary_source_name(event, extraction_result.source_name)
                                available_archive_files = sorted(extraction_result.archive_file_map.keys())
                                t_analyze = time.monotonic()
                                report_md = await self.analysis_service.analyze_with_llm(
                                    event=event,
                                    source_name=extraction_result.source_name,
                                    strategy=extraction_result.strategy,
                                    content=extracted_for_llm,
                                    available_archive_files=available_archive_files,
                                    analyze_provider_id=analyze_provider_id,
                                    run_id=run_id,
                                    deadline=deadline,
                                )
                                logger.info(f"[mc_log][{run_id}] 阶段分析: {time.monotonic() - t_analyze:.2f}s, ok={bool(report_md)}")
                                if not report_md:
                                    logger.warning(f"[mc_log][{run_id}] 最终分析未返回内容")
                                    metrics_data["needs_more_info"] = True
                                    result = event.plain_result(self._msg("analyze_failed_logged"))
                                    result.stop_event()
                                    terminal_result = result
                                else:
                                    report_md = self.privacy_service.guard_for_output(report_md)
                                    metrics_data.update(self.metrics_service.extract_metrics_from_report(report_md))
                                    analysis_result = AnalysisResult(
                                        report_md=report_md,
                                        code_blocks_text=self.rendering_adapter.build_code_blocks_message(report_md),
                                    )
                                    t_render = time.monotonic()
                                    render_mode, render_payload = await self.rendering_adapter.render_report(
                                        analysis_result.report_md,
                                        run_id=run_id,
                                        deadline=deadline,
                                    )
                                    render_result = RenderResult(render_mode=render_mode, render_payload=render_payload)
                                    logger.info(f"[mc_log][{run_id}] 阶段渲染: {time.monotonic() - t_render:.2f}s, mode={render_result.render_mode}")
                                    elapsed = time.monotonic() - started
                                    logger.info(f"[mc_log][{run_id}] 分析完成，总耗时: {elapsed:.2f}s")
                                    if render_result.render_mode == "text":
                                        summary = self.rendering_adapter.build_summary_text(
                                            source_name=extraction_result.source_name,
                                            strategy=extraction_result.strategy,
                                            elapsed=elapsed,
                                        )
                                        response = event.plain_result(summary + "\n\n" + render_result.render_payload)
                                    else:
                                        response = self.rendering_adapter.build_forward_response(
                                            event=event,
                                            source_name=extraction_result.source_name,
                                            strategy=extraction_result.strategy,
                                            elapsed=elapsed,
                                            render_mode=render_result.render_mode,
                                            render_payload=render_result.render_payload,
                                        )
                                    response.stop_event()
                                    terminal_result = response
                                    if analysis_result.code_blocks_text:
                                        code_result = event.plain_result(analysis_result.code_blocks_text)
                                        code_result.stop_event()
                                        terminal_extra_result = code_result
            except BudgetExceeded as exc:
                logger.warning(f"[mc_log][{run_id}] 资源预算超限: {exc}")
                metrics_data["needs_more_info"] = True
                result = event.plain_result(self._msg("file_too_large"))
                result.stop_event()
                terminal_result = result
            except TimeoutError as exc:
                logger.warning(f"[mc_log][{run_id}] 全局超时: {exc}")
                metrics_data["needs_more_info"] = True
                result = event.plain_result(self._msg("global_timeout"))
                result.stop_event()
                terminal_result = result
            except Exception as exc:
                logger.error(f"[mc_log][{run_id}] 处理流程异常: {exc}", exc_info=True)
                metrics_data["needs_more_info"] = True
                result = event.plain_result(self._msg("analyze_failed_retry"))
                result.stop_event()
                terminal_result = result
            finally:
                await self.write_metrics(metrics_data)
                self.runtime.clear_active_archive_file_map(event)
                if work_dir:
                    await self.file_adapter.safe_remove_dir(work_dir, deadline=deadline)
                await self.runtime.stop_debug_capture(run_id)
                await self.runtime.sync_latest_debug_copy(lambda src, dst: self.file_adapter.copy_path(src, dst))
        finally:
            if watchdog_task:
                watchdog_task.cancel()
                try:
                    await watchdog_task
                except Exception:
                    pass
            self.runtime.reset_run_id(run_token)
            await release_slot_once("finally")

        if terminal_result is not None:
            yield terminal_result
        if terminal_extra_result is not None:
            yield terminal_extra_result
