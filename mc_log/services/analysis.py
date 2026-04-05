from __future__ import annotations

import asyncio
import json

import aiohttp
from astrbot.api import logger


class AnalysisService:
    def __init__(self, context, config_manager, runtime, prompt_manager, tool_registry, metrics_service):
        self.context = context
        self.config_manager = config_manager
        self.runtime = runtime
        self.prompt_manager = prompt_manager
        self.tool_registry = tool_registry
        self.metrics_service = metrics_service

    def _cfg(self):
        return self.config_manager.get()

    def is_retryable_api_error(self, exc: Exception) -> bool:
        if isinstance(exc, aiohttp.ClientResponseError):
            status = int(getattr(exc, "status", 0) or 0)
            return status == 429 or 500 <= status <= 599

        text = str(exc or "").lower()
        if not text:
            return False
        patterns = (
            " 429",
            " 500",
            " 502",
            " 503",
            " 504",
            "status code: 429",
            "status code: 500",
            "status code: 502",
            "status code: 503",
            "status code: 504",
            "rate limit",
            "too many requests",
            "service unavailable",
            "server error",
            "internal server error",
            "bad gateway",
            "gateway timeout",
            "overloaded",
            "temporarily unavailable",
            "upstream",
            "candidate.content.parts 为空",
            "candidate.content.parts empty",
        )
        return any(pattern in text for pattern in patterns)

    async def call_with_api_retry(self, call_name: str, coro_factory, run_id: str = "", deadline: float | None = None, response_validator=None):
        tries = max(1, int(self._cfg().get("api_retry_limit", 1)) + 1)
        last_exc = None
        for idx in range(tries):
            self.runtime.ensure_not_timed_out(deadline, run_id=run_id, stage=f"{call_name}_try_{idx + 1}")
            try:
                resp = await coro_factory()
                if response_validator:
                    ok, reason = response_validator(resp)
                    if not ok:
                        if idx >= tries - 1:
                            raise RuntimeError(f"{call_name} invalid response: {reason}")
                        delay = min(0.4 * (2**idx), 2.0)
                        time_left = self.runtime.time_left(deadline)
                        if time_left <= 0:
                            raise TimeoutError("global timeout reached during api retry")
                        sleep_sec = min(delay, max(0.0, time_left - 0.05))
                        logger.warning(f"[mc_log][{run_id}] {call_name} 返回空内容，准备重试 {idx + 1}/{tries - 1}: {reason}")
                        if sleep_sec > 0:
                            await asyncio.sleep(sleep_sec)
                        continue
                return resp
            except Exception as exc:
                last_exc = exc
                is_retryable = self.is_retryable_api_error(exc)
                if not is_retryable or idx >= tries - 1:
                    raise
                delay = min(0.4 * (2**idx), 2.0)
                time_left = self.runtime.time_left(deadline)
                if time_left <= 0:
                    raise TimeoutError("global timeout reached during api retry") from exc
                sleep_sec = min(delay, max(0.0, time_left - 0.05))
                logger.warning(f"[mc_log][{run_id}] {call_name} 服务端异常，准备重试 {idx + 1}/{tries - 1}: {exc}")
                if sleep_sec > 0:
                    await asyncio.sleep(sleep_sec)
        if last_exc:
            raise last_exc
        raise RuntimeError(f"{call_name} failed without exception")

    def extract_llm_text_with_diag(self, llm_resp) -> tuple[str, dict]:
        text = str(getattr(llm_resp, "completion_text", "") or "").strip()
        diag = {"resp_type": type(llm_resp).__name__, "completion_len": len(text)}

        candidates = getattr(llm_resp, "candidates", None)
        if isinstance(candidates, list):
            diag["candidate_count"] = len(candidates)
            fallback_parts: list[str] = []
            for cand in candidates[:4]:
                content = cand.get("content") if isinstance(cand, dict) else getattr(cand, "content", None)
                parts = content.get("parts") if isinstance(content, dict) else getattr(content, "parts", None)
                if isinstance(parts, list):
                    for part in parts:
                        if isinstance(part, str):
                            part_text = part
                        elif isinstance(part, dict):
                            part_text = str(part.get("text") or part.get("content") or "")
                        else:
                            part_text = str(getattr(part, "text", "") or getattr(part, "content", "") or "")
                        if part_text.strip():
                            fallback_parts.append(part_text.strip())
                content_text = content.get("text") if isinstance(content, dict) else getattr(content, "text", "")
                if isinstance(content_text, str) and content_text.strip():
                    fallback_parts.append(content_text.strip())
                cand_text = cand.get("text") if isinstance(cand, dict) else getattr(cand, "text", "")
                if isinstance(cand_text, str) and cand_text.strip():
                    fallback_parts.append(cand_text.strip())
            merged = "\n".join(fallback_parts).strip()
            diag["candidates_fallback_len"] = len(merged)
            if not text and merged:
                text = merged
            first = candidates[0] if candidates else None
            if first is not None:
                diag["candidate_finish_reason"] = first.get("finish_reason") if isinstance(first, dict) else getattr(first, "finish_reason", None)
                diag["candidate_block_reason"] = first.get("block_reason") if isinstance(first, dict) else getattr(first, "block_reason", None)

        if not text:
            for attr in ("text", "output_text"):
                value = getattr(llm_resp, attr, None)
                if isinstance(value, str) and value.strip():
                    text = value.strip()
                    diag[f"{attr}_len"] = len(text)
                    break
        diag["final_text_len"] = len(text)
        return text, diag

    def validate_llm_response_not_empty(self, llm_resp, run_id: str, stage: str) -> tuple[bool, str]:
        text, diag = self.extract_llm_text_with_diag(llm_resp)
        if text:
            return True, "ok"
        logger.warning(
            f"[mc_log][{run_id}] {stage} LLM响应为空(candidate.content.parts可能为空): {json.dumps(diag, ensure_ascii=False)}"
        )
        return False, "empty_text_parts"

    async def analyze_with_llm(
        self,
        event,
        source_name: str,
        strategy: str,
        content: str,
        available_archive_files: list[str] | None,
        analyze_provider_id: str,
        run_id: str = "",
        deadline: float | None = None,
    ) -> str | None:
        toolset = self.tool_registry.build_toolset(["search_mc_sites", "read_archive_file"])
        system_prompt = self.prompt_manager.get_prompt("analyze_system")
        analyze_user_tpl = self.prompt_manager.get_prompt("analyze_user")
        if not system_prompt or not analyze_user_tpl:
            logger.error("[mc_log] 最终分析失败：分析提示词缺失")
            return None
        user_prompt = self.prompt_manager.render_prompt(
            analyze_user_tpl,
            {
                "source_name": source_name,
                "strategy": strategy,
                "content": content,
                "available_files": "\n".join(available_archive_files or []),
            },
        )
        user_prompt = user_prompt + "\n\n" + self.tool_registry.build_available_files_prompt_block(available_archive_files or [])
        tool_failed = False
        tool_failure_reason = ""
        tool_names = []
        if toolset:
            try:
                tool_names = [tool.name for tool in toolset.tools]
            except Exception:
                tool_names = []
        logger.info(
            f"[mc_log][{run_id}] 最终分析请求: provider={analyze_provider_id}, "
            f"system_chars={len(system_prompt)}, prompt_chars={len(user_prompt)}, "
            f"tools={tool_names}, max_steps={self._cfg()['max_tool_calls']}, archive_files={len(available_archive_files or [])}"
        )
        try:
            timeout_sec = min(float(self._cfg()["analyze_llm_timeout_sec"]), max(0.1, self.runtime.time_left(deadline)))
            llm_resp = None
            try:
                llm_resp = await asyncio.wait_for(
                    self.call_with_api_retry(
                        call_name="analyze_tool_loop_agent",
                        run_id=run_id,
                        deadline=deadline,
                        response_validator=lambda response: self.validate_llm_response_not_empty(
                            response,
                            run_id=run_id,
                            stage="analyze",
                        ),
                        coro_factory=lambda: self.context.tool_loop_agent(
                            event=event,
                            chat_provider_id=analyze_provider_id,
                            system_prompt=system_prompt,
                            prompt=user_prompt,
                            tools=toolset,
                            max_steps=self._cfg()["max_tool_calls"],
                            tool_call_timeout=self._cfg()["tool_timeout_sec"],
                        ),
                    ),
                    timeout=timeout_sec,
                )
            except Exception as exc:
                tool_failed = True
                tool_failure_reason = str(exc)
                logger.warning(f"[mc_log][{run_id}] 工具循环失败，准备降级为无工具分析: {exc}")
                llm_resp = None

            if llm_resp is None:
                tool_note = "【工具状态】本次工具调用失败或超时；禁止依赖工具补全或臆测，仅基于日志证据给出最稳妥的低风险建议。"
                fallback_prompt = user_prompt + "\n\n" + tool_note
                llm_resp = await asyncio.wait_for(
                    self.call_with_api_retry(
                        call_name="analyze_llm_generate_fallback",
                        run_id=run_id,
                        deadline=deadline,
                        response_validator=lambda response: self.validate_llm_response_not_empty(
                            response,
                            run_id=run_id,
                            stage="analyze_fallback",
                        ),
                        coro_factory=lambda: self.context.llm_generate(
                            chat_provider_id=analyze_provider_id,
                            system_prompt=system_prompt,
                            prompt=fallback_prompt,
                        ),
                    ),
                    timeout=timeout_sec,
                )
            text, diag = self.extract_llm_text_with_diag(llm_resp)
            text = text.strip()
            if text:
                suspect_reason = self.metrics_service.detect_suspect_analyze_text(text)
                if suspect_reason:
                    logger.warning(f"[mc_log][{run_id}] 最终分析返回疑似异常占位文本，触发无工具重试: reason={suspect_reason}")
                    fallback_prompt = (
                        user_prompt
                        + "\n\n"
                        + "【工具状态】上一次模型响应疑似异常占位文本；请忽略该异常信息，不要转述内部报错。"
                        "仅基于日志证据输出结构化结论；若证据不足请标注 UNCERTAIN。"
                    )
                    try:
                        fallback_resp = await asyncio.wait_for(
                            self.call_with_api_retry(
                                call_name="analyze_llm_generate_suspect_retry",
                                run_id=run_id,
                                deadline=deadline,
                                response_validator=lambda response: self.validate_llm_response_not_empty(
                                    response,
                                    run_id=run_id,
                                    stage="analyze_suspect_retry",
                                ),
                                coro_factory=lambda: self.context.llm_generate(
                                    chat_provider_id=analyze_provider_id,
                                    system_prompt=system_prompt,
                                    prompt=fallback_prompt,
                                ),
                            ),
                            timeout=timeout_sec,
                        )
                        fb_text, _ = self.extract_llm_text_with_diag(fallback_resp)
                        fb_text = fb_text.strip()
                        if fb_text:
                            logger.info(f"[mc_log][{run_id}] 疑似异常占位文本重试成功: chars={len(fb_text)}")
                            text = fb_text
                    except Exception as fb_exc:
                        logger.warning(f"[mc_log][{run_id}] 疑似异常占位文本重试失败，保留原结果: {fb_exc}")
                logger.info(f"[mc_log][{run_id}] 最终分析返回: chars={len(text)}")
                if tool_failed:
                    logger.info(f"[mc_log][{run_id}] 工具失败降级原因: {tool_failure_reason}")
                return text
            logger.error(f"[mc_log][{run_id}] 最终分析失败：LLM返回空内容, diag={json.dumps(diag, ensure_ascii=False)}")
        except asyncio.TimeoutError:
            logger.error(f"[mc_log][{run_id}] 最终分析LLM超时: {self._cfg()['analyze_llm_timeout_sec']}s")
            if self.runtime.time_left(deadline) <= 0:
                raise TimeoutError("global timeout reached during analyze")
        except Exception as exc:
            logger.error(f"[mc_log][{run_id}] 最终分析LLM异常: {exc}", exc_info=True)
        return None
