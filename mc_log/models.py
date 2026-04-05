from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class BudgetExceeded(RuntimeError):
    pass


class FlowDone(RuntimeError):
    pass


@dataclass(frozen=True)
class MessageTexts:
    accepted_notice: str = ""
    download_failed: str = ""
    rate_limited: str = ""
    queue_busy: str = ""
    file_too_large: str = ""
    no_extractable_content: str = ""
    analyze_failed_logged: str = ""
    analyze_failed_retry: str = ""
    prompt_missing: str = ""
    provider_not_configured: str = ""
    global_timeout: str = ""
    html_render_fallback_notice: str = ""
    text_render_fallback_notice: str = ""
    forward_sender_name: str = ""
    summary_template: str = ""

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> "MessageTexts":
        values = mapping or {}
        return cls(
            accepted_notice=str(values.get("accepted_notice", "") or ""),
            download_failed=str(values.get("download_failed", "") or ""),
            rate_limited=str(values.get("rate_limited", "") or ""),
            queue_busy=str(values.get("queue_busy", "") or ""),
            file_too_large=str(values.get("file_too_large", "") or ""),
            no_extractable_content=str(values.get("no_extractable_content", "") or ""),
            analyze_failed_logged=str(values.get("analyze_failed_logged", "") or ""),
            analyze_failed_retry=str(values.get("analyze_failed_retry", "") or ""),
            prompt_missing=str(values.get("prompt_missing", "") or ""),
            provider_not_configured=str(values.get("provider_not_configured", "") or ""),
            global_timeout=str(values.get("global_timeout", "") or ""),
            html_render_fallback_notice=str(values.get("html_render_fallback_notice", "") or ""),
            text_render_fallback_notice=str(values.get("text_render_fallback_notice", "") or ""),
            forward_sender_name=str(values.get("forward_sender_name", "") or ""),
            summary_template=str(values.get("summary_template", "") or ""),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "accepted_notice": self.accepted_notice,
            "download_failed": self.download_failed,
            "rate_limited": self.rate_limited,
            "queue_busy": self.queue_busy,
            "file_too_large": self.file_too_large,
            "no_extractable_content": self.no_extractable_content,
            "analyze_failed_logged": self.analyze_failed_logged,
            "analyze_failed_retry": self.analyze_failed_retry,
            "prompt_missing": self.prompt_missing,
            "provider_not_configured": self.provider_not_configured,
            "global_timeout": self.global_timeout,
            "html_render_fallback_notice": self.html_render_fallback_notice,
            "text_render_fallback_notice": self.text_render_fallback_notice,
            "forward_sender_name": self.forward_sender_name,
            "summary_template": self.summary_template,
        }

    def get(self, key: str, default: str = "") -> str:
        return self.to_dict().get(key, default)


@dataclass(frozen=True)
class PluginConfig:
    values: dict[str, Any]
    messages: MessageTexts

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def msg(self, key: str, default: str = "") -> str:
        value = self.messages.get(key, default)
        return value if value else default

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.values)
        data["messages"] = self.messages.to_dict()
        return data


@dataclass
class RunContext:
    event: Any
    file_comp: Any
    is_archive: bool
    run_id: str
    started_at: float
    deadline: float | None = None
    work_dir: Path | None = None


@dataclass
class ExtractionResult:
    content: str
    source_name: str
    strategy: str
    archive_file_map: dict[str, Path] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    report_md: str | None
    code_blocks_text: str = ""


@dataclass
class RenderResult:
    render_mode: str
    render_payload: str


@dataclass
class RunOutcome:
    terminal_result: Any = None
    terminal_extra_result: Any = None
    metrics_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchiveSessionState:
    file_map: dict[str, Path] = field(default_factory=dict)
    tool_calls: int = 0
    primary_source_name: str = ""
