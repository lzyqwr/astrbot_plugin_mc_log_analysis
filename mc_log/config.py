from __future__ import annotations

from typing import Any

from .models import MessageTexts, PluginConfig


DEFAULT_CONFIG = {
    "global_timeout_sec": 300,
    "rate_limit_user_sec": 60,
    "queue_wait_sec": 8,
    "max_concurrent_jobs": 2,
    "max_concurrent_io": 1,
    "max_input_file_bytes": 32 * 1024 * 1024,
    "max_archive_file_count": 160,
    "max_archive_single_file_bytes": 12 * 1024 * 1024,
    "max_archive_total_bytes": 32 * 1024 * 1024,
    "max_gz_output_bytes": 16 * 1024 * 1024,
    "must_keep_window_lines": 30,
    "diag_version": "1.0.0",
    "metrics_enabled": True,
    "metrics_path": "audit_metrics.jsonl",
    "analyze_select_provider": "",
    "session_whitelist": [],
    "render_mode": "html_to_image",
    "image_width": 640,
    "full_read_char_limit": 140000,
    "total_char_limit": 140000,
    "max_tool_calls": 6,
    "tool_timeout_sec": 120,
    "tool_retry_limit": 1,
    "api_retry_limit": 1,
    "tool_snippet_chars": 600,
    "read_archive_file_limit": 1,
    "analyze_llm_timeout_sec": 240,
    "html_render_timeout_sec": 30,
    "file_download_timeout_sec": 30,
    "messages": {
        "accepted_notice": "已接收文件，正在分析，请稍候。",
        "download_failed": "日志文件下载失败，请稍后重试。",
        "rate_limited": "请求过于频繁，请稍后再试。",
        "queue_busy": "当前任务较多，请稍后再试。",
        "file_too_large": "文件过大，超过处理上限，请拆分或压缩后重试。",
        "no_extractable_content": "未在文件中识别到可分析的日志内容。",
        "analyze_failed_logged": "日志分析失败，已记录日志，请联系管理员检查检查。",
        "analyze_failed_retry": "日志分析失败，请联系管理员检查。",
        "prompt_missing": "日志分析模板缺失，请联系管理员检查 assets 目录。",
        "provider_not_configured": "请先在插件配置中填写 analyze_select_provider。",
        "global_timeout": "日志分析超时，请稍后重试。",
        "html_render_fallback_notice": "[提示] HTML 渲染失败，已降级发送原始文本。",
        "text_render_fallback_notice": "[提示] 渲染不可用，已降级为纯文本发送。",
        "forward_sender_name": "MC日志分析",
        "summary_template": "分析完成，耗时 {elapsed:.2f} 秒\n文件: {source_name}\n策略: {strategy}",
    },
}

MAX_ARCHIVE_FILE_COUNT = 160
MAX_ARCHIVE_SINGLE_FILE_BYTES = 12 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 32 * 1024 * 1024
MAX_GZ_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_ARCHIVE_TOOL_CHARS = 35000
MAX_CODE_BLOCK_MESSAGE_CHARS = 16000
MAX_DEBUG_PREVIEW_CHARS = 500
PROMPT_FILES = {
    "analyze_system": "analyze_system.txt",
    "analyze_user": "analyze_user.txt",
}


def read_conf_value(raw_config: Any, key: str, default: Any) -> Any:
    if raw_config is None:
        return default
    try:
        if hasattr(raw_config, "get"):
            value = raw_config.get(key, default)
        else:
            value = raw_config[key]
        return default if value is None else value
    except Exception:
        return default


def normalize_messages_config(raw_messages: Any) -> MessageTexts:
    defaults = DEFAULT_CONFIG["messages"]
    out = dict(defaults)
    if isinstance(raw_messages, dict):
        for key, value in raw_messages.items():
            if key in out and value is not None:
                out[key] = str(value)
    return MessageTexts.from_mapping(out)


def normalize_string_list_config(raw_values: Any) -> list[str]:
    if not isinstance(raw_values, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for value in raw_values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def load_plugin_config(raw_config: Any) -> PluginConfig:
    cfg = dict(DEFAULT_CONFIG)
    for key in cfg:
        cfg[key] = read_conf_value(raw_config, key, cfg[key])
    cfg["global_timeout_sec"] = max(10, int(cfg.get("global_timeout_sec", 300)))
    cfg["rate_limit_user_sec"] = max(0, int(cfg.get("rate_limit_user_sec", 60)))
    cfg["queue_wait_sec"] = max(0, int(cfg.get("queue_wait_sec", 8)))
    cfg["max_concurrent_jobs"] = max(1, int(cfg.get("max_concurrent_jobs", 2)))
    cfg["max_concurrent_io"] = max(1, int(cfg.get("max_concurrent_io", 1)))
    cfg["max_input_file_bytes"] = max(1024 * 1024, int(cfg.get("max_input_file_bytes", 32 * 1024 * 1024)))
    cfg["max_archive_file_count"] = max(1, int(cfg.get("max_archive_file_count", 160)))
    cfg["max_archive_single_file_bytes"] = max(
        1024 * 1024, int(cfg.get("max_archive_single_file_bytes", 12 * 1024 * 1024))
    )
    cfg["max_archive_total_bytes"] = max(
        cfg["max_archive_single_file_bytes"],
        int(cfg.get("max_archive_total_bytes", 32 * 1024 * 1024)),
    )
    cfg["max_gz_output_bytes"] = max(1024 * 1024, int(cfg.get("max_gz_output_bytes", 16 * 1024 * 1024)))
    cfg["must_keep_window_lines"] = max(5, int(cfg.get("must_keep_window_lines", 30)))
    cfg["diag_version"] = str(cfg.get("diag_version", "1.0.0") or "1.0.0")
    cfg["metrics_enabled"] = bool(cfg.get("metrics_enabled", True))
    cfg["metrics_path"] = str(cfg.get("metrics_path", "audit_metrics.jsonl") or "audit_metrics.jsonl")
    cfg["analyze_select_provider"] = str(cfg.get("analyze_select_provider", "") or "").strip()
    cfg["session_whitelist"] = normalize_string_list_config(cfg.get("session_whitelist"))
    cfg["image_width"] = max(320, int(cfg.get("image_width", 640)))
    cfg["full_read_char_limit"] = max(10000, int(cfg["full_read_char_limit"]))
    cfg["total_char_limit"] = max(3000, int(cfg["total_char_limit"]))
    cfg["max_tool_calls"] = max(1, int(cfg["max_tool_calls"]))
    cfg["tool_timeout_sec"] = max(2, int(cfg["tool_timeout_sec"]))
    cfg["tool_retry_limit"] = max(0, int(cfg["tool_retry_limit"]))
    cfg["api_retry_limit"] = max(0, int(cfg.get("api_retry_limit", 1)))
    cfg["tool_snippet_chars"] = max(200, int(cfg.get("tool_snippet_chars", 600)))
    cfg["read_archive_file_limit"] = max(0, int(cfg.get("read_archive_file_limit", 1)))
    cfg["analyze_llm_timeout_sec"] = max(5, int(cfg["analyze_llm_timeout_sec"]))
    cfg["html_render_timeout_sec"] = max(5, int(cfg["html_render_timeout_sec"]))
    cfg["file_download_timeout_sec"] = max(5, int(cfg["file_download_timeout_sec"]))
    raw_render_mode = str(cfg["render_mode"]).lower().strip()
    if raw_render_mode in {"html", "html_to_image"}:
        cfg["render_mode"] = "html_to_image"
    elif raw_render_mode in {"text", "text_to_image"}:
        cfg["render_mode"] = "text_to_image"
    else:
        cfg["render_mode"] = "html_to_image"
    messages = normalize_messages_config(cfg.get("messages"))
    cfg["messages"] = messages.to_dict()
    return PluginConfig(values=cfg, messages=messages)


class ConfigManager:
    def __init__(self, raw_config: Any = None):
        self._raw_config = raw_config
        self._current = load_plugin_config(raw_config)

    def set_raw_config(self, raw_config: Any):
        self._raw_config = raw_config

    def reload(self) -> PluginConfig:
        self._current = load_plugin_config(self._raw_config)
        return self._current

    def get(self) -> PluginConfig:
        return self._current

    def msg(self, key: str, default: str = "") -> str:
        return self._current.msg(key, default)
