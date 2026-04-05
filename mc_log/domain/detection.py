from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse


ARCHIVE_EXTS = {".zip", ".gz"}
TEXT_EXTS = {".txt", ".log"}
ARCHIVE_NAME_KEYS = ("错误报告", "日志", "log")
TEXT_NAME_KEYS = ("crash", "hs_err", "latest", "debug", "fcl", "pcl", "游戏崩溃", "日志", "log")


def is_file_component(component) -> bool:
    return hasattr(component, "get_file") or any(hasattr(component, name) for name in ("name", "file_", "url"))


def pick_target_file(event):
    files = [comp for comp in event.get_messages() if is_file_component(comp)]
    if not files:
        return None
    for file_comp in files:
        name = detect_file_name(event, file_comp).lower()
        ext = Path(name).suffix.lower()
        if ext in ARCHIVE_EXTS and any(key in name for key in ARCHIVE_NAME_KEYS):
            return file_comp, True
        if ext in TEXT_EXTS and any(key in name for key in TEXT_NAME_KEYS):
            return file_comp, False
    return None


def detect_file_name(event, file_comp) -> str:
    name = str(getattr(file_comp, "name", "") or "").strip()
    if name:
        return name

    file_field = str(getattr(file_comp, "file_", "") or "").strip()
    if file_field:
        base = Path(file_field).name
        if base:
            return base

    url = str(getattr(file_comp, "url", "") or "").strip()
    if url:
        parsed = urlparse(url)
        base = Path(parsed.path).name
        if base:
            return base

    try:
        raw = getattr(event.message_obj, "raw_message", None)
        if raw and isinstance(raw, dict):
            segs = raw.get("message", [])
            if isinstance(segs, list):
                for seg in segs:
                    if isinstance(seg, dict) and seg.get("type") == "file":
                        data = seg.get("data", {})
                        if isinstance(data, dict):
                            raw_name = str(data.get("file_name") or data.get("name") or data.get("file") or "").strip()
                            if raw_name:
                                return raw_name
    except Exception:
        pass
    return ""


def detect_file_name_dummy(file_comp) -> str:
    name = str(getattr(file_comp, "name", "") or "").strip()
    if name:
        return name
    file_field = str(getattr(file_comp, "file_", "") or "").strip()
    if file_field:
        base = Path(file_field).name
        if base:
            return base
    url = str(getattr(file_comp, "url", "") or "").strip()
    if url:
        parsed = urlparse(url)
        base = Path(parsed.path).name
        if base:
            return base
    return ""


def sanitize_uploaded_name(name: str) -> str:
    return re.sub(r"[^\w\-.]", "_", name or "upload.log")


def looks_like_multiple_paths(value: str) -> bool:
    if not value:
        return False
    if re.search(r"[\n\r,;]", value):
        return True
    if re.search(r"\s+(and|以及|和)\s+", value, flags=re.IGNORECASE):
        return True
    return False


def resolve_archive_file_request(file_map: dict[str, Path], requested: str) -> tuple[str, Path | None, str]:
    if requested in file_map:
        return requested, file_map[requested], ""

    req_lower = requested.lower()
    exact_ci = [(key, path) for key, path in file_map.items() if key.lower() == req_lower]
    if len(exact_ci) == 1:
        return exact_ci[0][0], exact_ci[0][1], ""

    req_name = Path(requested).name.lower()
    by_name = [(key, path) for key, path in file_map.items() if Path(key).name.lower() == req_name]
    if len(by_name) == 1:
        return by_name[0][0], by_name[0][1], ""
    if len(by_name) > 1:
        options = "\n".join(f"- {key}" for key, _ in by_name[:20])
        return "", None, f"匹配到多个同名文件，请使用完整路径：\n{options}"

    suffix = [(key, path) for key, path in file_map.items() if key.lower().endswith("/" + req_lower)]
    if len(suffix) == 1:
        return suffix[0][0], suffix[0][1], ""
    if len(suffix) > 1:
        options = "\n".join(f"- {key}" for key, _ in suffix[:20])
        return "", None, f"匹配到多个候选文件，请使用完整路径：\n{options}"

    available = "\n".join(f"- {key}" for key in list(file_map.keys())[:30])
    return "", None, f"未找到 `{requested}`。可用文件示例：\n{available}"
