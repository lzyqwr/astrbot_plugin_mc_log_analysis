from __future__ import annotations

import re
from pathlib import Path
from typing import Awaitable, Callable, Iterable

from astrbot.api import logger


MAX_SECTION_LINES = 400
STRATEGY_C_HEAD_TAIL_LINES = 180
STRATEGY_C_WINDOW_LINES = 8
ERROR_LINE_RE = re.compile(
    r"(ERROR|WARN|Exception|Caused by|FATAL|Failed|Stacktrace|Problematic frame|siginfo)",
    re.IGNORECASE,
)
TIMESTAMP_LINE_RE = re.compile(r"^\s*(\[\d{2}:\d{2}:\d{2}\]|\d{4}-\d{2}-\d{2}|\[\d{2}[/:]\d{2}[/:]\d{2})")
CAUSE_LINE_RE = re.compile(r"^\s*Caused by:", re.I)
VERSION_LINE_RE = re.compile(
    r"\b(Minecraft\s*(?:Version|version)|MC\s*Version|Loader\s*Version|Forge\s*Version|Fabric\s*Loader|"
    r"Quilt\s*Loader|NeoForge|Java\s*(?:Version|version)|JVM\s*Version|Runtime\s*Version)\b",
    re.I,
)
CRASH_SAVED_RE = re.compile(r"Crash report saved to", re.I)
LOG_PREFIX_RE = re.compile(r"^\s*(?:\[[^\]]+\]\s*)+")
STACKTRACE_SEED_RE = re.compile(
    r"(Exception|Caused by|Stacktrace|NoClassDefFoundError|ClassNotFoundException|NoSuchMethodError|"
    r"NoSuchFieldError|Problematic frame|siginfo|Mixin.*(?:error|fail))",
    re.I,
)
STACKTRACE_CONT_RE = re.compile(
    r"(^at\s+\S+\(.*\)$|^\.\.\. \d+ more$|^Suppressed:|^Caused by:|^Exception in thread\b|"
    r"^[A-Za-z0-9_.$]+(?:Exception|Error|Throwable|LinkageError)(?::|\b)|^-- .* --$|^Description:|"
    r"^Failure message:|^Mod File:|^Suspected Mods:|^Stacktrace:)",
    re.I,
)
DEPENDENCY_LINE_RE = re.compile(
    r"(Missing(?: or unsupported mandatory)? dependencies|Could not find required mod|requires .+\{|"
    r"minimum version|dependency|depends on)",
    re.I,
)
CLASSLOAD_LINE_RE = re.compile(
    r"(NoClassDefFoundError|ClassNotFoundException|NoSuchMethodError|NoSuchFieldError|ClassMetadataNotFoundException|"
    r"Mixin(?: apply)? failed|MixinTransformerError|Failed to (?:load|apply|initialize)|"
    r"Could not execute entrypoint|Failed to create mod instance)",
    re.I,
)
MOD_INFO_LINE_RE = re.compile(
    r"(\bmod(?:id| file| list)?\b|Mixin|Fabric|Forge|NeoForge|Quilt|Loader|entrypoint|coremod|transformer)",
    re.I,
)
NOISE_LINE_PATTERNS = (
    re.compile(r"\b(texture|sprite|atlas|baked model|modelbakery|font|glyph|shader|particle)\b", re.I),
    re.compile(r"\b(soundengine|openal|channel access|loaded sound|playing sound|audio stream)\b", re.I),
    re.compile(r"\b(reloading resources|resource reload|registered resource pack|found resource pack|resourcemanager)\b", re.I),
    re.compile(r"\b(recipe book|recipe manager|loaded recipe|tag loader|advancement|loot table|language load)\b", re.I),
    re.compile(r"\b(preparing spawn area|download terrain|chunk render|render distance|stitching texture atlas)\b", re.I),
    re.compile(r"^\s*<[^>]{1,32}>\s+", re.I),
)


class ExtractionDomain:
    def __init__(self, cfg_getter: Callable[[], object]):
        self._cfg_getter = cfg_getter

    def _cfg(self):
        return self._cfg_getter()

    async def strategy_a_extract(
        self,
        path: Path,
        kind: str,
        read_text_with_fallback: Callable[..., Awaitable[str]],
        deadline: float | None = None,
    ) -> str:
        cfg = self._cfg()
        content = await read_text_with_fallback(path, deadline=deadline)
        if len(content) <= cfg["full_read_char_limit"]:
            return self.apply_budget_with_must_keep(content, content, cfg["total_char_limit"])
        key_sections = self.extract_strategy_a_key_sections(content, kind)
        merged = "\n\n".join(key_sections).strip()
        if not merged:
            merged = self.build_error_focused_text(content)
        return self.apply_budget_with_must_keep(merged, content, cfg["total_char_limit"])

    async def strategy_b_extract(
        self,
        path: Path,
        read_text_with_fallback: Callable[..., Awaitable[str]],
        deadline: float | None = None,
    ) -> str:
        cfg = self._cfg()
        content = await read_text_with_fallback(path, deadline=deadline)
        lines = content.splitlines()
        if len(lines) <= 1000:
            return self.apply_budget_with_must_keep(content, content, cfg["total_char_limit"])
        selected = set(range(0, 500))
        selected.update(range(max(0, len(lines) - 500), len(lines)))
        selected.update(self.collect_error_window_indexes(lines, window=8, max_hits=300))
        merged = self.compose_lines_with_gaps(lines, sorted(selected))
        return self.apply_budget_with_must_keep(merged, content, cfg["total_char_limit"])

    async def strategy_c_extract(
        self,
        path: Path,
        read_text_with_fallback: Callable[..., Awaitable[str]],
        run_id: str = "",
        deadline: float | None = None,
    ) -> str:
        cfg = self._cfg()
        content = await read_text_with_fallback(path, deadline=deadline)
        reduced = self.build_strategy_c_regex_text(content, run_id=run_id)
        return self.apply_budget_with_must_keep(reduced, content, cfg["total_char_limit"])

    def build_archive_file_map(self, files: Iterable[Path], root_dir: Path) -> dict[str, Path]:
        root = root_dir.resolve()
        mapped: dict[str, Path] = {}
        for path in files:
            try:
                resolved = path.resolve()
            except Exception:
                continue
            try:
                display = str(resolved.relative_to(root)).replace("\\", "/")
            except Exception:
                display = path.name
            if not display:
                display = path.name
            if display in mapped:
                suffix = 2
                new_name = f"{display} [{suffix}]"
                while new_name in mapped:
                    suffix += 1
                    new_name = f"{display} [{suffix}]"
                display = new_name
            mapped[display] = resolved
        return mapped

    def pick_priority_file(self, files: Iterable[Path]) -> Path | None:
        priority = [
            ("hs_err", "A"),
            ("crash", "A"),
            ("latest", "C"),
            ("游戏崩溃", "C"),
            ("日志", "B"),
            ("log", "B"),
            ("debug", "B"),
            ("fcl", "C"),
            ("pcl", "C"),
        ]
        files_list = list(files)
        lowered = [(path, path.name.lower()) for path in files_list]
        logger.info(f"[mc_log] 归档候选文件: {[path.name for path in files_list]}")
        for key, _ in priority:
            for path, name in lowered:
                if key in name:
                    logger.info(f"[mc_log] 归档优先命中: key={key}, file={path.name}")
                    return path
        logger.debug(f"[mc_log] 归档内未找到优先关键词文件: {[path.name for path in files_list]}")
        return None

    def strategy_from_text_name(self, name_lower: str) -> str:
        if "hs_err" in name_lower or "crash" in name_lower:
            return "A"
        if "游戏崩溃" in name_lower:
            return "C"
        if "latest" in name_lower or "fcl" in name_lower or "pcl" in name_lower:
            return "C"
        if "debug" in name_lower or "日志" in name_lower or "log" in name_lower:
            return "B"
        return "C"

    def extract_strategy_a_key_sections(self, content: str, kind: str) -> list[str]:
        lines = content.splitlines()
        patterns_hs_err = [
            ("fatal error", re.compile(r"fatal error|A fatal error has been detected", re.I)),
            ("problematic frame", re.compile(r"Problematic frame|siginfo", re.I)),
            ("java/native frames", re.compile(r"Java frames|Native frames", re.I)),
            ("vm arguments", re.compile(r"VM Arguments|Command Line", re.I)),
            ("system info", re.compile(r"OS:|CPU:|Memory:|Host:", re.I)),
        ]
        patterns_crash = [
            ("header/description", re.compile(r"Description:|---- Minecraft Crash Report ----", re.I)),
            ("stacktrace", re.compile(r"Stacktrace:|Exception Details:", re.I)),
            ("caused by chain", re.compile(r"Caused by:", re.I)),
            ("suspected mods", re.compile(r"Suspected Mods|Mod File|Mod List", re.I)),
            ("system details", re.compile(r"System Details|-- System Details --", re.I)),
        ]
        patterns = patterns_hs_err if kind == "hs_err" else patterns_crash
        sections = []
        for label, pattern in patterns:
            idx = self.first_match_index(lines, pattern)
            if idx is None:
                continue
            block = lines[idx : idx + MAX_SECTION_LINES]
            if not block:
                continue
            sections.append(f"[{label}]\n" + "\n".join(block))
        return sections

    def first_match_index(self, lines: list[str], pattern: re.Pattern) -> int | None:
        for index, line in enumerate(lines):
            if pattern.search(line):
                return index
        return None

    def normalize_log_line(self, line: str) -> str:
        normalized = LOG_PREFIX_RE.sub("", line or "")
        return normalized.lstrip(": ").strip()

    def is_strategy_c_key_line(self, line: str, normalized: str | None = None) -> bool:
        text = normalized if normalized is not None else self.normalize_log_line(line)
        return bool(
            ERROR_LINE_RE.search(text)
            or VERSION_LINE_RE.search(text)
            or CRASH_SAVED_RE.search(text)
            or DEPENDENCY_LINE_RE.search(text)
            or CLASSLOAD_LINE_RE.search(text)
            or MOD_INFO_LINE_RE.search(text)
        )

    def is_strategy_c_noise_line(self, line: str, normalized: str | None = None) -> bool:
        text = normalized if normalized is not None else self.normalize_log_line(line)
        return any(pattern.search(text) for pattern in NOISE_LINE_PATTERNS)

    def is_stacktrace_continuation_line(self, line: str, normalized: str | None = None) -> bool:
        text = normalized if normalized is not None else self.normalize_log_line(line)
        return bool(STACKTRACE_CONT_RE.search(text) or CLASSLOAD_LINE_RE.search(text))

    def is_strategy_c_block_seed(self, line: str, normalized: str | None = None) -> bool:
        text = normalized if normalized is not None else self.normalize_log_line(line)
        return bool(STACKTRACE_SEED_RE.search(text))

    def expand_indexes(self, size: int, center: int, radius: int) -> set[int]:
        if size <= 0:
            return set()
        lo = max(0, center - radius)
        hi = min(size, center + radius + 1)
        return set(range(lo, hi))

    def collect_strategy_c_block_indexes(self, lines: list[str], start_index: int) -> set[int]:
        selected = {start_index}
        for index in range(start_index + 1, len(lines)):
            normalized = self.normalize_log_line(lines[index])
            if not normalized:
                break
            if TIMESTAMP_LINE_RE.search(lines[index]) and not self.is_stacktrace_continuation_line(lines[index], normalized):
                break
            if not self.is_stacktrace_continuation_line(lines[index], normalized):
                break
            selected.add(index)
        return selected

    def filter_strategy_c_noise_lines(self, lines: list[str]) -> tuple[list[str], int, int]:
        filtered: list[str] = []
        key_hits = 0
        noise_removed = 0
        for line in lines:
            normalized = self.normalize_log_line(line)
            is_key = self.is_strategy_c_key_line(line, normalized)
            if is_key:
                key_hits += 1
            if self.is_strategy_c_noise_line(line, normalized) and not is_key:
                noise_removed += 1
                continue
            filtered.append(line)
        return filtered, key_hits, noise_removed

    def collect_strategy_c_indexes(self, lines: list[str]) -> tuple[set[int], int]:
        selected = set(range(0, min(len(lines), STRATEGY_C_HEAD_TAIL_LINES)))
        selected.update(range(max(0, len(lines) - STRATEGY_C_HEAD_TAIL_LINES), len(lines)))
        key_hits = 0
        for index, line in enumerate(lines):
            normalized = self.normalize_log_line(line)
            if not self.is_strategy_c_key_line(line, normalized):
                continue
            key_hits += 1
            selected.update(self.expand_indexes(len(lines), index, STRATEGY_C_WINDOW_LINES))
            if self.is_strategy_c_block_seed(line, normalized):
                selected.update(self.collect_strategy_c_block_indexes(lines, index))
        return selected, key_hits

    def should_fallback_strategy_c(self, reduced_text: str, key_hits: int, filtered_lines: list[str], original_lines: list[str]) -> bool:
        if not reduced_text.strip():
            return True
        if key_hits <= 0:
            return True
        if len(original_lines) >= 300 and len(reduced_text.splitlines()) < 20:
            return True
        if len(filtered_lines) >= 300 and len(reduced_text) < 500:
            return True
        return False

    def build_strategy_c_regex_text(self, text: str, run_id: str = "") -> str:
        lines = text.splitlines()
        if not lines:
            return ""

        filtered_lines, raw_key_hits, noise_removed = self.filter_strategy_c_noise_lines(lines)
        if not filtered_lines:
            logger.warning(f"[mc_log][{run_id}] C策略正则去噪后为空，回退到错误聚焦提取")
            return self.build_error_focused_text(text)

        selected, filtered_key_hits = self.collect_strategy_c_indexes(filtered_lines)
        reduced = self.compose_lines_with_gaps(filtered_lines, sorted(selected))
        if self.should_fallback_strategy_c(reduced, filtered_key_hits, filtered_lines, lines):
            logger.warning(
                f"[mc_log][{run_id}] C策略正则去噪有效命中不足，回退到错误聚焦提取: "
                f"raw_key_hits={raw_key_hits}, filtered_key_hits={filtered_key_hits}, filtered_lines={len(filtered_lines)}"
            )
            return self.build_error_focused_text(text)

        logger.info(
            f"[mc_log][{run_id}] C策略正则去噪完成: total_lines={len(lines)}, filtered_lines={len(filtered_lines)}, "
            f"selected_lines={len(selected)}, raw_key_hits={raw_key_hits}, filtered_key_hits={filtered_key_hits}, "
            f"noise_removed={noise_removed}"
        )
        return reduced

    def collect_error_window_indexes(self, lines: list[str], window: int, max_hits: int) -> list[int]:
        hits = []
        for index, line in enumerate(lines):
            if ERROR_LINE_RE.search(line):
                hits.append(index)
                if len(hits) >= max_hits:
                    break
        selected = set()
        for index in hits:
            lo = max(0, index - window)
            hi = min(len(lines), index + window + 1)
            selected.update(range(lo, hi))
        return sorted(selected)

    def compose_lines_with_gaps(self, lines: list[str], indexes: list[int]) -> str:
        if not indexes:
            return ""
        result = []
        prev = -2
        for idx in indexes:
            if idx < 0 or idx >= len(lines):
                continue
            if idx != prev + 1:
                result.append("...[中间内容已省略]...")
            result.append(lines[idx])
            prev = idx
        return "\n".join(result).strip()

    def extract_must_keep_windows(self, content: str, window: int) -> list[str]:
        if not content:
            return []
        lines = content.splitlines()
        if not lines:
            return []
        max_window = max(1, int(window))

        earliest_fatal = None
        for index, line in enumerate(lines):
            if re.search(r"\b(FATAL|ERROR)\b", line, re.IGNORECASE):
                earliest_fatal = index
                break

        cause_indexes = [index for index, line in enumerate(lines) if CAUSE_LINE_RE.search(line)]
        deepest_cause = max(cause_indexes) if cause_indexes else None
        version_indexes = [index for index, line in enumerate(lines) if VERSION_LINE_RE.search(line)]
        crash_saved_indexes = [index for index, line in enumerate(lines) if CRASH_SAVED_RE.search(line)]

        def window_block(center: int | None, label: str) -> str | None:
            if center is None:
                return None
            lo = max(0, center - max_window)
            hi = min(len(lines), center + max_window + 1)
            block = lines[lo:hi]
            if not block:
                return None
            return f"[must_keep:{label}]\n" + "\n".join(block)

        blocks = []
        for label, center in (("deepest_cause", deepest_cause), ("earliest_fatal", earliest_fatal)):
            block = window_block(center, label)
            if block:
                blocks.append(block)

        if version_indexes:
            seen = set()
            for index in version_indexes[:6]:
                if index in seen:
                    continue
                seen.add(index)
                block = window_block(index, "version_line")
                if block:
                    blocks.append(block)

        if crash_saved_indexes:
            for index in crash_saved_indexes[:2]:
                block = window_block(index, "crash_saved")
                if block:
                    blocks.append(block)
        return blocks

    def merge_with_must_keep(self, base_text: str, content: str) -> str:
        cfg = self._cfg()
        window = int(cfg.get("must_keep_window_lines", 30))
        must_keep_blocks = self.extract_must_keep_windows(content, window)
        if not must_keep_blocks:
            return base_text
        header = "【MustKeep Evidence】"
        return "\n\n".join([header] + must_keep_blocks + ["【Selected Evidence】", base_text])

    def apply_budget_with_must_keep(self, base_text: str, content: str, limit: int) -> str:
        merged = self.merge_with_must_keep(base_text, content)
        if len(merged) <= limit:
            return merged
        cfg = self._cfg()
        window = int(cfg.get("must_keep_window_lines", 30))
        must_keep_blocks = self.extract_must_keep_windows(content, window)
        if not must_keep_blocks:
            return merged[:limit]
        header = "【MustKeep Evidence】"
        must_keep = "\n\n".join([header] + must_keep_blocks)
        if len(must_keep) >= limit:
            return must_keep[:limit]
        remain = limit - len(must_keep) - len("\n\n【Selected Evidence】\n")
        clipped = base_text[: max(0, remain)]
        return must_keep + "\n\n【Selected Evidence】\n" + clipped

    def apply_total_budget(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        focused = self.build_error_focused_text(text)
        if len(focused) <= limit:
            return focused
        return focused[:limit]

    def build_error_focused_text(self, text: str) -> str:
        lines = text.splitlines()
        if not lines:
            return ""
        selected = set()
        selected.update(range(0, min(len(lines), 180)))
        selected.update(range(max(0, len(lines) - 180), len(lines)))
        selected.update(self.collect_error_window_indexes(lines, window=8, max_hits=400))
        ts_count = 0
        for index, line in enumerate(lines):
            if TIMESTAMP_LINE_RE.search(line):
                selected.add(index)
                ts_count += 1
                if ts_count >= 240:
                    break
        merged = self.compose_lines_with_gaps(lines, sorted(selected))
        return merged or "\n".join(lines[:500])
