from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Iterable

from astrbot.api import logger


MAX_SECTION_LINES = 400
STRATEGY_C_HEAD_TAIL_LINES = 180
STRATEGY_C_WINDOW_LINES = 8
STRATEGY_C_STRONG_SCORE = 4
ERROR_LINE_RE = re.compile(
    r"(ERROR|WARN|Exception|Caused by|FATAL|Failed|Stacktrace|Problematic frame|siginfo)",
    re.IGNORECASE,
)
TIMESTAMP_LINE_RE = re.compile(
    r"^\s*(?:"
    r"\[\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?\]|"
    r"\[\d{1,2}[A-Za-z]{3}\d{4}\s+\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?\]|"
    r"\[\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?\]|"
    r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?|"
    r"\[\d{2}[/:]\d{2}[/:]\d{2}(?:\.\d{1,3})?\]"
    r")",
    re.I,
)
CAUSE_LINE_RE = re.compile(r"^\s*Caused by(?:\s+\d+)?:", re.I)
VERSION_LINE_RE = re.compile(
    r"^\s*(?:"
    r"Loading\ Minecraft\b.*\b(?:Fabric\ Loader|Forge|NeoForge|Quilt\ Loader)\b|"
    r"Minecraft\ Version(?:\ ID)?\s*:|"
    r"Launched\ Version:|"
    r"(?:Fabric|Quilt)\s*Loader(?:\s*Version)?\s*:|"
    r"Forge(?:\s*Version)?\s*:|"
    r"NeoForge(?:\s*Version)?\s*:|"
    r"Java(?:\s*Version)?\s*:|"
    r"JVM\ Version:|"
    r"Runtime\ Version:|"
    r"Backend\ (?:library|API):"
    r")",
    re.I,
)
CRASH_SAVED_RE = re.compile(r"Crash report saved to", re.I)
MC_DIAGNOSTIC_LINE_RE = re.compile(
    r"(Exception caught during firing event|Loading errors encountered|Mod loading error has occurred|"
    r"Mod loading issue for|A potential solution has been determined|More details:|Details:|"
    r"InvocationTargetException|Failure message:|Exception message:)",
    re.I,
)
LOG_PREFIX_RE = re.compile(r"^\s*(?:\[[^\]]+\]\s*)+")
BRACKET_TOKEN_RE = re.compile(r"\[([^\]]+)\]")
LEVEL_TOKEN_RE = re.compile(r"^(.*?)/(TRACE|DEBUG|INFO|WARN|ERROR|FATAL)$", re.I)
LOG_EVENT_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"(?:\[\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?\]\s*)?"
    r"(?:\[[^\]/\]]+/(?:TRACE|DEBUG|INFO|WARN|ERROR|FATAL)\]\s*)+|"
    r"\[\d{1,2}[A-Za-z]{3}\d{4}\s+\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?\]\s*"
    r"(?:\[[^\]/\]]+/(?:TRACE|DEBUG|INFO|WARN|ERROR|FATAL)\]\s*)+"
    r")",
    re.I,
)
STACKTRACE_SEED_RE = re.compile(
    r"(Exception|Caused by(?:\s+\d+)?:|Stacktrace|NoClassDefFoundError|ClassNotFoundException|"
    r"ClassMetadataNotFoundException|NoSuchMethodError|NoSuchFieldError|Problematic frame|siginfo|"
    r"Mixin.*(?:error|fail)|MixinApplyError|InvalidMixinException|InjectionError|InvalidInjectionException|"
    r"Loading errors encountered|Mod loading error has occurred|Mod loading issue for|InvocationTargetException|"
    r"Could not execute entrypoint|Attempted to load class|Failed to validate datapack|"
    r"Errors? in currently selected data\s*packs?|shader compilation failed|failed to compile shader)",
    re.I,
)
STACKTRACE_CONT_RE = re.compile(
    r"(^at\s+\S+\(.*\)$|^\.\.\. \d+ more$|^Suppressed:|^Caused by(?:\s+\d+)?:|^Exception in thread\b|"
    r"^[A-Za-z0-9_.$]+(?:Exception|Error|Throwable|LinkageError)(?::|\b)|^-- .* --$|^Description:|"
    r"^Details:|^Failure message:|^Exception message:|^Mod File:|^Mod file:|^Mod version:|"
    r"^Suspected Mods:|^Stacktrace:|^More details:|^Unmet dependency listing:|"
    r"^A potential solution has been determined|^\s*-\s+.+(?:requires|missing|wrong version|Install|Replace)|"
    r"^Currently,\s+.+\s+is\s+not\s+installed)",
    re.I,
)
DEPENDENCY_HEADER_RE = re.compile(
    r"(Incompatible mod set!?|Some of your mods are incompatible|"
    r"Mod resolution encountered an incompatible mod set|"
    r"Missing(?: or unsupported mandatory)? dependencies|Unmet dependency listing|"
    r"A potential solution has been determined)",
    re.I,
)
DEPENDENCY_DETAIL_RE = re.compile(
    r"^\s*(?:-\s+(?:Install|Replace)\s+.+|-\s+Mod\s+'.+?'\s+\(.+?\).+?"
    r"(?:requires|breaks|conflicts).+|.+\bwhich\s+is\s+missing\b.*|"
    r".+\bbut\s+only\s+the\s+wrong\s+version\s+is\s+present\b.*|"
    r"Currently,\s+.+\s+is\s+not\s+installed\.?)$",
    re.I,
)
DEPENDENCY_LINE_RE = re.compile(
    r"(Missing(?: or unsupported mandatory)? dependencies|Could not find required mod|requires .+\{|"
    r"minimum version|dependency|depends on)",
    re.I,
)
CLASSLOAD_LINE_RE = re.compile(
    r"(NoClassDefFoundError|ClassNotFoundException|NoSuchMethodError|NoSuchFieldError|ClassMetadataNotFoundException|"
    r"AbstractMethodError|BootstrapMethodError|VerifyError|IncompatibleClassChangeError|IllegalAccessError|"
    r"UnsupportedClassVersionError|ExceptionInInitializerError|Mixin(?: apply)? failed|MixinApplyError|"
    r"MixinTransformerError|InvalidMixinException|InjectionError|InvalidInjectionException|EntrypointException|"
    r"Failed to (?:load|apply|initialize|create mod instance|parse|validate)|"
    r"Could not execute entrypoint|Attempted to load class|invalid dist|has failed to load correctly)",
    re.I,
)
MOD_INFO_LINE_RE = re.compile(
    r"(^\s*(?:Mod File|Mod file|Mod version|Mod List|Suspected Mods|Loaded mod|Found mod)\b|"
    r"\b(?:entrypoint|coremod|transformer)\b)",
    re.I,
)
DATAPACK_FAILURE_RE = re.compile(
    r"(Failed to validate datapack|Errors? in currently selected data\s*packs?|"
    r"failed to parse .*\.json|could(?:n't| not) parse (?:loot table|recipe|tag)|"
    r"could not load tag|duplicate key .*tags|loot table .* (?:error|failed|invalid))",
    re.I,
)
SHADER_FAILURE_RE = re.compile(
    r"(shader compilation failed|failed to compile shader|error compiling shader|OpenGL error|"
    r"GL error|program link failed|Iris encountered an issue trying to load the shader)",
    re.I,
)
LOW_VALUE_ASSET_RE = re.compile(
    r"\b(texture|sprite|atlas|baked model|modelbakery|font|glyph|particle|"
    r"loaded recipe|tag loader|advancement|language load|stitching texture atlas)\b",
    re.I,
)
RESOURCE_RELOAD_NOISE_RE = re.compile(
    r"\b(reloading resources|resource reload|registered resource pack|found resource pack|resourcemanager)\b",
    re.I,
)
NOISE_LINE_PATTERNS = (
    LOW_VALUE_ASSET_RE,
    re.compile(r"\b(soundengine|openal|channel access|loaded sound|playing sound|audio stream)\b", re.I),
    RESOURCE_RELOAD_NOISE_RE,
    re.compile(r"\b(recipe book|recipe manager)\b", re.I),
    re.compile(
        r"\b(preparing spawn area|download terrain|chunk render|render distance|stitching texture atlas)\b",
        re.I,
    ),
    re.compile(r"^\s*(?:\[[^\]]*CHAT[^\]]*\]\s*)?<[^>]{1,32}>\s+", re.I),
)
C_NAME_RE = re.compile(r"(latest|fcl|pcl|game[-_ ]?output|run[-_ ]?output|崩溃前|游戏崩溃)", re.I)
B_NAME_RE = re.compile(r"\bdebug\b|日志|log", re.I)
MINECRAFT_RUN_HINT_RE = re.compile(
    r"\b(Loading Minecraft|Launched Version|Minecraft Version|Fabric Loader|NeoForge|Forge|Quilt Loader)\b",
    re.I,
)
LOCAL_USER_PATH_RE = re.compile(r"(?i)([A-Z]:\\Users\\|/home/|/Users/)([^/\\\s]+)")
IP_ENDPOINT_RE = re.compile(
    r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?::\d{2,5})?\b"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class ParsedLogLine:
    raw: str
    body: str
    severity: str | None = None
    thread: str | None = None
    logger: str | None = None
    timestamp: str | None = None


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
        result = self.apply_budget_with_must_keep(reduced, content, cfg["total_char_limit"])
        return self.redact_strategy_c_output(result)

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

    async def pick_priority_file(
        self,
        files: Iterable[Path],
        read_text_with_fallback: Callable[..., Awaitable[str]] | None = None,
        deadline: float | None = None,
    ) -> Path | None:
        priority_groups = [
            ("hs_err",),
            ("crash",),
            ("latest", "游戏崩溃"),
            ("日志",),
            ("log",),
            ("debug",),
            ("fcl",),
            ("pcl",),
        ]
        files_list = list(files)
        lowered = [(path, path.name.lower()) for path in files_list]
        logger.info(f"[mc_log] 归档候选文件: {[path.name for path in files_list]}")
        for keys in priority_groups:
            hits = []
            for path, name in lowered:
                matched = [key for key in keys if key in name]
                if matched:
                    hits.append((path, max(matched, key=len)))
            if hits:
                ranked_hits = []
                for hit_index, (path, key) in enumerate(hits):
                    char_count = -1
                    if read_text_with_fallback is not None:
                        try:
                            char_count = len(await read_text_with_fallback(path, deadline=deadline))
                        except Exception as exc:
                            logger.warning(f"[mc_log] 归档候选文件读取失败: file={path.name}, error={exc}")
                    if char_count < 0:
                        try:
                            char_count = path.stat().st_size
                        except Exception:
                            char_count = 0
                    ranked_hits.append((path, key, char_count, hit_index))
                path, key, char_count, _ = max(
                    ranked_hits,
                    key=lambda item: (item[2], len(item[1]), -item[3]),
                )
                logger.info(f"[mc_log] 归档优先命中: key={key}, file={path.name}, chars={char_count}")
                return path
        logger.debug(f"[mc_log] 归档内未找到优先关键词文件: {[path.name for path in files_list]}")
        return None

    def strategy_from_text_name(self, name_lower: str) -> str:
        if "hs_err" in name_lower or "crash" in name_lower:
            return "A"
        if C_NAME_RE.search(name_lower):
            return "C"
        if B_NAME_RE.search(name_lower):
            return "B"
        return "C"

    def looks_like_minecraft_run_log(self, head_lines: list[str]) -> bool:
        hits = 0
        for line in head_lines[:80]:
            if TIMESTAMP_LINE_RE.search(line) or LOG_EVENT_PREFIX_RE.search(line):
                hits += 1
            if MINECRAFT_RUN_HINT_RE.search(line):
                hits += 2
        return hits >= 6

    def strategy_from_name_and_peek(self, name_lower: str, content: str | None = None) -> str:
        strategy = self.strategy_from_text_name(name_lower)
        if strategy != "B" or "debug" in name_lower:
            return strategy
        if content and self.looks_like_minecraft_run_log(content.splitlines()[:80]):
            return "C"
        return strategy

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

    def parse_log_line(self, line: str) -> ParsedLogLine:
        tokens = BRACKET_TOKEN_RE.findall((line or "")[:240])
        severity = None
        thread = None
        logger_name = None
        timestamp = None

        for token in tokens:
            token = token.strip()
            if TIMESTAMP_LINE_RE.match(f"[{token}]"):
                timestamp = token
                continue
            level_match = LEVEL_TOKEN_RE.match(token)
            if not level_match:
                continue
            if thread is None:
                thread = level_match.group(1)
            severity = level_match.group(2).upper()
            logger_name = level_match.group(1)

        return ParsedLogLine(
            raw=line or "",
            body=self.normalize_log_line(line),
            severity=severity,
            thread=thread,
            logger=logger_name,
            timestamp=timestamp,
        )

    def is_strategy_c_key_line(self, line: str, normalized: str | None = None) -> bool:
        return self.strategy_c_line_score(line, normalized) > 0

    def strategy_c_line_score(self, line: str, normalized: str | None = None) -> int:
        parsed = self.parse_log_line(line)
        text = normalized if normalized is not None else parsed.body
        score = 0
        if parsed.severity == "WARN":
            score += 1
        elif parsed.severity == "ERROR":
            score += 3
        elif parsed.severity == "FATAL":
            score += 5

        raw_or_body = f"{parsed.raw}\n{text}"
        if VERSION_LINE_RE.search(text) or VERSION_LINE_RE.search(parsed.raw):
            score += 1
        if MOD_INFO_LINE_RE.search(text):
            score += 1
        if CRASH_SAVED_RE.search(text):
            score += 4
        if re.search(r"\bFailed\b|Stacktrace", text, re.I):
            score += 2
        if re.search(r"\bFATAL\b|Exception|Caused by(?:\s+\d+)?:|Problematic frame|siginfo", raw_or_body, re.I):
            score += 4
        if DEPENDENCY_HEADER_RE.search(text) or DEPENDENCY_DETAIL_RE.search(text) or DEPENDENCY_LINE_RE.search(text):
            score += 5
        if CLASSLOAD_LINE_RE.search(text) or MC_DIAGNOSTIC_LINE_RE.search(text):
            score += 5
        if DATAPACK_FAILURE_RE.search(text) or SHADER_FAILURE_RE.search(text):
            score += 5
        return score

    def is_strategy_c_noise_line(self, line: str, normalized: str | None = None) -> bool:
        text = normalized if normalized is not None else self.normalize_log_line(line)
        return any(pattern.search(text) for pattern in NOISE_LINE_PATTERNS)

    def has_strategy_c_failure_context(self, lines: list[str], index: int, radius: int = 5) -> bool:
        lo = max(0, index - radius)
        hi = min(len(lines), index + radius + 1)
        for candidate in lines[lo:hi]:
            body = self.normalize_log_line(candidate)
            if SHADER_FAILURE_RE.search(body) or DATAPACK_FAILURE_RE.search(body):
                return True
        return False

    def is_stacktrace_continuation_line(self, line: str, normalized: str | None = None) -> bool:
        text = normalized if normalized is not None else self.normalize_log_line(line)
        return bool(
            STACKTRACE_CONT_RE.search(text)
            or CLASSLOAD_LINE_RE.search(text)
            or DEPENDENCY_DETAIL_RE.search(text)
            or DATAPACK_FAILURE_RE.search(text)
            or SHADER_FAILURE_RE.search(text)
        )

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
            if LOG_EVENT_PREFIX_RE.search(lines[index]) and not self.is_stacktrace_continuation_line(
                lines[index], normalized
            ):
                break
            if TIMESTAMP_LINE_RE.search(lines[index]) and not self.is_stacktrace_continuation_line(
                lines[index], normalized
            ):
                break
            if not self.is_stacktrace_continuation_line(lines[index], normalized):
                break
            selected.add(index)
        return selected

    def collect_strategy_c_noise_stats(self, lines: list[str]) -> tuple[int, int]:
        key_hits = 0
        noise_candidates = 0
        for line in lines:
            normalized = self.normalize_log_line(line)
            if self.is_strategy_c_key_line(line, normalized):
                key_hits += 1
            elif self.is_strategy_c_noise_line(line, normalized):
                noise_candidates += 1
        return key_hits, noise_candidates

    def prune_strategy_c_noise_indexes(self, lines: list[str], indexes: Iterable[int]) -> tuple[list[int], int]:
        selected: list[int] = []
        noise_removed = 0
        for index in sorted(indexes):
            if index < 0 or index >= len(lines):
                continue
            line = lines[index]
            normalized = self.normalize_log_line(line)
            is_reload_noise = bool(RESOURCE_RELOAD_NOISE_RE.search(normalized))
            keep_reload_context = is_reload_noise and self.has_strategy_c_failure_context(lines, index)
            if (
                self.is_strategy_c_noise_line(line, normalized)
                and self.strategy_c_line_score(line, normalized) <= 0
                and not keep_reload_context
            ):
                noise_removed += 1
                continue
            selected.append(index)
        return selected, noise_removed

    def collect_strategy_c_indexes(self, lines: list[str]) -> tuple[set[int], int, int]:
        selected = set(range(0, min(len(lines), STRATEGY_C_HEAD_TAIL_LINES)))
        selected.update(range(max(0, len(lines) - STRATEGY_C_HEAD_TAIL_LINES), len(lines)))
        key_hits = 0
        strong_key_hits = 0
        for index, line in enumerate(lines):
            normalized = self.normalize_log_line(line)
            score = self.strategy_c_line_score(line, normalized)
            if score <= 0:
                continue
            key_hits += 1
            if score >= STRATEGY_C_STRONG_SCORE:
                strong_key_hits += 1
            if score >= 9:
                radius = 20
            elif score >= STRATEGY_C_STRONG_SCORE:
                radius = 12
            else:
                radius = 4
            selected.update(self.expand_indexes(len(lines), index, radius))
            if self.is_strategy_c_block_seed(line, normalized):
                selected.update(self.collect_strategy_c_block_indexes(lines, index))
        return selected, key_hits, strong_key_hits

    def should_fallback_strategy_c(
        self,
        reduced_text: str,
        key_hits: int,
        strong_key_hits: int,
        original_lines: list[str],
    ) -> bool:
        if not reduced_text.strip():
            return True
        if key_hits <= 0:
            return True
        if len(original_lines) >= 300 and strong_key_hits <= 0:
            return True
        if strong_key_hits > 0:
            return False
        if len(original_lines) >= 300 and len(reduced_text.splitlines()) < 20:
            return True
        if len(original_lines) >= 300 and len(reduced_text) < 500:
            return True
        return False

    def build_strategy_c_regex_text(self, text: str, run_id: str = "") -> str:
        lines = text.splitlines()
        if not lines:
            return ""

        raw_key_hits, noise_candidates = self.collect_strategy_c_noise_stats(lines)
        selected, key_hits, strong_key_hits = self.collect_strategy_c_indexes(lines)
        pruned_selected, noise_removed = self.prune_strategy_c_noise_indexes(lines, selected)
        reduced = self.compose_lines_with_gaps(lines, pruned_selected)
        if self.should_fallback_strategy_c(reduced, key_hits, strong_key_hits, lines):
            logger.warning(
                f"[mc_log][{run_id}] C策略正则去噪有效命中不足，回退到错误聚焦提取: "
                f"raw_key_hits={raw_key_hits}, key_hits={key_hits}, strong_key_hits={strong_key_hits}, "
                f"noise_candidates={noise_candidates}"
            )
            return self.redact_strategy_c_output(self.build_error_focused_text(text))

        logger.info(
            f"[mc_log][{run_id}] C策略正则去噪完成: total_lines={len(lines)}, "
            f"selected_lines={len(selected)}, pruned_selected_lines={len(pruned_selected)}, "
            f"raw_key_hits={raw_key_hits}, key_hits={key_hits}, strong_key_hits={strong_key_hits}, "
            f"noise_candidates={noise_candidates}, noise_removed={noise_removed}"
        )
        return self.redact_strategy_c_output(reduced)

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
        version_indexes = [
            index
            for index, line in enumerate(lines)
            if VERSION_LINE_RE.search(line) or VERSION_LINE_RE.search(self.normalize_log_line(line))
        ]
        crash_saved_indexes = [index for index, line in enumerate(lines) if CRASH_SAVED_RE.search(line)]
        loader_anchor_patterns = (
            ("dependency", DEPENDENCY_HEADER_RE),
            ("mod_loading", MC_DIAGNOSTIC_LINE_RE),
            ("classload", CLASSLOAD_LINE_RE),
            ("datapack", DATAPACK_FAILURE_RE),
            ("shader", SHADER_FAILURE_RE),
        )
        loader_anchor_indexes: list[tuple[str, int]] = []
        for label, pattern in loader_anchor_patterns:
            index = self.first_match_index(
                [self.normalize_log_line(line) for line in lines],
                pattern,
            )
            if index is not None:
                loader_anchor_indexes.append((label, index))

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

        seen_loader_labels = set()
        for label, index in loader_anchor_indexes:
            if label in seen_loader_labels:
                continue
            seen_loader_labels.add(label)
            block = window_block(index, label)
            if block:
                blocks.append(block)
        return blocks

    def redact_strategy_c_output(self, text: str) -> str:
        if not text:
            return text
        out = LOCAL_USER_PATH_RE.sub(r"\1<user>", text)
        out = IP_ENDPOINT_RE.sub("<ip>", out)
        out = ANSI_RE.sub("", out)
        out = CONTROL_RE.sub("", out)
        return out

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
