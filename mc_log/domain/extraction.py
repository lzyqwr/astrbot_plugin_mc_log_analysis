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
DEFAULT_STRATEGY_C_MUST_KEEP_WINDOW_LINES = 12
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
FAILED_STACKTRACE_RE = re.compile(r"\bFailed\b|Stacktrace", re.I)
FATAL_EXCEPTION_RE = re.compile(
    r"\bFATAL\b|Exception|Caused by(?:\s+\d+)?:|Problematic frame|siginfo",
    re.I,
)
OOM_RE = re.compile(r"\b(?:OutOfMemoryError|Java heap space|GC overhead limit exceeded)\b", re.I)
WATCHDOG_RE = re.compile(
    r"(Watchdog.*Considering it to be crashed|took too long|"
    r"server watchdog|A single server tick took \d+)",
    re.I,
)
TICKING_KEEP_UP_RE = re.compile(
    r"(Can't keep up!.*(?:Ticking entity|Ticking block entity)|"
    r"(?:Ticking entity|Ticking block entity).*Can't keep up!)",
    re.I,
)
ORDERED_STOP_RE = re.compile(r"^\s*(?:Stopping!|Stopping server)\b", re.I)
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
    re.compile(r"CHAT_NOISE", re.I),
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
FOLD_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.I,
)
FOLD_HEX_ID_RE = re.compile(r"\b0x[0-9a-f]+\b", re.I)
FOLD_ENTITY_ID_RE = re.compile(r"\b(entity|id|entity id|eid)[=: #]+-?\d+\b", re.I)
FOLD_COORD_TRIPLE_RE = re.compile(
    r"(?<![\w.])-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?(?![\w.])"
)
FOLD_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")


@dataclass(frozen=True)
class ParsedLogLine:
    raw: str
    body: str
    severity: str | None = None
    thread: str | None = None
    logger: str | None = None
    timestamp: str | None = None


@dataclass(frozen=True)
class StrategyCLineRecord:
    index: int
    raw: str
    parsed: ParsedLogLine
    normalized: str
    score: int
    is_noise: bool


@dataclass
class OutputLineGroup:
    start: int
    end: int
    text: str
    repeat_count: int = 1
    repeat_label: str = "重复"


class ExtractionDomain:
    def __init__(self, cfg_getter: Callable[[], object]):
        self._cfg_getter = cfg_getter

    def _cfg(self):
        return self._cfg_getter()

    async def extract_path_by_strategy(
        self,
        path: Path,
        read_text_with_fallback: Callable[..., Awaitable[str]],
        run_id: str = "",
        deadline: float | None = None,
    ) -> tuple[str, str]:
        name_lower = path.name.lower()
        strategy = self.strategy_from_text_name(name_lower)
        if strategy == "B" and "debug" not in name_lower:
            try:
                peek_content = await read_text_with_fallback(path, deadline=deadline)
                strategy = self.strategy_from_name_and_peek(name_lower, peek_content)
            except Exception as exc:
                logger.warning(
                    f"[mc_log][{run_id}] 文本文件内容嗅探失败，沿用文件名策略: file={path.name}, error={exc}"
                )
        logger.info(
            f"[mc_log][{run_id}] 文本文件策略判定: file={path.name}, strategy={strategy}"
        )
        if strategy == "A":
            kind = "hs_err" if "hs_err" in name_lower else "crash"
            content = await self.strategy_a_extract(
                path,
                kind,
                read_text_with_fallback,
                deadline=deadline,
            )
            return content, strategy
        if strategy == "B":
            content = await self.strategy_b_extract(
                path, read_text_with_fallback, deadline=deadline
            )
            return content, strategy

        content = await self.strategy_c_extract(
            path,
            read_text_with_fallback,
            run_id=run_id,
            deadline=deadline,
        )
        return content, strategy

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
        return self.build_strategy_c_regex_text(
            content,
            run_id=run_id,
            budget_limit=cfg["total_char_limit"],
        )

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
        level_tokens: list[tuple[str, str]] = []
        fallback_logger = None
        saw_level_token = False

        for token in tokens:
            token = token.strip()
            if TIMESTAMP_LINE_RE.match(f"[{token}]"):
                timestamp = token
                continue
            level_match = LEVEL_TOKEN_RE.match(token)
            if level_match:
                saw_level_token = True
                level_tokens.append((level_match.group(1), level_match.group(2).upper()))
                continue
            if saw_level_token and fallback_logger is None:
                fallback_logger = token

        if level_tokens:
            thread = level_tokens[0][0]
            severity = level_tokens[-1][1]
            if len(level_tokens) > 1:
                logger_name = level_tokens[-1][0]
            else:
                logger_name = fallback_logger

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

    def strategy_c_line_score(
        self,
        line: str,
        normalized: str | None = None,
        parsed: ParsedLogLine | None = None,
    ) -> int:
        parsed = parsed or self.parse_log_line(line)
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
        if FAILED_STACKTRACE_RE.search(text):
            score += 2
        if FATAL_EXCEPTION_RE.search(raw_or_body):
            score += 4
        if DEPENDENCY_HEADER_RE.search(text) or DEPENDENCY_DETAIL_RE.search(text) or DEPENDENCY_LINE_RE.search(text):
            score += 5
        if CLASSLOAD_LINE_RE.search(text) or MC_DIAGNOSTIC_LINE_RE.search(text):
            score += 5
        if DATAPACK_FAILURE_RE.search(text) or SHADER_FAILURE_RE.search(text):
            score += 5
        if OOM_RE.search(raw_or_body):
            score += 8
        if WATCHDOG_RE.search(raw_or_body):
            score += 8
        if TICKING_KEEP_UP_RE.search(raw_or_body):
            score += 6
        if ORDERED_STOP_RE.search(text):
            score += 4
        return score

    def is_strategy_c_noise_line(self, line: str, normalized: str | None = None) -> bool:
        text = normalized if normalized is not None else self.normalize_log_line(line)
        return any(pattern.search(text) for pattern in NOISE_LINE_PATTERNS)

    def build_strategy_c_records(self, lines: list[str]) -> list[StrategyCLineRecord]:
        records: list[StrategyCLineRecord] = []
        for index, line in enumerate(lines):
            parsed = self.parse_log_line(line)
            normalized = parsed.body
            score = self.strategy_c_line_score(line, normalized, parsed)
            is_noise = self.is_strategy_c_noise_line(line, normalized)
            records.append(
                StrategyCLineRecord(
                    index=index,
                    raw=line,
                    parsed=parsed,
                    normalized=normalized,
                    score=score,
                    is_noise=is_noise,
                )
            )
        return records

    def has_strategy_c_failure_context(
        self,
        lines: list[str],
        index: int,
        radius: int = 5,
        records: list[StrategyCLineRecord] | None = None,
    ) -> bool:
        size = len(records) if records is not None else len(lines)
        lo = max(0, index - radius)
        hi = min(size, index + radius + 1)
        for offset in range(lo, hi):
            body = records[offset].normalized if records is not None else self.normalize_log_line(lines[offset])
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

    def collect_strategy_c_block_indexes_from_records(
        self,
        records: list[StrategyCLineRecord],
        start_index: int,
    ) -> set[int]:
        selected = {start_index}
        for index in range(start_index + 1, len(records)):
            record = records[index]
            normalized = record.normalized
            if not normalized:
                break
            if LOG_EVENT_PREFIX_RE.search(record.raw) and not self.is_stacktrace_continuation_line(
                record.raw, normalized
            ):
                break
            if TIMESTAMP_LINE_RE.search(record.raw) and not self.is_stacktrace_continuation_line(
                record.raw, normalized
            ):
                break
            if not self.is_stacktrace_continuation_line(record.raw, normalized):
                break
            selected.add(index)
        return selected

    def collect_strategy_c_noise_stats(self, lines: list[str]) -> tuple[int, int]:
        return self.collect_strategy_c_noise_stats_from_records(
            self.build_strategy_c_records(lines)
        )

    def collect_strategy_c_noise_stats_from_records(
        self, records: list[StrategyCLineRecord]
    ) -> tuple[int, int]:
        key_hits = 0
        noise_candidates = 0
        for record in records:
            if record.score > 0:
                key_hits += 1
            elif record.is_noise:
                noise_candidates += 1
        return key_hits, noise_candidates

    def prune_strategy_c_noise_indexes(self, lines: list[str], indexes: Iterable[int]) -> tuple[list[int], int]:
        return self.prune_strategy_c_noise_indexes_from_records(
            self.build_strategy_c_records(lines), indexes
        )

    def prune_strategy_c_noise_indexes_from_records(
        self,
        records: list[StrategyCLineRecord],
        indexes: Iterable[int],
    ) -> tuple[list[int], int]:
        selected: list[int] = []
        noise_removed = 0
        for index in sorted(indexes):
            if index < 0 or index >= len(records):
                continue
            record = records[index]
            normalized = record.normalized
            is_reload_noise = bool(RESOURCE_RELOAD_NOISE_RE.search(normalized))
            keep_reload_context = is_reload_noise and self.has_strategy_c_failure_context(
                [], index, records=records
            )
            if (
                record.is_noise
                and record.score <= 0
                and not keep_reload_context
            ):
                noise_removed += 1
                continue
            selected.append(index)
        return selected, noise_removed

    def collect_strategy_c_indexes(self, lines: list[str]) -> tuple[set[int], int, int]:
        return self.collect_strategy_c_indexes_from_records(self.build_strategy_c_records(lines))

    def collect_strategy_c_indexes_from_records(
        self, records: list[StrategyCLineRecord]
    ) -> tuple[set[int], int, int]:
        selected = set(range(0, min(len(records), STRATEGY_C_HEAD_TAIL_LINES)))
        selected.update(range(max(0, len(records) - STRATEGY_C_HEAD_TAIL_LINES), len(records)))
        key_hits = 0
        strong_key_hits = 0
        for index, record in enumerate(records):
            score = record.score
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
            selected.update(self.expand_indexes(len(records), index, radius))
            if self.is_strategy_c_block_seed(record.raw, record.normalized):
                selected.update(self.collect_strategy_c_block_indexes_from_records(records, index))
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

    def should_compact_log_prefixes(self) -> bool:
        return bool(self._cfg().get("strategy_c_compact_prefix", True))

    def should_keep_strategy_c_timestamps(self) -> bool:
        return bool(self._cfg().get("strategy_c_keep_timestamps", False))

    def format_log_line_for_output(
        self,
        line: str,
        parsed: ParsedLogLine | None = None,
        compact_prefix: bool | None = None,
    ) -> str:
        if compact_prefix is None:
            compact_prefix = self.should_compact_log_prefixes()
        if not compact_prefix:
            return line
        parsed = parsed or self.parse_log_line(line)
        if not parsed.severity:
            return line

        parts: list[str] = []
        if self.should_keep_strategy_c_timestamps() and parsed.timestamp:
            parts.append(parsed.timestamp)

        thread = (parsed.thread or "").strip()
        if thread and thread.lower() not in {"main", "render thread"}:
            parts.append(thread)

        level_part = parsed.severity
        if parsed.logger:
            level_part = f"{level_part} {parsed.logger}"
        parts.append(level_part)
        return f"[{'/'.join(parts)}]: {parsed.body}"

    def normalize_line_for_body_fold(self, text: str) -> str:
        folded = FOLD_UUID_RE.sub("<uuid>", text or "")
        folded = FOLD_HEX_ID_RE.sub("<id>", folded)
        folded = FOLD_ENTITY_ID_RE.sub(r"\1=<id>", folded)
        folded = FOLD_COORD_TRIPLE_RE.sub("<coords>", folded)
        folded = FOLD_NUMBER_RE.sub("<num>", folded)
        return re.sub(r"\s+", " ", folded).strip().lower()

    def normalize_line_for_tail_fold(self, record: StrategyCLineRecord) -> str:
        without_timestamp = TIMESTAMP_LINE_RE.sub("", record.raw, count=1)
        return without_timestamp.strip()

    def fold_strategy_c_output_groups(
        self,
        records: list[StrategyCLineRecord],
        indexes: Iterable[int],
    ) -> list[OutputLineGroup]:
        groups: list[OutputLineGroup] = []
        tail_start = max(0, len(records) - STRATEGY_C_HEAD_TAIL_LINES)
        prev_key: str | None = None
        prev_mode = ""

        for index in sorted(indexes):
            if index < 0 or index >= len(records):
                continue
            record = records[index]
            in_tail = index >= tail_start
            if in_tail:
                mode = "tail"
                key = self.normalize_line_for_tail_fold(record)
            elif record.score >= STRATEGY_C_STRONG_SCORE:
                mode = "body_exact"
                key = record.normalized
            else:
                mode = "body"
                key = self.normalize_line_for_body_fold(record.normalized)

            can_fold = False
            if groups and mode == prev_mode and key == prev_key:
                if mode == "tail":
                    can_fold = index == groups[-1].end + 1
                else:
                    can_fold = True

            if can_fold:
                groups[-1].end = index
                groups[-1].repeat_count += 1
                groups[-1].repeat_label = "连续重复" if mode == "tail" else "重复"
            else:
                groups.append(
                    OutputLineGroup(
                        start=index,
                        end=index,
                        text=self.format_log_line_for_output(record.raw, record.parsed),
                        repeat_label="连续重复" if mode == "tail" else "重复",
                    )
                )
            prev_key = key
            prev_mode = mode
        return groups

    def compose_line_groups_with_gaps(self, groups: list[OutputLineGroup]) -> str:
        if not groups:
            return ""
        result: list[str] = []
        prev_end = -2
        for group in groups:
            if group.start != prev_end + 1:
                result.append("...[中间内容已省略]...")
            text = group.text
            if group.repeat_count > 1:
                text = f"{text} ({group.repeat_label} {group.repeat_count} 次)"
            result.append(text)
            prev_end = group.end
        return "\n".join(result).strip()

    def compose_strategy_c_records_with_gaps(
        self,
        records: list[StrategyCLineRecord],
        indexes: Iterable[int],
    ) -> str:
        return self.compose_line_groups_with_gaps(
            self.fold_strategy_c_output_groups(records, indexes)
        )

    def join_strategy_c_parts(self, termination_note: str, body: str) -> str:
        return "\n".join(part for part in (termination_note, body) if part)

    def compose_strategy_c_budgeted_text(
        self,
        records: list[StrategyCLineRecord],
        indexes: Iterable[int],
        termination_note: str,
        limit: int | None = None,
    ) -> str:
        selected = sorted(set(indexes))
        body = self.compose_strategy_c_records_with_gaps(records, selected)
        full = self.join_strategy_c_parts(termination_note, body)
        if not limit or len(full) <= limit:
            return full

        tail_start = max(0, len(records) - STRATEGY_C_HEAD_TAIL_LINES)
        head_indexes = [index for index in selected if index < STRATEGY_C_HEAD_TAIL_LINES]
        tail_indexes = [index for index in selected if index >= tail_start]
        strong_middle_indexes = [
            index
            for index in selected
            if STRATEGY_C_HEAD_TAIL_LINES <= index < tail_start
            and records[index].score >= STRATEGY_C_STRONG_SCORE
        ]

        protected_indexes = sorted(set(head_indexes + strong_middle_indexes + tail_indexes))
        protected = self.join_strategy_c_parts(
            termination_note,
            self.compose_strategy_c_records_with_gaps(records, protected_indexes),
        )
        if len(protected) <= limit:
            return protected

        head_tail_indexes = sorted(set(head_indexes + tail_indexes))
        head_tail = self.join_strategy_c_parts(
            termination_note,
            self.compose_strategy_c_records_with_gaps(records, head_tail_indexes),
        )
        if len(head_tail) <= limit:
            return head_tail

        tail_text = self.join_strategy_c_parts(
            termination_note,
            self.compose_strategy_c_records_with_gaps(records, tail_indexes),
        )
        if len(tail_text) <= limit:
            lo = 0
            hi = len(head_indexes)
            best = tail_indexes
            while lo <= hi:
                mid = (lo + hi) // 2
                candidate_indexes = sorted(set(head_indexes[:mid] + tail_indexes))
                candidate = self.join_strategy_c_parts(
                    termination_note,
                    self.compose_strategy_c_records_with_gaps(records, candidate_indexes),
                )
                if len(candidate) <= limit:
                    best = candidate_indexes
                    lo = mid + 1
                else:
                    hi = mid - 1
            return self.join_strategy_c_parts(
                termination_note,
                self.compose_strategy_c_records_with_gaps(records, best),
            )

        lo = 0
        hi = len(tail_indexes)
        best_tail: list[int] = []
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate_indexes = tail_indexes[len(tail_indexes) - mid :]
            candidate = self.join_strategy_c_parts(
                termination_note,
                self.compose_strategy_c_records_with_gaps(records, candidate_indexes),
            )
            if len(candidate) <= limit:
                best_tail = candidate_indexes
                lo = mid + 1
            else:
                hi = mid - 1
        if best_tail:
            return self.join_strategy_c_parts(
                termination_note,
                self.compose_strategy_c_records_with_gaps(records, best_tail),
            )
        return termination_note[:limit]

    def collect_error_focused_indexes(self, lines: list[str]) -> set[int]:
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
        return selected

    def collect_must_keep_indexes_from_records(
        self,
        records: list[StrategyCLineRecord],
        window: int,
    ) -> set[int]:
        if not records:
            return set()
        max_window = min(
            DEFAULT_STRATEGY_C_MUST_KEEP_WINDOW_LINES,
            max(1, int(window)),
        )
        lines = [record.raw for record in records]
        normalized_lines = [record.normalized for record in records]

        centers: list[int] = []
        earliest_fatal = None
        for record in records:
            if record.parsed.severity in {"FATAL", "ERROR"}:
                earliest_fatal = record.index
                break
        if earliest_fatal is not None:
            centers.append(earliest_fatal)

        cause_indexes = [
            record.index for record in records if CAUSE_LINE_RE.search(record.normalized)
        ]
        if cause_indexes:
            centers.append(max(cause_indexes))

        version_indexes = [
            record.index
            for record in records
            if VERSION_LINE_RE.search(record.raw) or VERSION_LINE_RE.search(record.normalized)
        ]
        centers.extend(version_indexes[:3])

        crash_saved_indexes = [
            record.index for record in records if CRASH_SAVED_RE.search(record.normalized)
        ]
        centers.extend(crash_saved_indexes[:1])

        loader_anchor_patterns = (
            DEPENDENCY_HEADER_RE,
            MC_DIAGNOSTIC_LINE_RE,
            CLASSLOAD_LINE_RE,
            DATAPACK_FAILURE_RE,
            SHADER_FAILURE_RE,
            OOM_RE,
            WATCHDOG_RE,
            TICKING_KEEP_UP_RE,
            ORDERED_STOP_RE,
        )
        for pattern in loader_anchor_patterns:
            index = self.first_match_index(normalized_lines, pattern)
            if index is not None:
                centers.append(index)

        selected: set[int] = set()
        for center in centers:
            selected.update(self.expand_indexes(len(lines), center, max_window))
        return selected

    def strategy_c_termination_note(self, records: list[StrategyCLineRecord]) -> str:
        if not records:
            return ""

        category = ""
        for record in reversed(records):
            raw_or_body = f"{record.raw}\n{record.normalized}"
            if OOM_RE.search(raw_or_body):
                category = "OOM"
            elif WATCHDOG_RE.search(raw_or_body):
                category = "watchdog卡死"
            elif TICKING_KEEP_UP_RE.search(raw_or_body):
                category = "tick卡死"
            elif ORDERED_STOP_RE.search(record.normalized):
                category = "有序关闭"
            if category:
                break

        last_record = None
        for record in reversed(records):
            if record.normalized and not record.is_noise:
                last_record = record
                break
        if last_record is None:
            return ""

        last_logger = last_record.parsed.logger or last_record.parsed.thread or "unknown"
        last_line = self.format_log_line_for_output(last_record.raw, last_record.parsed)
        if not category:
            category = "无声中断"
            hint = "；尾部无异常块/崩溃报告落点，真凶大概率在独立 hs_err_pid*.log 或启动器/系统日志"
        else:
            hint = ""
        return f"[termination] type={category}; last_logger={last_logger}; last_non_noise={last_line}{hint}"

    def build_strategy_c_regex_text(
        self,
        text: str,
        run_id: str = "",
        budget_limit: int | None = None,
    ) -> str:
        lines = text.splitlines()
        if not lines:
            return ""

        records = self.build_strategy_c_records(lines)
        raw_key_hits, noise_candidates = self.collect_strategy_c_noise_stats_from_records(records)
        selected, key_hits, strong_key_hits = self.collect_strategy_c_indexes_from_records(records)
        must_keep_window = int(
            self._cfg().get("must_keep_window_lines", DEFAULT_STRATEGY_C_MUST_KEEP_WINDOW_LINES)
        )
        must_keep_indexes = self.collect_must_keep_indexes_from_records(
            records, must_keep_window
        )
        selected_with_must_keep = set(selected) | must_keep_indexes
        unified_selected, noise_removed = self.prune_strategy_c_noise_indexes_from_records(
            records, selected_with_must_keep
        )
        termination_note = self.strategy_c_termination_note(records)
        reduced = self.compose_strategy_c_budgeted_text(
            records,
            unified_selected,
            termination_note,
            budget_limit,
        )
        if self.should_fallback_strategy_c(reduced, key_hits, strong_key_hits, lines):
            logger.warning(
                f"[mc_log][{run_id}] C策略正则去噪有效命中不足，回退到错误聚焦提取: "
                f"raw_key_hits={raw_key_hits}, key_hits={key_hits}, strong_key_hits={strong_key_hits}, "
                f"noise_candidates={noise_candidates}"
            )
            fallback_selected, _ = self.prune_strategy_c_noise_indexes_from_records(
                records,
                self.collect_error_focused_indexes(lines),
            )
            fallback = self.compose_strategy_c_budgeted_text(
                records,
                fallback_selected,
                termination_note,
                budget_limit,
            )
            return self.redact_strategy_c_output(fallback)

        logger.info(
            f"[mc_log][{run_id}] C策略正则去噪完成: total_lines={len(lines)}, "
            f"selected_lines={len(selected)}, must_keep_lines={len(must_keep_indexes)}, "
            f"unified_selected_lines={len(unified_selected)}, "
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

    def compose_lines_with_gaps(
        self,
        lines: list[str],
        indexes: list[int],
        compact_prefix: bool | None = None,
    ) -> str:
        if not indexes:
            return ""
        result = []
        prev = -2
        for idx in indexes:
            if idx < 0 or idx >= len(lines):
                continue
            if idx != prev + 1:
                result.append("...[中间内容已省略]...")
            result.append(
                self.format_log_line_for_output(lines[idx], compact_prefix=compact_prefix)
            )
            prev = idx
        return "\n".join(result).strip()

    def extract_must_keep_windows(self, content: str, window: int) -> list[str]:
        if not content:
            return []
        lines = content.splitlines()
        if not lines:
            return []
        records = self.build_strategy_c_records(lines)
        indexes = sorted(self.collect_must_keep_indexes_from_records(records, window))
        if not indexes:
            return []
        return ["[must_keep:deduped]\n" + self.compose_lines_with_gaps(lines, indexes)]

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
        selected = self.collect_error_focused_indexes(lines)
        merged = self.compose_lines_with_gaps(lines, sorted(selected))
        return merged or "\n".join(lines[:500])
