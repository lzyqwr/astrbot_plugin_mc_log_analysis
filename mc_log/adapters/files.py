from __future__ import annotations

import asyncio
import codecs
import gzip
import re
import shutil
import traceback
import uuid
import zipfile
from pathlib import Path, PurePosixPath

import aiohttp
from astrbot.api import logger

from ..config import (
    MAX_ARCHIVE_FILE_COUNT,
    MAX_ARCHIVE_SINGLE_FILE_BYTES,
    MAX_ARCHIVE_TOTAL_BYTES,
    MAX_GZ_OUTPUT_BYTES,
)
from ..domain.detection import detect_file_name_dummy, sanitize_uploaded_name
from ..models import BudgetExceeded


TEXT_LOG_HINT_RE = re.compile(
    r"(ERROR|WARN|INFO|Exception|Caused by|Minecraft|Loader|Forge|Fabric|NeoForge|Quilt|"
    r"Version|\.jar|\.log|\[[^\]]+\])",
    re.I,
)


class FileAdapter:
    def __init__(self, config_manager, runtime):
        self.config_manager = config_manager
        self.runtime = runtime

    def _cfg(self):
        return self.config_manager.get()

    async def copy_path(self, src: Path, dst: Path, deadline: float | None = None):
        await self.runtime.run_io_in_thread("copy_file", self.copy_file_blocking, src, dst, deadline=deadline)

    async def download_to_workdir(self, file_comp, work_dir: Path, deadline: float | None = None) -> Path | None:
        self.runtime.ensure_not_timed_out(deadline, stage="download_prepare")
        timeout = None if deadline is None else max(0.1, self.runtime.time_left(deadline))
        src = await asyncio.wait_for(file_comp.get_file(allow_return_url=True), timeout=timeout)
        if not src:
            logger.warning("[mc_log] 获取文件源失败：file_comp.get_file 返回空")
            return None

        detected_name = detect_file_name_dummy(file_comp)
        safe_name = sanitize_uploaded_name(detected_name or "upload.log")
        dst = work_dir / safe_name
        logger.info(
            f"[mc_log] 文件源已解析: detected_name={detected_name}, safe_name={safe_name}, "
            f"src_type={'url' if str(src).startswith(('http://', 'https://')) else 'path'}"
        )

        if str(src).startswith("http://") or str(src).startswith("https://"):
            try:
                timeout_sec = min(float(self._cfg()["file_download_timeout_sec"]), max(0.1, self.runtime.time_left(deadline)))
                logger.info(f"[mc_log] 准备下载远程文件: url={src}, dst={dst.name}, timeout={timeout_sec:.2f}s")
                await self.download_url_to_path(url=str(src), dst=dst, timeout_sec=timeout_sec)
                if dst.exists():
                    logger.info(f"[mc_log] 远程文件下载完成: bytes={dst.stat().st_size}, dst={dst.name}")
                return dst if dst.exists() else None
            except Exception as exc:
                if self.runtime.time_left(deadline) <= 0:
                    raise TimeoutError("global timeout reached during download") from exc
                logger.error(f"[mc_log] 远程文件下载失败: {exc}", exc_info=True)
                return None

        self.runtime.ensure_not_timed_out(deadline, stage="download_copy")
        src_path = Path(src)
        if not src_path.exists():
            logger.warning(f"[mc_log] 本地文件源不存在: {src_path}")
            return None
        max_bytes = int(self._cfg().get("max_input_file_bytes", 32 * 1024 * 1024))
        if src_path.stat().st_size > max_bytes:
            raise BudgetExceeded("local file exceeds max_input_file_bytes")
        if not safe_name or safe_name == "upload.log":
            safe_name = sanitize_uploaded_name(src_path.name or "upload.log")
            dst = work_dir / safe_name
        await self.runtime.run_io_in_thread("copy_local_file", self.copy_file_blocking, src_path, dst, deadline=deadline)
        logger.info(f"[mc_log] 已复制本地文件到工作目录: src={src_path.name}, dst={dst.name}, bytes={dst.stat().st_size}")
        return dst

    def copy_file_blocking(self, src_path: Path, dst: Path, deadline: float | None, cancel_event):
        self.runtime.check_deadline_or_cancel(deadline, cancel_event, "copy_file_start")
        dst.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with open(src_path, "rb") as src, open(dst, "wb") as out:
            while True:
                chunk = src.read(512 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
                if written % (4 * 1024 * 1024) == 0:
                    self.runtime.check_deadline_or_cancel(deadline, cancel_event, "copy_file_loop")
        try:
            shutil.copystat(src_path, dst)
        except Exception:
            pass

    async def read_bytes(self, path: Path, deadline: float | None = None) -> bytes:
        return await self.runtime.run_io_in_thread("read_bytes", self.read_bytes_blocking, path, deadline=deadline)

    def read_bytes_blocking(self, path: Path, deadline: float | None, cancel_event) -> bytes:
        self.runtime.check_deadline_or_cancel(deadline, cancel_event, "read_bytes_start")
        data = path.read_bytes()
        self.runtime.check_deadline_or_cancel(deadline, cancel_event, "read_bytes_done")
        return data

    async def download_url_to_path(self, url: str, dst: Path, timeout_sec: float):
        headers = {"User-Agent": "AstrBot-MC-Log-Analyzer/1.0"}
        timeout = aiohttp.ClientTimeout(total=timeout_sec)
        async with aiohttp.ClientSession(trust_env=True, timeout=timeout) as session:
            async with session.get(url, headers=headers) as resp:
                resp.raise_for_status()
                max_bytes = int(self._cfg().get("max_input_file_bytes", 32 * 1024 * 1024))
                content_len = resp.headers.get("Content-Length")
                if content_len and content_len.isdigit() and int(content_len) > max_bytes:
                    raise BudgetExceeded("remote file exceeds max_input_file_bytes")
                dst.parent.mkdir(parents=True, exist_ok=True)
                with open(dst, "wb") as file_obj:
                    written = 0
                    async for chunk in resp.content.iter_chunked(64 * 1024):
                        if chunk:
                            written += len(chunk)
                            if written > max_bytes:
                                raise BudgetExceeded("remote file exceeds max_input_file_bytes")
                            file_obj.write(chunk)

    def read_text_with_fallback_blocking(self, path: Path, deadline: float | None, cancel_event) -> str:
        self.runtime.check_deadline_or_cancel(deadline, cancel_event, "read_text_start")
        raw = path.read_bytes()
        self.runtime.check_deadline_or_cancel(deadline, cancel_event, "read_text_after_read")
        text, diag = self._decode_text_with_diagnostics(raw)
        log_fn = logger.warning if diag["used_replace"] else logger.info
        log_fn(
            f"[mc_log][text] 文本解码完成: file={path.name}, bytes={diag['raw_bytes']}, chars={diag['char_count']}, "
            f"encoding={diag['encoding']}, bom={diag['bom']}, utf16_hint={diag['utf16_hint']}, "
            f"used_replace={diag['used_replace']}"
        )
        return text

    def _decode_text_with_diagnostics(self, raw: bytes) -> tuple[str, dict[str, object]]:
        bom_encoding, bom_name = self._detect_text_bom(raw)
        utf16_hint = False
        if bom_encoding:
            text = raw.decode(bom_encoding)
            return text, {
                "encoding": bom_encoding,
                "bom": bom_name,
                "utf16_hint": False,
                "used_replace": False,
                "raw_bytes": len(raw),
                "char_count": len(text),
            }

        utf16_hint = self._looks_like_utf16_without_bom(raw)
        candidates: list[dict[str, object]] = []
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            candidate = self._build_decode_candidate(raw, encoding)
            if candidate is not None:
                candidates.append(candidate)
        if utf16_hint:
            for encoding in ("utf-16-le", "utf-16-be"):
                candidate = self._build_decode_candidate(raw, encoding)
                if candidate is not None:
                    candidates.append(candidate)
        if candidates:
            best = max(candidates, key=lambda item: item["score"])
            text = str(best["text"])
            return text, {
                "encoding": best["encoding"],
                "bom": "",
                "utf16_hint": utf16_hint,
                "used_replace": False,
                "raw_bytes": len(raw),
                "char_count": len(text),
            }

        text = raw.decode("utf-8", errors="replace")
        return text, {
            "encoding": "utf-8-replace",
            "bom": "",
            "utf16_hint": utf16_hint,
            "used_replace": True,
            "raw_bytes": len(raw),
            "char_count": len(text),
        }

    def _detect_text_bom(self, raw: bytes) -> tuple[str | None, str]:
        if raw.startswith(codecs.BOM_UTF32_LE):
            return "utf-32", "utf-32-le"
        if raw.startswith(codecs.BOM_UTF32_BE):
            return "utf-32", "utf-32-be"
        if raw.startswith(codecs.BOM_UTF8):
            return "utf-8-sig", "utf-8"
        if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
            return "utf-16", "utf-16"
        return None, ""

    def _looks_like_utf16_without_bom(self, raw: bytes) -> bool:
        sample = raw[:4096]
        if len(sample) < 4:
            return False
        if len(sample) % 2:
            sample = sample[:-1]
        if len(sample) < 4:
            return False

        even_zero = sum(1 for index in range(0, len(sample), 2) if sample[index] == 0)
        odd_zero = sum(1 for index in range(1, len(sample), 2) if sample[index] == 0)
        pair_count = len(sample) // 2
        dominant = max(even_zero, odd_zero)
        minority = min(even_zero, odd_zero)
        if dominant < max(2, pair_count // 20):
            return False
        if dominant < (minority * 3) and (dominant - minority) < 2:
            return False
        return True

    def _build_decode_candidate(self, raw: bytes, encoding: str) -> dict[str, object] | None:
        try:
            text = raw.decode(encoding)
        except Exception:
            return None
        return {
            "encoding": encoding,
            "text": text,
            "score": self._decoded_text_quality_score(text),
        }

    def _decoded_text_quality_score(self, text: str) -> float:
        if not text:
            return float("-inf")

        length = max(len(text), 1)
        null_count = text.count("\x00")
        replacement_count = text.count("\ufffd")
        control_count = 0
        common_count = 0
        rare_count = 0

        for char in text:
            code = ord(char)
            if char in "\r\n\t":
                common_count += 1
                continue
            if code == 0 or code == 0xFFFD:
                continue
            if code < 32 or 0x7F <= code < 0xA0:
                control_count += 1
                continue
            if self._is_common_text_codepoint(code):
                common_count += 1
            else:
                rare_count += 1

        common_ratio = common_count / length
        rare_ratio = rare_count / length
        bad_ratio = (null_count + replacement_count + control_count) / length
        log_hits = len(TEXT_LOG_HINT_RE.findall(text[:4000]))
        return common_ratio * 100.0 + min(log_hits, 12) * 3.0 - rare_ratio * 25.0 - bad_ratio * 80.0

    def _is_common_text_codepoint(self, code: int) -> bool:
        if 0x20 <= code <= 0x7E:
            return True
        if 0x3000 <= code <= 0x303F:
            return True
        if 0x3040 <= code <= 0x30FF:
            return True
        if 0x3400 <= code <= 0x4DBF:
            return True
        if 0x4E00 <= code <= 0x9FFF:
            return True
        if 0xAC00 <= code <= 0xD7AF:
            return True
        if 0xFF01 <= code <= 0xFFEE:
            return True
        return False

    async def read_text_with_fallback(self, path: Path, deadline: float | None = None) -> str:
        return await self.runtime.run_io_in_thread(
            "read_text_with_fallback",
            self.read_text_with_fallback_blocking,
            path,
            deadline=deadline,
        )

    async def safe_extract_zip(self, zip_path: Path, out_dir: Path, deadline: float | None = None) -> list[Path]:
        return await self.runtime.run_io_in_thread("safe_extract_zip", self.safe_extract_zip_blocking, zip_path, out_dir, deadline=deadline)

    def normalize_zip_member_path(self, member_name: str) -> Path | None:
        raw = str(member_name or "").replace("\\", "/").strip()
        if not raw:
            return None
        if raw.startswith("/") or raw.startswith("//"):
            return None
        if re.match(r"^[A-Za-z]:", raw):
            return None

        pure_path = PurePosixPath(raw)
        parts: list[str] = []
        for part in pure_path.parts:
            if part in ("", "."):
                continue
            if part == ".." or part.endswith(":"):
                return None
            parts.append(part)
        if not parts:
            return None
        return Path(*parts)

    def is_within_directory(self, target: Path, base: Path) -> bool:
        try:
            target.relative_to(base)
            return True
        except ValueError:
            return False

    def safe_extract_zip_blocking(self, zip_path: Path, out_dir: Path, deadline: float | None, cancel_event) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        extracted: list[Path] = []
        total_bytes = 0
        cfg = self._cfg()
        max_files = int(cfg.get("max_archive_file_count", MAX_ARCHIVE_FILE_COUNT))
        max_single = int(cfg.get("max_archive_single_file_bytes", MAX_ARCHIVE_SINGLE_FILE_BYTES))
        max_total = int(cfg.get("max_archive_total_bytes", MAX_ARCHIVE_TOTAL_BYTES))

        with zipfile.ZipFile(zip_path, "r") as zf:
            infos = zf.infolist()
            if len(infos) > max_files:
                raise BudgetExceeded("archive entry count exceeded")

            base = out_dir.resolve()
            for idx, info in enumerate(infos):
                if idx % 8 == 0:
                    self.runtime.check_deadline_or_cancel(deadline, cancel_event, "safe_extract_zip_entries")
                if info.is_dir():
                    logger.debug(f"[mc_log][zip] skip dir: {info.filename}")
                    continue
                if self.zipinfo_is_link(info):
                    logger.debug(f"[mc_log][zip] skip link/special: {info.filename}")
                    continue
                if info.file_size > max_single:
                    logger.debug(f"[mc_log][zip] skip big file: {info.filename}, size={info.file_size}")
                    continue
                total_bytes += info.file_size
                if total_bytes > max_total:
                    raise BudgetExceeded("archive total size exceeded")

                safe_rel = self.normalize_zip_member_path(info.filename)
                if safe_rel is None:
                    logger.warning(f"[mc_log][zip] skip unsafe path: {info.filename}")
                    continue
                target = (base / safe_rel).resolve()
                if not self.is_within_directory(target, base):
                    logger.warning(f"[mc_log][zip] skip escaped path after normalize: {info.filename}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)

                written = 0
                with zf.open(info, "r") as src, open(target, "wb") as dst:
                    while True:
                        chunk = src.read(64 * 1024)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written % (2 * 1024 * 1024) == 0:
                            self.runtime.check_deadline_or_cancel(deadline, cancel_event, "safe_extract_zip_write")
                        if written > max_single:
                            raise BudgetExceeded("archive single file extracted size exceeded")
                        dst.write(chunk)
                extracted.append(target)
                logger.debug(f"[mc_log][zip] extracted: {info.filename}, size={written}")
        return extracted

    def zipinfo_is_link(self, info: zipfile.ZipInfo) -> bool:
        mode = (info.external_attr >> 16) & 0xF000
        return mode in {0xA000, 0x6000, 0x2000, 0x1000, 0xC000}

    async def safe_extract_gz(self, gz_path: Path, out_dir: Path, deadline: float | None = None) -> Path | None:
        return await self.runtime.run_io_in_thread("safe_extract_gz", self.safe_extract_gz_blocking, gz_path, out_dir, deadline=deadline)

    def safe_extract_gz_blocking(self, gz_path: Path, out_dir: Path, deadline: float | None, cancel_event) -> Path | None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = gz_path.stem or f"unzipped_{uuid.uuid4().hex[:6]}.log"
        target = out_dir / sanitize_uploaded_name(out_name)
        size = 0
        max_bytes = int(self._cfg().get("max_gz_output_bytes", MAX_GZ_OUTPUT_BYTES))
        with gzip.open(gz_path, "rb") as src, open(target, "wb") as dst:
            while True:
                chunk = src.read(64 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size % (2 * 1024 * 1024) == 0:
                    self.runtime.check_deadline_or_cancel(deadline, cancel_event, "safe_extract_gz_write")
                if size > max_bytes:
                    raise BudgetExceeded("gz extracted size exceeded")
                dst.write(chunk)
        return target if target.exists() else None

    async def safe_remove_dir(self, path: Path, deadline: float | None = None):
        await self.runtime.run_io_in_thread("safe_remove_dir", self.safe_remove_dir_blocking, path, deadline=deadline)

    def safe_remove_dir_blocking(self, path: Path, deadline: float | None, cancel_event):
        try:
            self.runtime.check_deadline_or_cancel(deadline, cancel_event, "safe_remove_dir_start")
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        except Exception:
            logger.warning("[mc_log] 删除临时目录失败:\n" + traceback.format_exc())
