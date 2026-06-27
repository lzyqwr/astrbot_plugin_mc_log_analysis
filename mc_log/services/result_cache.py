from __future__ import annotations

import asyncio
import time

from astrbot.api import logger


class ResultCache:
    """按 uid 缓存最近的分析结果文本，TTL 默认 30 分钟，仅存内存。"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self._store: dict[str, list[dict]] = {}
        self._lock = asyncio.Lock()

    def _cfg(self):
        return self.config_manager.get()

    def _ttl(self) -> float:
        return max(60.0, float(self._cfg().get("result_cache_ttl_sec", 1800)))

    def _max_per_uid(self) -> int:
        return max(1, int(self._cfg().get("result_cache_max_per_uid", 10)))

    def _is_expired(self, entry: dict, now: float) -> bool:
        return (now - float(entry.get("stored_at", 0.0))) > self._ttl()

    def _prune_uid(self, entries: list[dict], now: float) -> list[dict]:
        return [entry for entry in entries if not self._is_expired(entry, now)]

    async def store(self, uid: str, text: str) -> None:
        normalized = str(uid or "").strip()
        if not normalized or not text:
            return
        now = time.monotonic()
        async with self._lock:
            entries = self._prune_uid(self._store.get(normalized, []), now)
            entries.insert(0, {"text": str(text), "stored_at": now})
            cap = self._max_per_uid()
            if len(entries) > cap:
                entries = entries[:cap]
            self._store[normalized] = entries
        logger.info(
            f"[mc_log][cache] 已缓存分析结果: uid={normalized}, count={len(entries)}, ttl={self._ttl():.0f}s"
        )

    async def get_all(self, uid: str) -> list[dict]:
        normalized = str(uid or "").strip()
        if not normalized:
            return []
        now = time.monotonic()
        async with self._lock:
            entries = self._prune_uid(self._store.get(normalized, []), now)
            if entries:
                self._store[normalized] = entries
            else:
                self._store.pop(normalized, None)
            return list(entries)

    async def has_recent(self, uid: str, within_seconds: float = 600.0) -> bool:
        """是否存在在 within_seconds 秒内存储的缓存条目。"""
        normalized = str(uid or "").strip()
        if not normalized or within_seconds <= 0:
            return False
        now = time.monotonic()
        async with self._lock:
            entries = self._prune_uid(self._store.get(normalized, []), now)
            if entries:
                self._store[normalized] = entries
            else:
                self._store.pop(normalized, None)
            for entry in entries:
                if (now - float(entry.get("stored_at", 0.0))) <= within_seconds:
                    return True
        return False

    async def cleanup_expired(self) -> None:
        now = time.monotonic()
        async with self._lock:
            for uid in list(self._store.keys()):
                entries = self._prune_uid(self._store.get(uid, []), now)
                if entries:
                    self._store[uid] = entries
                else:
                    self._store.pop(uid, None)
