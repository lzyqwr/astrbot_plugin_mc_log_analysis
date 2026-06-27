from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

from astrbot.api import logger
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

_RETENTION_DAYS = 90


class TokenStatsService:
    """记录并查询 LLM token 用量,持久化到 JSON。"""

    def __init__(self):
        data_path = Path(get_astrbot_data_path())
        self._file = data_path / "plugin_data" / "mc_log_token_stats.json"
        self._records: list[dict] = []
        self._lock = asyncio.Lock()
        self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self):
        try:
            if self._file.exists():
                raw = self._file.read_text(encoding="utf-8").strip()
                if raw:
                    data = json.loads(raw)
                    records = data.get("records", []) if isinstance(data, dict) else []
                    if isinstance(records, list):
                        self._records = [r for r in records if isinstance(r, dict)]
        except Exception as exc:
            logger.warning(f"[mc_log] token_stats 加载失败,重新开始: {exc}")
            self._records = []

    def _save_sync(self):
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            self._file.write_text(
                json.dumps({"records": self._records}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning(f"[mc_log] token_stats 保存失败: {exc}")

    def _cleanup_old(self, now: float | None = None):
        if not self._records:
            return
        cutoff = (now or time.time()) - _RETENTION_DAYS * 86400
        self._records = [r for r in self._records if float(r.get("ts", 0)) >= cutoff]

    # ------------------------------------------------------------------ public API
    async def record(
        self,
        uid: str,
        session_id: str,
        input_tokens: int,
        output_tokens: int,
        source: str = "framework",
    ) -> None:
        """记录一次 LLM 调用的 token 用量。"""
        entry = {
            "ts": time.time(),
            "uid": str(uid or ""),
            "session_id": str(session_id or ""),
            "input": int(input_tokens or 0),
            "output": int(output_tokens or 0),
            "source": str(source or "framework"),
        }
        async with self._lock:
            self._records.append(entry)
            self._cleanup_old(entry["ts"])
            self._save_sync()

    async def get_stats(self) -> dict:
        """返回各项聚合统计。"""
        now = time.time()
        async with self._lock:
            self._cleanup_old(now)
            records = list(self._records)

        total_in = total_out = 0
        month_in = month_out = 0
        yesterday_in = yesterday_out = 0
        six_hour_in = six_hour_out = 0
        per_user: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0})
        per_session: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0})

        month_cutoff = now - 30 * 86400
        yesterday_start = now - ((int(now) % 86400) + 86400)
        six_hour_cutoff = now - 6 * 3600

        for r in records:
            ts = float(r.get("ts", 0))
            inp = int(r.get("input", 0))
            out = int(r.get("output", 0))
            uid = str(r.get("uid", ""))
            sid = str(r.get("session_id", ""))

            total_in += inp
            total_out += out

            if ts >= month_cutoff:
                month_in += inp
                month_out += out

            if yesterday_start <= ts < yesterday_start + 86400:
                yesterday_in += inp
                yesterday_out += out

            if ts >= six_hour_cutoff:
                six_hour_in += inp
                six_hour_out += out

            if uid:
                per_user[uid]["input"] += inp
                per_user[uid]["output"] += out
            if sid:
                per_session[sid]["input"] += inp
                per_session[sid]["output"] += out

        def total(d):
            return d["input"] + d["output"]

        return {
            "total": {"input": total_in, "output": total_out, "sum": total_in + total_out},
            "month": {"input": month_in, "output": month_out, "sum": month_in + month_out},
            "yesterday": {"input": yesterday_in, "output": yesterday_out, "sum": yesterday_in + yesterday_out},
            "six_hour": {"input": six_hour_in, "output": six_hour_out, "sum": six_hour_in + six_hour_out},
            "per_user": {
                uid: {"input": v["input"], "output": v["output"], "sum": total(v)}
                for uid, v in sorted(per_user.items(), key=lambda kv: total(kv[1]), reverse=True)
            },
            "per_session": {
                sid: {"input": v["input"], "output": v["output"], "sum": total(v)}
                for sid, v in sorted(per_session.items(), key=lambda kv: total(kv[1]), reverse=True)
            },
            "record_count": len(records),
        }
