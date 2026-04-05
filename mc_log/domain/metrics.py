from __future__ import annotations

import re


class MetricsService:
    def extract_metrics_from_report(self, report_md: str) -> dict:
        text = report_md or ""
        claim = ""
        needs_more = False
        guard_flags = []

        match = re.search(r"核心问题：\s*(.+)", text)
        if match:
            claim = match.group(1).strip()
        if re.search(r"\bUNCERTAIN\b", text):
            needs_more = True
        if re.search(r"补充确认：\s*无", text):
            pass
        elif re.search(r"补充确认：", text):
            needs_more = True

        if needs_more:
            guard_flags.append("needs_more_info")

        stability_key = self.build_stability_key(claim)
        return {
            "claim_type": claim or "",
            "guard_flags": guard_flags,
            "needs_more_info": needs_more,
            "root_cause_stability_key": stability_key,
        }

    def build_stability_key(self, claim: str) -> str:
        base = re.sub(r"\s+", " ", str(claim or "").lower()).strip()
        if not base:
            return ""
        base = re.sub(r"\b\d+(\.\d+)*\b", "x", base)
        base = re.sub(r"\b[A-F0-9]{6,}\b", "x", base)
        base = base[:120]
        return base

    def detect_suspect_analyze_text(self, text: str) -> str:
        sample = str(text or "").strip()
        if not sample:
            return "empty"
        lower = sample.lower()
        patterns = (
            "candidate.content.parts",
            "api 返回的",
            "chat model",
            "request error",
            "tool_loop",
            "模型请求失败",
            "模型响应异常",
        )
        for pattern in patterns:
            if pattern in sample or pattern in lower:
                return pattern
        if len(sample) <= 120 and ("请稍后重试" in sample or "分析失败" in sample):
            return "short_failure_text"
        return ""
