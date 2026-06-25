from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class GoldExtractionCase:
    name: str
    loader: str
    kind: str
    text: str
    gold_root_regexes: list[str]
    gold_support_regexes: list[str]
    culprit_ids: list[str]
    forbidden_noise_regexes: list[str]
    gold_signature_regexes: list[str] = field(default_factory=list)


def load_gold_cases(path: Path) -> list[GoldExtractionCase]:
    cases: list[GoldExtractionCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cases.append(GoldExtractionCase(**row))
    return cases


def regex_recall(patterns: list[str], text: str) -> float:
    if not patterns:
        return 1.0
    hits = sum(bool(re.search(pattern, text, re.I | re.M)) for pattern in patterns)
    return hits / len(patterns)


def culprit_recall(culprits: list[str], text: str) -> float:
    if not culprits:
        return 1.0
    lowered = text.lower()
    hits = sum(culprit.lower() in lowered for culprit in culprits)
    return hits / len(culprits)


def noise_leak(patterns: list[str], text: str) -> float:
    if not patterns:
        return 0.0
    hits = sum(bool(re.search(pattern, text, re.I | re.M)) for pattern in patterns)
    return hits / len(patterns)


def evaluate_extractor(
    extractor: Callable[[str], str],
    cases: list[GoldExtractionCase],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for case in cases:
        output = extractor(case.text)
        row = {
            "name": case.name,
            "loader": case.loader,
            "kind": case.kind,
            "chars": len(output),
            "root_recall": regex_recall(case.gold_root_regexes, output),
            "support_recall": regex_recall(case.gold_support_regexes, output),
            "signature_recall": regex_recall(case.gold_signature_regexes, output),
            "culprit_recall": culprit_recall(case.culprit_ids, output),
            "noise_leak": noise_leak(case.forbidden_noise_regexes, output),
        }
        row["success"] = (
            row["root_recall"] == 1.0
            and row["signature_recall"] == 1.0
            and row["culprit_recall"] == 1.0
            and row["noise_leak"] <= 0.25
        )
        rows.append(row)

    if not rows:
        return {
            "rows": [],
            "macro_root_recall": 0.0,
            "macro_support_recall": 0.0,
            "macro_signature_recall": 0.0,
            "macro_culprit_recall": 0.0,
            "macro_noise_leak": 0.0,
            "case_success_rate": 0.0,
        }

    return {
        "rows": rows,
        "macro_root_recall": _mean(rows, "root_recall"),
        "macro_support_recall": _mean(rows, "support_recall"),
        "macro_signature_recall": _mean(rows, "signature_recall"),
        "macro_culprit_recall": _mean(rows, "culprit_recall"),
        "macro_noise_leak": _mean(rows, "noise_leak"),
        "case_success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
        "by_loader": summarize_rows(rows, "loader"),
        "by_kind": summarize_rows(rows, "kind"),
    }


def summarize_rows(rows: list[dict[str, object]], key: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return {group: summarize_group(group_rows) for group, group_rows in grouped.items()}


def summarize_group(rows: list[dict[str, object]]) -> dict[str, float]:
    return {
        "case_count": float(len(rows)),
        "macro_root_recall": _mean(rows, "root_recall"),
        "macro_support_recall": _mean(rows, "support_recall"),
        "macro_signature_recall": _mean(rows, "signature_recall"),
        "macro_culprit_recall": _mean(rows, "culprit_recall"),
        "macro_noise_leak": _mean(rows, "noise_leak"),
        "case_success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
    }


def _mean(rows: list[dict[str, object]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)
