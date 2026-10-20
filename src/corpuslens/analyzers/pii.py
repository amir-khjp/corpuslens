"""Regex-based PII detection (emails, phone numbers, IPs, credit cards, SSNs).

Deliberately dependency-free so it always runs. A spaCy-NER-based analyzer
for names/addresses is a good follow-up contribution (see CONTRIBUTING.md).
"""

from __future__ import annotations

import re

import pandas as pd

from corpuslens.analyzers.base import Analyzer, AnalyzerResult, Chart

PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"(?<!\d)(\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"),
    "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "credit_card": re.compile(r"(?<!\d)(?:\d[ -]*?){13,16}(?!\d)"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


class PiiAnalyzer(Analyzer):
    name = "pii"

    def run(self, df: pd.DataFrame, text_col: str, **kwargs) -> AnalyzerResult:
        texts = df[text_col].fillna("").astype(str)

        counts = {kind: 0 for kind in PATTERNS}
        flagged_rows = 0
        for text in texts:
            hit = False
            for kind, pattern in PATTERNS.items():
                matches = pattern.findall(text)
                if matches:
                    counts[kind] += len(matches)
                    hit = True
            if hit:
                flagged_rows += 1

        summary = {
            "rows_with_pii": flagged_rows,
            "pii_row_ratio": round(flagged_rows / max(len(texts), 1), 4),
            "matches_by_type": counts,
        }

        chart = Chart(
            title="PII matches by type",
            kind="bar",
            labels=list(counts.keys()),
            values=list(counts.values()),
        )

        warnings = []
        if flagged_rows:
            warnings.append(
                f"{flagged_rows} row(s) ({summary['pii_row_ratio']:.1%}) contain "
                "patterns that look like PII. Review before sharing this dataset."
            )

        return AnalyzerResult(name=self.name, summary=summary, charts=[chart], warnings=warnings)
