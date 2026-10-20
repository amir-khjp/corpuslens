"""Class-balance statistics for an optional label column."""

from __future__ import annotations

import pandas as pd

from corpuslens.analyzers.base import Analyzer, AnalyzerResult, Chart


class LabelBalanceAnalyzer(Analyzer):
    name = "label_balance"

    def run(self, df: pd.DataFrame, text_col: str, label_col: str | None = None, **kwargs) -> AnalyzerResult:
        if not label_col or label_col not in df.columns:
            return AnalyzerResult(name=self.name, summary={"enabled": False})

        counts = df[label_col].value_counts(dropna=False)
        total = int(counts.sum())
        imbalance_ratio = round(float(counts.max() / max(counts.min(), 1)), 2) if len(counts) else 0.0

        summary = {
            "enabled": True,
            "n_classes": len(counts),
            "class_counts": {str(k): int(v) for k, v in counts.items()},
            "imbalance_ratio": imbalance_ratio,
        }

        chart = Chart(
            title=f"Class distribution ({label_col})",
            kind="bar",
            labels=[str(k) for k in counts.index],
            values=[int(v) for v in counts.values],
        )

        warnings = []
        if imbalance_ratio >= 10:
            warnings.append(
                f"Class imbalance ratio is {imbalance_ratio}:1 — the majority class "
                f"dominates {total} rows."
            )

        return AnalyzerResult(name=self.name, summary=summary, charts=[chart], warnings=warnings)
