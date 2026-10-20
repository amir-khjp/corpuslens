"""Length, language, and vocabulary-diversity statistics."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from corpuslens.analyzers.base import Analyzer, AnalyzerResult, Chart


def _detect_language(sample: list[str]) -> dict[str, int]:
    try:
        from langdetect import LangDetectException, detect
    except ImportError:  # pragma: no cover
        return {}

    counts: Counter[str] = Counter()
    for text in sample:
        if not text or not text.strip():
            continue
        try:
            counts[detect(text)] += 1
        except LangDetectException:
            counts["unknown"] += 1
    return dict(counts)


class BasicStatsAnalyzer(Analyzer):
    name = "basic_stats"

    def run(self, df: pd.DataFrame, text_col: str, **kwargs) -> AnalyzerResult:
        texts = df[text_col].fillna("").astype(str)
        lengths_chars = texts.str.len()
        lengths_words = texts.str.split().apply(len)

        empty_count = int((lengths_chars == 0).sum())
        vocab = set()
        for text in texts:
            vocab.update(text.lower().split())

        sample = texts.sample(min(len(texts), 200), random_state=0).tolist() if len(texts) else []
        lang_counts = _detect_language(sample)

        summary = {
            "n_rows": len(df),
            "n_empty": empty_count,
            "mean_chars": round(float(lengths_chars.mean()) if len(df) else 0.0, 1),
            "median_chars": round(float(lengths_chars.median()) if len(df) else 0.0, 1),
            "mean_words": round(float(lengths_words.mean()) if len(df) else 0.0, 1),
            "vocab_size": len(vocab),
            "type_token_ratio": round(len(vocab) / max(int(lengths_words.sum()), 1), 4),
            "languages_detected": lang_counts,
        }

        hist, edges = np.histogram(lengths_words, bins=10)
        chart = Chart(
            title="Document length (words)",
            kind="histogram",
            labels=[f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(hist))],
            values=[int(v) for v in hist],
        )

        warnings = []
        if empty_count:
            warnings.append(f"{empty_count} row(s) have empty text in '{text_col}'.")

        return AnalyzerResult(name=self.name, summary=summary, charts=[chart], warnings=warnings)
