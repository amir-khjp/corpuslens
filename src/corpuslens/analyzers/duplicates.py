"""Exact and near-duplicate detection.

Near-duplicates are found with TF-IDF cosine similarity rather than a
transformer embedding model, so CorpusLens stays installable without heavy
GPU dependencies. See CONTRIBUTING.md for a "swap in sentence-transformers"
good-first-issue if you want higher recall.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from corpuslens.analyzers.base import Analyzer, AnalyzerResult, Chart

NEAR_DUP_THRESHOLD = 0.9
MAX_ROWS_FOR_PAIRWISE = 5000


class DuplicateAnalyzer(Analyzer):
    name = "duplicates"

    def run(self, df: pd.DataFrame, text_col: str, **kwargs) -> AnalyzerResult:
        texts = df[text_col].fillna("").astype(str)
        n = len(texts)

        exact_dup_mask = texts.duplicated(keep="first")
        exact_dup_count = int(exact_dup_mask.sum())

        near_dup_pairs: list[dict] = []
        warnings = []
        if 1 < n <= MAX_ROWS_FOR_PAIRWISE:
            non_empty = texts[texts.str.strip() != ""]
            if len(non_empty) > 1:
                vectorizer = TfidfVectorizer(max_features=5000)
                matrix = vectorizer.fit_transform(non_empty)
                sims = cosine_similarity(matrix)
                idx = non_empty.index.tolist()
                seen = set()
                for i in range(len(idx)):
                    for j in range(i + 1, len(idx)):
                        is_exact = non_empty.iloc[i] == non_empty.iloc[j]
                        if (
                            not is_exact
                            and sims[i, j] >= NEAR_DUP_THRESHOLD
                            and idx[i] not in seen
                            and idx[j] not in seen
                        ):
                            near_dup_pairs.append(
                                {
                                    "row_a": idx[i],
                                    "row_b": idx[j],
                                    "similarity": round(float(sims[i, j]), 3),
                                }
                            )
                            seen.add(idx[j])
        elif n > MAX_ROWS_FOR_PAIRWISE:
            warnings.append(
                f"Skipped near-duplicate detection: {n} rows exceeds the "
                f"{MAX_ROWS_FOR_PAIRWISE}-row pairwise-similarity limit."
            )

        summary = {
            "exact_duplicates": exact_dup_count,
            "exact_duplicate_ratio": round(exact_dup_count / max(n, 1), 4),
            "near_duplicate_pairs": len(near_dup_pairs),
        }

        chart = Chart(
            title="Duplicate breakdown",
            kind="bar",
            labels=["Unique", "Exact duplicates", "Near duplicates"],
            values=[max(0, n - exact_dup_count - len(near_dup_pairs)), exact_dup_count, len(near_dup_pairs)],
        )

        if exact_dup_count:
            warnings.append(
                f"{exact_dup_count} exact duplicate row(s) found "
                f"({summary['exact_duplicate_ratio']:.1%} of the dataset)."
            )

        return AnalyzerResult(
            name=self.name,
            summary=summary,
            charts=[chart],
            table=near_dup_pairs[:50] or None,
            warnings=warnings,
        )
