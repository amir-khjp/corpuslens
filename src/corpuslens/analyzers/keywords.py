"""Corpus-level keyword extraction via TF-IDF."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from corpuslens.analyzers.base import Analyzer, AnalyzerResult, Chart

TOP_K = 20


class KeywordAnalyzer(Analyzer):
    name = "keywords"

    def run(self, df: pd.DataFrame, text_col: str, **kwargs) -> AnalyzerResult:
        texts = df[text_col].fillna("").astype(str)
        non_empty = texts[texts.str.strip() != ""]

        if len(non_empty) < 2:
            return AnalyzerResult(
                name=self.name,
                summary={"top_keywords": []},
                warnings=["Not enough non-empty text to extract keywords."],
            )

        vectorizer = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(non_empty)
        scores = matrix.sum(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        ranked = sorted(zip(terms, scores), key=lambda t: t[1], reverse=True)[:TOP_K]

        top_keywords = [{"term": term, "score": round(float(score), 3)} for term, score in ranked]

        chart = Chart(
            title="Top keywords (TF-IDF)",
            kind="bar",
            labels=[t for t, _ in ranked[:10]],
            values=[round(float(s), 2) for _, s in ranked[:10]],
        )

        return AnalyzerResult(
            name=self.name,
            summary={"top_keywords": top_keywords},
            charts=[chart],
            table=top_keywords,
        )
