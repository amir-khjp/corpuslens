"""Topic clustering via TF-IDF + KMeans, with optional LLM-generated labels."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from corpuslens.analyzers.base import Analyzer, AnalyzerResult, Chart
from corpuslens.llm.base import LLMBackend

MIN_ROWS_FOR_CLUSTERING = 10
MAX_CLUSTERS = 8
SAMPLES_PER_CLUSTER = 3


class ClusteringAnalyzer(Analyzer):
    name = "clustering"

    def run(
        self,
        df: pd.DataFrame,
        text_col: str,
        llm_backend: LLMBackend | None = None,
        **kwargs,
    ) -> AnalyzerResult:
        texts = df[text_col].fillna("").astype(str)
        non_empty = texts[texts.str.strip() != ""]

        if len(non_empty) < MIN_ROWS_FOR_CLUSTERING:
            return AnalyzerResult(
                name=self.name,
                summary={"n_clusters": 0},
                warnings=[f"Need at least {MIN_ROWS_FOR_CLUSTERING} non-empty rows to cluster."],
            )

        n_clusters = min(MAX_CLUSTERS, max(2, len(non_empty) // 20))

        vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
        matrix = vectorizer.fit_transform(non_empty)
        km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        cluster_ids = km.fit_predict(matrix)

        terms = vectorizer.get_feature_names_out()
        centers = km.cluster_centers_

        clusters = []
        for cid in range(n_clusters):
            member_mask = cluster_ids == cid
            size = int(member_mask.sum())
            top_term_idx = centers[cid].argsort()[::-1][:8]
            top_terms = [terms[i] for i in top_term_idx]
            samples = non_empty[member_mask].head(SAMPLES_PER_CLUSTER).tolist()

            label = " / ".join(top_terms[:3])
            if llm_backend is not None:
                try:
                    label = llm_backend.label_cluster(top_terms, samples) or label
                except Exception:  # pragma: no cover - never let LLM errors break the report
                    pass

            clusters.append(
                {
                    "cluster_id": cid,
                    "label": label,
                    "size": size,
                    "top_terms": top_terms,
                    "samples": samples,
                }
            )

        clusters.sort(key=lambda c: c["size"], reverse=True)

        chart = Chart(
            title="Cluster sizes",
            kind="pie",
            labels=[c["label"] for c in clusters],
            values=[c["size"] for c in clusters],
        )

        return AnalyzerResult(
            name=self.name,
            summary={"n_clusters": n_clusters, "clusters": clusters},
            charts=[chart],
            table=clusters,
        )
