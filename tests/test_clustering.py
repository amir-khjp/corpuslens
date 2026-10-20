import pandas as pd

from corpuslens.analyzers.clustering import ClusteringAnalyzer


def test_clustering_skips_small_datasets(sample_df):
    result = ClusteringAnalyzer().run(sample_df, "text")
    assert result.summary["n_clusters"] == 0
    assert result.warnings


def test_clustering_runs_on_larger_dataset():
    texts = [f"Finance report number {i} about quarterly revenue growth." for i in range(15)]
    texts += [f"Weather update {i}: sunny skies and light wind today." for i in range(15)]
    df = pd.DataFrame({"text": texts})
    result = ClusteringAnalyzer().run(df, "text")
    assert result.summary["n_clusters"] >= 2
    assert len(result.summary["clusters"]) == result.summary["n_clusters"]
