from corpuslens.analyzers.basic_stats import BasicStatsAnalyzer


def test_basic_stats_counts_rows_and_empties(sample_df):
    result = BasicStatsAnalyzer().run(sample_df, "text")
    assert result.summary["n_rows"] == len(sample_df)
    assert result.summary["n_empty"] == 1
    assert result.summary["vocab_size"] > 0
    assert result.charts[0].kind == "histogram"
