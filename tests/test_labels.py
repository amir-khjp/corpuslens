from corpuslens.analyzers.labels import LabelBalanceAnalyzer


def test_label_balance_disabled_without_label_col(sample_df):
    result = LabelBalanceAnalyzer().run(sample_df, "text", label_col=None)
    assert result.summary["enabled"] is False


def test_label_balance_counts_classes(sample_df):
    result = LabelBalanceAnalyzer().run(sample_df, "text", label_col="category")
    assert result.summary["enabled"] is True
    assert result.summary["n_classes"] == 5
