from corpuslens.analyzers.duplicates import DuplicateAnalyzer


def test_detects_exact_duplicate(sample_df):
    result = DuplicateAnalyzer().run(sample_df, "text")
    assert result.summary["exact_duplicates"] == 1


def test_detects_near_duplicates(sample_df):
    result = DuplicateAnalyzer().run(sample_df, "text")
    assert result.summary["near_duplicate_pairs"] >= 1
