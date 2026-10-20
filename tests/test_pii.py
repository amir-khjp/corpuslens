from corpuslens.analyzers.pii import PiiAnalyzer


def test_detects_email_and_phone(sample_df):
    result = PiiAnalyzer().run(sample_df, "text")
    assert result.summary["matches_by_type"]["email"] == 1
    assert result.summary["matches_by_type"]["phone"] == 1
    assert result.summary["rows_with_pii"] == 1
