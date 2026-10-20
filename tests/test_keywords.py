from corpuslens.analyzers.keywords import KeywordAnalyzer


def test_extracts_keywords(sample_df):
    result = KeywordAnalyzer().run(sample_df, "text")
    terms = [kw["term"] for kw in result.summary["top_keywords"]]
    assert len(terms) > 0
