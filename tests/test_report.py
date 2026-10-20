import json

from corpuslens.report import CorpusReport


def test_report_runs_and_exports_json(sample_df, tmp_path):
    report = CorpusReport(sample_df, text_col="text", label_col="category")
    report.run()
    payload = json.loads(report.to_json())
    assert payload["n_rows"] == len(sample_df)
    assert len(payload["analyzers"]) == 6


def test_report_renders_html(sample_df, tmp_path):
    report = CorpusReport(sample_df, text_col="text")
    report.run()
    out_path = tmp_path / "report.html"
    html = report.to_html(out_path)
    assert out_path.exists()
    assert "CorpusLens Report" in html


def test_report_rejects_missing_column(sample_df):
    try:
        CorpusReport(sample_df, text_col="nope")
        assert False, "expected ValueError"
    except ValueError:
        pass
