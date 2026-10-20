"""Command-line entry point: `corpuslens report data.csv --text-col text`."""

from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd

from corpuslens.report import CorpusReport


def _load_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise click.ClickException(f"Unsupported file type: {suffix}. Use .csv, .jsonl, .json, or .parquet.")


@click.group()
@click.version_option(package_name="corpuslens")
def main() -> None:
    """CorpusLens: automated EDA and data-mining reports for text datasets."""


@main.command()
@click.argument("data_path", type=click.Path(exists=True, path_type=Path))
@click.option("--text-col", required=True, help="Name of the column containing text.")
@click.option("--label-col", default=None, help="Optional column with class labels, for balance stats.")
@click.option("--output", "-o", default="corpuslens_report.html", help="Output HTML path.")
@click.option("--json-output", default=None, help="Optional path to also write a JSON report.")
@click.option(
    "--llm",
    default=None,
    type=click.Choice(["none", "anthropic", "openai"]),
    help="LLM backend for cluster labeling. Defaults to auto-detecting from API key env vars.",
)
def report(data_path: Path, text_col: str, label_col: str | None, output: str, json_output: str | None, llm: str | None) -> None:
    """Generate an HTML (and optional JSON) report for DATA_PATH."""
    df = _load_dataframe(data_path)
    if text_col not in df.columns:
        raise click.ClickException(f"Column {text_col!r} not found. Available columns: {list(df.columns)}")

    corpus_report = CorpusReport(df, text_col=text_col, label_col=label_col, llm_backend=llm)
    corpus_report.run()
    corpus_report.to_html(output)
    click.echo(f"Report written to {output}")

    if json_output:
        corpus_report.to_json(json_output)
        click.echo(f"JSON report written to {json_output}")


if __name__ == "__main__":
    sys.exit(main())
