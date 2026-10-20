"""Orchestrates analyzers and renders the HTML/JSON report."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from corpuslens.analyzers import DEFAULT_ANALYZERS, Analyzer
from corpuslens.llm import LLMBackend, get_backend

TEMPLATE_DIR = Path(__file__).parent / "templates"


class CorpusReport:
    """Runs a set of analyzers over a text dataset and renders a report.

    Example:
        >>> report = CorpusReport(df, text_col="text", label_col="category")
        >>> report.run()
        >>> report.to_html("report.html")
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_col: str,
        label_col: str | None = None,
        analyzers: list[Analyzer] | None = None,
        llm_backend: LLMBackend | str | None = None,
        title: str = "CorpusLens Report",
    ):
        if text_col not in df.columns:
            raise ValueError(f"Column {text_col!r} not found in DataFrame. Columns: {list(df.columns)}")

        self.df = df
        self.text_col = text_col
        self.label_col = label_col
        self.analyzers = analyzers if analyzers is not None else list(DEFAULT_ANALYZERS)
        self.title = title
        self.llm_backend = llm_backend if isinstance(llm_backend, LLMBackend) else get_backend(llm_backend)
        self._results: list[dict[str, Any]] = []

    def run(self) -> CorpusReport:
        self._results = []
        for analyzer in self.analyzers:
            result = analyzer.run(
                self.df,
                self.text_col,
                label_col=self.label_col,
                llm_backend=self.llm_backend,
            )
            self._results.append(asdict(result))
        return self

    @property
    def results(self) -> list[dict[str, Any]]:
        if not self._results:
            raise RuntimeError("Call .run() before accessing results.")
        return self._results

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "n_rows": len(self.df),
            "text_col": self.text_col,
            "label_col": self.label_col,
            "analyzers": self.results,
        }

    def to_json(self, path: str | Path | None = None) -> str:
        payload = json.dumps(self.to_dict(), indent=2, default=str)
        if path is not None:
            Path(path).write_text(payload, encoding="utf-8")
        return payload

    def to_html(self, path: str | Path) -> str:
        env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            autoescape=select_autoescape(["html"]),
        )
        template = env.get_template("report.html.jinja")
        html = template.render(report=self.to_dict())
        Path(path).write_text(html, encoding="utf-8")
        return html
