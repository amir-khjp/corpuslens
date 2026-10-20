"""Base interface that every analyzer implements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Chart:
    """A single chart to render in the report."""

    title: str
    kind: str  # "bar", "histogram", "pie"
    labels: list[str]
    values: list[float]


@dataclass
class AnalyzerResult:
    """Structured output of an analyzer, consumed by the report renderer."""

    name: str
    summary: dict[str, Any] = field(default_factory=dict)
    charts: list[Chart] = field(default_factory=list)
    table: list[dict[str, Any]] | None = None
    warnings: list[str] = field(default_factory=list)


class Analyzer:
    """Subclass this and implement `run` to add a new analysis to CorpusLens.

    This is the main extension point for contributors — see CONTRIBUTING.md
    for the "good first issue" list of analyzers that are still missing.
    """

    name: str = "analyzer"

    def run(self, df: pd.DataFrame, text_col: str, **kwargs: Any) -> AnalyzerResult:
        raise NotImplementedError
