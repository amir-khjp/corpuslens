"""Default backend: no API calls, no cost, no network dependency."""

from __future__ import annotations

from corpuslens.llm.base import LLMBackend


class NoOpBackend(LLMBackend):
    def summarize(self, text: str) -> str:
        return text[:200] + ("..." if len(text) > 200 else "")

    def label_cluster(self, keywords: list[str], samples: list[str]) -> str:
        return " / ".join(keywords[:3])
