"""Anthropic (Claude) backend. Requires `pip install corpuslens[anthropic]`
and an ANTHROPIC_API_KEY environment variable.
"""

from __future__ import annotations

import os

from corpuslens.llm.base import LLMBackend

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class AnthropicBackend(LLMBackend):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "AnthropicBackend requires the 'anthropic' package: "
                "pip install corpuslens[anthropic]"
            ) from exc

        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._model = model

    def _complete(self, prompt: str, max_tokens: int = 200) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text").strip()

    def summarize(self, text: str) -> str:
        prompt = f"Summarize this dataset description in one or two plain-English sentences:\n\n{text}"
        return self._complete(prompt, max_tokens=120)

    def label_cluster(self, keywords: list[str], samples: list[str]) -> str:
        sample_block = "\n".join(f"- {s[:150]}" for s in samples)
        prompt = (
            "Give a short (3-6 word) topic label for a cluster of text documents.\n"
            f"Top keywords: {', '.join(keywords)}\n"
            f"Example documents:\n{sample_block}\n\n"
            "Respond with only the label, no punctuation or explanation."
        )
        return self._complete(prompt, max_tokens=20)
