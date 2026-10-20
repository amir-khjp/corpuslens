# Contributing to CorpusLens

Thanks for considering a contribution! CorpusLens is intentionally built
around a small, uniform plugin interface so it's easy to extend.

## Setup

```bash
git clone https://github.com/amir-khjp/corpuslens.git
cd corpuslens
pip install -e ".[dev]"
pytest -q
ruff check src tests
```

## Adding a new analyzer

Every analysis (duplicate detection, PII, clustering, ...) is a subclass of
`corpuslens.analyzers.base.Analyzer` with one method to implement:

```python
from corpuslens.analyzers.base import Analyzer, AnalyzerResult, Chart

class MyAnalyzer(Analyzer):
    name = "my_analyzer"

    def run(self, df, text_col, **kwargs) -> AnalyzerResult:
        ...
        return AnalyzerResult(name=self.name, summary={...}, charts=[...])
```

Register it in `src/corpuslens/analyzers/__init__.py`'s `DEFAULT_ANALYZERS`
list, add a test under `tests/`, and open a PR.

## Adding a new LLM backend

Subclass `corpuslens.llm.base.LLMBackend` (see `anthropic_backend.py` for the
reference implementation) and wire it into `get_backend()` in
`src/corpuslens/llm/__init__.py`.

## Good first issues

- **spaCy-based PII analyzer**: extend `PiiAnalyzer` (or add a new analyzer)
  to detect names/addresses via NER, as an optional extra so the core stays
  dependency-light.
- **Sentence-transformer embeddings**: add an optional embedding-based
  backend for `DuplicateAnalyzer`/`ClusteringAnalyzer` for higher recall than
  TF-IDF, gated behind an extras install.
- **Sentiment/toxicity analyzer**: a new `Analyzer` using a lightweight
  lexicon or an optional model.
- **Markdown/plain-text report export**: an alternative to `to_html`/`to_json`
  for piping into CI logs or PR comments.
- **Streaming/chunked loading**: support datasets too large to fit in memory
  via `pandas` chunked reads.

Open an issue before starting large changes so we can agree on the approach.

## Code style

- Format/lint with `ruff check src tests`.
- Keep core dependencies light; anything heavy (transformers, spaCy models)
  should be an optional extra in `pyproject.toml`.
- Every analyzer needs at least one test in `tests/`.
