# CorpusLens

**Automated EDA and data-mining reports for text datasets — like `pandas-profiling`, but for NLP data.**

Point CorpusLens at any CSV/JSONL dataset with a text column and get back a
single HTML report covering duplicate detection, PII leaks, topic clusters,
keyword extraction, language mix, and class balance — in one command, with
no API key required.

```bash
pip install corpuslens
corpuslens report reviews.csv --text-col text --label-col category
```

Opens `corpuslens_report.html` with:

- **Length & language stats** — word/char distributions, vocabulary diversity, language mix
- **Duplicate detection** — exact duplicates + near-duplicates (TF-IDF cosine similarity), the #1 cause of dataset leakage
- **PII detection** — emails, phone numbers, IPs, credit-card-shaped strings, flagged before you share a dataset
- **Topic clustering** — TF-IDF + KMeans clusters with auto-extracted keywords
- **Class balance** — distribution and imbalance ratio for an optional label column
- **Optional LLM-powered cluster labels** — plug in Claude or GPT to turn `"revenue / quarter / forecast"` into `"Quarterly earnings"` (fully optional — the core report works with zero API calls)

## Why

Tabular data has `pandas-profiling`/`ydata-profiling`. Text data mostly
doesn't have an equivalent — yet text datasets are exactly where silent
problems (duplicated training examples, leaked PII, lopsided topic mix) are
easiest to miss and most expensive to ship.

## Install

```bash
pip install corpuslens          # core (no LLM dependency)
pip install corpuslens[anthropic]  # + Claude-powered cluster labels
pip install corpuslens[openai]     # + GPT-powered cluster labels
```

## Usage

### CLI

```bash
corpuslens report data.csv --text-col text --label-col category -o report.html
```

```
Usage: corpuslens report [OPTIONS] DATA_PATH

Options:
  --text-col TEXT        Name of the column containing text.  [required]
  --label-col TEXT        Optional column with class labels, for balance stats.
  -o, --output TEXT       Output HTML path.
  --json-output TEXT      Optional path to also write a JSON report.
  --llm [none|anthropic|openai]
                          LLM backend for cluster labeling.
```

### Python API

```python
import pandas as pd
from corpuslens import CorpusReport

df = pd.read_csv("data.csv")
report = CorpusReport(df, text_col="text", label_col="category")
report.run()

report.to_html("report.html")
report.to_json("report.json")       # for CI pipelines / programmatic checks
```

Each analyzer result is a plain dict, so you can fail a CI job on, say,
too many exact duplicates:

```python
dup_ratio = next(a for a in report.results if a["name"] == "duplicates")["summary"]["exact_duplicate_ratio"]
assert dup_ratio < 0.05, "Too many duplicate rows in training data!"
```

## Design

CorpusLens is built around a small `Analyzer` plugin interface — every
section of the report (duplicates, PII, clustering, ...) is an independent,
testable class. Adding a new analysis is a ~30-line PR. See
[CONTRIBUTING.md](CONTRIBUTING.md).

The core install has **no heavy ML dependencies** (no torch, no
transformers) — clustering and near-duplicate detection use TF-IDF, not
embeddings, so `pip install corpuslens` stays fast. LLM backends
(Claude/GPT) are opt-in extras used only to improve cluster labels, never
required for the report to run.

## Status

Early and actively developed. API may change before 1.0. Issues and PRs
welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for good first issues
(spaCy-based PII, embedding-based clustering, sentiment analysis, and more).

## License

MIT
