# Exons-Detect

Exons-Detect is a PyTorch and Hugging Face based text detector that scores a passage by comparing the token-level behavior and hidden-state divergence of two causal language models.

This repository is a cleaned, GitHub-ready open-source packaging of the original code you provided. It keeps the core weighted scoring idea while adding a reusable Python package, a command-line interface, tests, and standard project metadata.

## Highlights

- Reusable Python package under `src/exons_detect`
- CLI for batch scoring JSON datasets
- Typed, documented implementation of the weighted detector
- `pyproject.toml` based packaging
- Unit tests for the numerical scoring utilities

## Repository Layout

```text
Exons-Detect/
|-- src/exons_detect/
|   |-- __init__.py
|   |-- cli.py
|   |-- detector.py
|   |-- metrics.py
|   `-- utils.py
|-- tests/
|   |-- test_metrics.py
|   `-- test_package.py
|-- .gitignore
|-- CONTRIBUTING.md
|-- CITATION.cff
|-- LICENSE
|-- README.md
`-- pyproject.toml
```

## Installation

```bash
git clone <your-repo-url>
cd Exons-Detect
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e .
```



```bash
set HF_TOKEN=your_token_here
```

## Quick Start

### Python API

```python
from exons_detect import ExonsDetect

detector = ExonsDetect(
    observer_name_or_path="tiiuae/falcon-7b",
    performer_name_or_path="tiiuae/falcon-7b-instruct",
)

score = detector.score_text("This is a sample passage.")
print(score)
```

### Command Line

Given an input JSON file such as:

```json
{
  "human_text": ["example one", "example two"]
}
```

run:

```bash
exons-detect ^
  --input data.json ^
  --output scores.json ^
  --text-key human_text ^
  --observer tiiuae/falcon-7b ^
  --performer tiiuae/falcon-7b-instruct
```

The output JSON format is:

```json
{
  "text": ["example one", "example two"],
  "predictions": [0.91, 1.04]
}
```

## Method Overview

Exons-Detect computes a weighted score using:

1. Token-level perplexity style signals from a performer model
2. Token-level weighted entropy between observer and performer distributions
3. Layer-wise hidden-state divergence aggregated across the last `hidden_num` layers

The final score is computed as a ratio:

```text
weighted_perplexity / weighted_entropy
```

where the token weights are derived from hidden-state cosine divergence after thresholding and exponential scaling.

## Notes

- The default model names are examples and may require significant GPU memory.
- Both models must use identical tokenizers.
- The provided tests validate the scoring math, not end-to-end model downloads.
- For a publication release, you may want to add model cards, benchmark scripts, and dataset links specific to your paper.

## Development

```bash
pip install -e .[dev]
pytest
```

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE).
