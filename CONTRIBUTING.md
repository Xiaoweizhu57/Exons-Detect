# Contributing

Thanks for contributing to Exons-Detect.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e .[dev]
```

## Development Guidelines

- Keep public APIs typed and documented.
- Add or update tests for behavior changes.
- Prefer small, focused pull requests.
- Do not commit model weights or large datasets.

## Testing

```bash
pytest
```

## Issues and Pull Requests

When opening an issue or PR, please include:

- A short problem statement
- Reproduction steps if relevant
- Expected and actual behavior
- Environment details for model-loading issues
