# Contributing

## Setup

```bash
cd agent-surge
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all,dev]"
pre-commit install
```

Requires Python >= 3.13. Pre-commit runs ruff (lint + format) and mypy on every commit -- see `pyproject.toml` for config.

## Tests

```bash
pytest tests/
pytest tests/capacity/                 # capacity module only
pytest tests/core/test_workload.py -v  # single file
```

`tests/core/test_multi_turn_feeding.py` includes a live vLLM smoke test for the
empty-assistant-content `finish` handoff. Run it only when you have a direct
OpenAI-compatible vLLM endpoint with tool calling enabled:

```bash
AGENTSURGE_LIVE_VLLM_URL=http://host:8000 \
AGENTSURGE_LIVE_VLLM_MODEL=served-model-name \
python3.13 -m pytest tests/core/test_multi_turn_feeding.py -k live_vllm -v
```

## Pull requests

One feature or fix per PR. Add tests. Run `pytest tests/` before submitting. Public contributions should target https://github.com/moreh-io/agent-surge.
