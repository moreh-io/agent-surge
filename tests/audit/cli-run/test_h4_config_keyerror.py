"""H4: Missing vllm/url/model in config raises KeyError; CLI overrides
should win even if config is missing the key.

Audit: cli.md H4 — `_build_runner_config` does `cfg["vllm"]["url"]` which
crashes with an opaque traceback if the YAML omits `vllm:`. The fix:
* `cfg.get("vllm", {})` so missing config doesn't raise.
* Resolve via `getattr(args, "vllm_url", None) or vllm.get("url")`.
* If still missing, raise a typed `ConfigError` (not bare KeyError) with
  a message naming the offending key.

Same for sweep.py and slo.py.
"""

from __future__ import annotations

import argparse

import pytest


def _min_args(**kwargs):
    defaults = dict(
        vllm_url=None,
        model=None,
        preset=None,
        config=None,
        backend=None,
        no_metrics=True,
        seed=42,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_build_runner_config_no_vllm_section_with_cli_override():
    """Config missing `vllm:` entirely, but --vllm-url and --model passed:
    must succeed (CLI overrides win), not raise KeyError.
    """
    from agentsurge.cli._helpers import _build_runner_config

    args = _min_args(vllm_url="http://override:8000", model="m-override")
    cfg = {"workload": {"n_turns": 1}, "runner": {}}  # NO vllm key

    bc = _build_runner_config(args, cfg)
    assert bc.vllm_url == "http://override:8000"
    assert bc.model == "m-override"


def test_build_runner_config_missing_url_raises_typed_error():
    """No --vllm-url and config missing `vllm.url`: must raise a typed
    ConfigError (or ValueError naming the missing key), not bare KeyError.
    """
    from agentsurge.cli._helpers import _build_runner_config

    args = _min_args(vllm_url=None, model="m")
    cfg = {"workload": {"n_turns": 1}, "runner": {}, "vllm": {"model": "m"}}  # no url

    with pytest.raises(Exception) as exc:
        _build_runner_config(args, cfg)
    # Must NOT be a bare KeyError — must mention vllm.url or "url"
    assert not isinstance(exc.value, KeyError) or "vllm" in str(exc.value), (
        f"expected typed error naming vllm.url, got {type(exc.value).__name__}: {exc.value}"
    )
    msg = str(exc.value).lower()
    assert "vllm" in msg or "url" in msg, f"error must mention vllm/url: {exc.value!r}"


def test_build_runner_config_missing_model_raises_typed_error():
    """No --model and config missing `vllm.model`: typed error."""
    from agentsurge.cli._helpers import _build_runner_config

    args = _min_args(vllm_url="http://x", model=None)
    cfg = {
        "workload": {"n_turns": 1},
        "runner": {},
        "vllm": {"url": "http://x"},  # no model
    }

    with pytest.raises(Exception) as exc:
        _build_runner_config(args, cfg)
    msg = str(exc.value).lower()
    assert "model" in msg or "vllm" in msg, f"error must mention vllm/model: {exc.value!r}"
    assert not isinstance(exc.value, KeyError) or "vllm" in str(exc.value)
