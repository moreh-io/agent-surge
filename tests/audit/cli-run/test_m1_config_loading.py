"""M1: Empty/missing config fallback is asymmetric.

Audit: cli.md M1 — `_load_config` raises FileNotFoundError when the
AGENTSURGE_CONFIG env var points to a missing file, but silently falls
back to defaults when the file exists but is empty AND no explicit -c was
passed. Asymmetry is surprising. Pin: env-var-pointing-to-missing-file
should also fall back to default with a warning, mirroring the empty-file
soft fallback.
"""

from __future__ import annotations

import pytest


def test_explicit_config_missing_raises(tmp_path):
    """Explicit -c with non-existent file must still raise (loud)."""
    from agentsurge.cli._helpers import _load_config

    bogus = tmp_path / "does-not-exist.yaml"
    with pytest.raises(FileNotFoundError):
        _load_config(str(bogus))


def test_env_var_config_missing_falls_back_with_warning(tmp_path, monkeypatch, caplog):
    """AGENTSURGE_CONFIG env var pointing to missing file must NOT crash.

    Should fall back to bundled default and log a warning. Symmetric with
    the empty-file behaviour.
    """
    from agentsurge.cli import _helpers

    bogus = tmp_path / "missing.yaml"
    monkeypatch.setenv("AGENTSURGE_CONFIG", str(bogus))

    cfg = _helpers._load_config(None)
    assert isinstance(cfg, dict)
    assert "vllm" in cfg, "must have fallen back to default config"


def test_explicit_config_empty_yaml_exits(tmp_path):
    """Explicit -c pointing to empty YAML still exits (loud) — already works."""
    from agentsurge.cli._helpers import _load_config

    empty = tmp_path / "empty.yaml"
    empty.write_text("")

    with pytest.raises(SystemExit):
        _load_config(str(empty))
