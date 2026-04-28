"""CLI flag plumbing for --frontend / --frontend-* into BenchmarkConfig.frontend."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest


def _parse_run(argv_tail: list[str]):
    """Parse ``agentsurge run ...`` and return the resulting argparse Namespace."""
    import agentsurge.cli._app as app_mod

    captured: dict = {}

    def fake_cmd_run(args):
        captured["args"] = args

    test_argv = ["agentsurge", "run", *argv_tail]
    with (
        patch.object(sys, "argv", test_argv),
        patch("agentsurge.cli._app.cmd_run", side_effect=fake_cmd_run),
    ):
        app_mod.main()
    return captured["args"]


def _build_cfg(args):
    from agentsurge.cli._helpers import _build_runner_config

    cfg_yaml = {"vllm": {"url": "http://vllm:8000", "model": "m"}, "runner": {}}
    return _build_runner_config(args, cfg_yaml)


def test_default_direct_has_no_frontend_settings():
    args = _parse_run(["--backend", "mock", "--n-sessions", "1", "--no-metrics"])
    cfg = _build_cfg(args)
    assert cfg.frontend_name == "direct"
    assert cfg.frontend is None


def test_echo_with_model_and_extra_env():
    args = _parse_run(
        [
            "--backend",
            "mock",
            "--frontend",
            "echo",
            "--frontend-model",
            "gpt-test",
            "--frontend-extra-env",
            "A=1",
            "--frontend-extra-env",
            "B=2",
        ]
    )
    cfg = _build_cfg(args)
    assert cfg.frontend_name == "echo"
    assert cfg.frontend is not None
    assert cfg.frontend.model == "gpt-test"
    assert cfg.frontend.extra_env == (("A", "1"), ("B", "2"))


def test_extra_env_malformed_exits_2(capsys):
    args = _parse_run(
        [
            "--backend",
            "mock",
            "--frontend",
            "echo",
            "--frontend-extra-env",
            "A",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        _build_cfg(args)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "frontend-extra-env" in err or "KEY=VALUE" in err


def test_unknown_frontend_choice_exits_2(capsys):
    with pytest.raises(SystemExit) as exc:
        _parse_run(["--backend", "mock", "--frontend", "bogus"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "frontend" in err.lower()


def test_frontend_codex_round_trips_through_frontend_name():
    args = _parse_run(["--backend", "mock", "--frontend", "codex"])
    cfg = _build_cfg(args)
    assert cfg.frontend is not None
    assert cfg.frontend.name == "codex"
    assert cfg.frontend_name == "codex"
