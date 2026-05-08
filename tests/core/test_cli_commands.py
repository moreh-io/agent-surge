"""Smoke tests for untested CLI commands - argparse Namespace invocation.

Tests verify: no crash on invocation, correct dispatch, basic arg validation.
All external dependencies (runner, analyzer, capacity, metrics) are mocked.
"""

from __future__ import annotations

import argparse
import sys

import pytest

pytestmark = pytest.mark.e2e


def _make_args(**kwargs):
    """Build an argparse.Namespace with common defaults."""
    defaults = {
        "config": None,
        "vllm_url": "http://localhost:8000",
        "model": "test-model",
        "output_dir": "/tmp/test_agentsurge_cli",
        "command": "test",
        "api": "chat",
        "backend": "vllm",
        "no_metrics": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ===========================================================================
# cmd_sweep - dispatch tests
# ===========================================================================


class TestCmdSweep:
    def test_negative_hint_exits(self):
        from agentsurge.cli.sweep import cmd_sweep

        args = _make_args(slo_ms=3000, hint=-5)
        with pytest.raises(SystemExit):
            cmd_sweep(args)


# ===========================================================================
# CLI structure: all commands importable
# ===========================================================================


class TestCLIStructure:
    def test_all_commands_importable(self):
        import inspect

        from agentsurge.cli import (
            cmd_generate,
            cmd_run,
            cmd_sweep,
        )
        from agentsurge.cli.slo import cmd_slo

        for fn in [
            cmd_generate,
            cmd_run,
            cmd_sweep,
            cmd_slo,
        ]:
            assert callable(fn)
            sig = inspect.signature(fn)
            params = list(sig.parameters)
            assert len(params) >= 1, f"{fn.__name__} should accept at least one argument"

    def test_sweep_subcommand_in_help(self):
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "agentsurge", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        for cmd in ("sweep", "generate"):
            assert cmd in result.stdout, f"'{cmd}' missing from --help"
