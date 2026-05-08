"""M2: No range validation on durations / rates / counts.

Audit: cli.md M2 — `--duration -1`, `--arrival-rate 0`, `--n-sessions 0`,
`--ramp-duration 1000 --duration 10` all parse cleanly. Pin: a validation
pass at the top of cmd_run / cmd_sweep / cmd_slo asserts:
  duration >= 0
  ramp_duration <= duration when both > 0
  arrival_rate > 0 (or 'auto' from preset)
  n_sessions > 0
  max_tokens > 0
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest


def _run_cli(argv: list[str]) -> None:
    from agentsurge.cli._app import main

    with patch.object(sys, "argv", ["agentsurge", *argv]):
        main()


@pytest.mark.parametrize(
    "extra,err_substr",
    [
        (["--duration", "-1"], "duration"),
        (["--arrival-rate", "0"], "arrival-rate"),
        (["--n-sessions", "0"], "n-sessions"),
        (["--max-tokens", "-5"], "max-tokens"),
    ],
)
def test_invalid_numeric_inputs_exit_with_clear_error(extra, err_substr, capsys, tmp_path):
    """Each invalid value must fail fast with a message naming the flag."""
    argv = [
        "run",
        "--backend",
        "mock",
        "--output-dir",
        str(tmp_path),
        "--no-metrics",
        "--tool-mode",
        "off",
        "--tokenizer",
        "none",
        "--ignore-replay-output-length",
        *extra,
    ]
    with pytest.raises(SystemExit) as exc:
        _run_cli(argv)
    assert exc.value.code != 0
    captured = capsys.readouterr()
    msg = (captured.err + captured.out).lower()
    assert err_substr in msg, f"error must mention {err_substr!r}: {msg!r}"


def test_ramp_longer_than_duration_rejected(capsys, tmp_path):
    """--ramp-duration > --duration is meaningless and must be rejected."""
    argv = [
        "run",
        "--backend",
        "mock",
        "--output-dir",
        str(tmp_path),
        "--no-metrics",
        "--tool-mode",
        "off",
        "--tokenizer",
        "none",
        "--ignore-replay-output-length",
        "--duration",
        "10",
        "--ramp-duration",
        "1000",
    ]
    with pytest.raises(SystemExit) as exc:
        _run_cli(argv)
    assert exc.value.code != 0
    captured = capsys.readouterr()
    assert "ramp" in (captured.err + captured.out).lower()
