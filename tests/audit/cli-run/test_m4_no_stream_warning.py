"""M4: --no-stream silently disables TPOT measurement.

Audit: cli.md M4 — `--no-stream` makes TTFT == E2E and TPOT goes to ~0,
but no warning is emitted and no metadata flag marks the run as
non-streaming. Pin: emit a stderr warning at run start when --no-stream.
"""

from __future__ import annotations

import sys
from unittest.mock import patch


def _run_cli(argv: list[str]) -> int:
    from agentsurge.cli._app import main

    with patch.object(sys, "argv", ["agentsurge", *argv]):
        try:
            main()
            return 0
        except SystemExit as exc:
            return int(exc.code or 0)


def test_no_stream_emits_stderr_warning(capsys, tmp_path):
    """--no-stream should produce a single-line stderr warning at run start."""
    rc = _run_cli(
        [
            "run",
            "--backend",
            "mock",
            "--output-dir",
            str(tmp_path),
            "--no-metrics",
            "--tool-mode",
            "off",
            "--n-sessions",
            "1",
            "--n-turns",
            "1",
            "--synthetic-tokens-per-turn",
            "16",
            "--tokenizer",
            "none",
            "--ignore-replay-output-length",
            "--no-stream",
        ]
    )
    assert rc == 0, f"run failed with rc={rc}"
    captured = capsys.readouterr()
    combined = (captured.err + captured.out).lower()
    # Must mention non-streaming and TPOT
    assert "non-streaming" in combined or "no-stream" in combined, (
        f"--no-stream warning missing from output: {combined!r}"
    )
    assert "tpot" in combined, f"--no-stream warning must mention TPOT impact: {combined!r}"
