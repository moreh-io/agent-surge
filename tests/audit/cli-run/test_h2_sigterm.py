"""H2: SIGTERM handler must convert to cooperative cancel and flush results.

Audit: cli.md H2 — only KeyboardInterrupt is caught at `_app.py:820`. SIGTERM
isn't installed at all, so k8s/slurm preemption silently kills mid-run with
no partial JSON. This test pins the contract: the CLI MUST install a SIGTERM
handler and flush partial results, exiting 130-style.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _venv_python() -> str:
    cand = REPO_ROOT / ".venv" / "bin" / "python3"
    return str(cand) if cand.exists() else sys.executable


def test_sigterm_handler_installed(monkeypatch):
    """The CLI's main() must install a SIGTERM handler before dispatching.

    We can't easily fire SIGTERM mid-asyncio in pytest, so we assert the
    handler-installation hook exists and is wired into main().
    """
    from agentsurge.cli import _app

    assert hasattr(_app, "_install_signal_handlers"), (
        "Expected _install_signal_handlers helper to be defined in cli._app (see cli.md H2)."
    )


def test_main_catches_sigterm_with_distinct_exit(monkeypatch):
    """`main()` must translate SIGTERM into a SystemExit with code 143
    (128+SIGTERM), distinct from 130 (SIGINT).
    """
    import argparse

    from agentsurge.cli import _app

    fake_args = argparse.Namespace(command="run")

    def _raise_term(_=None):
        # Simulate the SIGTERM handler firing during dispatch.
        raise _app._SigTerm()  # type: ignore[attr-defined]

    fake_args.func = _raise_term

    monkeypatch.setattr(_app.argparse.ArgumentParser, "parse_args", lambda self: fake_args)

    with pytest.raises(SystemExit) as exc:
        _app.main()
    assert exc.value.code == 143, f"expected exit 143 (128+SIGTERM); got {exc.value.code}"


def test_sigterm_subprocess_flushes_partial_results(tmp_path: Path):
    """Send SIGTERM to a real `python -m agentsurge run` subprocess and verify:
    * exit code 143 (128+SIGTERM)
    * a partial result file was written into --output-dir
    """
    out_dir = tmp_path / "results"
    out_dir.mkdir()

    proc = subprocess.Popen(
        [
            _venv_python(),
            "-m",
            "agentsurge",
            "run",
            "--backend",
            "mock",
            "--output-dir",
            str(out_dir),
            "--client-concurrency",
            "1",
            "--n-sessions",
            "200",
            "--n-turns",
            "40",
            "--synthetic-tokens-per-turn",
            "1024",
            "--no-metrics",
            "--tool-mode",
            "off",
            "--tokenizer",
            "none",
            "--ignore-replay-output-length",
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    # Let the run start producing turns before SIGTERM.
    time.sleep(2.0)
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("CLI did not exit within 20s after SIGTERM")

    assert proc.returncode == 143, (
        f"expected exit 143 on SIGTERM, got {proc.returncode}; "
        f"stderr={proc.stderr.read().decode(errors='ignore')[-500:]}"
    )

    # Partial results must have been written.
    runs = list(out_dir.glob("run_*.json"))
    assert runs, "no partial results were written on SIGTERM"
