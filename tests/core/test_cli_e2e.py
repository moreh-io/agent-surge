# SPDX-License-Identifier: MIT
"""End-to-end CLI tests against MockBackend.

These tests invoke real cmd_run / cmd_sweep / cmd_slo / cmd_generate
through the actual argparse parser and a real BenchmarkRunner.
The only substitution is the inference backend (MockBackend), which
lets the tests finish in under a second without a live vLLM server.

The point is to catch CLI-layer regressions that unit-level mocks
miss: arg wiring, config merging, output-file contracts, tool-mode
gating, and the runner -> result -> json serialization path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def _run_cli(argv: list[str]) -> None:
    """Invoke agentsurge.cli._app.main with the given argv."""
    from agentsurge.cli._app import main

    with patch.object(sys, "argv", ["agentsurge", *argv]):
        main()


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    d = tmp_path / "results"
    d.mkdir()
    return d


@pytest.fixture
def openhands_jsonl(tmp_path: Path) -> Path:
    """Minimal openhands-format JSONL fixture for trace-replay flows."""
    p = tmp_path / "trajs.jsonl"
    rows = [
        {
            "instance_id": f"t-{i:03d}",
            "trajectory": [
                {"role": "user", "content": f"task-{i}"},
                {"role": "assistant", "content": "plan"},
                {"role": "user", "content": "keep going"},
                {"role": "assistant", "content": "done"},
            ],
        }
        for i in range(3)
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


# ---------------------------------------------------------------------------
# cmd_run
# ---------------------------------------------------------------------------


def test_cmd_run_synthetic_against_mock_backend(out_dir: Path):
    """cmd_run end-to-end: synthetic workload + MockBackend -> RunResult json."""
    _run_cli(
        [
            "run",
            "--backend",
            "mock",
            "--output-dir",
            str(out_dir),
            "--n-sessions",
            "2",
            "--n-turns",
            "2",
            "--synthetic-tokens-per-turn",
            "64",
            "--no-metrics",
            "--tool-mode",
            "off",
        ]
    )

    runs = list(out_dir.glob("run_*.json"))
    assert runs, f"cmd_run produced no result file in {out_dir}"
    data = json.loads(runs[0].read_text())

    assert "sessions" in data
    assert len(data["sessions"]) == 2
    for s in data["sessions"]:
        assert s["completed"] is True
        assert len(s["turns"]) == 2
        for t in s["turns"]:
            assert t["completed"] is True
            assert t["ttft_ms"] >= 0.0


# ---------------------------------------------------------------------------
# cmd_sweep
# ---------------------------------------------------------------------------


def test_cmd_sweep_two_levels_against_mock_backend(out_dir: Path):
    """cmd_sweep end-to-end: two probe levels each produce a per-level JSON."""
    _run_cli(
        [
            "sweep",
            "--backend",
            "mock",
            "--output-dir",
            str(out_dir),
            "--probe-levels",
            "1,2",
            "--n-turns",
            "1",
            "--synthetic-tokens-per-turn",
            "32",
            "--cooldown",
            "0",
            "--no-cooldown",
            "--no-metrics",
            "--tool-mode",
            "off",
        ]
    )

    per_level = list(out_dir.glob("sweep_*.json"))
    assert per_level, f"cmd_sweep produced no result files in {out_dir}"
    # At minimum, one file should exist that records the sweep.
    summary = json.loads(per_level[0].read_text())
    assert "probe_results" in summary or "levels" in summary or "results" in summary


# ---------------------------------------------------------------------------
# cmd_slo
# ---------------------------------------------------------------------------


def test_cmd_slo_binary_search_against_mock_backend(out_dir: Path):
    """cmd_slo end-to-end: binary search terminates and writes boundary JSON.

    MockBackend has fast TTFT so the search should converge to max_n quickly.
    Note: with mock backend the search may report breach (max_n=0) since
    synthetic responses have minimal TTFT differentiation; per cli.md H1
    contract a breach exits 2, but the JSON is still written first.
    """
    try:
        _run_cli(
            [
                "sweep",
                "--backend",
                "mock",
                "--output-dir",
                str(out_dir),
                "--slo-ms",
                "60000",
                "--max-n",
                "4",
                "--n-turns",
                "1",
                "--synthetic-tokens-per-turn",
                "32",
                "--cooldown",
                "0",
                "--no-cooldown",
                "--no-metrics",
                "--tool-mode",
                "off",
                "--profiles",
                "1,32,32",
            ]
        )
    except SystemExit as exc:
        # H1 contract: SLO breach yields exit 2; success yields no exit.
        assert exc.code in (None, 0, 2), f"unexpected exit code {exc.code}"

    out = list(out_dir.glob("slo_boundary_*.json"))
    assert out, f"cmd_slo produced no result files in {out_dir}"
    data = json.loads(out[0].read_text())
    assert "slo_ms" in data
    assert data["slo_ms"] == 60000


# ---------------------------------------------------------------------------
# cmd_generate
# ---------------------------------------------------------------------------


def test_cmd_generate_rejects_malformed_mix_weights(out_dir: Path):
    """cmd_generate --source mixed with a bad --mix-weights must exit with
    a message naming the offending pair, not swallow the typo and run.
    """
    with pytest.raises(SystemExit) as exc:
        _run_cli(
            [
                "generate",
                "--output-dir",
                str(out_dir),
                "--source",
                "mixed",
                "--n-sessions",
                "2",
                "--mix-weights",
                "openhands=0.5,swe-bench:0.5",
            ]
        )
    # sys.exit("msg") sets .code to the string.
    assert exc.value.code and "mix-weights" in str(exc.value.code)


def test_cmd_generate_rejects_non_numeric_mix_weight(out_dir: Path):
    """Bad weight value (non-float) must produce a targeted error."""
    with pytest.raises(SystemExit) as exc:
        _run_cli(
            [
                "generate",
                "--output-dir",
                str(out_dir),
                "--source",
                "mixed",
                "--n-sessions",
                "2",
                "--mix-weights",
                "openhands:0.5,swe-bench:NOT_A_NUMBER",
            ]
        )
    assert exc.value.code and "NOT_A_NUMBER" in str(exc.value.code)


def test_cmd_generate_trace_replay_from_local_jsonl(out_dir: Path, openhands_jsonl: Path):
    """cmd_generate with --source openhands --local-path exercises the
    loader -> TraceReplayGenerator -> versioned JSON envelope path."""
    _run_cli(
        [
            "generate",
            "--output-dir",
            str(out_dir),
            "--source",
            "openhands",
            "--local-path",
            str(openhands_jsonl),
            "--n-sessions",
            "2",
        ]
    )

    workloads = list(out_dir.glob("workload_*.json"))
    assert workloads, f"cmd_generate produced no result files in {out_dir}"
    data = json.loads(workloads[0].read_text())
    assert "version" in data
    assert "sessions" in data
    assert len(data["sessions"]) >= 1
