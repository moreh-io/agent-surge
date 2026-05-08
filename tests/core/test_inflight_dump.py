# SPDX-License-Identifier: MIT
"""Tests for the inflight per-turn JSONL dump writer."""

import json
import threading
from pathlib import Path

import pytest

from agentsurge._inflight_dump import InflightDumpWriter
from agentsurge.types import TurnResult


def _make_turn(session: str, idx: int, finish: str = "stop") -> TurnResult:
    return TurnResult(
        session_id=session,
        turn_index=idx,
        completed=True,
        ttft_ms=12.5 + idx,
        total_ms=100.0 + idx,
        output_tokens=50 + idx,
        response_text=f"reply {session}/{idx}",
        finish_reason=finish,
    )


class TestInflightDumpWriter:
    def test_writes_one_line_per_turn(self, tmp_path: Path) -> None:
        path = tmp_path / "turns.jsonl"
        writer = InflightDumpWriter(path, flush_every=10)
        writer.start()
        try:
            for i in range(25):
                writer.on_turn(_make_turn("s1", i))
        finally:
            writer.stop()

        lines = path.read_text().splitlines()
        assert len(lines) == 25
        parsed = [json.loads(ln) for ln in lines]
        assert [p["turn"] for p in parsed] == list(range(25))
        assert all(p["session"] == "s1" for p in parsed)

    def test_line_schema(self, tmp_path: Path) -> None:
        path = tmp_path / "turns.jsonl"
        writer = InflightDumpWriter(path)
        writer.start()
        try:
            writer.on_turn(_make_turn("abc", 3, finish="length"))
        finally:
            writer.stop()

        rec = json.loads(path.read_text().strip())
        assert rec == {
            "session": "abc",
            "turn": 3,
            "tokens": 53,
            "ttft_ms": 15.5,
            "response_text": "reply abc/3",
            "finish_reason": "length",
        }

    def test_flush_on_shutdown_keeps_partial_buffer(self, tmp_path: Path) -> None:
        path = tmp_path / "turns.jsonl"
        writer = InflightDumpWriter(path, flush_every=100)
        writer.start()
        try:
            # Only 3 turns — far below flush_every. Must still land on disk after stop().
            for i in range(3):
                writer.on_turn(_make_turn("s2", i))
        finally:
            writer.stop()

        lines = path.read_text().splitlines()
        assert len(lines) == 3

    def test_concurrent_sessions_no_corruption(self, tmp_path: Path) -> None:
        path = tmp_path / "turns.jsonl"
        writer = InflightDumpWriter(path, flush_every=50)
        writer.start()

        n_sessions = 8
        turns_per_session = 30

        def feed(session: str) -> None:
            for i in range(turns_per_session):
                writer.on_turn(_make_turn(session, i))

        threads = [threading.Thread(target=feed, args=(f"s{k}",)) for k in range(n_sessions)]
        try:
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            writer.stop()

        lines = path.read_text().splitlines()
        assert len(lines) == n_sessions * turns_per_session
        parsed = [json.loads(ln) for ln in lines]  # valid JSON per line
        counts: dict[str, int] = {}
        for p in parsed:
            counts[p["session"]] = counts.get(p["session"], 0) + 1
        assert all(c == turns_per_session for c in counts.values())
        assert len(counts) == n_sessions

    def test_stop_is_idempotent(self, tmp_path: Path) -> None:
        path = tmp_path / "turns.jsonl"
        writer = InflightDumpWriter(path)
        writer.start()
        writer.on_turn(_make_turn("s", 0))
        writer.stop()
        writer.stop()  # must not raise

    def test_on_turn_after_stop_is_noop(self, tmp_path: Path) -> None:
        path = tmp_path / "turns.jsonl"
        writer = InflightDumpWriter(path)
        writer.start()
        writer.on_turn(_make_turn("s", 0))
        writer.stop()
        # Late-arriving callback after shutdown must not raise and must not re-open the file.
        writer.on_turn(_make_turn("s", 1))
        lines = path.read_text().splitlines()
        assert len(lines) == 1


class TestBenchmarkConfigInflightDumpFlag:
    def test_default_is_false(self) -> None:
        from agentsurge.types.results import BenchmarkConfig

        cfg = BenchmarkConfig(vllm_url="http://localhost:8000", model="m")
        assert cfg.inflight_dump is False

    def test_explicit_true(self) -> None:
        from agentsurge.types.results import BenchmarkConfig

        cfg = BenchmarkConfig(vllm_url="http://localhost:8000", model="m", inflight_dump=True)
        assert cfg.inflight_dump is True


class TestCliFlagWiring:
    def _run_parser(self):
        from agentsurge.cli import _app as app_mod

        return app_mod._run_args()

    def test_flag_default_off(self) -> None:
        p = self._run_parser()
        args = p.parse_args([])
        assert getattr(args, "enable_inflight_dump", False) is False

    def test_flag_on(self) -> None:
        p = self._run_parser()
        args = p.parse_args(["--enable-inflight-dump"])
        assert args.enable_inflight_dump is True


class TestPathDerivation:
    def test_path_with_dot_json_in_directory(self, tmp_path: Path) -> None:
        """Concern #4: str.replace clobbers .json in parent dir components."""
        weird_dir = tmp_path / "bench.json"
        weird_dir.mkdir()
        base = str(weird_dir / "run_xyz.json")

        # Emulate the BUGGY derivation in cli/run.py
        buggy_jsonl = base.replace(".json", "_turns.jsonl")
        # Correct derivation using Path.with_suffix
        from pathlib import Path as _Path

        correct_jsonl = str(_Path(base).with_suffix("")) + "_turns.jsonl"

        # Buggy version replaces ALL occurrences — the directory gets mangled
        assert buggy_jsonl != correct_jsonl, (
            "Expected buggy str.replace to produce a different (wrong) path"
        )
        # Correct path keeps the directory component intact
        assert correct_jsonl == str(weird_dir / "run_xyz_turns.jsonl")

    def test_path_normal_case_unchanged(self, tmp_path: Path) -> None:
        """Correct derivation matches old behaviour for a normal path."""
        from pathlib import Path as _Path

        base = str(tmp_path / "run_20240101.json")
        correct_jsonl = str(_Path(base).with_suffix("")) + "_turns.jsonl"
        assert correct_jsonl == str(tmp_path / "run_20240101_turns.jsonl")


class TestStopBeforeStart:
    def test_stop_before_start_is_no_op(self, tmp_path: Path) -> None:
        """Concern #5: stop() on unstarted writer must not poison _stopped."""
        writer = InflightDumpWriter(tmp_path / "t.jsonl")
        writer.stop()  # should be a no-op
        writer.start()
        writer.on_turn(_make_turn("s", 0))
        writer.stop()
        lines = (tmp_path / "t.jsonl").read_text().splitlines()
        assert len(lines) == 1, (
            f"Expected 1 line but got {len(lines)} — stop()-before-start() poisoned _stopped"
        )


class TestStopRace:
    def test_stop_races_feeder_no_corruption(self, tmp_path: Path) -> None:
        """Concern #2: feeder thread runs concurrently with stop() on the main thread.

        Invariant: every written line is valid JSON (no partial / corrupted records).
        The exact count is intentionally not asserted — the race window is lossy by design.
        """
        path = tmp_path / "race.jsonl"
        writer = InflightDumpWriter(path, flush_every=10)
        writer.start()

        n_turns = 200
        feeder_done = threading.Event()

        def feeder() -> None:
            for i in range(n_turns):
                writer.on_turn(_make_turn("race", i))
            feeder_done.set()

        t = threading.Thread(target=feeder)
        t.start()

        # Race: stop() on the main thread while feeder is still running.
        writer.stop()

        t.join(timeout=5.0)
        assert feeder_done.is_set(), "feeder thread did not finish within 5 s"

        # File may not exist if stop() won the race before any writes.
        if not path.exists() or path.stat().st_size == 0:
            return

        lines = path.read_text().splitlines()
        assert len(lines) > 0, "Expected at least some lines before the race completed"
        assert len(lines) <= n_turns, "More lines than total turns — double-write bug"

        for ln in lines:
            rec = json.loads(ln)  # raises if line is not valid JSON
            assert rec["session"] == "race"


@pytest.fixture
def fake_turn() -> TurnResult:
    return _make_turn("fx", 0)
