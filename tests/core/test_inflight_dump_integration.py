# SPDX-License-Identifier: MIT
"""End-to-end integration: BenchmarkRunner → InflightDumpWriter → file.

Unit tests in ``test_inflight_dump.py`` exercise the writer in isolation.
This module drives the full runner chain with MockBackend to verify that
the streamed JSONL lines and the final consolidated result stay in
agreement on turn count and schema.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agentsurge._inflight_dump import InflightDumpWriter
from agentsurge.backends.mock import MockBackend, MockConfig
from agentsurge.runner import BenchmarkRunner
from agentsurge.types import ReplaySession
from agentsurge.types.results import BenchmarkConfig


def _run_with_writer(tmp_path: Path, n_sessions: int, turns_per_session: int):
    jsonl_path = tmp_path / "turns.jsonl"
    writer = InflightDumpWriter(jsonl_path)
    writer.start()

    cfg = BenchmarkConfig(
        vllm_url="http://mock",
        model="mock",
        max_concurrency=4,
        max_tokens=8,
        arrival_pattern="constant",
        arrival_rate=1000.0,
        duration=0.0,
        ignore_replay_output_length=True,
        skip_tokenizer_load=True,
        seed=0,
    )
    backend = MockBackend(MockConfig(output_tokens=4, ttft_ms=0.0, inter_token_ms=0.0))

    sessions = []
    for s in range(n_sessions):
        per_turn_messages = [
            [{"role": "user", "content": f"hi {s} t{t}"}] for t in range(turns_per_session)
        ]
        sessions.append(
            ReplaySession(
                session_id=f"s{s}",
                turn_messages=per_turn_messages,
                metadata={},
            )
        )

    runner = BenchmarkRunner(cfg, backend=backend, on_turn=writer.on_turn)
    try:
        result = asyncio.run(runner.run(sessions))
    finally:
        writer.stop()

    return result, jsonl_path


class TestInflightDumpIntegration:
    def test_jsonl_matches_final_result_turn_count(self, tmp_path: Path) -> None:
        result, jsonl_path = _run_with_writer(tmp_path, n_sessions=3, turns_per_session=4)

        total_turns = sum(len(s.turns) for s in result.sessions)
        lines = jsonl_path.read_text().splitlines()
        assert total_turns > 0
        assert len(lines) == total_turns

    def test_jsonl_schema_matches_spec(self, tmp_path: Path) -> None:
        _, jsonl_path = _run_with_writer(tmp_path, n_sessions=2, turns_per_session=2)

        required_keys = {"session", "turn", "tokens", "ttft_ms", "response_text", "finish_reason"}
        for line in jsonl_path.read_text().splitlines():
            rec = json.loads(line)  # valid JSON per line
            assert set(rec.keys()) == required_keys

    def test_jsonl_tokens_match_turn_output_tokens(self, tmp_path: Path) -> None:
        result, jsonl_path = _run_with_writer(tmp_path, n_sessions=2, turns_per_session=3)

        by_key = {}
        for s in result.sessions:
            for t in s.turns:
                by_key[(s.session_id, t.turn_index)] = t.output_tokens

        for line in jsonl_path.read_text().splitlines():
            rec = json.loads(line)
            key = (rec["session"], rec["turn"])
            assert key in by_key, f"JSONL line for unknown turn {key}"
            assert rec["tokens"] == by_key[key]

    def test_writer_off_produces_no_file(self, tmp_path: Path) -> None:
        """Symmetry check: when no writer is wired, no JSONL is produced."""
        cfg = BenchmarkConfig(
            vllm_url="http://mock",
            model="mock",
            max_concurrency=2,
            max_tokens=8,
            arrival_pattern="constant",
            arrival_rate=1000.0,
            ignore_replay_output_length=True,
            skip_tokenizer_load=True,
            seed=0,
        )
        backend = MockBackend(MockConfig(output_tokens=2, ttft_ms=0.0, inter_token_ms=0.0))
        sessions = [
            ReplaySession(
                session_id="s0",
                turn_messages=[[{"role": "user", "content": "hi"}]],
                metadata={},
            )
        ]
        runner = BenchmarkRunner(cfg, backend=backend)  # no on_turn
        asyncio.run(runner.run(sessions))

        # No InflightDumpWriter was started — the tmp dir should be untouched.
        assert not any(tmp_path.iterdir())


class TestInflightDumpOrderingAndFanOut:
    def test_per_session_turn_order_is_monotonic(self, tmp_path: Path) -> None:
        """Concern #3a: within each session, JSONL lines must appear in
        monotonically-increasing turn_index order."""
        result, jsonl_path = _run_with_writer(tmp_path, n_sessions=4, turns_per_session=5)

        lines = jsonl_path.read_text().splitlines()
        assert len(lines) > 0

        # Group by session and check monotonically increasing turn indices.
        by_session: dict[str, list[int]] = {}
        for ln in lines:
            rec = json.loads(ln)
            by_session.setdefault(rec["session"], []).append(rec["turn"])

        for sid, turn_indices in by_session.items():
            for i in range(1, len(turn_indices)):
                assert turn_indices[i] > turn_indices[i - 1], (
                    f"Session {sid!r}: turn_index went {turn_indices[i - 1]} → "
                    f"{turn_indices[i]} (not monotonically increasing)"
                )

    def test_fan_out_turns_all_appear_in_jsonl(self, tmp_path: Path) -> None:
        """Concern #3b: sessions with fan_out=3 produce 3 TurnResults per turn.
        Total JSONL lines must equal the total TurnResults in the RunResult."""
        jsonl_path = tmp_path / "turns.jsonl"
        writer = InflightDumpWriter(jsonl_path)
        writer.start()

        fan_out = 3
        n_sessions = 2
        turns_per_session = 2

        cfg = BenchmarkConfig(
            vllm_url="http://mock",
            model="mock",
            max_concurrency=16,
            max_tokens=8,
            arrival_pattern="constant",
            arrival_rate=1000.0,
            duration=0.0,
            ignore_replay_output_length=True,
            skip_tokenizer_load=True,
            seed=0,
        )
        backend = MockBackend(MockConfig(output_tokens=4, ttft_ms=0.0, inter_token_ms=0.0))

        sessions = []
        for s in range(n_sessions):
            per_turn_messages = [
                [{"role": "user", "content": f"hi {s} t{t}"}] for t in range(turns_per_session)
            ]
            sessions.append(
                ReplaySession(
                    session_id=f"fs{s}",
                    turn_messages=per_turn_messages,
                    metadata={},
                    fan_out=fan_out,
                )
            )

        runner = BenchmarkRunner(cfg, backend=backend, on_turn=writer.on_turn)
        try:
            result = asyncio.run(runner.run(sessions))
        finally:
            writer.stop()

        total_turns_in_result = sum(len(s.turns) for s in result.sessions)
        lines = jsonl_path.read_text().splitlines()

        # Each session has turns_per_session turns × fan_out calls each.
        expected = n_sessions * turns_per_session * fan_out
        assert total_turns_in_result == expected, (
            f"RunResult has {total_turns_in_result} TurnResults, expected {expected}"
        )
        assert len(lines) == total_turns_in_result, (
            f"JSONL has {len(lines)} lines but RunResult has {total_turns_in_result} TurnResults"
        )
