"""Tests for the agentsurge.run() and agentsurge.arun() public Python API."""

from __future__ import annotations

import pytest

from agentsurge import arun, run
from agentsurge._api import _build_config
from agentsurge.types import BenchmarkConfig, ReplaySession, RunResult, SessionResult, TurnResult


class TestRunMockBackend:
    def test_run_mock_backend_returns_run_result(self):
        result = run(
            backend="mock",
            vllm_url="http://x",
            model="m",
            n_sessions=3,
            n_turns=2,
        )
        assert isinstance(result, RunResult)
        assert len(result.sessions) == 3
        for sr in result.sessions:
            assert isinstance(sr, SessionResult)
            assert sr.n_turns == 2
            assert sr.completed

    def test_run_with_custom_sessions(self):
        sessions = [
            ReplaySession(
                session_id="custom-0",
                turn_messages=[
                    [{"role": "user", "content": "hello"}],
                    [
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "hi"},
                        {"role": "user", "content": "bye"},
                    ],
                ],
            ),
            ReplaySession(
                session_id="custom-1",
                turn_messages=[
                    [{"role": "user", "content": "one"}],
                ],
            ),
        ]
        result = run(
            backend="mock",
            vllm_url="http://x",
            model="m",
            sessions=sessions,
        )
        assert len(result.sessions) == 2
        ids = {sr.session_id for sr in result.sessions}
        assert ids == {"custom-0", "custom-1"}
        turns_by_id = {sr.session_id: sr.n_turns for sr in result.sessions}
        assert turns_by_id["custom-0"] == 2
        assert turns_by_id["custom-1"] == 1

    def test_run_with_preset(self):
        result = run(
            preset="stress-default",
            backend="mock",
            vllm_url="http://x",
            model="m",
            n_sessions=2,
            n_turns=1,
        )
        assert isinstance(result, RunResult)
        assert len(result.sessions) == 2
        assert result.config.get("arrival_pattern") == "burst"

    def test_run_callbacks_invoked(self):
        turn_results: list[TurnResult] = []
        session_results: list[SessionResult] = []

        run(
            backend="mock",
            vllm_url="http://x",
            model="m",
            n_sessions=2,
            n_turns=3,
            on_turn=lambda tr: turn_results.append(tr),
            on_session=lambda sr: session_results.append(sr),
        )
        assert len(session_results) == 2
        assert len(turn_results) == 6
        assert all(isinstance(tr, TurnResult) for tr in turn_results)
        assert all(isinstance(sr, SessionResult) for sr in session_results)

    def test_run_deterministic_seed(self):
        r1 = run(backend="mock", vllm_url="http://x", model="m", n_sessions=5, n_turns=3, seed=99)
        r2 = run(backend="mock", vllm_url="http://x", model="m", n_sessions=5, n_turns=3, seed=99)
        ids_1 = [sr.session_id for sr in r1.sessions]
        ids_2 = [sr.session_id for sr in r2.sessions]
        assert ids_1 == ids_2
        turns_1 = [(t.session_id, t.turn_index) for t in r1.all_turn_results()]
        turns_2 = [(t.session_id, t.turn_index) for t in r2.all_turn_results()]
        assert turns_1 == turns_2


class TestArun:
    @pytest.mark.asyncio
    async def test_arun_works(self):
        result = await arun(
            backend="mock",
            vllm_url="http://x",
            model="m",
            n_sessions=2,
            n_turns=2,
        )
        assert isinstance(result, RunResult)
        assert len(result.sessions) == 2
        for sr in result.sessions:
            assert sr.n_turns == 2
            assert sr.completed


class TestRunValidation:
    def test_run_missing_url_raises(self):
        with pytest.raises(ValueError, match="vllm_url is required"):
            run(model="m", backend="mock")

    def test_run_missing_model_raises(self):
        with pytest.raises(ValueError, match="model is required"):
            run(vllm_url="http://x", backend="mock")


class TestBuildConfig:
    def test_build_config_preset_loaded(self):
        cfg = _build_config(
            vllm_url="http://x",
            model="m",
            config=None,
            preset="stress-default",
            max_concurrency=None,
            max_tokens=None,
            temperature=None,
            arrival_pattern=None,
            no_metrics=False,
            seed=None,
            backend_name="mock",
        )
        assert isinstance(cfg, BenchmarkConfig)
        assert cfg.vllm_url == "http://x"
        assert cfg.model == "m"
        assert cfg.preset == "stress-default"
        assert cfg.arrival_pattern == "burst"

    def test_build_config_explicit_kwargs_win_where_applicable(self):
        cfg = _build_config(
            vllm_url="http://x",
            model="m",
            config=None,
            preset="stress-default",
            max_concurrency=7,
            max_tokens=128,
            temperature=0.9,
            arrival_pattern="constant",
            no_metrics=True,
            seed=77,
            backend_name=None,
        )
        assert cfg.max_concurrency == 7
        assert cfg.max_tokens == 128
        assert cfg.temperature == 0.9
        assert cfg.arrival_pattern == "constant"
        assert cfg.seed == 77
        assert cfg.no_metrics is True
        assert cfg.preset == "stress-default"
        assert cfg.vllm_url == "http://x"

    def test_build_config_explicit_args_override_preset(self):
        """Explicit caller args must win over preset-defined values (H1 fix)."""
        cfg = _build_config(
            vllm_url="http://x",
            model="m",
            config=None,
            preset="stress-default",
            max_concurrency=None,
            max_tokens=512,
            temperature=0.7,
            arrival_pattern="constant",
            no_metrics=False,
            seed=99,
            backend_name=None,
        )
        assert cfg.max_tokens == 512
        assert cfg.temperature == 0.7
        assert cfg.arrival_pattern == "constant"
        assert cfg.seed == 99

    def test_build_config_from_existing_config(self):
        base = BenchmarkConfig(vllm_url="http://base", model="base-model", max_tokens=512)
        cfg = _build_config(
            vllm_url=None,
            model=None,
            config=base,
            preset=None,
            max_concurrency=None,
            max_tokens=None,
            temperature=None,
            arrival_pattern=None,
            no_metrics=False,
            seed=None,
            backend_name=None,
        )
        assert cfg.vllm_url == "http://base"
        assert cfg.model == "base-model"
        assert cfg.max_tokens == 512
