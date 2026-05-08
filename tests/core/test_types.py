"""Trimmed types tests: canonical types, BenchmarkConfig, RunResult, validation."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from unittest import mock

import pytest

from agentsurge.exceptions import WorkloadValidationError
from agentsurge.io import validate_workload
from agentsurge.types import (
    RunResult,
    SessionResult,
    TurnResult,
)

# ===========================================================================
# Canonical type location (single loop)
# ===========================================================================


class TestSessionResultCompleted:
    """`SessionResult.completed` must not silently mark placeholder / failed
    sessions as successes via the `all([])` quirk."""

    def test_empty_turns_is_not_completed(self):
        s = SessionResult(session_id="crashed")
        assert s.completed is False

    def test_failed_metadata_overrides_completed(self):
        # A session whose turns all completed but was marked failed by the
        # dispatcher (e.g. post-hoc cleanup crash) must not count as completed.
        t = TurnResult(session_id="x", turn_index=0, completed=True)
        s = SessionResult(
            session_id="x",
            turns=[t],
            metadata={"failed": True, "error": "RuntimeError: boom"},
        )
        assert s.completed is False

    def test_all_turns_completed_without_failed_flag(self):
        t1 = TurnResult(session_id="x", turn_index=0, completed=True)
        t2 = TurnResult(session_id="x", turn_index=1, completed=True)
        s = SessionResult(session_id="x", turns=[t1, t2])
        assert s.completed is True

    def test_any_turn_incomplete_means_not_completed(self):
        t1 = TurnResult(session_id="x", turn_index=0, completed=True)
        t2 = TurnResult(session_id="x", turn_index=1, completed=False)
        s = SessionResult(session_id="x", turns=[t1, t2])
        assert s.completed is False


def test_canonical_type_locations():
    """All shared types originate from agentsurge.types and are re-exported identically."""
    from agentsurge import types as types_mod

    checks = [
        ("agentsurge.loaders.base", "Turn"),
        ("agentsurge.loaders.base", "Session"),
        ("agentsurge.loaders.base", "Trajectory"),
        ("agentsurge.runner", "TurnResult"),
        ("agentsurge.runner", "SessionResult"),
        ("agentsurge.runner", "RunResult"),
        ("agentsurge.runner", "BenchmarkConfig"),
        ("agentsurge.metrics", "MetricsSnapshot"),
        ("agentsurge.metrics", "LmcacheV1Metrics"),
    ]
    import importlib

    for mod_path, name in checks:
        mod = importlib.import_module(mod_path)
        canonical = getattr(types_mod, name)
        reexported = getattr(mod, name)
        assert canonical is reexported, f"{mod_path}.{name} is not agentsurge.types.{name}"


# ===========================================================================
# BenchmarkConfig.from_yaml
# ===========================================================================


@pytest.fixture
def tmp_yaml(tmp_path):
    def _write(content: str) -> str:
        p = tmp_path / "config.yaml"
        p.write_text(textwrap.dedent(content))
        return str(p)

    return _write


@pytest.fixture
def default_yaml_path():
    p = Path(__file__).resolve().parent.parent.parent / "agentsurge" / "configs" / "default.yaml"
    if p.exists():
        return str(p)
    pytest.skip("configs/default.yaml not found")


class TestFromYaml:
    def test_bundled_default(self, default_yaml_path):
        from agentsurge.types import BenchmarkConfig

        cfg = BenchmarkConfig.from_yaml()
        assert cfg.vllm_url == "http://localhost:8000"
        assert cfg.arrival_pattern == "poisson"

    def test_explicit_path(self, tmp_yaml):
        from agentsurge.types import BenchmarkConfig

        path = tmp_yaml("""\
            vllm:
              url: "http://my-server:9000"
              model: "my-model"
            runner:
              max_concurrency: 42
        """)
        cfg = BenchmarkConfig.from_yaml(path)
        assert cfg.vllm_url == "http://my-server:9000"
        assert cfg.max_concurrency == 42

    def test_missing_path_raises(self):
        from agentsurge.types import BenchmarkConfig

        with pytest.raises(FileNotFoundError):
            BenchmarkConfig.from_yaml("/no/such/file.yaml")

    def test_env_var_fallback(self, tmp_yaml):
        from agentsurge.types import BenchmarkConfig

        path = tmp_yaml('vllm:\n  url: "http://env:8000"\n  model: "m"\n')
        with mock.patch.dict(os.environ, {"AGENTSURGE_CONFIG": path}):
            cfg = BenchmarkConfig.from_yaml()
        assert cfg.vllm_url == "http://env:8000"

    def test_empty_yaml_uses_defaults(self, tmp_yaml):
        from agentsurge.types import BenchmarkConfig

        path = tmp_yaml("# empty\n")
        cfg = BenchmarkConfig.from_yaml(path)
        assert cfg.vllm_url == "http://localhost:8000"


class TestNoStream:
    def test_no_stream_from_yaml(self, tmp_yaml):
        from agentsurge.types import BenchmarkConfig

        path = tmp_yaml("""\
            vllm:
              url: "http://localhost:8000"
              model: "test"
            runner:
              no_stream: true
        """)
        cfg = BenchmarkConfig.from_yaml(path)
        assert cfg.no_stream is True


class TestFromYamlNewFields:
    def test_new_fields_from_yaml(self, tmp_yaml):
        from agentsurge.types import BenchmarkConfig

        path = tmp_yaml("""\
            vllm:
              url: "http://test:8000"
              model: "test-model"
            runner:
              tool_call_parser_fallback: "deepseek_text"
              save_responses: true
              tool_mode: "real"
        """)
        cfg = BenchmarkConfig.from_yaml(path)
        assert cfg.tool_call_parser_fallback == "deepseek_text"
        assert cfg.save_responses is True
        assert cfg.tool_mode == "real"


class TestFromYamlPresets:
    def test_preset_stress_default(self, default_yaml_path):
        from agentsurge.types import BenchmarkConfig

        cfg = BenchmarkConfig.from_yaml(preset="stress-default")
        assert cfg.preset == "stress-default"
        assert cfg.max_tokens == 256

    def test_preset_simulate_openhands(self, default_yaml_path):
        from agentsurge.types import BenchmarkConfig

        cfg = BenchmarkConfig.from_yaml(preset="simulate-openhands")
        assert cfg.arrival_pattern == "gamma"
        assert cfg.tool_delay is True
        assert cfg.think_time is not None

    def test_preset_alias_measure(self, default_yaml_path):
        from agentsurge.types import BenchmarkConfig

        cfg = BenchmarkConfig.from_yaml(preset="measure")
        assert cfg.use_model_reply_in_next_turn is True
        assert cfg.max_tokens == 32768

    def test_unknown_preset_raises(self, default_yaml_path):
        from agentsurge.types import BenchmarkConfig

        with pytest.raises(ValueError, match="Unknown preset"):
            BenchmarkConfig.from_yaml(preset="nonexistent")

    def test_preset_from_custom_yaml(self, tmp_yaml):
        from agentsurge.types import BenchmarkConfig

        path = tmp_yaml("""\
            vllm:
              url: "http://localhost:8000"
              model: "test-model"
            presets:
              my-preset:
                max_tokens: 512
                arrival_pattern: "poisson"
                think_time: [1.0, 0.5]
        """)
        cfg = BenchmarkConfig.from_yaml(path, preset="my-preset")
        assert cfg.max_tokens == 512
        assert cfg.think_time == (1.0, 0.5)


class TestFromYamlFalsyOverrides:
    """Explicit falsy values in overrides must not be silently dropped."""

    def test_empty_string_vllm_url_override(self, tmp_yaml):
        from agentsurge.types import BenchmarkConfig

        path = tmp_yaml("""\
            vllm:
              url: "http://base:8000"
              model: "base-model"
        """)
        cfg = BenchmarkConfig.from_yaml(path, vllm_url="")
        assert cfg.vllm_url == "", (
            f"empty-string vllm_url override was replaced by base value '{cfg.vllm_url}'"
        )

    def test_empty_string_model_override(self, tmp_yaml):
        from agentsurge.types import BenchmarkConfig

        path = tmp_yaml("""\
            vllm:
              url: "http://base:8000"
              model: "base-model"
        """)
        cfg = BenchmarkConfig.from_yaml(path, model="")
        assert cfg.model == "", (
            f"empty-string model override was replaced by base value '{cfg.model}'"
        )

    def test_empty_string_continue_prompt_override(self, tmp_yaml):
        from agentsurge.types import BenchmarkConfig

        path = tmp_yaml("""\
            vllm:
              url: "http://base:8000"
              model: "base-model"
        """)
        cfg = BenchmarkConfig.from_yaml(path, continue_prompt="")
        assert cfg.continue_prompt == "", (
            "empty-string continue_prompt override was replaced by default value"
        )


class TestModelFallback:
    """Verify three-level model resolution: override > vllm.model > 'default'."""

    def test_override_only_no_yaml_model(self, tmp_yaml):
        from agentsurge.types import BenchmarkConfig

        path = tmp_yaml("""\
            vllm:
              url: "http://localhost:8000"
              model: "m1"
        """)
        cfg = BenchmarkConfig.from_yaml(path, model="m2")
        assert cfg.model == "m2", f"expected 'm2' override beats yaml, got '{cfg.model}'"

    def test_yaml_model_null_falls_back_to_default(self, tmp_yaml):
        """Concern #6: YAML `model: null` yields Python None, which hits the
        'key absent' branch of the 3-branch model-resolution logic and falls
        back to 'default' (same as if the key were omitted entirely)."""
        from agentsurge.types import BenchmarkConfig

        path = tmp_yaml("""\
            vllm:
              url: "http://localhost:8000"
              model: null
        """)
        cfg = BenchmarkConfig.from_yaml(path)
        assert cfg.model == "default", (
            f"expected 'default' when vllm.model is null, got '{cfg.model}'"
        )


class TestFromYamlOverrides:
    def test_override_beats_preset(self, default_yaml_path):
        from agentsurge.types import BenchmarkConfig

        cfg = BenchmarkConfig.from_yaml(preset="stress-default", max_tokens=999)
        assert cfg.max_tokens == 999

    def test_override_all_fields(self, tmp_yaml):
        from agentsurge.types import BenchmarkConfig

        path = tmp_yaml("# minimal\n")
        cfg = BenchmarkConfig.from_yaml(
            path,
            vllm_url="http://test:8000",
            model="test-model",
            max_concurrency=200,
            max_tokens=512,
            temperature=0.9,
            arrival_pattern="poisson",
            think_time=(1.0, 0.5),
            tool_delay=True,
            seed=123,
        )
        assert cfg.vllm_url == "http://test:8000"
        assert cfg.max_tokens == 512
        assert cfg.think_time == (1.0, 0.5)
        assert cfg.seed == 123


class TestFromYamlArrivalRateAuto:
    def test_auto_arrival_rate(self, tmp_yaml):
        from agentsurge.types import BenchmarkConfig

        path = tmp_yaml("""\
            vllm:
              url: "http://localhost:8000"
              model: "test"
            runner:
              arrival_rate: "auto"
              max_concurrency: 100
        """)
        cfg = BenchmarkConfig.from_yaml(path)
        assert cfg.arrival_rate == 30.0


class TestFromYamlLmcache:
    def test_lmcache_enabled(self, tmp_yaml):
        from agentsurge.types import BenchmarkConfig

        path = tmp_yaml("""\
            vllm:
              url: "http://localhost:8000"
              model: "test"
            lmcache:
              url: "http://localhost:9003/metrics"
              enabled: true
        """)
        cfg = BenchmarkConfig.from_yaml(path)
        assert cfg.extra_metrics_urls == ["http://localhost:9003/metrics"]

    def test_lmcache_disabled(self, tmp_yaml):
        from agentsurge.types import BenchmarkConfig

        path = tmp_yaml("""\
            vllm:
              url: "http://localhost:8000"
              model: "test"
            lmcache:
              url: "http://localhost:9003/metrics"
              enabled: false
        """)
        cfg = BenchmarkConfig.from_yaml(path)
        assert cfg.extra_metrics_urls == []


class TestBenchmarkConfigValidation:
    def test_gamma_cv_zero_raises(self):
        from agentsurge.types import BenchmarkConfig

        with pytest.raises(ValueError, match="gamma_cv must be positive"):
            BenchmarkConfig(vllm_url="http://localhost:8000", model="test", gamma_cv=0.0)

    def test_gamma_cv_negative_raises(self):
        from agentsurge.types import BenchmarkConfig

        with pytest.raises(ValueError, match="gamma_cv must be positive"):
            BenchmarkConfig(vllm_url="http://localhost:8000", model="test", gamma_cv=-1.0)

    def test_gamma_cv_positive_accepted(self):
        from agentsurge.types import BenchmarkConfig

        cfg = BenchmarkConfig(vllm_url="http://localhost:8000", model="test", gamma_cv=0.5)
        assert cfg.gamma_cv == 0.5


# ===========================================================================
# RunResult
# ===========================================================================


def test_run_result_to_dict():
    t = TurnResult(
        session_id="s1",
        turn_index=0,
        completed=True,
        ttft_ms=100.0,
        total_ms=200.0,
        output_tokens=50,
    )
    s = SessionResult(session_id="s1", turns=[t], total_ms=200.0, llm_ms=200.0)
    rr = RunResult(sessions=[s], kv_util_peak=0.4, total_elapsed_s=3.0, config={"model": "gpt"})
    d = rr.to_dict()
    assert "config" in d and "kv_util_peak" in d and "sessions" in d and "ttft_values" in d
    assert d["kv_util_peak"] == pytest.approx(0.4)
    turn0 = d["sessions"][0]["turns"][0]
    assert turn0["ttft_ms"] == pytest.approx(100.0)
    assert turn0["total_ms"] == pytest.approx(200.0)
    assert turn0["tokens"] == 50
    assert turn0["completed"] is True


def test_run_result_ttft_values_only_ok():
    t_ok = TurnResult(session_id="s1", turn_index=0, completed=True, ttft_ms=100.0)
    t_err = TurnResult(session_id="s1", turn_index=1, completed=False, ttft_ms=0.0)
    sess = SessionResult(session_id="s1", turns=[t_ok, t_err])
    rr = RunResult(sessions=[sess])
    ttfts = rr.ttft_values()
    assert len(ttfts) == 1 and ttfts[0] == pytest.approx(100.0)


# ===========================================================================
# Validation
# ===========================================================================


def test_validate_workload_valid():
    """validate_workload accepts well-formed workload and rejects malformed entries."""
    data = [
        {"session_id": "s1", "turn_messages": [[{"role": "user", "content": "hi"}]], "metadata": {}}
    ]
    validate_workload(data)  # should not raise

    # Missing session_id should raise
    bad = [{"turn_messages": [[{"role": "user", "content": "hi"}]]}]
    with pytest.raises(WorkloadValidationError):
        validate_workload(bad)


def test_validate_workload_empty_raises():
    with pytest.raises(WorkloadValidationError):
        validate_workload([])


# ===========================================================================
# RunResult.to_dict – save_responses flag
# ===========================================================================


def _make_run_result_with_response_text(text: str = "some model output") -> RunResult:
    t = TurnResult(
        session_id="s1",
        turn_index=0,
        completed=True,
        ttft_ms=50.0,
        total_ms=100.0,
        output_tokens=10,
        response_text=text,
    )
    s = SessionResult(session_id="s1", turns=[t], total_ms=100.0)
    return RunResult(sessions=[s])


def test_to_dict_save_responses_false_excludes_response_text():
    rr = _make_run_result_with_response_text()
    d = rr.to_dict(save_responses=False)
    for session in d["sessions"]:
        for turn in session["turns"]:
            assert "response_text" not in turn


def test_to_dict_save_responses_true_includes_response_text():
    rr = _make_run_result_with_response_text("some model output")
    d = rr.to_dict(save_responses=True)
    turns = d["sessions"][0]["turns"]
    assert len(turns) == 1
    assert turns[0]["response_text"] == "some model output"


def test_to_dict_save_responses_true_truncates_long_text():
    rr = _make_run_result_with_response_text("x" * 5000)
    d = rr.to_dict(save_responses=True)
    turn = d["sessions"][0]["turns"][0]
    assert len(turn["response_text"]) == 2000


# ===========================================================================
# RunResult.to_dict – input_text_delta field
# ===========================================================================


def _make_run_result_with_input_text(
    input_text: str = "user question", response_text: str = ""
) -> RunResult:
    t = TurnResult(
        session_id="s1",
        turn_index=0,
        completed=True,
        ttft_ms=50.0,
        total_ms=100.0,
        output_tokens=10,
        input_text=input_text,
        response_text=response_text,
    )
    s = SessionResult(session_id="s1", turns=[t], total_ms=100.0)
    return RunResult(sessions=[s])


def test_to_dict_save_responses_includes_input_text_delta():
    rr = _make_run_result_with_input_text("user question")
    d = rr.to_dict(save_responses=True)
    turn = d["sessions"][0]["turns"][0]
    assert turn["input_text_delta"] == "user question"


def test_to_dict_save_responses_false_excludes_input_text_delta():
    rr = _make_run_result_with_input_text("user question")
    d = rr.to_dict(save_responses=False)
    turn = d["sessions"][0]["turns"][0]
    assert "input_text_delta" not in turn


def test_to_dict_input_text_delta_truncated():
    rr = _make_run_result_with_input_text("y" * 5000)
    d = rr.to_dict(save_responses=True)
    turn = d["sessions"][0]["turns"][0]
    assert len(turn["input_text_delta"]) == 2000


# ===========================================================================
# to_turns_csv – timing-anomaly turns must not write negative TPOT
# ===========================================================================


def test_to_turns_csv_timing_anomaly_tpot_blank(tmp_path):
    """A turn with total_ms <= ttft_ms must produce a blank tpot_ms cell, not a negative value."""
    import csv

    t_anomaly = TurnResult(
        session_id="s1",
        turn_index=0,
        completed=True,
        ttft_ms=200.0,
        total_ms=100.0,
        output_tokens=50,
    )
    s = SessionResult(session_id="s1", turns=[t_anomaly])
    rr = RunResult(sessions=[s])

    out = tmp_path / "turns.csv"
    rr.to_turns_csv(str(out))

    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    tpot_cell = rows[0]["tpot_ms"]
    assert tpot_cell == "", f"Expected blank tpot_ms for timing-anomaly turn, got {tpot_cell!r}"


# ===========================================================================
# ReplaySession subclass round-trip (H14)
# ===========================================================================


def test_map_reduce_session_round_trip():
    """MapReduceSession fields declared in _METADATA_FIELDS survive to_dict/from_dict."""
    from agentsurge.generators.burst import MapReduceSession

    original = MapReduceSession(
        session_id="sess-001",
        turn_messages=[[{"role": "user", "content": "hello"}]],
        pattern="map_reduce",
        agent_role="mapper",
        burst_group=3,
        phase="scatter",
    )
    d = original.to_dict()
    restored = MapReduceSession.from_dict(d)

    assert isinstance(restored, MapReduceSession)
    assert restored.pattern == "map_reduce"
    assert restored.agent_role == "mapper"
    assert restored.burst_group == 3
    assert restored.phase == "scatter"
    assert restored.session_id == "sess-001"
    assert "pattern" not in restored.metadata
    assert "agent_role" not in restored.metadata
