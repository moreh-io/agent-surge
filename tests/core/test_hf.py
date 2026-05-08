"""Tests for agentsurge.hf - HuggingFace dataset conversion utilities."""

import json
import tempfile

import pytest

from agentsurge.hf import (
    generate_run_id,
    is_hf_url,
    make_dataset_card,
    parse_hf_url,
    run_result_to_parquet,
    workload_to_parquet,
)
from agentsurge.types.results import RunResult, SessionResult, TurnResult
from agentsurge.types.trace import ReplaySession

# ── Fixtures ────────────────────────────────────────────────────


def _make_turn(session_id: str = "s0", turn_index: int = 0, **kwargs) -> TurnResult:
    defaults = dict(
        completed=True,
        ttft_ms=100.0,
        total_ms=500.0,
        output_tokens=64,
        input_tokens=256,
        response_text="Hello world",
        tool_calls=["bash", "read_file"],
        tool_valid=True,
        cached_tokens=128,
    )
    defaults.update(kwargs)
    return TurnResult(session_id=session_id, turn_index=turn_index, **defaults)


def _make_session(session_id: str = "s0", n_turns: int = 2) -> SessionResult:
    turns = [_make_turn(session_id=session_id, turn_index=i) for i in range(n_turns)]
    return SessionResult(
        session_id=session_id,
        turns=turns,
        total_ms=1000.0,
        llm_ms=900.0,
        expected_turns=n_turns,
        metadata={"source": "test"},
    )


def _make_result(n_sessions: int = 2, n_turns: int = 2) -> RunResult:
    sessions = [_make_session(f"s{i}", n_turns) for i in range(n_sessions)]
    return RunResult(
        sessions=sessions,
        total_elapsed_s=5.0,
        config={"model": "test-model", "max_concurrency": 10},
        isl_total=1024.0,
        osl_total=256.0,
        backend_metrics={"kv_util_peak": 0.85},
    )


def _make_replay_session(session_id: str = "rs0") -> ReplaySession:
    return ReplaySession(
        session_id=session_id,
        turn_messages=[
            [
                {"role": "system", "content": "You are a helper."},
                {"role": "user", "content": "Hello"},
            ],
            [
                {"role": "system", "content": "You are a helper."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
        ],
        metadata={"source": "openhands"},
        fan_out=1,
        lineage_id="root_1",
    )


# ── URL parsing ─────────────────────────────────────────────────


class TestHfUrl:
    def test_is_hf_url(self):
        assert is_hf_url("hf://agentsurge/agentsurge-workloads")
        assert not is_hf_url("/local/path/workload.json")
        assert not is_hf_url("https://huggingface.co/datasets/foo")

    def test_parse_hf_url_with_subpath(self):
        repo, sub = parse_hf_url("hf://agentsurge/agentsurge-workloads/qwen3.5-27b/agent-heavy")
        assert repo == "agentsurge/agentsurge-workloads"
        assert sub == "qwen3.5-27b/agent-heavy"

    def test_parse_hf_url_no_subpath(self):
        repo, sub = parse_hf_url("hf://agentsurge/agentsurge-workloads")
        assert repo == "agentsurge/agentsurge-workloads"
        assert sub == ""

    def test_parse_hf_url_invalid(self):
        with pytest.raises(ValueError, match="Invalid hf:// URL"):
            parse_hf_url("hf://only-one-part")


# ── Run ID ──────────────────────────────────────────────────────


class TestRunId:
    def test_deterministic(self):
        r = _make_result()
        id1 = generate_run_id(r)
        id2 = generate_run_id(r)
        assert id1 == id2
        assert len(id1) == 12

    def test_different_config_different_id(self):
        r1 = _make_result()
        r2 = _make_result()
        r2.config["max_concurrency"] = 999
        assert generate_run_id(r1) != generate_run_id(r2)


# ── Result → Parquet round-trip ─────────────────────────────────


@pytest.mark.e2e
class TestResultToParquet:
    def test_round_trip(self):
        import pandas as pd

        result = _make_result(n_sessions=3, n_turns=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = run_result_to_parquet(result, tmpdir, run_id="test123")

            # Check files exist
            assert paths["turns"].exists()
            assert paths["sessions"].exists()
            assert paths["meta"].exists()
            assert paths["readme"].exists()
            assert paths["license"].exists()
            assert paths["attribution"].exists()

            # Read turns back
            turns_df = pd.read_parquet(paths["turns"])
            assert len(turns_df) == 6  # 3 sessions x 2 turns
            assert turns_df["run_id"].iloc[0] == "test123"
            assert turns_df["ttft_ms"].iloc[0] == 100.0
            assert turns_df["output_tokens"].iloc[0] == 64

            # Check response_text is included
            assert turns_df["response_text"].iloc[0] == "Hello world"

            # Check tool_calls is JSON string
            tc = json.loads(turns_df["tool_calls"].iloc[0])
            assert tc == ["bash", "read_file"]

            # Check nullable columns
            assert turns_df["cached_tokens"].iloc[0] == 128

            # Read sessions back
            sessions_df = pd.read_parquet(paths["sessions"])
            assert len(sessions_df) == 3
            assert sessions_df["completed"].iloc[0] == True  # noqa: E712

            # Read meta
            meta = json.loads(paths["meta"].read_text())
            assert meta["run_id"] == "test123"
            assert meta["n_sessions"] == 3
            assert meta["backend_metrics"]["kv_util_peak"] == 0.85

            attribution = json.loads(paths["attribution"].read_text())
            assert attribution["repo_type"] == "results"
            assert attribution["source_counts"] == {"test": 3}
            assert "Local/private export by default" in attribution["sharing_note"]

    def test_nullable_columns_with_none(self):
        import pandas as pd

        t = _make_turn(cached_tokens=None, prompt_tokens_server=None, tool_valid=None)
        s = SessionResult(session_id="s0", turns=[t], total_ms=100.0)
        result = RunResult(sessions=[s], total_elapsed_s=1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = run_result_to_parquet(result, tmpdir)
            df = pd.read_parquet(paths["turns"])
            assert pd.isna(df["cached_tokens"].iloc[0])
            assert pd.isna(df["prompt_tokens_server"].iloc[0])


# ── Workload → Parquet round-trip ───────────────────────────────


@pytest.mark.e2e
class TestWorkloadToParquet:
    def test_round_trip(self):
        import pandas as pd

        sessions = [_make_replay_session("rs0"), _make_replay_session("rs1")]

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = workload_to_parquet(sessions, tmpdir)
            assert paths["workload"].exists()
            assert paths["readme"].exists()
            assert paths["license"].exists()
            assert paths["attribution"].exists()

            df = pd.read_parquet(paths["workload"])
            assert len(df) == 2
            assert df["session_id"].iloc[0] == "rs0"
            assert df["n_turns"].iloc[0] == 2

            # Verify turn_messages round-trip
            restored = json.loads(df["turn_messages"].iloc[0])
            assert len(restored) == 2
            assert restored[0][0]["role"] == "system"
            assert restored[1][-1]["content"] == "How are you?"

            # Verify metadata
            meta = json.loads(df["metadata"].iloc[0])
            assert meta["source"] == "openhands"

            # Verify lineage_id
            assert df["lineage_id"].iloc[0] == "root_1"

            readme = paths["readme"].read_text()
            license_notes = paths["license"].read_text()
            attribution = json.loads(paths["attribution"].read_text())
            assert "Local/private AgentSurge workloads export" in readme
            assert "nebius/SWE-rebench-openhands-trajectories" in readme
            assert "does not grant" in license_notes
            assert attribution["source_counts"] == {"openhands": 2}
            assert attribution["upstream_sources"][0]["license"] == "CC-BY-4.0"


# ── DatasetCard ─────────────────────────────────────────────────


class TestDatasetCard:
    def test_basic_card(self):
        card = make_dataset_card(
            "results",
            "Qwen/Qwen3.5-27B",
            "qwen3.5-27b/agent-heavy",
            n_sessions=50,
        )
        assert "license: other" in card
        assert "Qwen/Qwen3.5-27B" in card
        assert "agentsurge" in card
        assert "https://github.com/moreh-io/agent-surge" in card
        assert "## Dataset Description" in card
        assert "private/local by default" in card

    def test_card_with_attribution(self):
        card = make_dataset_card(
            "workloads",
            "Qwen/Qwen3.5-27B",
            "qwen3.5-27b/agent-heavy",
            upstream_sources=["openhands", "swe-bench-verified"],
            n_sessions=100,
        )
        assert "CC-BY-4.0" in card
        assert "MIT" in card
        assert "nebius/SWE-rebench-openhands-trajectories" in card
        assert "| Source |" in card
        assert "| License |" in card or "License" in card

    def test_card_odc_by_license(self):
        card = make_dataset_card(
            "results",
            "test-model",
            "test",
            license_id="odc-by",
            upstream_sources=["abc-bench"],
        )
        assert "license: odc-by" in card
        assert "ODC-BY" in card
        assert "creativecommons.org/licenses/by/4.0" not in card
        assert card.startswith("---"), "card should start with YAML front matter"

    def test_card_marks_unknown_attribution_without_license_claim(self):
        card = make_dataset_card(
            "workloads",
            "test-model",
            "test",
            upstream_sources=["unknown-source"],
        )
        assert "Unknown" in card
        assert "redistributable" in card
        assert "This dataset is licensed under" not in card
