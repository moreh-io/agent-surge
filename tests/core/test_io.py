"""Tests for agentsurge.io workload I/O and agentsurge.types.results output formats."""

import csv
import json

import pytest

from agentsurge.exceptions import WorkloadValidationError
from agentsurge.io import (
    _parse_workload_json,
    load_workload_sessions,
    save_corpus,
    validate_workload,
)
from agentsurge.types import WORKLOAD_SCHEMA_VERSION, ReplaySession
from agentsurge.types.results import RunResult, SessionResult, TurnResult


def _session_dict(sid="s1", content="hello"):
    return {
        "session_id": sid,
        "turn_messages": [[{"role": "user", "content": content}]],
        "metadata": {"source": "test"},
    }


def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# load_workload_sessions
# ---------------------------------------------------------------------------


class TestLoadWorkloadSessions:
    def test_legacy_format(self, tmp_path):
        p = tmp_path / "legacy.json"
        raw = [_session_dict("a", "q1"), _session_dict("b", "q2")]
        _write_json(p, raw)

        sessions = load_workload_sessions(p)

        assert len(sessions) == 2
        assert sessions[0].session_id == "a"
        assert sessions[1].session_id == "b"
        assert sessions[0].turn_messages == [[{"role": "user", "content": "q1"}]]
        assert sessions[0].metadata == {"source": "test"}
        assert sessions[0].fan_out == 1

    def test_versioned_format(self, tmp_path):
        p = tmp_path / "versioned.json"
        envelope = {
            "version": WORKLOAD_SCHEMA_VERSION,
            "sessions": [_session_dict("v1", "hi")],
        }
        _write_json(p, envelope)

        sessions = load_workload_sessions(p)

        assert len(sessions) == 1
        assert sessions[0].session_id == "v1"
        assert sessions[0].turn_messages[0][0]["content"] == "hi"

    def test_corpus_format(self, tmp_path):
        p = tmp_path / "corpus.json"
        envelope = {
            "version": WORKLOAD_SCHEMA_VERSION,
            "manifest": {"n_sessions": 1, "source_counts": {"test": 1}},
            "sessions": [_session_dict("c1", "corpus q")],
        }
        _write_json(p, envelope)

        sessions = load_workload_sessions(p)

        assert len(sessions) == 1
        assert sessions[0].session_id == "c1"
        assert sessions[0].turn_messages[0][0]["content"] == "corpus q"

    def test_preserves_pending_user_messages(self, tmp_path):
        p = tmp_path / "pending.json"
        raw = [_session_dict("p1", "q1") | {"pending_user_messages": ["u2", "u3"]}]
        _write_json(p, raw)

        sessions = load_workload_sessions(p)

        assert sessions[0].pending_user_messages == ["u2", "u3"]


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


class TestSaveAndLoadRoundTrip:
    def test_roundtrip_preserves_sessions(self, tmp_path):
        originals = [
            ReplaySession(
                session_id="rt1",
                turn_messages=[
                    [{"role": "system", "content": "sys"}, {"role": "user", "content": "u1"}],
                    [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "u1"},
                        {"role": "assistant", "content": "a1"},
                        {"role": "user", "content": "u2"},
                    ],
                ],
                metadata={"source": "roundtrip", "k": 42},
                fan_out=2,
                lineage_id="lin-1",
                parent_session_id="parent-1",
                branch_id=3,
                branch_depth=1,
            ),
        ]
        p = tmp_path / "rt.json"
        save_corpus(originals, p)
        loaded = load_workload_sessions(p)

        assert len(loaded) == 1
        s = loaded[0]
        assert s.session_id == "rt1"
        assert s.turn_messages == originals[0].turn_messages
        assert s.metadata == {"source": "roundtrip", "k": 42}
        assert s.fan_out == 2
        assert s.lineage_id == "lin-1"
        assert s.parent_session_id == "parent-1"
        assert s.branch_id == 3
        assert s.branch_depth == 1


# ---------------------------------------------------------------------------
# validate_workload
# ---------------------------------------------------------------------------


def _minimal_session(**overrides) -> dict:
    s = {"session_id": "s1", "turn_messages": [[{"role": "user", "content": "hi"}]]}
    s.update(overrides)
    return s


class TestValidateWorkload:
    def test_rejects_empty_list(self):
        with pytest.raises(WorkloadValidationError, match="no sessions"):
            validate_workload([])

    def test_rejects_missing_session_id(self):
        data = [{"turn_messages": [[{"role": "user", "content": "x"}]]}]
        with pytest.raises(WorkloadValidationError, match="session_id"):
            validate_workload(data)

    def test_rejects_missing_turn_messages(self):
        data = [{"session_id": "s1"}]
        with pytest.raises(WorkloadValidationError, match="turn_messages"):
            validate_workload(data)

    def test_rejects_none(self):
        with pytest.raises(WorkloadValidationError, match="is None"):
            validate_workload(None)

    def test_rejects_scalar(self):
        with pytest.raises(WorkloadValidationError, match="must be a dict or list"):
            validate_workload(42)

    def test_rejects_envelope_missing_sessions_key(self):
        with pytest.raises(WorkloadValidationError, match="missing 'sessions'"):
            validate_workload({"version": "1.0"})

    def test_rejects_envelope_non_list_sessions(self):
        with pytest.raises(WorkloadValidationError, match="'sessions' must be a list"):
            validate_workload({"version": "1.0", "sessions": "oops"})

    def test_rejects_envelope_non_dict_manifest(self):
        with pytest.raises(WorkloadValidationError, match="'manifest' must be a dict"):
            validate_workload(
                {"version": "1.0", "manifest": "oops", "sessions": [_minimal_session()]}
            )

    def test_rejects_unsupported_major_version(self):
        with pytest.raises(WorkloadValidationError, match="Unsupported workload schema"):
            validate_workload({"version": "999.0", "sessions": [_minimal_session()]})

    def test_rejects_malformed_version_string(self):
        # _major raises WorkloadValidationError for non-semver input.
        with pytest.raises(WorkloadValidationError, match="Invalid version string"):
            validate_workload({"version": "not-a-version", "sessions": [_minimal_session()]})

    def test_rejects_non_dict_session(self):
        with pytest.raises(WorkloadValidationError, match="must be a dict"):
            validate_workload(["not a dict"])

    def test_rejects_empty_session_id(self):
        with pytest.raises(WorkloadValidationError, match="invalid 'session_id'"):
            validate_workload([_minimal_session(session_id="")])

    def test_rejects_empty_turn_messages_list(self):
        with pytest.raises(WorkloadValidationError, match="non-empty list"):
            validate_workload([_minimal_session(turn_messages=[])])

    def test_rejects_turn_messages_not_list_of_lists(self):
        with pytest.raises(WorkloadValidationError, match="non-empty list of messages"):
            validate_workload([_minimal_session(turn_messages=[{"role": "user"}])])

    def test_rejects_message_missing_role(self):
        bad = _minimal_session(turn_messages=[[{"content": "hi"}]])
        with pytest.raises(WorkloadValidationError, match="missing required field 'role'"):
            validate_workload([bad])

    def test_rejects_message_missing_content(self):
        bad = _minimal_session(turn_messages=[[{"role": "user"}]])
        with pytest.raises(WorkloadValidationError, match="missing required field 'content'"):
            validate_workload([bad])

    def test_rejects_non_string_role(self):
        bad = _minimal_session(turn_messages=[[{"role": 123, "content": "hi"}]])
        with pytest.raises(WorkloadValidationError, match="non-string 'role'"):
            validate_workload([bad])

    def test_rejects_invalid_fan_out(self):
        with pytest.raises(WorkloadValidationError, match="'fan_out' must be a positive integer"):
            validate_workload([_minimal_session(fan_out=0)])

    def test_rejects_non_dict_metadata(self):
        with pytest.raises(WorkloadValidationError, match="'metadata' must be a dict"):
            validate_workload([_minimal_session(metadata="nope")])

    def test_rejects_pending_user_messages_string(self):
        with pytest.raises(WorkloadValidationError, match="'pending_user_messages' must be"):
            validate_workload([_minimal_session(pending_user_messages="u2")])

    def test_rejects_pending_user_messages_non_string_item(self):
        with pytest.raises(WorkloadValidationError, match="pending_user_messages\\[0\\]"):
            validate_workload([_minimal_session(pending_user_messages=[{"role": "user"}])])


# ---------------------------------------------------------------------------
# _parse_workload_json
# ---------------------------------------------------------------------------


class TestParseWorkloadJson:
    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Unrecognised workload JSON"):
            _parse_workload_json({"no_sessions_key": True})

    def test_string_raises(self):
        with pytest.raises((ValueError, TypeError)):
            _parse_workload_json("not a list or dict")


# ---------------------------------------------------------------------------
# RunResult.to_dict
# ---------------------------------------------------------------------------


class TestRunResultToDict:
    def _make_run_result(self):
        t1 = TurnResult(
            session_id="s1",
            turn_index=0,
            completed=True,
            ttft_ms=150.0,
            wall_ttft_ms=155.0,
            total_ms=300.0,
            output_tokens=40,
            input_tokens=100,
            input_messages=3,
            cached_tokens=80,
        )
        t2 = TurnResult(
            session_id="s1",
            turn_index=1,
            completed=True,
            ttft_ms=200.0,
            wall_ttft_ms=210.0,
            total_ms=400.0,
            output_tokens=60,
            input_tokens=200,
            input_messages=5,
        )
        s = SessionResult(
            session_id="s1",
            turns=[t1, t2],
            total_ms=700.0,
            llm_ms=700.0,
            expected_turns=2,
            metadata={"source": "test"},
        )
        return RunResult(
            sessions=[s],
            total_elapsed_s=5.0,
            config={"model": "test-model"},
            isl_total=300.0,
            osl_total=100.0,
            kv_util_peak=0.75,
            prefix_cache_hit_delta=8.0,
            prefix_cache_query_delta=10.0,
        )

    def test_roundtrip_all_keys(self):
        rr = self._make_run_result()
        d = rr.to_dict()

        assert d["config"] == {"model": "test-model"}
        assert d["isl_total"] == 300.0
        assert d["osl_total"] == 100.0
        assert d["total_elapsed_s"] == 5.0
        assert d["isl_osl_ratio"] == 3.0
        assert d["requests_per_s"] == 0.4
        assert d["prefix_cache_hit_rate"] == pytest.approx(0.8)
        assert d["n_sessions"] == 1
        assert d["n_completed_sessions"] == 1
        assert d["n_total_turns"] == 2
        assert d["kv_util_peak"] == pytest.approx(0.75)
        assert d["ttft_values"] == [150.0, 200.0]

        sess = d["sessions"][0]
        assert sess["session_id"] == "s1"
        assert sess["completed"] is True
        assert sess["n_turns"] == 2
        assert sess["expected_turns"] == 2
        assert sess["total_ms"] == 700.0
        assert sess["llm_ms"] == 700.0
        assert sess["metadata"] == {"source": "test"}

        turn0 = sess["turns"][0]
        assert turn0["turn"] == 0
        assert turn0["completed"] is True
        assert turn0["ttft_ms"] == 150.0
        assert turn0["wall_ttft_ms"] == 155.0
        assert turn0["total_ms"] == 300.0
        assert turn0["input_tokens"] == 100
        assert turn0["tokens"] == 40
        assert turn0["cached_tokens"] == 80
        assert turn0["error"] == ""
        assert turn0["retry_count"] == 0
        assert turn0["fan_out_count"] == 1

        turn1 = sess["turns"][1]
        assert "cached_tokens" not in turn1


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


class TestRunResultCsvOutput:
    def _make_run_result(self):
        t1 = TurnResult(
            session_id="s1",
            turn_index=0,
            completed=True,
            ttft_ms=100.5,
            wall_ttft_ms=105.5,
            total_ms=250.0,
            output_tokens=30,
            input_tokens=80,
            cached_tokens=50,
        )
        t2 = TurnResult(
            session_id="s1",
            turn_index=1,
            completed=False,
            ttft_ms=0.0,
            wall_ttft_ms=0.0,
            total_ms=50.0,
            output_tokens=0,
            input_tokens=120,
            error="timeout",
            retry_count=2,
            fan_out_count=3,
        )
        s = SessionResult(
            session_id="s1",
            turns=[t1, t2],
            total_ms=300.0,
            llm_ms=300.0,
            expected_turns=2,
        )
        return RunResult(
            sessions=[s],
            total_elapsed_s=2.0,
            isl_total=200.0,
            osl_total=30.0,
            kv_util_peak=0.55,
        )

    def test_turns_csv_columns_and_values(self, tmp_path):
        rr = self._make_run_result()
        csv_path = str(tmp_path / "turns.csv")
        rr.to_turns_csv(csv_path)

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2

        r0 = rows[0]
        assert r0["session_id"] == "s1"
        assert r0["turn"] == "0"
        assert r0["completed"] == "True"
        assert float(r0["ttft_ms"]) == pytest.approx(100.5)
        assert float(r0["wall_ttft_ms"]) == pytest.approx(105.5)
        assert float(r0["total_ms"]) == pytest.approx(250.0)
        assert r0["input_tokens"] == "80"
        assert r0["output_tokens"] == "30"
        assert r0["cached_tokens"] == "50"
        assert r0["error"] == ""

        r1 = rows[1]
        assert r1["completed"] == "False"
        assert r1["error"] == "timeout"
        assert r1["retry_count"] == "2"
        assert r1["fan_out_count"] == "3"
        assert r1["cached_tokens"] == ""

    def test_summary_csv_columns_and_values(self, tmp_path):
        rr = self._make_run_result()
        csv_path = str(tmp_path / "summary.csv")
        rr.to_summary_csv(csv_path)

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]

        expected_columns = set(RunResult._SUMMARY_CSV_COLUMNS)
        assert set(row.keys()) == expected_columns

        assert row["n_sessions"] == "1"
        assert row["n_completed"] == "0"
        assert row["n_turns"] == "2"
        assert float(row["total_elapsed_s"]) == pytest.approx(2.0)
        assert float(row["kv_util_peak"]) == pytest.approx(0.55)
        assert float(row["isl_total"]) == pytest.approx(200.0)
        assert float(row["osl_total"]) == pytest.approx(30.0)
        assert float(row["total_tok_per_s"]) == pytest.approx(115.0)
        assert float(row["isl_per_s"]) == pytest.approx(100.0)
        assert float(row["osl_per_s"]) == pytest.approx(15.0)
