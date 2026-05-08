"""Tests for agentsurge.cli._display pure formatting functions."""

from __future__ import annotations

from agentsurge.cli._display import (
    _classify_failure,
    _failure_breakdown,
    _fmt_latency_row,
    _percentile,
    print_run_summary,
)
from agentsurge.types.results import RunResult, SessionResult, TurnResult

# ---------------------------------------------------------------------------
# _percentile
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_empty_list(self):
        assert _percentile([], 50) == 0.0

    def test_single_value(self):
        assert _percentile([42.0], 50) == 42.0
        assert _percentile([42.0], 0) == 42.0
        assert _percentile([42.0], 100) == 42.0

    def test_two_values(self):
        assert _percentile([10.0, 20.0], 0) == 10.0
        assert _percentile([10.0, 20.0], 100) == 20.0

    def test_known_p50(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(vals, 50) == 3.0

    def test_p95_selects_near_end(self):
        vals = list(range(1, 101))
        result = _percentile(vals, 95)
        assert result >= 95.0

    def test_unsorted_input(self):
        assert _percentile([5.0, 1.0, 3.0], 50) == 3.0

    def test_p0_returns_min(self):
        assert _percentile([10.0, 20.0, 30.0], 0) == 10.0


# ---------------------------------------------------------------------------
# _fmt_latency_row
# ---------------------------------------------------------------------------


class TestFmtLatencyRow:
    def test_basic(self):
        row = _fmt_latency_row("TTFT", 100.0, 500.0, 900.0)
        assert "p50 100 ms" in row
        assert "p95 500 ms" in row
        assert "p99 900 ms" in row

    def test_zeros(self):
        row = _fmt_latency_row("TPOT", 0.0, 0.0, 0.0)
        assert "p50 0 ms" in row


# ---------------------------------------------------------------------------
# _classify_failure
# ---------------------------------------------------------------------------


class TestClassifyFailure:
    def test_context_overflow(self):
        assert _classify_failure("exceeded context length limit") == "context_overflow"
        assert _classify_failure("exceeds max_model_len") == "context_overflow"
        assert _classify_failure("maximum context window") == "context_overflow"

    def test_timeout(self):
        assert _classify_failure("request timeout after 30s") == "timeout"
        assert _classify_failure("connection timed out") == "timeout"
        assert _classify_failure("deadline exceeded") == "timeout"

    def test_connection(self):
        assert _classify_failure("connection refused") == "connection"
        assert _classify_failure("connection reset by peer") == "connection"

    def test_rate_limit(self):
        assert _classify_failure("rate limit exceeded") == "rate_limit"
        assert _classify_failure("HTTP 429 Too Many Requests") == "rate_limit"
        assert _classify_failure("too many requests") == "rate_limit"

    def test_oom(self):
        assert _classify_failure("OOM: out of memory") == "oom"
        assert _classify_failure("CUDA error: out of memory") == "oom"

    def test_server_error(self):
        assert _classify_failure("EngineCore crashed") == "server_error"
        assert _classify_failure("internal server error") == "server_error"

    def test_unknown(self):
        assert _classify_failure("something completely unexpected") == "error"

    def test_case_insensitive(self):
        assert _classify_failure("TIMEOUT") == "timeout"
        assert _classify_failure("Connection Refused") == "connection"


# ---------------------------------------------------------------------------
# _failure_breakdown
# ---------------------------------------------------------------------------


def _completed_session(sid: str = "ok") -> SessionResult:
    return SessionResult(
        session_id=sid,
        turns=[TurnResult(session_id=sid, turn_index=0, completed=True)],
    )


def _failed_session(sid: str, error: str) -> SessionResult:
    return SessionResult(
        session_id=sid,
        turns=[TurnResult(session_id=sid, turn_index=0, completed=False, error=error)],
    )


class TestFailureBreakdown:
    def test_all_ok(self):
        sessions = [_completed_session("a"), _completed_session("b")]
        assert _failure_breakdown(sessions) == {}

    def test_empty(self):
        assert _failure_breakdown([]) == {}

    def test_single_failure(self):
        sessions = [_failed_session("a", "request timeout")]
        assert _failure_breakdown(sessions) == {"timeout": 1}

    def test_mixed(self):
        sessions = [
            _completed_session("a"),
            _failed_session("b", "timeout reached"),
            _failed_session("c", "timeout hit"),
            _failed_session("d", "connection refused"),
        ]
        counts = _failure_breakdown(sessions)
        assert counts["timeout"] == 2
        assert counts["connection"] == 1
        assert "error" not in counts

    def test_failed_session_no_error_string(self):
        s = SessionResult(
            session_id="x",
            turns=[TurnResult(session_id="x", turn_index=0, completed=False, error="")],
        )
        counts = _failure_breakdown([s])
        assert counts == {"error": 1}


# ---------------------------------------------------------------------------
# print_run_summary (smoke test - verifies output is produced, no crash)
# ---------------------------------------------------------------------------


def _minimal_run_result(
    *,
    n_completed: int = 3,
    n_fail: int = 0,
    elapsed: float = 10.0,
) -> RunResult:
    sessions = []
    for i in range(n_completed):
        sessions.append(
            SessionResult(
                session_id=f"ok-{i}",
                turns=[
                    TurnResult(
                        session_id=f"ok-{i}",
                        turn_index=0,
                        completed=True,
                        ttft_ms=100.0 + i * 10,
                        total_ms=500.0 + i * 50,
                        output_tokens=20,
                        input_tokens=100,
                    )
                ],
            )
        )
    for i in range(n_fail):
        sessions.append(
            SessionResult(
                session_id=f"fail-{i}",
                turns=[
                    TurnResult(
                        session_id=f"fail-{i}",
                        turn_index=0,
                        completed=False,
                        error="timeout exceeded",
                    )
                ],
            )
        )
    return RunResult(
        sessions=sessions,
        total_elapsed_s=elapsed,
        isl_total=5000,
        osl_total=300,
        backend_metrics={
            "kv_util_peak": 0.75,
            "prefix_cache_hit_delta": 80.0,
            "prefix_cache_query_delta": 100.0,
        },
    )


class TestPrintRunSummary:
    def test_normal_run(self, capsys):
        result = _minimal_run_result()
        print_run_summary(result, model="test-model")
        out = capsys.readouterr().out
        assert "test-model" in out or "Sessions" in out

    def test_all_failed(self, capsys):
        result = _minimal_run_result(n_completed=0, n_fail=3)
        print_run_summary(result)
        out = capsys.readouterr().out
        assert "fail" in out.lower() or "error" in out.lower()

    def test_with_saved_files(self, capsys):
        result = _minimal_run_result()
        print_run_summary(result, saved_files=["/tmp/out.json"])
        out = capsys.readouterr().out
        assert "/tmp/out.json" in out

    def test_empty_sessions(self, capsys):
        result = RunResult(sessions=[], total_elapsed_s=0.0)
        print_run_summary(result)
        out = capsys.readouterr().out
        assert "0" in out or out == ""

    def test_high_error_rate(self, capsys):
        result = _minimal_run_result(n_completed=1, n_fail=4)
        print_run_summary(result)
        out = capsys.readouterr().out
        assert len(out) > 0

    def test_no_metrics_mode(self, capsys):
        result = _minimal_run_result()
        result.config["no_metrics"] = True
        print_run_summary(result)
        out = capsys.readouterr().out
        assert len(out) > 0
