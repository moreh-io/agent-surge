"""Tests for the Frontend Statistics block in agentsurge.cli._display."""

from __future__ import annotations

from agentsurge.cli._display import (
    _build_frontend_lines,
    _frontend_failure_summary,
    _frontend_provider_summary,
    print_run_summary,
)
from agentsurge.types.results import (
    FrontendMetrics,
    RunResult,
    SessionResult,
    TurnResult,
)


def _ok_turn(sid: str) -> TurnResult:
    return TurnResult(
        session_id=sid,
        turn_index=0,
        completed=True,
        ttft_ms=100.0,
        total_ms=500.0,
        output_tokens=20,
        input_tokens=100,
    )


def _session(sid: str, fm: FrontendMetrics | None) -> SessionResult:
    return SessionResult(
        session_id=sid,
        turns=[_ok_turn(sid)],
        frontend_metrics=fm,
    )


def _run_result(sessions: list[SessionResult]) -> RunResult:
    return RunResult(
        sessions=sessions,
        total_elapsed_s=10.0,
        isl_total=5000,
        osl_total=300,
        backend_metrics={
            "kv_util_peak": 0.5,
            "prefix_cache_hit_delta": 50.0,
            "prefix_cache_query_delta": 100.0,
        },
    )


def test_frontend_section_omitted_when_no_frontend_metrics(capsys):
    sessions = [_session(f"s-{i}", None) for i in range(3)]
    result = _run_result(sessions)
    print_run_summary(result, model="m")
    out = capsys.readouterr().out
    assert "Frontend statistics" not in out


def test_frontend_section_renders_basic_metrics(capsys):
    metrics = [
        FrontendMetrics(
            provider="echo",
            process_exit_code=0,
            process_wall_ms=wall,
            process_startup_to_first_event_ms=first,
            streaming_text_available=True,
            time_to_first_assistant_text_ms=fa,
            time_to_final_message_ms=final,
        )
        for wall, first, fa, final in [
            (100.0, 5.0, 10.0, 50.0),
            (200.0, 10.0, 20.0, 60.0),
            (300.0, 15.0, 30.0, 70.0),
        ]
    ]
    sessions = [_session(f"s-{i}", m) for i, m in enumerate(metrics)]
    result = _run_result(sessions)
    print_run_summary(result, model="m")
    out = capsys.readouterr().out

    assert "Frontend statistics" in out
    assert "Provider" in out
    assert "echo" in out
    assert "CLI processes launched" in out
    assert "CLI processes exited 0" in out
    assert "Time to first CLI event" in out
    assert "Time to first assistant text" in out
    assert "Time to final message" in out
    assert "Process wall time" in out
    # Percentile prefixes appear at least once per latency row.
    assert "p50" in out
    assert "p95" in out
    assert "p99" in out
    # Launched and exited-0 counts (3 each).
    assert "3" in out


def test_frontend_section_omits_assistant_text_when_no_streaming(capsys):
    metrics = [
        FrontendMetrics(
            provider="echo",
            process_exit_code=0,
            process_wall_ms=100.0 * (i + 1),
            process_startup_to_first_event_ms=5.0 * (i + 1),
            streaming_text_available=False,
            time_to_first_assistant_text_ms=None,
            time_to_final_message_ms=50.0 * (i + 1),
        )
        for i in range(3)
    ]
    sessions = [_session(f"s-{i}", m) for i, m in enumerate(metrics)]
    result = _run_result(sessions)
    print_run_summary(result, model="m")
    out = capsys.readouterr().out

    assert "Frontend statistics" in out
    assert "Time to first assistant text" not in out
    assert "Time to first CLI event" in out
    assert "Time to final message" in out
    assert "Process wall time" in out


def test_frontend_section_lists_top_failure_categories(capsys):
    metrics = [
        FrontendMetrics(
            provider="echo",
            process_exit_code=1,
            process_wall_ms=100.0,
            failure_category="timeout",
        ),
        FrontendMetrics(
            provider="echo",
            process_exit_code=1,
            process_wall_ms=100.0,
            failure_category="timeout",
        ),
        FrontendMetrics(
            provider="echo",
            process_exit_code=2,
            process_wall_ms=100.0,
            failure_category="spawn_error",
        ),
        FrontendMetrics(
            provider="echo",
            process_exit_code=0,
            process_wall_ms=100.0,
            failure_category=None,
        ),
    ]
    sessions = [_session(f"s-{i}", m) for i, m in enumerate(metrics)]
    result = _run_result(sessions)
    print_run_summary(result, model="m")
    out = capsys.readouterr().out

    assert "Frontend statistics" in out
    # Total failure count plus ordered category breakdown.
    assert "3 (timeout: 2, spawn_error: 1)" in out
    # Order check: 'timeout' category appears before 'spawn_error'.
    failures_line = next(line for line in out.splitlines() if "timeout: 2" in line)
    assert failures_line.index("timeout: 2") < failures_line.index("spawn_error: 1")


def test_frontend_section_handles_mixed_providers(capsys):
    providers = ["echo", "echo", "echo", "codex", "codex"]
    metrics = [
        FrontendMetrics(
            provider=p,
            process_exit_code=0,
            process_wall_ms=100.0,
        )
        for p in providers
    ]
    sessions = [_session(f"s-{i}", m) for i, m in enumerate(metrics)]
    result = _run_result(sessions)
    print_run_summary(result, model="m")
    out = capsys.readouterr().out

    assert "Frontend statistics" in out
    assert "echo (3), codex (2)" in out


# ---------------------------------------------------------------------------
# Direct unit checks on helpers.
# ---------------------------------------------------------------------------


class TestProviderSummary:
    def test_single_provider_returns_bare_name(self):
        m = [FrontendMetrics(provider="echo") for _ in range(3)]
        assert _frontend_provider_summary(m) == "echo"

    def test_sorts_descending_by_count(self):
        m = [FrontendMetrics(provider="codex")] * 2 + [FrontendMetrics(provider="echo")] * 3
        assert _frontend_provider_summary(m) == "echo (3), codex (2)"


class TestFailureSummary:
    def test_no_failures(self):
        m = [FrontendMetrics(provider="echo", failure_category=None) for _ in range(3)]
        assert _frontend_failure_summary(m) == "0"

    def test_top_three_only(self):
        m = (
            [FrontendMetrics(provider="echo", failure_category="timeout")] * 3
            + [FrontendMetrics(provider="echo", failure_category="spawn_error")] * 2
            + [FrontendMetrics(provider="echo", failure_category="nonzero_exit")] * 1
            + [FrontendMetrics(provider="echo", failure_category="other")] * 1
        )
        result = _frontend_failure_summary(m)
        assert result.startswith("7 (")
        assert "timeout: 3" in result
        assert "spawn_error: 2" in result
        # Only 3 categories listed; lower-count ones drop off.
        assert result.count(",") == 2


def test_build_frontend_lines_empty_when_all_none():
    sessions = [_session(f"s-{i}", None) for i in range(2)]
    result = _run_result(sessions)
    assert _build_frontend_lines(result) == []
