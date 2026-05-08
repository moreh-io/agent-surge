"""Tests for agentsurge.cli._timeline ASCII and matplotlib timeline functions."""

from __future__ import annotations

import pytest

from agentsurge.cli._timeline import save_session_timeline, save_session_timeline_ascii
from agentsurge.types.results import RunResult, SessionResult, TurnResult


def _make_session(
    session_id: str,
    start: float,
    end: float,
    total_ms: float = 0.0,
    llm_ms: float = 0.0,
    completed: bool = True,
    n_turns: int = 1,
) -> SessionResult:
    turns = [
        TurnResult(
            session_id=session_id, turn_index=i, completed=completed, ttft_ms=10.0, total_ms=100.0
        )
        for i in range(n_turns)
    ]
    return SessionResult(
        session_id=session_id,
        turns=turns,
        total_ms=total_ms or (end - start) * 1000,
        llm_ms=llm_ms or (end - start) * 500,
        start_time=start,
        end_time=end,
    )


def _make_result(sessions: list[SessionResult], elapsed: float = 0.0) -> RunResult:
    if not elapsed and sessions:
        elapsed = max((s.end_time for s in sessions), default=0.0)
    return RunResult(sessions=sessions, total_elapsed_s=elapsed)


# ---------------------------------------------------------------------------
# save_session_timeline_ascii
# ---------------------------------------------------------------------------


class TestAsciiTimelineCreatesOutput:
    def test_two_sessions(self):
        sessions = [
            _make_session("sess-aaa", 0.0, 5.0),
            _make_session("sess-bbb", 2.0, 8.0),
        ]
        result = _make_result(sessions, elapsed=10.0)
        text = save_session_timeline_ascii(result)
        assert text, "Expected non-empty output for 2 sessions"
        assert "\n" in text


class TestAsciiTimelineContent:
    def test_contains_session_ids(self):
        sessions = [
            _make_session("alpha1", 0.0, 3.0),
            _make_session("beta22", 1.0, 5.0),
        ]
        text = save_session_timeline_ascii(_make_result(sessions, elapsed=5.0))
        assert "alpha1" in text
        assert "beta22" in text

    def test_contains_header_with_count(self):
        sessions = [_make_session(f"s{i}", float(i), float(i + 2)) for i in range(4)]
        text = save_session_timeline_ascii(_make_result(sessions, elapsed=6.0))
        assert "4 sessions" in text

    def test_contains_time_labels(self):
        sessions = [_make_session("s0", 0.0, 10.0)]
        text = save_session_timeline_ascii(_make_result(sessions, elapsed=10.0))
        assert "0s" in text
        assert "10s" in text

    def test_contains_bar_characters(self):
        sessions = [_make_session("s0", 0.0, 5.0, llm_ms=2500.0, total_ms=5000.0)]
        text = save_session_timeline_ascii(_make_result(sessions, elapsed=5.0))
        assert "\u2588" in text or "\u2591" in text

    def test_long_session_id_truncated(self):
        sessions = [_make_session("very-long-session-id-123", 0.0, 5.0)]
        text = save_session_timeline_ascii(_make_result(sessions, elapsed=5.0))
        assert "very-long-session-id-123" not in text
        assert "id-123" in text


class TestAsciiTimelineEmptySessions:
    def test_no_sessions_returns_empty(self):
        result = _make_result([], elapsed=10.0)
        text = save_session_timeline_ascii(result)
        assert text == ""

    def test_sessions_without_end_time_returns_empty(self):
        s = SessionResult(session_id="x", start_time=0.0, end_time=0.0)
        result = _make_result([s], elapsed=5.0)
        text = save_session_timeline_ascii(result)
        assert text == ""


class TestAsciiTimelineSingleTurn:
    def test_single_turn_session(self):
        sessions = [_make_session("solo", 0.0, 2.0, n_turns=1)]
        text = save_session_timeline_ascii(_make_result(sessions, elapsed=2.0))
        assert "1 sessions" in text
        assert "solo" in text

    def test_single_turn_bar_present(self):
        sessions = [_make_session("s1", 1.0, 3.0, n_turns=1)]
        text = save_session_timeline_ascii(_make_result(sessions, elapsed=5.0))
        lines = text.split("\n")
        bar_lines = [ln for ln in lines if "\u2588" in ln or "\u2591" in ln]
        assert len(bar_lines) == 1


class TestAsciiTimelineOverflow:
    def test_overflow_message(self):
        sessions = [_make_session(f"s{i:03d}", float(i), float(i + 1)) for i in range(35)]
        text = save_session_timeline_ascii(_make_result(sessions, elapsed=36.0))
        assert "35 sessions" in text
        assert "5 more sessions" in text


class TestAsciiTimelineMinuteLabels:
    def test_minute_format(self):
        sessions = [_make_session("s0", 0.0, 120.0)]
        text = save_session_timeline_ascii(_make_result(sessions, elapsed=120.0))
        assert "2.0m" in text


class TestAsciiTimelineFailedSession:
    def test_failed_session_still_rendered(self):
        sessions = [_make_session("fail", 0.0, 3.0, completed=False)]
        text = save_session_timeline_ascii(_make_result(sessions, elapsed=3.0))
        assert "fail" in text


# ---------------------------------------------------------------------------
# save_session_timeline (matplotlib)
# ---------------------------------------------------------------------------


class TestMatplotlibTimeline:
    @pytest.fixture
    def _skip_no_matplotlib(self):
        pytest.importorskip("matplotlib")

    @pytest.mark.usefixtures("_skip_no_matplotlib")
    def test_creates_png(self, tmp_path):
        sessions = [
            _make_session("s1", 0.0, 3.0),
            _make_session("s2", 1.0, 5.0),
        ]
        result = _make_result(sessions, elapsed=5.0)
        out = tmp_path / "timeline.png"
        ret = save_session_timeline(result, str(out))
        assert ret == str(out)
        assert out.exists()
        assert out.stat().st_size > 0

    @pytest.mark.usefixtures("_skip_no_matplotlib")
    def test_no_valid_sessions_returns_empty(self, tmp_path):
        result = _make_result([], elapsed=10.0)
        out = tmp_path / "empty.png"
        ret = save_session_timeline(result, str(out))
        assert ret == ""
        assert not out.exists()
