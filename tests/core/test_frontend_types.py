from __future__ import annotations

import dataclasses
import json

from agentsurge.types import FrontendMetrics, RunResult, ServingTraceMetrics, SessionResult


def test_frontend_metrics_defaults():
    fm = FrontendMetrics(provider="echo")
    assert fm.process_exit_code is None
    assert fm.streaming_text_available is False
    assert fm.event_count == 0
    d = dataclasses.asdict(fm)
    round_tripped = json.loads(json.dumps(d))
    assert round_tripped["provider"] == "echo"
    assert round_tripped["process_exit_code"] is None
    assert round_tripped["streaming_text_available"] is False
    assert round_tripped["event_count"] == 0


def test_serving_trace_metrics_defaults():
    st = ServingTraceMetrics()
    assert st.available is False
    assert st.request_count == 0
    d = dataclasses.asdict(st)
    round_tripped = json.loads(json.dumps(d))
    assert round_tripped["available"] is False
    assert round_tripped["request_count"] == 0


def test_session_result_frontend_metrics():
    s = SessionResult(session_id="x")
    assert s.frontend_metrics is None

    fm = FrontendMetrics(provider="echo", event_count=5)
    s.frontend_metrics = fm
    assert s.frontend_metrics is fm

    d = dataclasses.asdict(s)
    assert d["frontend_metrics"]["provider"] == "echo"
    assert d["frontend_metrics"]["event_count"] == 5


def test_run_result_frontend_fields_default_to_none():
    r = RunResult()
    assert r.frontend_metrics is None
    assert r.serving_trace is None

    r2 = RunResult()
    r2.serving_trace = ServingTraceMetrics(available=True, request_count=3)
    assert r2.serving_trace.available is True


def test_run_result_to_dict_includes_frontend_fields():
    """RunResult.to_dict must surface frontend_metrics and serving_trace when set."""
    st = ServingTraceMetrics(available=True, request_count=7)
    r = RunResult(serving_trace=st)
    d = r.to_dict()
    assert "serving_trace" in d
    assert d["serving_trace"]["available"] is True
    assert d["serving_trace"]["request_count"] == 7
    assert "frontend_metrics" in d
    assert d["frontend_metrics"] is None

    r_base = RunResult()
    r_with_trace = RunResult(serving_trace=ServingTraceMetrics(available=True))
    assert r_base != r_with_trace
