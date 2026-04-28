from __future__ import annotations

import dataclasses
import json

from agentsurge.types import FrontendMetrics, ServingTraceMetrics, SessionResult


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
