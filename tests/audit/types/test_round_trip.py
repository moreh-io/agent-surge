# SPDX-License-Identifier: MIT
"""M2: MetricsSnapshot needs to_dict/from_dict; RunResult needs from_dict.

Without round-trip helpers, JSON consumers must reverse-engineer the schema
and adding backend metric keys silently drifts.
"""

from agentsurge.types.metrics import LmcacheV1Metrics, MetricsSnapshot
from agentsurge.types.results import RunResult, SessionResult, TurnResult


def test_metrics_snapshot_to_dict_round_trips():
    s = MetricsSnapshot(
        timestamp=1.5,
        kv_cache_usage_perc=0.42,
        num_requests_running=3,
        num_requests_waiting=1,
        isl_total=100.0,
        osl_total=200.0,
        custom_metrics={"foo": 1.0},
    )
    s.prefix_cache_hits_total = 5.0
    d = s.to_dict()
    assert isinstance(d, dict)
    s2 = MetricsSnapshot.from_dict(d)
    assert s2 == s


def test_metrics_snapshot_round_trip_preserves_lmcache_v1():
    s = MetricsSnapshot()
    s.lmcache_v1 = LmcacheV1Metrics(retrieve_requests=42)
    d = s.to_dict()
    s2 = MetricsSnapshot.from_dict(d)
    assert s2.backend_metrics["lmcache_v1"].retrieve_requests == 42


def test_run_result_from_dict_round_trips():
    r = RunResult(
        sessions=[
            SessionResult(
                session_id="s1",
                turns=[TurnResult(session_id="s1", turn_index=0, completed=True)],
                start_time=0.0,
                end_time=1.0,
            )
        ],
        total_elapsed_s=2.0,
        config={"model": "x"},
        isl_total=10,
        osl_total=20,
    )
    r.kv_util_peak = 0.5
    d = r.to_dict()
    r2 = RunResult.from_dict(d)
    assert r2.total_elapsed_s == r.total_elapsed_s
    assert r2.isl_total == r.isl_total
    assert r2.osl_total == r.osl_total
    assert r2.kv_util_peak == 0.5
    assert len(r2.sessions) == 1
    assert r2.sessions[0].session_id == "s1"
    assert r2.sessions[0].turns[0].turn_index == 0
