"""Audit H2: useful_retrieve_fraction mixes numerator and denominator domains.

The numerator (total_retrieved) sums hits across all stores including
GPU-sufficed (low-KV) stores; the denominator (high_pressure_tokens)
excludes them. Without per-event linkage between hits and stores, the
classifier cannot compute a true ratio. The fix is to mark the metric
explicitly as a saturation indicator and surface a flag so callers know
when the clamp at 1.0 is binding (meaning retrievals exceeded the
high-pressure pool, which is the symptom of the domain mix).
"""

from __future__ import annotations

from agentsurge.capacity.wasted_store import WastedStoreTracker


def test_useful_retrieve_saturated_flag_set_when_clamp_binds():
    """When low-KV stores are retrieved later, total_retrieved can exceed
    high_pressure_tokens. The report must flag this rather than silently
    pretending fraction == 1.0 is meaningful."""
    t = WastedStoreTracker(gpu_sufficient_kv_threshold=0.5)
    # snap0: baseline
    t.record_snapshot(
        0.0,
        5,
        0.3,
        {"stored_tokens": 0, "hit_tokens": 0, "cpu_evictions": 0, "disk_evictions": 0},
    )
    # snap1: store 100 tokens at low KV (gpu-sufficed pool)
    t.record_snapshot(
        1.0,
        5,
        0.3,
        {"stored_tokens": 100, "hit_tokens": 0, "cpu_evictions": 0, "disk_evictions": 0},
    )
    # snap2: store 100 more at high KV (high-pressure pool)
    t.record_snapshot(
        2.0,
        5,
        0.7,
        {"stored_tokens": 200, "hit_tokens": 0, "cpu_evictions": 0, "disk_evictions": 0},
    )
    # snap3: retrieve 500 tokens - WAY more than high_pressure_tokens (100)
    t.record_snapshot(
        3.0,
        5,
        0.7,
        {"stored_tokens": 200, "hit_tokens": 500, "cpu_evictions": 0, "disk_evictions": 0},
    )
    r = t.classify()
    # numerator (500) >> denominator (100), so the clamp is binding.
    assert r.high_pressure_tokens == 100
    assert r.total_retrieved_tokens == 500
    assert r.useful_retrieve_fraction == 1.0
    # The fraction is mathematically meaningless here; flag must say so.
    assert r.useful_retrieve_saturated is True


def test_useful_retrieve_not_saturated_when_within_pool():
    t = WastedStoreTracker(gpu_sufficient_kv_threshold=0.5)
    t.record_snapshot(
        0.0,
        5,
        0.85,
        {"stored_tokens": 0, "hit_tokens": 0, "cpu_evictions": 0, "disk_evictions": 0},
    )
    t.record_snapshot(
        1.0,
        5,
        0.85,
        {"stored_tokens": 1000, "hit_tokens": 0, "cpu_evictions": 0, "disk_evictions": 0},
    )
    t.record_snapshot(
        2.0,
        5,
        0.9,
        {"stored_tokens": 1000, "hit_tokens": 400, "cpu_evictions": 0, "disk_evictions": 0},
    )
    r = t.classify()
    assert r.high_pressure_tokens == 1000
    assert r.useful_retrieve_fraction == 0.4
    assert r.useful_retrieve_saturated is False
