"""Audit M2: first_external_hit_kv must distinguish "never observed" from 0.0.

The default-zero dataclass field collapses two distinct cases:
  - never observed: no external hit ever happened
  - observed at exactly kv_usage == 0.0
Use None as the "never observed" sentinel and only use the recommend-100%-gate
fallback when the value is None.
"""

from __future__ import annotations

from agentsurge.capacity.wasted_store import (
    WastedStoreReport,
    WastedStoreTracker,
    compute_admission_value,
)


def test_no_hits_reports_none():
    t = WastedStoreTracker()
    t.record_snapshot(0.0, 5, 0.9, {"stored_tokens": 0, "hit_tokens": 0})
    t.record_snapshot(1.0, 5, 0.9, {"stored_tokens": 100, "hit_tokens": 0})
    r = t.classify()
    assert r.first_external_hit_kv is None
    assert r.first_external_hit_n is None


def test_hit_at_zero_kv_is_distinguishable():
    """A real (sustained) hit at kv=0.0 must register as 0.0, not collapse to None."""
    t = WastedStoreTracker()
    t.record_snapshot(0.0, 3, 0.0, {"stored_tokens": 0, "hit_tokens": 0})
    # >= first_hit_min_tokens to clear the sustained-signal threshold (M5).
    t.record_snapshot(1.0, 3, 0.0, {"stored_tokens": 200, "hit_tokens": 200})
    r = t.classify()
    assert r.first_external_hit_kv == 0.0
    assert r.first_external_hit_n == 3


def test_admission_uses_100pct_gate_only_when_never_observed():
    # Never observed → fallback to 100%.
    res = compute_admission_value(
        WastedStoreReport(
            wasted_fraction=0.7,
            gpu_sufficed_fraction=0.0,
            first_external_hit_kv=None,
            first_external_hit_n=None,
        )
    )
    assert res["activation_kv_pct"] == 100.0

    # Hit at exactly 0.0 → activation_kv_pct must be 0.0, not 100.
    res2 = compute_admission_value(
        WastedStoreReport(
            wasted_fraction=0.7,
            gpu_sufficed_fraction=0.0,
            first_external_hit_kv=0.0,
            first_external_hit_n=2,
        )
    )
    assert res2["activation_kv_pct"] == 0.0
