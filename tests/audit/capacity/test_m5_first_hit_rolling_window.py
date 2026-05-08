"""Audit M5: first_external_hit must require a sustained signal, not a single
1-token blip.

A single retrieval at the wrong moment otherwise pins the activation_kv
threshold and makes the gate recommendation non-reproducible across
nominally-identical runs. Detect activation only when the cumulative
hit-token delta over a small window exceeds a non-trivial threshold.
"""

from __future__ import annotations

from agentsurge.capacity.wasted_store import WastedStoreTracker


def test_single_token_blip_does_not_register_as_activation():
    t = WastedStoreTracker()
    t.record_snapshot(0.0, 1, 0.10, {"stored_tokens": 0, "hit_tokens": 0})
    # Tiny 1-token bump at low KV: noise, must not pin activation here.
    t.record_snapshot(1.0, 1, 0.10, {"stored_tokens": 50, "hit_tokens": 1})
    t.record_snapshot(2.0, 1, 0.10, {"stored_tokens": 100, "hit_tokens": 1})
    t.record_snapshot(3.0, 1, 0.10, {"stored_tokens": 100, "hit_tokens": 1})
    # Sustained activity later, at a meaningful KV.
    t.record_snapshot(4.0, 5, 0.85, {"stored_tokens": 200, "hit_tokens": 200})
    t.record_snapshot(5.0, 5, 0.90, {"stored_tokens": 200, "hit_tokens": 500})
    r = t.classify()
    # Must reflect the sustained activation point, not the 1-token blip.
    assert r.first_external_hit_kv is not None
    assert r.first_external_hit_kv >= 0.85


def test_sustained_low_kv_activation_still_detected():
    t = WastedStoreTracker()
    t.record_snapshot(0.0, 2, 0.20, {"stored_tokens": 0, "hit_tokens": 0})
    # Sustained, real retrievals at low KV -- legitimate activation.
    t.record_snapshot(1.0, 2, 0.20, {"stored_tokens": 100, "hit_tokens": 200})
    t.record_snapshot(2.0, 2, 0.25, {"stored_tokens": 200, "hit_tokens": 400})
    t.record_snapshot(3.0, 2, 0.25, {"stored_tokens": 200, "hit_tokens": 600})
    r = t.classify()
    assert r.first_external_hit_kv is not None
    assert r.first_external_hit_kv == 0.20
