"""Audit M3: counter resets must not silently disappear.

When the backend restarts mid-run, stored_tokens drops back to zero. The
old `> 0` guard silently dropped the negative interval and `classify()`
took `last - first` from the very first snapshot, so all stores recorded
before the reset were lost. The fix: detect the regression, accumulate
totals across resets, and surface a `counter_reset_observed` flag.
"""

from __future__ import annotations

from agentsurge.capacity.wasted_store import WastedStoreTracker


def test_reset_observed_flag_set_on_regression():
    t = WastedStoreTracker()
    t.record_snapshot(0.0, 1, 0.9, {"stored_tokens": 0, "hit_tokens": 0})
    t.record_snapshot(1.0, 1, 0.9, {"stored_tokens": 1000, "hit_tokens": 0})
    # Backend restart: counters reset.
    t.record_snapshot(2.0, 1, 0.9, {"stored_tokens": 0, "hit_tokens": 0})
    t.record_snapshot(3.0, 1, 0.9, {"stored_tokens": 500, "hit_tokens": 0})
    r = t.classify()
    assert r.counter_reset_observed is True


def test_reset_observed_false_on_monotonic_run():
    t = WastedStoreTracker()
    t.record_snapshot(0.0, 1, 0.9, {"stored_tokens": 0, "hit_tokens": 0})
    t.record_snapshot(1.0, 1, 0.9, {"stored_tokens": 100, "hit_tokens": 0})
    t.record_snapshot(2.0, 1, 0.9, {"stored_tokens": 200, "hit_tokens": 50})
    r = t.classify()
    assert r.counter_reset_observed is False


def test_reset_accumulates_totals_across_checkpoint():
    """After a reset, the cumulative total must include both segments."""
    t = WastedStoreTracker()
    t.record_snapshot(0.0, 1, 0.9, {"stored_tokens": 0, "hit_tokens": 0})
    t.record_snapshot(1.0, 1, 0.9, {"stored_tokens": 1000, "hit_tokens": 0})
    # Reset.
    t.record_snapshot(2.0, 1, 0.9, {"stored_tokens": 0, "hit_tokens": 0})
    t.record_snapshot(3.0, 1, 0.9, {"stored_tokens": 500, "hit_tokens": 0})
    r = t.classify()
    # Old behaviour: max(0, 500-0) = 500. With checkpoint accumulation: 1000+500.
    assert r.total_stored_tokens == 1500
