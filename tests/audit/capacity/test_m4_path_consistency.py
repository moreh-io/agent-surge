"""Audit M4: WastedStoreTracker.classify() and compute_external_reuse must agree
on the canonical "external hit tokens" quantity when fed the same data.

Both paths claim to compute the same conceptual delta. The two should
round-trip equal totals when the same hit-token / stored-token series is
fed in via both surfaces.
"""

from __future__ import annotations

from agentsurge.capacity.wasted_store import (
    WastedStoreTracker,
    compute_external_reuse,
)


class _Snap:
    def __init__(self, isl_total: float) -> None:
        self.isl_total = isl_total


class _SyncCollector:
    """Collector stub that mirrors what the tracker sees.

    Uses ``hit_series``/``stored_series`` to compute v1 deltas from first to
    last entry, mimicking ``MetricsCollector.lmcache_v1_deltas`` semantics
    (clamped non-negative deltas).
    """

    memory_pressure = 0.0
    sglang_cache_hit_rate = -1.0

    def __init__(self, hit_series: list[int], stored_series: list[int]) -> None:
        self._hit_series = hit_series
        self._stored_series = stored_series
        self.snapshots = [_Snap(0.0), _Snap(1000.0)]

    def prefix_cache_delta(self):
        return (0.0, 0.0)

    def lmcache_v1_deltas(self):
        # Mirror MetricsCollector behaviour: max(0, last - first).
        hit_delta = max(0, self._hit_series[-1] - self._hit_series[0])
        stored_delta = max(0, self._stored_series[-1] - self._stored_series[0])
        return {"hit_tokens": hit_delta, "stored_tokens": stored_delta}

    def external_prefix_cache_delta(self):
        return (max(0, self._hit_series[-1] - self._hit_series[0]), 0.0)


def test_classify_and_compute_external_reuse_agree_on_monotonic_run():
    hit = [0, 100, 250, 400]
    stored = [0, 500, 800, 900]
    t = WastedStoreTracker()
    for i, (h, s) in enumerate(zip(hit, stored)):
        t.record_snapshot(float(i), 1, 0.85, {"stored_tokens": s, "hit_tokens": h})
    report = t.classify()

    collector = _SyncCollector(hit, stored)
    ext = compute_external_reuse(collector)

    assert report.total_retrieved_tokens == ext["external_hit_tokens"]
    assert report.total_stored_tokens == 900


def test_classify_and_compute_external_reuse_agree_after_reset():
    """A backend reset mid-run is the canonical divergence case (M3 + M4).
    classify() now accumulates across resets; compute_external_reuse only
    sees first/last v1 deltas. After M4's harmonisation, classify must at
    least be a *superset* of compute_external_reuse's count -- i.e. it
    must never silently undercount what the v1-deltas surface reports."""
    hit = [0, 200, 0, 300]
    stored = [0, 1000, 0, 600]
    t = WastedStoreTracker()
    for i, (h, s) in enumerate(zip(hit, stored)):
        t.record_snapshot(float(i), 1, 0.85, {"stored_tokens": s, "hit_tokens": h})
    report = t.classify()

    collector = _SyncCollector(hit, stored)
    ext = compute_external_reuse(collector)

    # classify (with M3 accumulation) sees 200 + 300 = 500 hit tokens.
    # compute_external_reuse sees max(0, 300-0) = 300.
    # The tracker must not undercount the v1-deltas-derived figure.
    assert report.total_retrieved_tokens >= ext["external_hit_tokens"]
    assert report.counter_reset_observed is True
