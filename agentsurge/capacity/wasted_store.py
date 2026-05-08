# SPDX-License-Identifier: MIT
"""Wasted-store lifecycle tracing -- track stored KV object fates.

Each KV object stored in LMCache has one of three fates:
  1. Retrieved: loaded back to GPU (useful store)
  2. Evicted: removed without retrieval (wasted store, wasted bandwidth)
  3. GPU-sufficed: never needed because GPU prefix cache covered it (redundant store)

This module provides accounting to classify stores by outcome,
enabling admission-control decisions.

"""

from dataclasses import dataclass, field

__all__ = [
    "StoreEvent",
    "WastedStoreReport",
    "WastedStoreTracker",
    "compute_admission_value",
]


@dataclass
class StoreEvent:
    """A single KV store operation."""

    timestamp: float = 0.0  # wall-clock time
    n_sessions: int = 0  # concurrency at time of store
    kv_usage: float = 0.0  # KV% at time of store
    tokens_stored: int = 0  # tokens in this store batch


@dataclass
class WastedStoreReport:
    """Summary of wasted-store analysis."""

    total_stored_tokens: int = 0
    total_retrieved_tokens: int = 0
    total_evicted_chunks: int = 0
    wasted_tokens: int = 0
    wasted_fraction: float = 0.0
    gpu_sufficed_tokens: int = 0
    gpu_sufficed_fraction: float = 0.0
    high_pressure_tokens: int = 0
    useful_retrieve_fraction: float = 0.0
    # True when total_retrieved exceeds high_pressure_tokens, i.e. the clamp at
    # 1.0 is binding and the fraction is mathematically meaningless. Stems from
    # numerator (all retrievals) and denominator (high-pressure stores only)
    # living in different domains; without per-event linkage we can only flag.
    useful_retrieve_saturated: bool = False
    # None means the "never observed" sentinel; 0.0 means a real hit at kv=0.
    first_external_hit_n: int | None = None
    first_external_hit_kv: float | None = None
    n_snapshots: int = 0
    n_store_events: int = 0
    # True when at least one snapshot reported strictly-decreasing counters,
    # indicating a backend restart mid-run. Totals are accumulated across the
    # reset, but the flag warns callers their numbers may still be incomplete
    # (snapshots between the last pre-reset sample and the reset are lost).
    counter_reset_observed: bool = False


@dataclass
class WastedStoreTracker:
    """Track lifecycle of stored KV objects across a benchmark run.

    Fed by periodic LMCache metrics snapshots. Classifies stores
    as useful, wasted, or GPU-sufficed based on retrieval patterns.
    """

    # Running counters from LMCache snapshots
    _snapshots: list[dict] = field(default_factory=list)
    _store_events: list[StoreEvent] = field(default_factory=list)
    _counter_reset_observed: bool = False

    # Classification thresholds.
    # Below gpu_sufficient_kv_threshold, APC hit rate >99% (empirical ablation).
    # LMCache stores at low KV% are redundant - GPU prefix cache handles everything.
    gpu_sufficient_kv_threshold: float = 0.5

    # First-hit detection requires a sustained signal so a 1-token blip can't
    # silently pin activation_kv. The first snapshot whose forward window
    # accumulates >= first_hit_min_tokens hit-token delta is treated as the
    # activation point.
    first_hit_min_tokens: int = 100
    first_hit_window_snapshots: int = 3

    def record_snapshot(
        self, timestamp: float, n_sessions: int, kv_usage: float, lmcache_metrics: dict
    ) -> None:
        """Record a metrics snapshot for lifecycle tracking.

        Call this periodically (e.g., every 0.3s from MetricsCollector).

        Args:
            timestamp: wall-clock time
            n_sessions: current concurrent sessions
            kv_usage: current KV cache usage fraction (0-1)
            lmcache_metrics: dict with keys like stored_tokens, hit_tokens,
                            cpu_evictions, disk_evictions, etc.
        """
        snapshot = {
            "timestamp": timestamp,
            "n_sessions": n_sessions,
            "kv_usage": kv_usage,
            **lmcache_metrics,
        }
        self._snapshots.append(snapshot)

        if len(self._snapshots) >= 2:
            prev = self._snapshots[-2]
            curr = self._snapshots[-1]
            for key in ("stored_tokens", "hit_tokens", "cpu_evictions", "disk_evictions"):
                if curr.get(key, 0) < prev.get(key, 0):
                    self._counter_reset_observed = True
                    break
            new_stored = curr.get("stored_tokens", 0) - prev.get("stored_tokens", 0)
            if new_stored > 0:
                self._store_events.append(
                    StoreEvent(
                        timestamp=timestamp,
                        n_sessions=n_sessions,
                        kv_usage=kv_usage,
                        tokens_stored=int(new_stored),
                    )
                )

    def classify(self) -> WastedStoreReport:
        """Classify all stores by outcome based on snapshot history.

        Logic:
        - total_stored = final stored_tokens
        - total_retrieved = final hit_tokens
        - total_evicted = final cpu_evictions + disk_evictions
        - wasted = stored - retrieved (tokens that were stored but never hit)
        - gpu_sufficed = stores that happened when kv_usage < threshold
          (APC was sufficient, LMCache store was redundant)
        """
        if not self._snapshots:
            return WastedStoreReport()

        # Accumulate positive deltas across consecutive snapshots so backend
        # restarts (counter resets) become checkpoints rather than data loss.
        def _sum_positive_delta(key: str) -> int:
            total = 0
            for i in range(1, len(self._snapshots)):
                d = self._snapshots[i].get(key, 0) - self._snapshots[i - 1].get(key, 0)
                if d > 0:
                    total += int(d)
            return total

        total_stored = _sum_positive_delta("stored_tokens")
        total_retrieved = _sum_positive_delta("hit_tokens")
        total_evictions = _sum_positive_delta("cpu_evictions") + _sum_positive_delta(
            "disk_evictions"
        )

        dead_tokens = max(0, total_stored - total_retrieved)

        gpu_sufficed_tokens = 0
        high_pressure_tokens = 0
        for event in self._store_events:
            if event.kv_usage < self.gpu_sufficient_kv_threshold:
                gpu_sufficed_tokens += event.tokens_stored
            else:
                high_pressure_tokens += event.tokens_stored

        # Require sustained activity: the first snapshot whose forward window
        # accumulates at least first_hit_min_tokens of hit-token delta. This
        # prevents a single 1-token retrieval from pinning the activation_kv.
        first_hit_snapshot = None
        n_snaps = len(self._snapshots)
        window = max(1, self.first_hit_window_snapshots)
        threshold = max(1, self.first_hit_min_tokens)
        for i in range(1, n_snaps):
            prev_hits = self._snapshots[i - 1].get("hit_tokens", 0)
            curr_hits = self._snapshots[i].get("hit_tokens", 0)
            if curr_hits <= prev_hits:
                continue
            window_end = min(n_snaps - 1, i + window - 1)
            cumulative = self._snapshots[window_end].get("hit_tokens", 0) - prev_hits
            # Negative cumulatives can occur after a counter reset; ignore those.
            if cumulative >= threshold:
                first_hit_snapshot = self._snapshots[i]
                break

        return WastedStoreReport(
            total_stored_tokens=int(total_stored),
            total_retrieved_tokens=int(total_retrieved),
            total_evicted_chunks=int(total_evictions),
            wasted_tokens=int(dead_tokens),
            wasted_fraction=dead_tokens / max(total_stored, 1),
            gpu_sufficed_tokens=int(gpu_sufficed_tokens),
            gpu_sufficed_fraction=gpu_sufficed_tokens / max(total_stored, 1),
            high_pressure_tokens=int(high_pressure_tokens),
            useful_retrieve_fraction=min(1.0, total_retrieved / high_pressure_tokens)
            if high_pressure_tokens > 0
            else 0.0,
            useful_retrieve_saturated=(
                high_pressure_tokens > 0 and total_retrieved > high_pressure_tokens
            ),
            first_external_hit_n=first_hit_snapshot.get("n_sessions", 0)
            if first_hit_snapshot
            else None,
            first_external_hit_kv=first_hit_snapshot.get("kv_usage", 0.0)
            if first_hit_snapshot
            else None,
            n_snapshots=len(self._snapshots),
            n_store_events=len(self._store_events),
            counter_reset_observed=self._counter_reset_observed,
        )

    def to_dict(self) -> dict:
        """Export full tracking data for JSON serialization."""
        report = self.classify()
        return {
            "report": {
                "total_stored_tokens": report.total_stored_tokens,
                "total_retrieved_tokens": report.total_retrieved_tokens,
                "wasted_tokens": report.wasted_tokens,
                "wasted_fraction": round(report.wasted_fraction, 4),
                "gpu_sufficed_tokens": report.gpu_sufficed_tokens,
                "gpu_sufficed_fraction": round(report.gpu_sufficed_fraction, 4),
                "high_pressure_tokens": report.high_pressure_tokens,
                "useful_retrieve_fraction": round(report.useful_retrieve_fraction, 4),
                "useful_retrieve_saturated": report.useful_retrieve_saturated,
                "first_external_hit_n": report.first_external_hit_n,
                "first_external_hit_kv": (
                    round(report.first_external_hit_kv, 4)
                    if report.first_external_hit_kv is not None
                    else None
                ),
                "total_evicted_chunks": report.total_evicted_chunks,
            },
            "store_events": [
                {
                    "ts": e.timestamp,
                    "n": e.n_sessions,
                    "kv": round(e.kv_usage, 4),
                    "tokens": e.tokens_stored,
                }
                for e in self._store_events
            ],
            "n_snapshots": len(self._snapshots),
        }


# Admission-control decision thresholds.
# Extracted as named constants so they can be overridden per-deployment.
_USEFUL_THRESHOLD = 0.1  # store if >10% of stores are useful (minimum ROI)
_HIGH_WASTE_THRESHOLD = 0.9  # >90% waste → recommend DISABLE
_MODERATE_WASTE_THRESHOLD = 0.5  # >50% waste → recommend GATE


def compute_admission_value(
    report: WastedStoreReport,
    *,
    useful_threshold: float = _USEFUL_THRESHOLD,
    high_waste_threshold: float = _HIGH_WASTE_THRESHOLD,
    moderate_waste_threshold: float = _MODERATE_WASTE_THRESHOLD,
) -> dict:
    """Compute admission-control decision metrics from wasted-store report.

    Args:
        report: WastedStoreReport from WastedStoreTracker.classify().
        useful_threshold: minimum useful_ratio to recommend storing (default 0.1).
        high_waste_threshold: waste_ratio above which DISABLE is recommended (default 0.9).
        moderate_waste_threshold: waste_ratio above which GATE is recommended (default 0.5).

    Returns:
        - should_store: bool -- True if storing is net-positive
        - waste_ratio: float -- fraction of store bandwidth wasted
        - activation_kv: float -- KV% above which storing becomes useful
        - recommendation: str -- human-readable admission recommendation
    """
    # Conservative union upper bound: min(1, a+b) when per-event overlap is
    # unknown. max(a, b) is the lower bound and biases toward ENABLE; the
    # audit prefers the upper bound so recommendations don't silently
    # undercount waste.
    _wasted = max(0.0, report.wasted_fraction)
    _gpu_sufficed = max(0.0, report.gpu_sufficed_fraction)
    waste_ratio = min(1.0, _wasted + _gpu_sufficed)
    useful_ratio = 1.0 - waste_ratio

    should_store = useful_ratio > useful_threshold

    # None means the tracker never observed an external hit -- recommend
    # gating at 100% (effectively never store). A real hit at kv=0.0 is
    # legitimate and must round-trip as 0.0%.
    activation_kv = 1.0 if report.first_external_hit_kv is None else report.first_external_hit_kv

    if waste_ratio > high_waste_threshold:
        recommendation = f"DISABLE: >{high_waste_threshold * 100:.0f}% of stores are wasted (GPU-sufficed or wasted)"
    elif waste_ratio > moderate_waste_threshold:
        recommendation = f"GATE: enable only above KV={activation_kv * 100:.0f}%"
    else:
        recommendation = "ENABLE: majority of stores are useful"

    return {
        "should_store": should_store,
        "waste_ratio": round(waste_ratio, 4),
        "useful_ratio": round(useful_ratio, 4),
        "activation_kv_pct": round(activation_kv * 100, 1),
        "recommendation": recommendation,
    }


def compute_external_reuse(
    collector,
    kv_bytes_per_token: int = 512,
) -> dict:
    """Compute external APC-miss reuse metrics from a MetricsCollector.

    Args:
        collector: MetricsCollector with snapshots from a completed run.
        kv_bytes_per_token: Approximate KV cache bytes per token (K+V combined).
    """
    apc_hits, apc_queries = collector.prefix_cache_delta()
    # SGLang: prefix_cache_delta clamps to (0,0) since cache_hit_rate is point-in-time.
    # Detect that case so callers can tell which denominator definition was used.
    sglang_rate = getattr(collector, "sglang_cache_hit_rate", -1.0)
    is_sglang = sglang_rate >= 0
    useful_retrieve_basis = "total_prompt" if is_sglang else "apc_miss"

    v1 = collector.lmcache_v1_deltas()
    stored_known: bool
    if v1:
        tier_hits = 0
        tiers = v1.get("tiers")
        if isinstance(tiers, dict):
            tier_hits = sum(
                tier.get("hit_tokens", 0) for tier in tiers.values() if isinstance(tier, dict)
            )
        ext_hits = max(v1.get("hit_tokens", 0), tier_hits)
        stored = v1.get("stored_tokens", 0)
        stored_known = True
    else:
        ext_hit_delta, _ = collector.external_prefix_cache_delta()
        ext_hits = ext_hit_delta
        stored = None
        stored_known = False

    if stored_known:
        dead: int | None = max(0, stored - ext_hits)
        dead_frac: float | None = dead / max(stored, 1)
    else:
        dead = None
        dead_frac = None

    first_snap = collector.snapshots[0] if collector.snapshots else None
    last_snap = collector.snapshots[-1] if collector.snapshots else None
    total_prompt = 0.0
    if first_snap and last_snap:
        total_prompt = max(0.0, last_snap.isl_total - first_snap.isl_total)

    if is_sglang:
        # apc_hits unavailable as a delta on SGLang; denominator is total_prompt.
        denom = total_prompt
    else:
        denom = total_prompt - apc_hits
    useful_frac = ext_hits / max(denom, 1) if denom > 0 else 0.0

    return {
        "apc_hit_tokens": apc_hits,
        "external_hit_tokens": ext_hits,
        "wasted_tokens": dead,
        "wasted_fraction": round(dead_frac, 4) if dead_frac is not None else None,
        "useful_retrieve_fraction": round(min(1.0, useful_frac), 4),
        "useful_retrieve_basis": useful_retrieve_basis,
        "gpu_tax_bytes": dead * kv_bytes_per_token if dead is not None else None,
        "memory_pressure": round(collector.memory_pressure, 4),
    }
