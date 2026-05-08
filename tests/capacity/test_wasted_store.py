"""Tests for agentsurge.capacity.wasted_store module.

Trimmed: scenario-based tests for tracker, decision logic for admission.
"""

from __future__ import annotations

import pytest

from agentsurge.capacity.wasted_store import (
    WastedStoreReport,
    WastedStoreTracker,
    compute_admission_value,
    compute_external_reuse,
)

# ---------------------------------------------------------------------------
# WastedStoreTracker - all scenarios
# ---------------------------------------------------------------------------


class TestWastedStoreTracker:
    def test_empty(self):
        assert WastedStoreTracker().classify().total_stored_tokens == 0

    def test_all_retrieved(self):
        t = WastedStoreTracker()
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
            {"stored_tokens": 500, "hit_tokens": 0, "cpu_evictions": 0, "disk_evictions": 0},
        )
        t.record_snapshot(
            2.0,
            5,
            0.95,
            {"stored_tokens": 500, "hit_tokens": 500, "cpu_evictions": 0, "disk_evictions": 0},
        )
        r = t.classify()
        assert r.total_stored_tokens == 500
        assert r.wasted_tokens == 0
        assert r.first_external_hit_n == 5
        assert r.first_external_hit_kv == pytest.approx(0.95)

    def test_all_wasted(self):
        t = WastedStoreTracker()
        t.record_snapshot(0.0, 5, 0.9, {"stored_tokens": 0, "hit_tokens": 0})
        t.record_snapshot(1.0, 5, 0.9, {"stored_tokens": 200, "hit_tokens": 0})
        r = t.classify()
        assert r.wasted_fraction == pytest.approx(1.0)
        # No external hit ever observed → None sentinel, not zero.
        assert r.first_external_hit_n is None
        assert r.first_external_hit_kv is None

    def test_gpu_sufficed(self):
        t = WastedStoreTracker(gpu_sufficient_kv_threshold=0.5)
        t.record_snapshot(0.0, 2, 0.3, {"stored_tokens": 0, "hit_tokens": 0})
        t.record_snapshot(1.0, 2, 0.3, {"stored_tokens": 100, "hit_tokens": 0})
        t.record_snapshot(2.0, 2, 0.8, {"stored_tokens": 200, "hit_tokens": 0})
        r = t.classify()
        assert r.gpu_sufficed_tokens == 100
        assert r.gpu_sufficed_fraction == pytest.approx(0.5)

    def test_store_events_and_evictions(self):
        t = WastedStoreTracker()
        t.record_snapshot(
            0.0, 1, 0.9, {"stored_tokens": 0, "cpu_evictions": 0, "disk_evictions": 0}
        )
        t.record_snapshot(
            1.0, 1, 0.9, {"stored_tokens": 100, "cpu_evictions": 5, "disk_evictions": 2}
        )
        t.record_snapshot(
            2.0, 1, 0.9, {"stored_tokens": 200, "cpu_evictions": 5, "disk_evictions": 2}
        )
        r = t.classify()
        assert r.n_store_events == 2
        assert r.total_evicted_chunks == 7

    def test_useful_retrieve_fraction_clamped(self):
        t = WastedStoreTracker(gpu_sufficient_kv_threshold=0.5)
        t.record_snapshot(
            0.0,
            5,
            0.3,
            {"stored_tokens": 0, "hit_tokens": 0, "cpu_evictions": 0, "disk_evictions": 0},
        )
        t.record_snapshot(
            1.0,
            5,
            0.6,
            {"stored_tokens": 100, "hit_tokens": 0, "cpu_evictions": 0, "disk_evictions": 0},
        )
        t.record_snapshot(
            2.0,
            5,
            0.7,
            {"stored_tokens": 200, "hit_tokens": 500, "cpu_evictions": 0, "disk_evictions": 0},
        )
        r = t.classify()
        assert r.useful_retrieve_fraction <= 1.0

    def test_to_dict(self):
        t = WastedStoreTracker()
        t.record_snapshot(0.0, 5, 0.9, {"stored_tokens": 0, "hit_tokens": 0})
        t.record_snapshot(1.0, 5, 0.9, {"stored_tokens": 100, "hit_tokens": 50})
        d = t.to_dict()
        assert d["report"]["total_stored_tokens"] == 100
        assert d["report"]["total_retrieved_tokens"] == 50
        assert len(d["store_events"]) > 0


# ---------------------------------------------------------------------------
# compute_admission_value() - decision logic
# ---------------------------------------------------------------------------


class TestComputeAdmissionValue:
    def test_low_waste_recommends_enable(self):
        enable = compute_admission_value(
            WastedStoreReport(
                total_stored_tokens=1000,
                total_retrieved_tokens=600,
                wasted_fraction=0.3,
                gpu_sufficed_fraction=0.1,
                useful_retrieve_fraction=0.6,
            )
        )
        assert "ENABLE" in enable["recommendation"]
        assert enable["should_store"] is True

    def test_high_waste_recommends_disable(self):
        disable = compute_admission_value(
            WastedStoreReport(
                total_stored_tokens=1000,
                wasted_fraction=0.95,
                gpu_sufficed_fraction=0.0,
                useful_retrieve_fraction=0.0,
            )
        )
        assert "DISABLE" in disable["recommendation"]
        assert disable["should_store"] is False

    def test_moderate_waste_recommends_gate(self):
        # waste_ratio = min(1, 0.4+0.3) = 0.7, which is > 0.5 (GATE) and < 0.9 (DISABLE)
        gate = compute_admission_value(
            WastedStoreReport(
                total_stored_tokens=1000,
                wasted_fraction=0.4,
                gpu_sufficed_fraction=0.3,
                first_external_hit_kv=0.85,
            )
        )
        assert "GATE" in gate["recommendation"]

    def test_waste_and_useful_sum(self):
        result = compute_admission_value(
            WastedStoreReport(wasted_fraction=0.3, gpu_sufficed_fraction=0.2)
        )
        assert result["waste_ratio"] + result["useful_ratio"] == pytest.approx(1.0)

    def test_union_upper_bound_when_overlap_unknown(self):
        # wasted_fraction=0.6, gpu_sufficed_fraction=0.6: the two sets may
        # overlap or be disjoint; per-event linkage isn't tracked, so the
        # admission decision uses the conservative union upper bound:
        # min(1.0, 0.6+0.6) = 1.0. This biases toward DISABLE rather than
        # silently under-counting waste (audit M1).
        result = compute_admission_value(
            WastedStoreReport(
                total_stored_tokens=1000,
                wasted_fraction=0.6,
                gpu_sufficed_fraction=0.6,
            )
        )
        assert result["waste_ratio"] == pytest.approx(1.0)
        assert "DISABLE" in result["recommendation"]


# ---------------------------------------------------------------------------
# compute_external_reuse() - snapshot-timing jitter guard
# ---------------------------------------------------------------------------


class _Snap:
    """Minimal snapshot stub with isl_total attribute."""

    def __init__(self, isl_total: float) -> None:
        self.isl_total = isl_total


class _MockCollector:
    """Minimal collector stub for compute_external_reuse."""

    def __init__(
        self,
        *,
        apc_hits: float,
        total_prompt: float,
        ext_hits: float,
        stored: float = 0,
        memory_pressure: float = 0.0,
    ) -> None:
        self._apc_hits = apc_hits
        self._ext_hits = ext_hits
        self._stored = stored
        self.memory_pressure = memory_pressure
        self.snapshots = [_Snap(0.0), _Snap(total_prompt)]

    def prefix_cache_delta(self) -> tuple[float, float]:
        return self._apc_hits, 0.0

    def lmcache_v1_deltas(self) -> dict:
        return {"hit_tokens": self._ext_hits, "stored_tokens": self._stored}

    def external_prefix_cache_delta(self) -> tuple[float, float]:
        return self._ext_hits, 0.0


class _MockCollectorV1TierOnly(_MockCollector):
    """Collector stub with v1 tier hits but zero aggregate hit_tokens."""

    def lmcache_v1_deltas(self) -> dict:
        return {
            "hit_tokens": 0,
            "stored_tokens": self._stored,
            "tiers": {
                "cpu": {"hit_tokens": self._ext_hits},
                "disk": {"hit_tokens": 0},
                "remote": {"hit_tokens": 0},
            },
        }


class _MockCollectorNoV1:
    """Collector stub where v1 LMCache data is unavailable (empty dict)."""

    def __init__(
        self,
        *,
        apc_hits: float,
        total_prompt: float,
        ext_hits: float,
        ext_query: float = 0,
        memory_pressure: float = 0.0,
    ) -> None:
        self._apc_hits = apc_hits
        self._ext_hits = ext_hits
        self._ext_query = ext_query
        self.memory_pressure = memory_pressure
        self.snapshots = [_Snap(0.0), _Snap(total_prompt)]

    def prefix_cache_delta(self) -> tuple[float, float]:
        return self._apc_hits, 0.0

    def lmcache_v1_deltas(self) -> dict:
        return {}

    def external_prefix_cache_delta(self) -> tuple[float, float]:
        return self._ext_hits, self._ext_query


class TestComputeExternalReuse:
    def test_useful_retrieve_fraction_clamped_when_ext_hits_exceeds_apc_miss(self):
        # apc_miss = total_prompt - apc_hits = 100 - 90 = 10
        # ext_hits = 50 → raw fraction = 50/10 = 5.0 without the clamp
        collector = _MockCollector(apc_hits=90, total_prompt=100, ext_hits=50)
        result = compute_external_reuse(collector)
        assert result["useful_retrieve_fraction"] <= 1.0, (
            f"useful_retrieve_fraction={result['useful_retrieve_fraction']} exceeds 1.0"
        )

    def test_fallback_path_wasted_fraction_is_none_when_stored_unknown(self):
        # v1 LMCache data unavailable → stored-token count is unknown.
        # With genuine retrievals happening (ext_hits=200), hardcoding stored=0
        # would give dead=max(0, 0-200)=0 and wasted_fraction=0.0, which is
        # misleadingly optimistic. Option B: return None for stored-dependent fields.
        collector = _MockCollectorNoV1(
            apc_hits=100,
            total_prompt=500,
            ext_hits=200,
            ext_query=250,
        )
        result = compute_external_reuse(collector)
        assert result["wasted_fraction"] is None, (
            f"Expected wasted_fraction=None on v1 fallback path, got {result['wasted_fraction']!r}"
        )
        assert result["wasted_tokens"] is None, (
            f"Expected wasted_tokens=None on v1 fallback path, got {result['wasted_tokens']!r}"
        )

    def test_v1_tier_only_hits_count_as_external_reuse(self):
        collector = _MockCollectorV1TierOnly(
            apc_hits=100,
            total_prompt=500,
            ext_hits=200,
            stored=300,
        )
        result = compute_external_reuse(collector)
        assert result["external_hit_tokens"] == 200
        assert result["wasted_tokens"] == 100
