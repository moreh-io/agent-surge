"""Audit M1: compute_admission_value's waste_ratio must use the union upper bound.

Using max(wasted_fraction, gpu_sufficed_fraction) is the *lower* bound of
the union of the two waste sets, biasing the recommendation toward ENABLE
even when actual waste could be near 100%. The audit recommends using
min(1.0, wasted + gpu_sufficed) so the recommendation is conservative.
"""

from __future__ import annotations

import pytest

from agentsurge.capacity.wasted_store import (
    WastedStoreReport,
    compute_admission_value,
)


def test_disjoint_fractions_sum_to_union_upper_bound():
    """When the two fractions are disjoint, max under-counts the union by half."""
    result = compute_admission_value(
        WastedStoreReport(
            total_stored_tokens=1000,
            wasted_fraction=0.6,
            gpu_sufficed_fraction=0.6,
        )
    )
    # Under max(): waste_ratio=0.6 → useful_ratio=0.4 → ENABLE.
    # Under union upper bound: min(1.0, 0.6+0.6)=1.0 → useful_ratio=0.0 → DISABLE.
    assert result["waste_ratio"] == pytest.approx(1.0)
    assert "DISABLE" in result["recommendation"]
    assert result["should_store"] is False


def test_pure_wasted_no_gpu_sufficed_unchanged():
    """When gpu_sufficed_fraction is zero the metric must equal wasted_fraction."""
    result = compute_admission_value(
        WastedStoreReport(wasted_fraction=0.3, gpu_sufficed_fraction=0.0)
    )
    assert result["waste_ratio"] == pytest.approx(0.3)


def test_clamped_at_one():
    result = compute_admission_value(
        WastedStoreReport(wasted_fraction=0.7, gpu_sufficed_fraction=0.7)
    )
    assert result["waste_ratio"] == pytest.approx(1.0)
