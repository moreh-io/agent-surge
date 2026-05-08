# SPDX-License-Identifier: MIT
"""[audit:misc-core#H2] Reject non-monotonic cumulative bucket counts.

After ``_bucket_delta`` subtracts two snapshots, a counter reset or
out-of-order Prometheus scrape can yield a "cumulative" series where
later le buckets carry SMALLER counts than earlier ones. The function
previously interpolated happily and returned plausible-looking but
meaningless percentiles. We want NaN in that case so downstream SLO
compares treat it as "no data" instead of "passing".
"""

from __future__ import annotations

import math

from agentsurge.metrics import histogram_percentiles


def test_non_monotonic_cumulative_returns_nan() -> None:
    # le=1 had 5 cumulative samples, le=2 had 3 — physically impossible
    # for a real Prometheus histogram, indicates server reset/scrape skew.
    result = histogram_percentiles({"1": 5, "2": 3}, 5, (50, 95, 99))
    for p in (50, 95, 99):
        assert math.isnan(result[f"p{p}"]), (
            f"expected NaN for non-monotonic buckets at p{p}, got {result[f'p{p}']!r}"
        )


def test_monotonic_still_works() -> None:
    # Sanity: ordinary monotonic buckets must still produce real percentiles.
    result = histogram_percentiles({"1": 5, "2": 10}, 10, (50,))
    assert not math.isnan(result["p50"])
