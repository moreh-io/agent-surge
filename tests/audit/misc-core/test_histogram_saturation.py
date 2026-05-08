# SPDX-License-Identifier: MIT
"""[audit:misc-core#H1] histogram_percentiles must not silently saturate.

When the cumulative count of finite buckets is less than ``count`` (the
distribution's tail is in the +Inf bucket), the function previously
returned the largest finite bound. That looks like a real number and is
indistinguishable from a measured value, so saturation is silently
hidden. Returning +inf surfaces the saturation.
"""

from __future__ import annotations

import math

from agentsurge.metrics import histogram_percentiles


def test_p99_returns_inf_when_tail_in_plus_inf_bucket() -> None:
    # 5 total samples, but only 2 land in finite buckets (le="1", le="2")
    # The other 3 sit in +Inf. Previously p99 returned 2.0 (top finite bound)
    # which is a confident but meaningless number.
    result = histogram_percentiles({"1": 1, "2": 1}, 5, (99,))
    assert math.isinf(result["p99"]) and result["p99"] > 0, (
        f"expected +inf for saturated histogram, got {result['p99']!r}"
    )


def test_p50_within_finite_buckets_still_works() -> None:
    # Sanity check: ordinary case still interpolates correctly.
    result = histogram_percentiles({"1": 5, "2": 10}, 10, (50,))
    # 50% of 10 = 5; cum_count at le=1 is 5, so result is exactly 1.0
    assert result["p50"] == 1.0
