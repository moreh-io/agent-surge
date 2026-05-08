"""Audit H8: TracePool._pick_nearest must surface silent undershoots.

When all snippets are short and target_tokens is large, the safety=10 loop
terminates well below the target with no signal. The fix exposes a counter
(or similar diagnostic) so callers can detect systematic undershoot.
"""

from __future__ import annotations

import random

from agentsurge.generators.trace_pool import Snippet, TracePool


def _tiny_pool() -> TracePool:
    # 5 short snippets of ~50 tokens each.
    snippets = [Snippet(tool_type="bash", tok_estimate=50, body="x" * 200) for _ in range(5)]
    return TracePool(snippets=snippets)


def test_pick_nearest_undershoot_exposed_via_counter():
    pool = _tiny_pool()
    rng = random.Random(0)
    # Request 5000 tokens; safety=10 caps concat to ~500 tokens — undershoot.
    pool.sample(rng, tool_type="bash", target_tokens=5000)
    # Fix exposes either an attribute counter or a public method to read it.
    count = getattr(pool, "undershoot_count", None)
    assert count is not None, (
        "TracePool must expose an `undershoot_count` attribute so callers "
        "can detect silent target-token undershoots in _pick_nearest."
    )
    assert count >= 1, (
        f"expected undershoot_count >= 1 after sampling 5000 tokens from a pool "
        f"of ~50-token snippets; got {count}"
    )
