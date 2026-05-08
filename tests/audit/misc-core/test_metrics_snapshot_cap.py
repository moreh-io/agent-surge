# SPDX-License-Identifier: MIT
"""[audit:misc-core#M1] MetricsCollector snapshots must be bounded.

Long ``--duration`` runs at 0.3 s sample interval grow ``snapshots``
without bound (~12 k entries / hour). Each snapshot carries multiple
bucket dicts. The runner copies the full list into the result JSON.

The fix introduces a configurable ``max_snapshots`` cap that retains
the first snapshot (needed for cumulative-delta calculations) plus the
most recent N-1 entries. Without a cap, the list grows linearly.
"""

from __future__ import annotations

from agentsurge.metrics import MetricsCollector, MetricsSnapshot


def test_max_snapshots_caps_list_size() -> None:
    coll = MetricsCollector(vllm_url="http://example", enabled=False, max_snapshots=5)
    # Simulate the poll-loop appending 20 snapshots through the bounded helper.
    for i in range(20):
        coll._append_snapshot(MetricsSnapshot(timestamp=float(i)))
    assert len(coll.snapshots) <= 5, f"snapshots list exceeded cap: {len(coll.snapshots)}"
    # First snapshot must be retained for cumulative-delta calculations.
    assert coll.snapshots[0].timestamp == 0.0
    # Most recent must always be present.
    assert coll.snapshots[-1].timestamp == 19.0


def test_unbounded_when_max_snapshots_zero() -> None:
    coll = MetricsCollector(vllm_url="http://example", enabled=False, max_snapshots=0)
    for i in range(50):
        coll._append_snapshot(MetricsSnapshot(timestamp=float(i)))
    assert len(coll.snapshots) == 50
