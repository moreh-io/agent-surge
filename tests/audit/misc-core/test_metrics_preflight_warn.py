# SPDX-License-Identifier: MIT
"""[audit:misc-core#M5] MetricsCollector.start preflight-failure must warn.

When ``fetch_metrics`` returns ``None`` on the immediate pre-flight
snapshot but ``enabled=True``, the collector previously proceeded
silently. Backend-mode runs then ship "passing" reports with
``kv_util_peak=0.0`` and empty timeseries. The fix logs a single
high-priority warning so users can correlate the empty fields with the
failed initial fetch.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import patch

from agentsurge.metrics import MetricsCollector


def test_preflight_failure_logs_warning(caplog) -> None:
    coll = MetricsCollector(vllm_url="http://example", enabled=True, interval=10.0)

    async def _no_op_fetch(*_a, **_k):
        return None

    with (
        patch("agentsurge.metrics.fetch_metrics", _no_op_fetch),
        caplog.at_level(logging.WARNING, logger="agentsurge.metrics"),
    ):

        async def run() -> None:
            await coll.start()
            # Don't let the poll loop spin meaningfully.
            await coll.stop()

        asyncio.run(run())

    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("preflight" in m.lower() or "initial fetch" in m.lower() for m in msgs), (
        f"expected a preflight-failure warning, got: {msgs!r}"
    )
