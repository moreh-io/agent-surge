"""M7: MockBackend.call_log / _call_count must reflect non-overridden calls only.

The pre-fix behaviour increments the counter and appends to call_log before
checking the on_send_turn callback override. That makes:
1. call_log misreport "successful" calls when a callback intercepts every call.
2. fail_after consume budget for callback-overridden calls.

Move the increment/log to happen only when the default mock logic runs
(after the callback short-circuit returns None or is absent).
"""

from __future__ import annotations

import asyncio

from agentsurge.backends.mock import MockBackend, MockConfig
from agentsurge.types import TurnResult


def test_callback_override_does_not_increment_counter():
    """A callback that intercepts every call must not consume fail_after budget."""

    def cb(session_id, turn_index, messages, config):
        return TurnResult(
            session_id=session_id,
            turn_index=turn_index,
            completed=True,
            ttft_ms=1.0,
            output_tokens=1,
            input_messages=len(messages),
        )

    cfg = MockConfig(on_send_turn=cb, fail_after=2)

    async def _run():
        async with MockBackend(cfg) as backend:
            results = [await backend.send_turn("s1", i, []) for i in range(5)]
            return backend, results

    backend, results = asyncio.run(_run())
    # All 5 must complete via callback - none should hit fail_after path
    assert all(r.completed for r in results), [r.error for r in results]
    # call_log must not record callback-overridden calls
    assert len(backend.call_log) == 0
