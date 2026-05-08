# SPDX-License-Identifier: MIT
"""[audit:misc-core#M6] _retry_loop must not retry on deterministic errors.

A TurnResult with ``error="empty response: 0 output tokens"`` is a
deterministic outcome (server returned an empty completion, e.g. the
``max_tokens`` boundary case). Retrying with exponential backoff is
pointless and inflates TTFT-shaped session timing by 1-4 s per affected
turn. Only retry on transient signals (HTTP 5xx, network timeouts,
connection failures).
"""

from __future__ import annotations

from agentsurge.runner import BenchmarkRunner
from agentsurge.types import BenchmarkConfig, TurnResult


def _mk_runner(max_retries: int) -> BenchmarkRunner:
    cfg = BenchmarkConfig(
        model="dummy",
        vllm_url="http://localhost:9999",
        max_retries=max_retries,
        ignore_replay_output_length=True,
        skip_tokenizer_load=True,
    )
    return BenchmarkRunner(cfg)


def test_retry_loop_does_not_retry_on_empty_response() -> None:
    import asyncio
    import unittest.mock as _mock

    runner = _mk_runner(max_retries=3)
    calls = 0

    async def send_fn() -> TurnResult:
        nonlocal calls
        calls += 1
        return TurnResult(
            session_id="s",
            turn_index=0,
            completed=False,
            ttft_ms=0.0,
            total_ms=10.0,
            output_tokens=0,
            input_messages=1,
            input_tokens=1,
            error="empty response: 0 output tokens",
        )

    real_sleep = asyncio.sleep

    async def fast_sleep(_d: float) -> None:
        await real_sleep(0)

    with _mock.patch("agentsurge.runner.asyncio.sleep", fast_sleep):
        asyncio.run(runner._retry_loop(send_fn))
    assert calls == 1, (
        f"empty-response is deterministic and must not be retried; "
        f"send_fn was invoked {calls} times"
    )


def test_retry_loop_still_retries_on_http_5xx() -> None:
    import asyncio

    runner = _mk_runner(max_retries=2)
    calls = 0

    async def send_fn() -> TurnResult:
        nonlocal calls
        calls += 1
        return TurnResult(
            session_id="s",
            turn_index=0,
            completed=False,
            ttft_ms=0.0,
            total_ms=10.0,
            output_tokens=0,
            input_messages=1,
            input_tokens=1,
            error="HTTP 503: backend overloaded",
        )

    # Patch asyncio.sleep so the retry backoff doesn't slow the test.
    real_sleep = asyncio.sleep

    async def fast_sleep(_duration: float) -> None:
        await real_sleep(0)

    runner_loop = runner._retry_loop
    import unittest.mock as _mock

    with _mock.patch("agentsurge.runner.asyncio.sleep", fast_sleep):
        asyncio.run(runner_loop(send_fn))

    assert calls == 3, f"transient HTTP 503 must be retried; got calls={calls}"
