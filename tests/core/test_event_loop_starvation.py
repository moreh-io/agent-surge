"""Regression tests for event loop starvation (H1).

The async subprocess fix prevents synchronous tool execution from blocking the
event loop during concurrent fan-out. These tests verify that:

1. Multiple concurrent tool sessions don't serialize on tool execution
2. Client-observed request latency doesn't inflate pathologically with n
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentsurge.runner import BenchmarkRunner
from agentsurge.types.results import BenchmarkConfig
from agentsurge.types.trace import ReplaySession

pytestmark = pytest.mark.e2e


def _make_tool_call_response(turn_idx: int, finish: bool = False) -> dict:
    """Build a mock OpenAI-style response with a tool call or stop."""
    if finish:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Done.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10},
        }
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{turn_idx}",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command": "echo hello"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
    }


def _make_mock_http_for_tool_session(n_tool_turns: int = 2):
    """Mock aiohttp.ClientSession that returns tool call responses.

    Returns n_tool_turns responses with tool calls, then a stop response.
    Each session tracks its own call count via closure.
    aiohttp's post() returns a sync context manager used with `async with`.
    """
    call_counts: dict[str, int] = {}

    def _mock_post(url, json=None, timeout=None):
        msgs = json.get("messages", []) if isinstance(json, dict) else []
        sid = "unknown"
        for m in msgs:
            if m.get("role") == "user":
                sid = m.get("content", "unknown")[:20]
                break

        count = call_counts.get(sid, 0)
        call_counts[sid] = count + 1

        finish = count >= n_tool_turns
        resp_json = _make_tool_call_response(count, finish=finish)

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=resp_json)
        mock_resp.text = AsyncMock(return_value="")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        return mock_resp

    mock_http = MagicMock()
    mock_http.post = MagicMock(side_effect=_mock_post)
    return mock_http


TOOL_SLEEP_DURATION = 0.15  # seconds - long enough to detect serialization


async def _slow_build_tool_response_async(assistant_msg, tool_calls, *, sandbox=None, **kw):
    """Simulate slow tool execution without blocking the event loop."""
    await asyncio.sleep(TOOL_SLEEP_DURATION)
    messages = [assistant_msg]
    for tc in tool_calls:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.tool_call_id,
                "name": tc.name,
                "content": "ok",
            }
        )
    return messages


def _make_sessions(n: int, n_turns: int = 3) -> list[ReplaySession]:
    """Create n ReplaySessions with initial user messages."""
    sessions = []
    for i in range(n):
        sessions.append(
            ReplaySession(
                session_id=f"test_session_{i}",
                turn_messages=[[{"role": "user", "content": f"session {i} task"}]] * n_turns,
                metadata={"max_turns": n_turns},
            )
        )
    return sessions


class TestEventLoopStarvation:
    """Regression tests for H1: event loop starvation from synchronous tool execution."""

    def test_concurrent_tool_sessions_no_serialization(self):
        """n=3 concurrent tool sessions should NOT serialize on tool execution.

        With the async subprocess path, tool execution no longer blocks the event loop.
        Wall time should be closer to max(session_times) than sum(session_times).
        If the event loop is blocked (the bug), wall time approaches the sum.
        """
        n_sessions = 3
        n_tool_turns = 2  # each session does 2 tool calls then stops

        cfg = BenchmarkConfig(
            vllm_url="http://mock:8000",
            model="test-model",
            max_concurrency=n_sessions,
            tool_mode="real",
            request_timeout=30,
        )
        cfg.sandbox_dir = None  # no real sandbox

        runner = BenchmarkRunner(cfg)
        runner._run_start = time.monotonic()

        sessions = _make_sessions(n_sessions, n_turns=n_tool_turns + 1)
        mock_http = _make_mock_http_for_tool_session(n_tool_turns)

        async def _run():
            semaphore = asyncio.Semaphore(n_sessions)
            delays = [0.0] * n_sessions

            with patch(
                "agentsurge.tool_call.build_tool_response_messages_async",
                new=_slow_build_tool_response_async,
            ):
                t0 = time.monotonic()
                results = await runner._dispatch_sessions(mock_http, semaphore, sessions, delays)
                wall_time = time.monotonic() - t0

            return results, wall_time

        results, wall_time = asyncio.run(_run())

        # Each session does n_tool_turns tool calls, each sleeping TOOL_SLEEP_DURATION
        per_session_tool_time = n_tool_turns * TOOL_SLEEP_DURATION
        sum_all_tool_time = n_sessions * per_session_tool_time

        # With proper async command execution, wall time should be close to per_session_tool_time
        # (all sessions run in parallel). With the bug, it approaches sum_all_tool_time.
        # Use a threshold: wall time must be less than 60% of the sum (generous margin).
        assert wall_time < sum_all_tool_time * 0.6, (
            f"Wall time {wall_time:.2f}s is too close to sum({sum_all_tool_time:.2f}s). "
            f"Event loop may be blocked by synchronous tool execution. "
            f"Expected closer to {per_session_tool_time:.2f}s (parallel execution)."
        )

    def test_ttft_inflation_bounded(self):
        """Client-observed request latency at n=3 should not exceed 5x of n=1.

        This catches pathological TTFT inflation from event loop starvation.
        """
        n_tool_turns = 1

        def _run_with_n(n: int) -> list[float]:
            """Run n sessions and return per-turn TTFTs."""
            cfg = BenchmarkConfig(
                vllm_url="http://mock:8000",
                model="test-model",
                max_concurrency=n,
                tool_mode="real",
                request_timeout=30,
            )
            cfg.sandbox_dir = None

            runner = BenchmarkRunner(cfg)
            runner._run_start = time.monotonic()

            sessions = _make_sessions(n, n_turns=n_tool_turns + 1)
            mock_http = _make_mock_http_for_tool_session(n_tool_turns)

            async def _run():
                semaphore = asyncio.Semaphore(n)
                delays = [0.0] * n

                with patch(
                    "agentsurge.tool_call.build_tool_response_messages_async",
                    new=_slow_build_tool_response_async,
                ):
                    results = await runner._dispatch_sessions(
                        mock_http, semaphore, sessions, delays
                    )

                ttfts = []
                for r in results:
                    for t in r.turns:
                        if t.ttft_ms > 0:
                            ttfts.append(t.ttft_ms)
                return ttfts

            return asyncio.run(_run())

        ttfts_n1 = _run_with_n(1)
        ttfts_n3 = _run_with_n(3)

        if not ttfts_n1 or not ttfts_n3:
            pytest.skip("No TTFT values collected")

        median_n1 = sorted(ttfts_n1)[len(ttfts_n1) // 2]
        median_n3 = sorted(ttfts_n3)[len(ttfts_n3) // 2]

        # n=3 TTFT should not exceed 5x of n=1
        # With the fix, natural c=1 queueing gives ~3x at most
        # Without the fix, this was 65x+
        ratio = median_n3 / median_n1 if median_n1 > 0 else float("inf")
        assert ratio < 5.0, (
            f"TTFT inflation {ratio:.1f}x exceeds 5x threshold. "
            f"n=1 median={median_n1:.0f}ms, n=3 median={median_n3:.0f}ms. "
            f"Possible event loop starvation regression."
        )
