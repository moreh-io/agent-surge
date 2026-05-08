# SPDX-License-Identifier: MIT
"""Tests for pending_user_messages feeding in the dynamic tool-loop runner."""

from __future__ import annotations

import os
from unittest.mock import patch

import aiohttp
import pytest

from agentsurge.runner import BenchmarkRunner, _make_ssl_kwargs
from agentsurge.tool_call import CODING_TOOLS
from agentsurge.types import ReplaySession
from agentsurge.types.results import BenchmarkConfig, TurnResult


def _make_config(**overrides) -> BenchmarkConfig:
    defaults = dict(
        vllm_url="http://mock",
        model="mock-model",
        max_concurrency=1,
        tool_mode="off",
        tool_call_parser_fallback="off",
        sanitize_truncated_tool_calls=False,
        max_tokens=256,
        request_timeout=10,
        continue_turn_on="never",
        ignore_replay_output_length=True,  # avoid tokenizer load
        skip_tokenizer_load=True,
    )
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)


def _chat_response(
    *,
    content: str = "",
    tool_calls: list[dict] | None = None,
) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "resp-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _make_session(*, pending: list[str] | None = None, n_turn_messages: int = 1) -> ReplaySession:
    base_messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
    ]
    turn_messages = [list(base_messages) for _ in range(n_turn_messages)]
    return ReplaySession(
        session_id="sess-1",
        turn_messages=turn_messages,
        metadata={"max_turns": 5, "tools": []},
        pending_user_messages=list(pending or []),
    )


def _finish_only_tools() -> list[dict]:
    return [tool for tool in CODING_TOOLS if tool["function"]["name"] == "finish"]


def _make_live_config_or_skip() -> BenchmarkConfig:
    url = os.getenv("AGENTSURGE_LIVE_VLLM_URL")
    model = os.getenv("AGENTSURGE_LIVE_VLLM_MODEL")
    if not url or not model:
        pytest.skip(
            "set AGENTSURGE_LIVE_VLLM_URL and AGENTSURGE_LIVE_VLLM_MODEL "
            "to run live vLLM multi-turn feeding tests"
        )
    return _make_config(
        vllm_url=url,
        model=model,
        max_tokens=32,
        temperature=0.0,
        request_timeout=180,
        extra_body={"tool_choice": {"type": "function", "function": {"name": "finish"}}},
    )


class _SequencedExecutor:
    """Stub for `_execute_single_tool_turn` returning a scripted sequence."""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.captured_messages: list[list[dict]] = []

    async def __call__(
        self,
        http,
        semaphore,
        session_id,
        turn_idx,
        messages,
        endpoint,
        tools,
        validator,
        session_meta,
    ):
        # Record a copy so later mutations don't hide the evidence.
        self.captured_messages.append([dict(m) for m in messages])
        if not self._responses:
            raise AssertionError("No more scripted responses")
        response = self._responses.pop(0)
        turn_result = TurnResult(
            session_id=session_id,
            turn_index=turn_idx,
            completed=True,
            total_ms=1.0,
            response_text=response["choices"][0]["message"].get("content", ""),
            input_tokens=response["usage"]["prompt_tokens"],
            output_tokens=response["usage"]["completion_tokens"],
        )
        return turn_result, response


@pytest.mark.asyncio
class TestPendingUserMessageFeeding:
    async def _run(self, cfg, session, responses):
        runner = BenchmarkRunner(config=cfg)
        runner._run_start = 0.0
        executor = _SequencedExecutor(responses)
        import asyncio

        semaphore = asyncio.Semaphore(1)
        # Use ``new=executor`` (not ``side_effect=``) because patch.object
        # auto-wraps async attributes in AsyncMock, and AsyncMock treats
        # side_effect's return value as the coroutine's return — which would
        # hand back the unawaited coroutine instead of its result.
        with patch.object(runner, "_execute_single_tool_turn", new=executor):
            result = await runner._run_tool_session(
                http=None, semaphore=semaphore, session=session, delay=0.0
            )
        return result, executor

    async def test_sanitized_finish_does_not_terminate_session(self):
        """Sanitize contract: a truncated/malformed tool call should flow
        through tool-response building so the tool reports the error back
        to the model and the session continues. A truncated `finish` must
        NOT trigger early termination."""
        cfg = _make_config(sanitize_truncated_tool_calls=True)
        session = ReplaySession(
            session_id="sess-1",
            turn_messages=[
                [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "u1"},
                ]
                for _ in range(3)
            ],
            metadata={"max_turns": 5, "tools": CODING_TOOLS},
            pending_user_messages=[],
        )

        # Turn 0: truncated finish (JSON unterminated) → sanitize kicks in.
        # Turn 1: after seeing the tool error, model emits a proper finish.
        responses = [
            _chat_response(
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "finish",
                            "arguments": '{"summary": "truncated',
                        },
                    }
                ]
            ),
            _chat_response(
                tool_calls=[
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "finish", "arguments": "{}"},
                    }
                ]
            ),
        ]
        result, executor = await self._run(cfg, session, responses)
        assert len(result.turns) >= 2, (
            f"sanitized finish terminated session prematurely; got {len(result.turns)} turn(s)"
        )

    async def test_finish_with_empty_queue_still_ends_session(self):
        """Regression guard: existing finish-terminates behavior preserved."""
        cfg = _make_config()
        session = _make_session(pending=[], n_turn_messages=3)
        responses = [
            _chat_response(
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "finish", "arguments": "{}"},
                    }
                ]
            ),
        ]
        result, executor = await self._run(cfg, session, responses)
        assert len(result.turns) == 1
        assert len(executor.captured_messages) == 1

    async def test_finish_with_pending_queue_is_overridden(self):
        """When finish tool is called but queue has items, session continues
        with the next queued user message instead of breaking."""
        cfg = _make_config()
        session = _make_session(pending=["u2"], n_turn_messages=3)
        responses = [
            _chat_response(
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "finish", "arguments": "{}"},
                    }
                ]
            ),
            _chat_response(
                tool_calls=[
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "finish", "arguments": "{}"},
                    }
                ]
            ),
        ]
        result, executor = await self._run(cfg, session, responses)
        assert len(result.turns) == 2, "should run two turns, one per user message"
        # Turn 2's messages must contain the injected u2
        turn2_msgs = executor.captured_messages[1]
        assert any(m.get("role") == "user" and m.get("content") == "u2" for m in turn2_msgs), (
            f"expected u2 in turn 2 messages, got {turn2_msgs}"
        )

    async def test_text_only_never_with_pending_continues_with_queue(self):
        """With continue_turn_on='never' and a pending queue, a text-only
        response feeds the next user message instead of terminating."""
        cfg = _make_config(continue_turn_on="never")
        session = _make_session(pending=["u2"], n_turn_messages=3)
        responses = [
            _chat_response(content="I think we should do X."),
            _chat_response(
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "finish", "arguments": "{}"},
                    }
                ]
            ),
        ]
        result, executor = await self._run(cfg, session, responses)
        assert len(result.turns) == 2
        turn2_msgs = executor.captured_messages[1]
        # Should contain both the assistant's text-only reply and the queued user msg
        assert any(m.get("role") == "assistant" for m in turn2_msgs)
        assert any(m.get("role") == "user" and m.get("content") == "u2" for m in turn2_msgs)

    async def test_text_only_text_only_mode_prefers_pending_over_continue_prompt(
        self,
    ):
        """With continue_turn_on='text-only' and a pending queue, the queued
        user message is used instead of cfg.continue_prompt."""
        cfg = _make_config(continue_turn_on="text-only", continue_prompt="GENERIC CONTINUE")
        session = _make_session(pending=["u2-real"], n_turn_messages=3)
        responses = [
            _chat_response(content="text-only reply"),
            _chat_response(
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "finish", "arguments": "{}"},
                    }
                ]
            ),
        ]
        result, executor = await self._run(cfg, session, responses)
        turn2_msgs = executor.captured_messages[1]
        assert any(m.get("role") == "user" and m.get("content") == "u2-real" for m in turn2_msgs)
        assert not any(
            m.get("role") == "user" and m.get("content") == "GENERIC CONTINUE" for m in turn2_msgs
        )

    async def test_queue_consumed_in_fifo_order(self):
        cfg = _make_config()
        session = _make_session(pending=["u2", "u3"], n_turn_messages=5)
        responses = [
            _chat_response(
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "finish", "arguments": "{}"},
                    }
                ]
            ),
            _chat_response(
                tool_calls=[
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "finish", "arguments": "{}"},
                    }
                ]
            ),
            _chat_response(
                tool_calls=[
                    {
                        "id": "c3",
                        "type": "function",
                        "function": {"name": "finish", "arguments": "{}"},
                    }
                ]
            ),
        ]
        result, executor = await self._run(cfg, session, responses)
        assert len(result.turns) == 3
        turn2_msgs = executor.captured_messages[1]
        turn3_msgs = executor.captured_messages[2]
        assert [m["content"] for m in turn2_msgs if m.get("role") == "user"][-1] == "u2"
        assert [m["content"] for m in turn3_msgs if m.get("role") == "user"][-1] == "u3"

    async def test_finish_after_queue_drained_ends_session(self):
        cfg = _make_config()
        session = _make_session(pending=["u2"], n_turn_messages=3)
        responses = [
            _chat_response(content="text-only reply"),
            _chat_response(
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "finish", "arguments": "{}"},
                    }
                ]
            ),
        ]
        result, executor = await self._run(cfg, session, responses)
        # Queue had 1 entry; first turn used it, second turn's finish is honored.
        assert len(result.turns) == 2

    async def test_max_turns_is_extended_by_pending_count(self):
        """If base max_turns from metadata is small, pending queue still drains."""
        cfg = _make_config()
        session = ReplaySession(
            session_id="sess-1",
            turn_messages=[
                [{"role": "system", "content": "sys"}, {"role": "user", "content": "u1"}]
            ],
            metadata={"max_turns": 1, "tools": []},
            pending_user_messages=["u2", "u3"],
        )
        responses = [
            _chat_response(
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "finish", "arguments": "{}"},
                    }
                ]
            ),
            _chat_response(
                tool_calls=[
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "finish", "arguments": "{}"},
                    }
                ]
            ),
            _chat_response(
                tool_calls=[
                    {
                        "id": "c3",
                        "type": "function",
                        "function": {"name": "finish", "arguments": "{}"},
                    }
                ]
            ),
        ]
        result, executor = await self._run(cfg, session, responses)
        assert len(result.turns) == 3, "base max_turns=1 + 2 pending = 3 total"


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_live_vllm_finish_with_empty_assistant_content_continues_pending_queue():
    """Real vLLM smoke: empty-content finish responses must still feed the queue."""
    import asyncio

    cfg = _make_live_config_or_skip()
    runner = BenchmarkRunner(config=cfg)
    runner._run_start = 0.0
    pending_msg = "SECOND TURN SENTINEL: call finish immediately with no assistant text."
    session = ReplaySession(
        session_id="live-empty-finish",
        turn_messages=[
            [
                {
                    "role": "system",
                    "content": (
                        "You are a tool-calling assistant. Emit only tool calls and no "
                        "assistant text."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Call the finish tool immediately. Do not write any assistant text."
                    ),
                },
            ]
        ],
        metadata={"max_turns": 1, "tools": _finish_only_tools()},
        pending_user_messages=[pending_msg],
    )

    connector = aiohttp.TCPConnector(limit=0, **_make_ssl_kwargs(cfg.vllm_url))
    async with aiohttp.ClientSession(connector=connector) as http:
        result = await runner._run_tool_session(
            http=http,
            semaphore=asyncio.Semaphore(1),
            session=session,
            delay=0.0,
        )

    assert len(result.turns) == 2
    assert result.turns[0].completed is True
    assert result.turns[0].tool_calls == ["finish"]
    assert not result.turns[0].response_text
    assert result.turns[1].input_text == pending_msg
    assert result.turns[1].tool_calls == ["finish"]
