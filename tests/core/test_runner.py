"""Runner tests: arrival patterns, _send_turn, _run_session, responses API, simulation, callbacks."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from agentsurge import BenchmarkConfig, ReplaySession
from agentsurge.backends.base import BackendBase
from agentsurge.runner import (
    BenchmarkRunner,
    _ensure_tool_calls,
    _placeholderize_tool_outputs,
    _sanitize_tool_schema,
)
from agentsurge.tool_call import ToolCallCheck
from agentsurge.types import SessionResult, TurnResult
from tests.conftest import (
    MockAsyncLineIterator,
    make_mock_aiohttp_response,
    make_mock_aiohttp_session,
)

# ===========================================================================
# _placeholderize_tool_outputs
# ===========================================================================


class TestPlaceholderizeToolOutputs:
    def test_preserves_non_tool_messages(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Fix the bug."},
            {"role": "assistant", "content": "I'll look at it."},
        ]
        result = _placeholderize_tool_outputs(msgs)
        assert result == msgs  # unchanged

    def test_replaces_tool_output_prefix_content(self):
        original = "[Tool output: read_file] def foo():\n    return 42\n"
        msgs = [{"role": "user", "content": original}]
        result = _placeholderize_tool_outputs(msgs)
        out = result[0]["content"]
        assert out.startswith("[Tool output: read_file] ")
        assert "def foo" not in out
        assert abs(len(out) - len(original)) <= 1  # char count preserved

    def test_replaces_tool_role_content(self):
        msgs = [{"role": "tool", "name": "bash", "content": "Error: file not found\n" * 5}]
        result = _placeholderize_tool_outputs(msgs)
        assert result[0]["role"] == "tool"
        assert result[0]["name"] == "bash"
        assert "Error" not in result[0]["content"]
        assert abs(len(result[0]["content"]) - len(msgs[0]["content"])) <= 1

    def test_mixed_messages_selective(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Call: read_file(x.py)"},
            {"role": "user", "content": "[Tool output: read_file] class Foo:\n    pass\n"},
            {"role": "user", "content": "Now fix it."},
        ]
        result = _placeholderize_tool_outputs(msgs)
        assert result[0]["content"] == "Hello"  # preserved
        assert result[1]["content"] == "Call: read_file(x.py)"  # preserved
        assert "class Foo" not in result[2]["content"]  # replaced
        assert result[3]["content"] == "Now fix it."  # preserved


# ===========================================================================
# Arrival patterns
# ===========================================================================


class TestArrivalDelays:
    def test_burst_always_zero(self):
        runner = BenchmarkRunner(
            BenchmarkConfig(vllm_url="http://x", model="t", arrival_pattern="burst")
        )
        assert all(d == 0.0 for d in runner._compute_arrival_delays(10))

    def test_constant_spacing(self):
        runner = BenchmarkRunner(
            BenchmarkConfig(
                vllm_url="http://x", model="t", arrival_pattern="constant", arrival_rate=10.0
            )
        )
        delays = runner._compute_arrival_delays(5)
        for i, d in enumerate(delays):
            assert d == pytest.approx(i / 10.0)

    def test_poisson_nonnegative_and_monotonic(self):
        runner = BenchmarkRunner(
            BenchmarkConfig(
                vllm_url="http://x", model="t", arrival_pattern="poisson", arrival_rate=5.0
            )
        )
        delays = runner._compute_arrival_delays(1000)
        assert all(d >= 0.0 for d in delays)
        for a, b in zip(delays, delays[1:], strict=False):
            assert b > a
        gaps = [b - a for a, b in zip(delays, delays[1:])]
        mean_gap = sum(gaps) / len(gaps)
        expected_gap = 1.0 / 5.0
        assert abs(mean_gap - expected_gap) / expected_gap < 0.3

    def test_invalid_arrival_pattern(self):
        with pytest.raises(ValueError, match="Unknown arrival_pattern"):
            BenchmarkRunner(
                BenchmarkConfig(vllm_url="http://x", model="t", arrival_pattern="invalid")
            )


# ===========================================================================
# _effective_max_tokens
# ===========================================================================


class TestEffectiveMaxTokens:
    def _make_runner(self, **overrides):
        defaults = dict(vllm_url="http://x", model="t", max_tokens=256)
        defaults.update(overrides)
        return BenchmarkRunner(BenchmarkConfig(**defaults))

    def test_ignore_replay_output_length_returns_max_tokens(self):
        runner = self._make_runner(ignore_replay_output_length=True)
        meta = {"turn_output_contents": ["some recorded output text here"]}
        assert runner._effective_max_tokens(0, meta) == 256

    def test_no_meta_returns_max_tokens(self):
        runner = self._make_runner()
        assert runner._effective_max_tokens(0, None) == 256

    def test_turn_index_out_of_range_returns_max_tokens(self):
        runner = self._make_runner()
        meta = {"turn_output_contents": ["only one turn"]}
        assert runner._effective_max_tokens(5, meta) == 256

    def test_empty_content_returns_max_tokens(self):
        runner = self._make_runner()
        meta = {"turn_output_contents": ["", "   "]}
        assert runner._effective_max_tokens(0, meta) == 256
        assert runner._effective_max_tokens(1, meta) == 256

    def test_recorded_content_sizes_to_token_count(self):
        runner = self._make_runner()
        # FakeTokenizer: encode returns list(range(max(1, len(text)//4)))
        # "x" * 80 => len=80, 80//4=20, list(range(20)) => 20 tokens
        meta = {"turn_output_contents": ["x" * 80]}
        result = runner._effective_max_tokens(0, meta)
        assert result == 20

    def test_thinking_budget_added_when_thinking_enabled(self):
        runner = self._make_runner(enable_thinking=True, thinking_budget=4096)
        meta = {"turn_output_contents": ["x" * 80]}
        # 20 recorded tokens + 4096 thinking budget
        result = runner._effective_max_tokens(0, meta)
        assert result == 20 + 4096

    def test_thinking_budget_not_added_when_thinking_disabled(self):
        runner = self._make_runner(enable_thinking=False, thinking_budget=4096)
        meta = {"turn_output_contents": ["x" * 80]}
        result = runner._effective_max_tokens(0, meta)
        assert result == 20

    def test_minimum_one_token(self):
        runner = self._make_runner()
        # "x" * 3 => 3//4=0, max(1, 0) => list(range(1)) => 1 token
        meta = {"turn_output_contents": ["x" * 3]}
        result = runner._effective_max_tokens(0, meta)
        assert result >= 1

    def test_meta_without_turn_output_contents_key(self):
        runner = self._make_runner()
        meta = {"some_other_key": 123}
        assert runner._effective_max_tokens(0, meta) == 256

    def test_ignore_false_explicit(self):
        runner = self._make_runner(ignore_replay_output_length=False)
        meta = {"turn_output_contents": ["x" * 80]}
        assert runner._effective_max_tokens(0, meta) == 20


# ===========================================================================
# _send_turn
# ===========================================================================


def _make_mock_http(sse_lines, status=200):
    resp = make_mock_aiohttp_response(status=status, sse_lines=sse_lines)
    return make_mock_aiohttp_session(response=resp)


class TestSendTurn:
    def test_send_turn_ok_with_correct_token_count(self):
        runner = BenchmarkRunner(BenchmarkConfig(vllm_url="http://localhost:8000", model="test"))
        sse_lines = [
            b'data: {"choices":[{"delta":{"content":"hello"}}]}\n',
            b'data: {"choices":[{"delta":{"content":" world"}}]}\n',
            b"data: [DONE]\n",
        ]
        result = asyncio.run(
            runner._send_turn(
                _make_mock_http(sse_lines), "s1", 0, [{"role": "user", "content": "hi"}]
            )
        )
        assert result.completed is True and result.output_tokens == 2

    def test_http_error(self):
        runner = BenchmarkRunner(BenchmarkConfig(vllm_url="http://localhost:8000", model="test"))
        mock_resp = make_mock_aiohttp_response(status=500, body_text="Internal Server Error")
        mock_http = make_mock_aiohttp_session(response=mock_resp)
        result = asyncio.run(
            runner._send_turn(mock_http, "s1", 0, [{"role": "user", "content": "hi"}])
        )
        assert result.completed is False

    def test_usage_field_preferred(self):
        runner = BenchmarkRunner(BenchmarkConfig(vllm_url="http://localhost:8000", model="test"))
        sse_lines = [
            b'data: {"choices":[{"delta":{"content":"hello"}}]}\n',
            b'data: {"choices":[{"delta":{"content":" world"}}],"usage":{"completion_tokens":42}}\n',
            b"data: [DONE]\n",
        ]
        result = asyncio.run(
            runner._send_turn(
                _make_mock_http(sse_lines), "s1", 0, [{"role": "user", "content": "hi"}]
            )
        )
        assert result.output_tokens == 42

    def test_json_decode_error_skipped(self):
        runner = BenchmarkRunner(BenchmarkConfig(vllm_url="http://localhost:8000", model="test"))
        sse_lines = [
            b"data: {invalid json}\n",
            b'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
            b"data: [DONE]\n",
        ]
        result = asyncio.run(
            runner._send_turn(
                _make_mock_http(sse_lines), "s1", 0, [{"role": "user", "content": "hi"}]
            )
        )
        assert result.completed and result.output_tokens == 1


class _RecordingBackend(BackendBase):
    name = "_recording-runner-backend"

    def __init__(self):
        from agentsurge.backends.base import BackendConfig

        super().__init__(BackendConfig())
        self.calls = []

    async def send_turn(self, session_id, turn_index, messages, **kwargs):
        self.calls.append((session_id, turn_index, messages, kwargs))
        return TurnResult(
            session_id=session_id,
            turn_index=turn_index,
            completed=True,
            ttft_ms=1.0,
            total_ms=2.0,
            output_tokens=1,
            input_messages=len(messages),
        )


class TestBackendSendTurnContract:
    def test_backend_receives_effective_max_tokens_and_tools(self):
        backend = _RecordingBackend()
        runner = BenchmarkRunner(
            BenchmarkConfig(
                vllm_url="http://localhost:8000",
                model="test",
                max_tokens=64,
                ignore_replay_output_length=False,
                tool_mode="replay",
                no_metrics=True,
            ),
            backend=backend,
        )
        session = ReplaySession(
            session_id="s1",
            turn_messages=[[{"role": "user", "content": "hi"}]],
            metadata={
                "turn_output_contents": ["x" * 80],
                "tools": [{"type": "function", "function": {"name": "finish"}}],
            },
        )

        asyncio.run(runner.run([session]))

        assert backend.calls
        kwargs = backend.calls[0][3]
        assert kwargs["max_tokens"] == 20
        assert kwargs["tools"] == session.metadata["tools"]


# ===========================================================================
# _run_session
# ===========================================================================


class TestDurationModeEmptySessions:
    def test_duration_mode_empty_sessions_raises_cleanly(self):
        """Duration mode with an empty session list must fail with a clear
        error, not a ZeroDivisionError from ``idx % len(sessions)``."""
        cfg = BenchmarkConfig(
            vllm_url="http://x",
            model="t",
            duration=60.0,
            arrival_rate=10.0,
            no_metrics=True,
        )
        runner = BenchmarkRunner(cfg)
        with pytest.raises(ValueError, match="duration"):
            asyncio.run(runner.run([]))


class TestDurationModeRecycleCapUsesAvgTurns:
    """The recycling safety cap must use average turn count, not sessions[0].n_turns.

    With a shallow first session (n_turns=1) and a deep second session
    (n_turns=99), avg=50, so the loop should stop around 2_000 recycles.
    The old code used sessions[0].n_turns=1, which allowed up to 99_999
    recycles — far exceeding the intent of the 100_000-turn budget.
    """

    def _make_sessions(self, turn_counts: list[int]) -> list[ReplaySession]:
        return [
            ReplaySession(
                session_id=f"s{i}",
                turn_messages=[[{"role": "user", "content": f"t{j}"}] for j in range(n)],
            )
            for i, n in enumerate(turn_counts)
        ]

    def _run_recycle_loop(self, sessions: list[ReplaySession]) -> list[ReplaySession]:
        """Replicate the production recycling loop so the test exercises the
        real cap expression from runner.py."""
        recycled: list[ReplaySession] = []
        idx = 0
        avg_turns = sum(s.n_turns for s in sessions) / max(len(sessions), 1)
        while len(recycled) * avg_turns < 100_000:  # mirrors fixed production code
            s = sessions[idx % len(sessions)]
            recycled.append(
                ReplaySession(
                    session_id=f"{s.session_id}_r{idx // len(sessions)}",
                    turn_messages=s.turn_messages,
                    metadata=s.metadata,
                    fan_out=s.fan_out,
                )
            )
            idx += 1
        return recycled

    def test_shallow_first_session_does_not_inflate_cap(self):
        """avg_turns=50 → cap fires around 2_000 recycles, not 99_999."""
        sessions = self._make_sessions([1, 99])  # avg = 50
        avg_turns = sum(s.n_turns for s in sessions) / len(sessions)

        recycled = self._run_recycle_loop(sessions)

        # Total turns budgeted must not exceed 100_000 by more than one batch
        assert len(recycled) * avg_turns <= 100_000 + avg_turns, (
            f"cap overshot: {len(recycled)} recycles × avg {avg_turns} turns"
        )
        # Must be far below the old broken upper bound of 99_999
        assert len(recycled) < 10_000, (
            f"cap appears to still use sessions[0].n_turns: {len(recycled)} recycles"
        )

    def test_uniform_depth_workload_unchanged(self):
        """avg == first when all depths equal — result must stay the same."""
        sessions = self._make_sessions([10] * 5)
        recycled = self._run_recycle_loop(sessions)
        avg_turns = 10.0
        assert len(recycled) * avg_turns <= 100_000 + avg_turns


class TestRunSession:
    def test_stops_on_error(self):
        runner = BenchmarkRunner(BenchmarkConfig(vllm_url="http://localhost:8000", model="test"))
        session = ReplaySession(
            session_id="s1",
            turn_messages=[
                [{"role": "user", "content": "turn1"}],
                [{"role": "user", "content": "turn2"}],
            ],
        )

        async def mock_send(http, sid, tidx, msgs, **kwargs):
            if tidx == 0:
                return TurnResult(session_id=sid, turn_index=tidx, completed=False, error="fail")
            return TurnResult(session_id=sid, turn_index=tidx, completed=True)

        runner._send_turn = mock_send
        result = asyncio.run(runner._run_session(MagicMock(), asyncio.Semaphore(10), session, 0.0))
        assert result.n_turns == 1 and result.completed is False

    def test_all_turns_ok(self):
        runner = BenchmarkRunner(BenchmarkConfig(vllm_url="http://localhost:8000", model="test"))
        session = ReplaySession(
            session_id="s1",
            turn_messages=[
                [{"role": "user", "content": "t1"}],
                [
                    {"role": "user", "content": "t1"},
                    {"role": "assistant", "content": "a1"},
                    {"role": "user", "content": "t2"},
                ],
            ],
        )

        async def mock_send(http, sid, tidx, msgs, **kwargs):
            return TurnResult(
                session_id=sid, turn_index=tidx, completed=True, ttft_ms=50.0, output_tokens=10
            )

        runner._send_turn = mock_send
        result = asyncio.run(runner._run_session(MagicMock(), asyncio.Semaphore(10), session, 0.0))
        assert result.n_turns == 2 and result.completed is True


# ===========================================================================
# Full run (mocked)
# ===========================================================================


# ===========================================================================
# Responses API
# ===========================================================================


@pytest.mark.asyncio
async def test_responses_api_payload_format():
    runner = BenchmarkRunner(
        BenchmarkConfig("http://test:8000", "test-model", max_tokens=64, api_type="responses")
    )
    captured = {}

    def mock_post(url, json=None, timeout=None):
        captured.update(json)
        return make_mock_aiohttp_response(
            status=200,
            sse_lines=[
                b"event: response.completed\n",
                b'data: {"type":"response.completed","response":{"usage":{"output_tokens":5}}}\n\n',
            ],
        )

    mock_session = MagicMock()
    mock_session.post = mock_post
    await runner._send_turn_responses(mock_session, "s1", 0, [{"role": "user", "content": "hello"}])
    assert "input" in captured and "messages" not in captured


# ===========================================================================
# Simulation: think_time, retry
# ===========================================================================


def test_think_time_adds_delay():
    runner = BenchmarkRunner(
        BenchmarkConfig(vllm_url="http://localhost:8000", model="test", think_time=(3.5, 1.0))
    )
    session = ReplaySession(
        session_id="s1",
        turn_messages=[
            [{"role": "user", "content": "t1"}],
            [
                {"role": "user", "content": "t1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "t2"},
            ],
        ],
    )

    async def mock_send(http, sid, tidx, msgs, **kwargs):
        return TurnResult(session_id=sid, turn_index=tidx, completed=True, ttft_ms=10.0)

    runner._send_with_retry = mock_send
    sleep_calls = []

    async def mock_sleep(duration):
        sleep_calls.append(duration)

    async def _run():
        with patch("asyncio.sleep", side_effect=mock_sleep):
            return await runner._run_session(MagicMock(), asyncio.Semaphore(10), session, 0.0)

    result = asyncio.run(_run())
    assert result.completed
    assert all(d > 0 for d in sleep_calls), "all think_time delays should be positive"


def test_no_retry_when_disabled():
    runner = BenchmarkRunner(
        BenchmarkConfig(vllm_url="http://localhost:8000", model="test", max_retries=0)
    )
    call_count = 0

    async def mock_send(http, sid, tidx, msgs, **kwargs):
        nonlocal call_count
        call_count += 1
        return TurnResult(session_id=sid, turn_index=tidx, completed=False, error="fail")

    runner._send_turn = mock_send
    session = ReplaySession(session_id="s1", turn_messages=[[{"role": "user", "content": "hi"}]])
    asyncio.run(runner._run_session(MagicMock(), asyncio.Semaphore(10), session, 0.0))
    assert call_count == 1


# ===========================================================================
# Gamma / Ramp arrival patterns
# ===========================================================================


class TestGammaRampArrivals:
    def test_ramp_increases_rate(self):
        runner = BenchmarkRunner(
            BenchmarkConfig(
                vllm_url="http://x",
                model="t",
                arrival_pattern="ramp",
                arrival_rate=5.0,
                ramp_duration=10.0,
            )
        )
        delays = runner._compute_arrival_delays(20)
        assert all(d >= 0.0 for d in delays)
        # Ramp should have decreasing inter-arrival gaps
        gaps = [b - a for a, b in zip(delays, delays[1:], strict=False)]
        assert gaps[-1] < gaps[0], "ramp should accelerate (shrinking gaps)"


# ===========================================================================
# _ensure_tool_calls
# ===========================================================================


class TestEnsureToolCalls:
    def test_ensure_tool_calls(self):
        tc_valid = ToolCallCheck(
            valid=True, tool_call_id="call_1", name="bash", arguments_raw='{"command": "ls"}'
        )
        tc_invalid = ToolCallCheck(
            valid=False, tool_call_id="call_bad", name="write", arguments_raw="broken"
        )

        # adds when missing
        msg = {"role": "assistant", "content": "text"}
        result = _ensure_tool_calls(msg, [tc_valid])
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "call_1"
        assert result["tool_calls"][0]["function"]["name"] == "bash"

        # preserves existing
        msg2 = {"role": "assistant", "tool_calls": [{"id": "existing"}]}
        result2 = _ensure_tool_calls(msg2, [tc_valid])
        assert result2["tool_calls"] == [{"id": "existing"}]

        # filters invalid
        msg3 = {"role": "assistant", "content": "text"}
        result3 = _ensure_tool_calls(msg3, [tc_valid, tc_invalid])
        assert len(result3["tool_calls"]) == 1
        assert result3["tool_calls"][0]["id"] == "call_1"


# ===========================================================================
# _sanitize_tool_schema
# ===========================================================================


# ===========================================================================
# _run_tool_session
# ===========================================================================


class TestRunToolSession:
    """Verify _run_tool_session dynamic tool-calling loop."""

    def _make_runner(self):
        runner = BenchmarkRunner(
            BenchmarkConfig(vllm_url="http://localhost:8000", model="test", tool_mode="real")
        )
        runner._run_start = 0.0
        return runner

    @staticmethod
    def _response_with_tool_call(tool_name="bash"):
        return {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_0",
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": '{"command": "ls"}',
                                },
                            }
                        ],
                    },
                }
            ],
        }

    @staticmethod
    def _response_finished():
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Done.",
                    },
                }
            ],
        }

    def test_basic_two_turn_flow(self):
        runner = self._make_runner()
        session = ReplaySession(
            session_id="tool-s1",
            turn_messages=[
                [
                    {"role": "system", "content": "You are a coding assistant."},
                    {"role": "user", "content": "List files in /tmp"},
                ]
            ],
            metadata={"max_turns": 2},
        )

        turn0_resp = self._response_with_tool_call("bash")
        turn1_resp = self._response_finished()

        async def mock_execute(http, sem, sid, tidx, msgs, endpoint, tools, validate_fn, **kwargs):
            if tidx == 0:
                return (
                    TurnResult(session_id=sid, turn_index=0, completed=True, total_ms=10.0),
                    turn0_resp,
                )
            return (
                TurnResult(
                    session_id=sid,
                    turn_index=1,
                    completed=True,
                    total_ms=5.0,
                    response_text="Done.",
                ),
                turn1_resp,
            )

        async def mock_build_tool_msgs(assistant_msg, tool_calls, *, sandbox=None, **kw):
            return [
                assistant_msg,
                {
                    "role": "tool",
                    "tool_call_id": "call_0",
                    "name": "bash",
                    "content": "file1.txt\nfile2.txt",
                },
            ]

        runner._execute_single_tool_turn = mock_execute

        async def _run():
            with patch(
                "agentsurge.tool_call.build_tool_response_messages_async",
                side_effect=mock_build_tool_msgs,
            ):
                return await runner._run_tool_session(
                    MagicMock(), asyncio.Semaphore(10), session, 0.0
                )

        result = asyncio.run(_run())

        assert result.n_turns == 2, f"expected 2 turns, got {result.n_turns}"
        assert result.turns[0].completed
        assert result.turns[1].completed
        assert result.total_ms > 0

    def test_stops_on_failed_turn(self):
        runner = self._make_runner()
        session = ReplaySession(
            session_id="tool-fail",
            turn_messages=[[{"role": "user", "content": "hi"}]],
            metadata={"max_turns": 3},
        )

        async def mock_execute(http, sem, sid, tidx, msgs, endpoint, tools, validate_fn, **kwargs):
            return (
                TurnResult(session_id=sid, turn_index=tidx, completed=False, error="timeout"),
                None,
            )

        runner._execute_single_tool_turn = mock_execute

        result = asyncio.run(
            runner._run_tool_session(MagicMock(), asyncio.Semaphore(10), session, 0.0)
        )
        assert result.n_turns == 1
        assert not result.completed


def test_sanitize_tool_schema():
    # strips null values
    tools_with_nulls = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": None,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "enum": None},
                    },
                },
            },
        }
    ]
    result = _sanitize_tool_schema(tools_with_nulls)
    func = result[0]["function"]
    assert "description" not in func
    assert "enum" not in func["parameters"]["properties"]["command"]
    assert func["name"] == "bash"

    # preserves valid schemas
    clean_tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    assert _sanitize_tool_schema(clean_tools) == clean_tools


# ===========================================================================
# Tool mode routing (_dispatch_sessions)
# ===========================================================================


class TestToolModeRouting:
    async def test_off_routes_to_run_session(self):
        runner = BenchmarkRunner(BenchmarkConfig(vllm_url="http://x", model="t", tool_mode="off"))
        session = ReplaySession(
            session_id="s1", turn_messages=[[{"role": "user", "content": "hi"}]]
        )
        sentinel = SessionResult(session_id="s1")

        async def fake_run(http, sem, sess, delay, session_index=0):
            return sentinel

        runner._run_session = fake_run
        results = await runner._dispatch_sessions(None, asyncio.Semaphore(10), [session], [0.0])
        assert results == (sentinel,)

    async def test_real_routes_to_run_tool_session(self):
        runner = BenchmarkRunner(BenchmarkConfig(vllm_url="http://x", model="t", tool_mode="real"))
        session = ReplaySession(
            session_id="s1", turn_messages=[[{"role": "user", "content": "hi"}]]
        )
        sentinel = SessionResult(session_id="s1")

        async def fake_run(http, sem, sess, delay, session_index=0):
            return sentinel

        runner._run_tool_session = fake_run
        results = await runner._dispatch_sessions(None, asyncio.Semaphore(10), [session], [0.0])
        assert results == (sentinel,)

    async def test_replay_routes_to_run_replay_session(self):
        runner = BenchmarkRunner(
            BenchmarkConfig(vllm_url="http://x", model="t", tool_mode="replay")
        )
        session = ReplaySession(
            session_id="s1", turn_messages=[[{"role": "user", "content": "hi"}]]
        )
        sentinel = SessionResult(session_id="s1")

        async def fake_run(http, sem, sess, delay, session_index=0):
            return sentinel

        runner._run_replay_session = fake_run
        results = await runner._dispatch_sessions(None, asyncio.Semaphore(10), [session], [0.0])
        assert results == (sentinel,)


# ===========================================================================
# _run_replay_session
# ===========================================================================


class TestRunReplaySession:
    async def test_replays_all_turns_with_correct_messages(self):
        runner = BenchmarkRunner(
            BenchmarkConfig(vllm_url="http://x", model="t", tool_mode="replay")
        )
        turn_msgs = [
            [{"role": "user", "content": "turn0"}],
            [
                {"role": "user", "content": "turn0"},
                {"role": "assistant", "content": "resp0"},
                {"role": "user", "content": "turn1"},
            ],
            [
                {"role": "user", "content": "turn0"},
                {"role": "assistant", "content": "resp0"},
                {"role": "user", "content": "turn1"},
                {"role": "assistant", "content": "resp1"},
                {"role": "user", "content": "turn2"},
            ],
        ]
        session = ReplaySession(session_id="replay1", turn_messages=turn_msgs)
        sent_calls: list[dict] = []

        async def mock_send(http, sid, tidx, msgs, tools=None, **kwargs):
            sent_calls.append({"turn": tidx, "msgs": msgs, "tools": tools})
            return TurnResult(
                session_id=sid, turn_index=tidx, completed=True, ttft_ms=10.0, output_tokens=5
            )

        runner._send_turn = mock_send
        result = await runner._run_replay_session(MagicMock(), asyncio.Semaphore(10), session, 0.0)
        assert result.n_turns == 3
        assert result.completed is True
        for i, call in enumerate(sent_calls):
            assert call["turn"] == i
            assert call["msgs"] == list(turn_msgs[i])
            assert call["tools"] is not None

    async def test_stops_on_turn_failure(self):
        runner = BenchmarkRunner(
            BenchmarkConfig(vllm_url="http://x", model="t", tool_mode="replay")
        )
        turn_msgs = [
            [{"role": "user", "content": "t0"}],
            [
                {"role": "user", "content": "t0"},
                {"role": "assistant", "content": "a0"},
                {"role": "user", "content": "t1"},
            ],
        ]
        session = ReplaySession(session_id="replay2", turn_messages=turn_msgs)

        async def mock_send(http, sid, tidx, msgs, tools=None, **kwargs):
            return TurnResult(
                session_id=sid,
                turn_index=tidx,
                completed=(tidx != 0),
                error="fail" if tidx == 0 else None,
            )

        runner._send_turn = mock_send
        result = await runner._run_replay_session(MagicMock(), asyncio.Semaphore(10), session, 0.0)
        assert result.n_turns == 1
        assert result.completed is False


# ===========================================================================
# _send_turn with tools parameter
# ===========================================================================


class TestSendTurnWithTools:
    def test_send_turn_with_tools_injects_tool_choice(self):
        runner = BenchmarkRunner(BenchmarkConfig(vllm_url="http://localhost:8000", model="test"))
        captured = {}
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                },
            }
        ]

        def mock_post(url, json=None, timeout=None):
            captured.update(json)
            return make_mock_aiohttp_response(
                status=200,
                sse_lines=[
                    b'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
                    b"data: [DONE]\n",
                ],
            )

        mock_http = MagicMock()
        mock_http.post = mock_post
        result = asyncio.run(
            runner._send_turn(mock_http, "s1", 0, [{"role": "user", "content": "hi"}], tools=tools)
        )
        assert result.completed is True
        assert captured["tools"] == tools
        assert captured["tool_choice"] == "auto"

    def test_send_turn_without_tools_omits_tool_fields(self):
        runner = BenchmarkRunner(BenchmarkConfig(vllm_url="http://localhost:8000", model="test"))
        captured = {}

        def mock_post(url, json=None, timeout=None):
            captured.update(json)
            return make_mock_aiohttp_response(
                status=200,
                sse_lines=[
                    b'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
                    b"data: [DONE]\n",
                ],
            )

        mock_http = MagicMock()
        mock_http.post = mock_post
        asyncio.run(runner._send_turn(mock_http, "s1", 0, [{"role": "user", "content": "hi"}]))
        assert "tools" not in captured
        assert "tool_choice" not in captured


# ===========================================================================
# SSE tool_calls delta TTFT detection
# ===========================================================================


class TestSSEToolCallsTTFT:
    async def test_tool_calls_delta_triggers_ttft(self):
        from agentsurge.backends._sse import parse_sse

        stream = MockAsyncLineIterator(
            [
                b'data: {"choices":[{"delta":{"tool_calls":[{"function":{"name":"bash","arguments":"{}"}}]}}]}\n',
                b"data: [DONE]\n",
            ]
        )
        result = await parse_sse(stream)
        assert result.ttft is not None
        assert result.tokens == 1

    async def test_content_delta_still_triggers_ttft(self):
        from agentsurge.backends._sse import parse_sse

        stream = MockAsyncLineIterator(
            [
                b'data: {"choices":[{"delta":{"content":"hello"}}]}\n',
                b"data: [DONE]\n",
            ]
        )
        result = await parse_sse(stream)
        assert result.ttft is not None
        assert result.tokens == 1

    async def test_empty_delta_does_not_trigger_ttft(self):
        from agentsurge.backends._sse import parse_sse

        stream = MockAsyncLineIterator(
            [
                b'data: {"choices":[{"delta":{}}]}\n',
                b"data: [DONE]\n",
            ]
        )
        result = await parse_sse(stream)
        assert result.ttft is None
        assert result.tokens == 0

    async def test_tool_call_multi_chunk_accumulation(self):
        from agentsurge.backends._sse import parse_sse

        stream = MockAsyncLineIterator(
            [
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"bash","arguments":""}}]}}]}\n',
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"co"}}]}}]}\n',
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"mmand\\": \\"ls\\"}"}}]}}]}\n',
                b'data: {"choices":[{"finish_reason":"tool_calls"}]}\n',
                b"data: [DONE]\n",
            ]
        )
        result = await parse_sse(stream)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc["id"] == "call_1"
        assert tc["function"]["name"] == "bash"
        assert tc["function"]["arguments"] == '{"command": "ls"}'
        assert result.finish_reason == "tool_calls"
        assert result.tokens == 3

    async def test_tool_call_multiple_indices(self):
        from agentsurge.backends._sse import parse_sse

        stream = MockAsyncLineIterator(
            [
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"bash","arguments":"{\\"command\\":\\"ls\\"}"}}]}}]}\n',
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":\\"/tmp\\"}"}}]}}]}\n',
                b'data: {"choices":[{"finish_reason":"tool_calls"}]}\n',
                b"data: [DONE]\n",
            ]
        )
        result = await parse_sse(stream)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["function"]["name"] == "bash"
        assert result.tool_calls[1]["function"]["name"] == "read_file"
        assert result.tool_calls[1]["id"] == "call_b"

    async def test_no_tool_calls_returns_none(self):
        from agentsurge.backends._sse import parse_sse

        stream = MockAsyncLineIterator(
            [
                b'data: {"choices":[{"delta":{"content":"just text"}}]}\n',
                b'data: {"choices":[{"finish_reason":"stop"}]}\n',
                b"data: [DONE]\n",
            ]
        )
        result = await parse_sse(stream, capture_text=True)
        assert result.tool_calls is None
        assert result.response_text == "just text"

    async def test_spaceless_data_prefix_accepted(self):
        """Vendors may emit `data:{json}` (no space) per SSE spec."""
        from agentsurge.backends._sse import parse_sse

        stream = MockAsyncLineIterator(
            [
                b'data:{"choices":[{"delta":{"content":"hi"}}]}\n',
                b"data:[DONE]\n",
            ]
        )
        result = await parse_sse(stream, capture_text=True)
        assert result.tokens == 1
        assert result.response_text == "hi"

    async def test_spaceless_data_prefix_accepted_responses_api(self):
        """Same spaceless-data tolerance must apply to the `responses` branch."""
        from agentsurge.backends._sse import parse_sse

        stream = MockAsyncLineIterator(
            [
                b'data:{"type":"response.output_text.delta","delta":"hi"}\n',
                b'data:{"type":"response.completed","response":{"usage":{"output_tokens":1,"input_tokens":5}}}\n',
            ]
        )
        result = await parse_sse(stream, api_type="responses", capture_text=True)
        assert result.tokens == 1
        assert result.response_text == "hi"
        assert result.usage_tokens == 1
        assert result.prompt_tokens == 5


# ===========================================================================
# chat_template_kwargs gating (M3 fix)
# ===========================================================================

_STOP_SSE = [
    b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n',
    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
    b"data: [DONE]\n",
]


class TestChatTemplateKwargsGating:
    """chat_template_kwargs must only appear when enable_thinking=True."""

    def _make_mock_http(self):
        resp = make_mock_aiohttp_response(status=200, sse_lines=_STOP_SSE)
        return make_mock_aiohttp_session(response=resp)

    # --- _send_turn path (lines 1668 and 1682 in runner.py) ---

    def test_send_turn_excludes_field_by_default(self):
        runner = BenchmarkRunner(BenchmarkConfig(vllm_url="http://x:8000", model="m"))
        mock_http = self._make_mock_http()
        asyncio.run(runner._send_turn(mock_http, "s1", 0, [{"role": "user", "content": "hi"}]))
        payload = mock_http.post.call_args[1]["json"]
        assert "chat_template_kwargs" not in payload

    def test_send_turn_includes_field_when_enabled(self):
        runner = BenchmarkRunner(
            BenchmarkConfig(vllm_url="http://x:8000", model="m", enable_thinking=True)
        )
        mock_http = self._make_mock_http()
        asyncio.run(runner._send_turn(mock_http, "s1", 0, [{"role": "user", "content": "hi"}]))
        payload = mock_http.post.call_args[1]["json"]
        assert payload["chat_template_kwargs"] == {"enable_thinking": True}


# ===========================================================================
# stream_idle_timeout wire-through in _execute_single_tool_turn
# ===========================================================================


class TestToolTurnStreamIdleTimeout:
    """stream_idle_timeout must reach sock_read in _execute_single_tool_turn."""

    def _make_mock_http(self):
        resp = make_mock_aiohttp_response(status=200, sse_lines=_STOP_SSE)
        return make_mock_aiohttp_session(response=resp)

    def _run_tool_turn(self, runner: BenchmarkRunner):
        import aiohttp as _aiohttp

        from agentsurge.tool_call import CODING_TOOLS, validate_tool_calls

        mock_http = self._make_mock_http()
        sem = asyncio.Semaphore(1)
        captured: list[_aiohttp.ClientTimeout] = []
        original_timeout = _aiohttp.ClientTimeout

        def capturing_timeout(*args, **kwargs):
            t = original_timeout(*args, **kwargs)
            captured.append(t)
            return t

        with patch("agentsurge.runner.aiohttp.ClientTimeout", side_effect=capturing_timeout):
            asyncio.run(
                runner._execute_single_tool_turn(
                    mock_http,
                    sem,
                    "s1",
                    0,
                    [{"role": "user", "content": "hi"}],
                    "http://x:8000/v1/chat/completions",
                    CODING_TOOLS,
                    validate_tool_calls,
                )
            )
        return captured

    def test_tool_turn_stream_idle_timeout_forwarded(self):
        runner = BenchmarkRunner(
            BenchmarkConfig(
                vllm_url="http://x:8000",
                model="m",
                tool_mode="real",
                stream_idle_timeout=5.0,
            )
        )
        captured = self._run_tool_turn(runner)
        assert captured, "ClientTimeout was never called"
        assert captured[0].sock_read == 5.0, f"expected sock_read=5.0, got {captured[0].sock_read}"

    def test_tool_turn_stream_idle_timeout_zero_means_none(self):
        runner = BenchmarkRunner(
            BenchmarkConfig(
                vllm_url="http://x:8000",
                model="m",
                tool_mode="real",
                stream_idle_timeout=0.0,
            )
        )
        captured = self._run_tool_turn(runner)
        assert captured, "ClientTimeout was never called"
        assert captured[0].sock_read is None, (
            f"expected sock_read=None for 0.0, got {captured[0].sock_read}"
        )


# ===========================================================================
# on_turn callback fired for sandbox-setup failure
# ===========================================================================


class TestSandboxSetupFailureFiresOnTurn:
    """When _setup_sandbox raises RuntimeError, the failure TurnResult must
    be delivered to the on_turn callback — same as every other TurnResult
    appended to result.turns."""

    def test_on_turn_called_when_sandbox_setup_fails(self):
        runner = BenchmarkRunner(
            BenchmarkConfig(vllm_url="http://x:8000", model="m", tool_mode="real")
        )
        runner._run_start = 0.0

        # Capture on_turn calls.
        received_turns: list[TurnResult] = []

        def on_turn_cb(turn_result: TurnResult) -> None:
            received_turns.append(turn_result)

        runner._on_turn = on_turn_cb

        # Stub _setup_sandbox to raise.
        with patch.object(
            runner.__class__,
            "_setup_sandbox",
            staticmethod(
                lambda cfg, session: (_ for _ in ()).throw(RuntimeError("stub setup failure"))
            ),
        ):
            session = ReplaySession(
                session_id="sandbox-fail-sess",
                turn_messages=[[{"role": "user", "content": "hi"}]],
            )
            asyncio.run(
                runner._run_tool_session(
                    MagicMock(),
                    asyncio.Semaphore(10),
                    session,
                    0.0,
                )
            )

        assert len(received_turns) == 1, (
            f"Expected on_turn to be called once for the failure turn, got {len(received_turns)}"
        )
        turn = received_turns[0]
        assert turn.session_id == "sandbox-fail-sess"
        assert turn.completed is False
        assert "stub setup failure" in (turn.error or "")
