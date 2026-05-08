"""Trimmed backend tests: base, mock, vllm, registry, SSE parsing."""

from __future__ import annotations

import asyncio
import json

import pytest

from agentsurge.backends.base import BackendBase, BackendConfig, DefaultRequestAdapter
from agentsurge.backends.mock import MockBackend, MockConfig
from agentsurge.backends.registry import backend_registry
from agentsurge.backends.vllm import VllmBackend
from agentsurge.backends.vllm._sse import stream_sse_ttft
from agentsurge.types import TurnResult

# ===========================================================================
# BackendConfig
# ===========================================================================


def test_backend_config_invalid_timeout():
    with pytest.raises(ValueError, match="request_timeout must be positive"):
        BackendConfig(request_timeout=0)


# ===========================================================================
# BackendBase ABC
# ===========================================================================


class _StubBackend(BackendBase):
    name = "stub"

    async def send_turn(self, session_id, turn_index, messages, **kwargs):
        return TurnResult(
            session_id=session_id,
            turn_index=turn_index,
            completed=True,
            ttft_ms=1.0,
            total_ms=2.0,
            output_tokens=10,
            input_messages=len(messages),
        )


def test_stub_backend_send_turn():
    cfg = BackendConfig()
    backend = _StubBackend(cfg)
    assert backend.name == "stub"
    result = asyncio.run(backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}]))
    assert result.completed is True and result.session_id == "s1"


def test_backend_async_context_manager():
    async def _run():
        async with _StubBackend(BackendConfig()) as backend:
            result = await backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}])
            assert result.completed
            assert result.output_tokens > 0
            assert result.session_id == "s1"

    asyncio.run(_run())


# ===========================================================================
# MockBackend
# ===========================================================================


class TestMockBackendHappyPath:
    def test_send_turn_ok(self):
        cfg = MockConfig(output_tokens=20, ttft_ms=3.0, total_ms=10.0)

        async def _run():
            async with MockBackend(cfg) as backend:
                return await backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}])

        result = asyncio.run(_run())
        assert result.completed is True
        assert result.output_tokens == 20
        assert result.ttft_ms == 3.0

    def test_default_config(self):
        cfg = MockConfig()
        assert cfg.latency_sec == 0.0
        assert cfg.output_tokens == 10

        async def _run():
            async with MockBackend() as backend:
                return await backend.send_turn("s1", 0, [])

        result = asyncio.run(_run())
        assert result.completed is True
        assert result.output_tokens == 10

    def test_multiple_turns_input_messages(self):
        async def _run():
            async with MockBackend(MockConfig(output_tokens=20)) as backend:
                r1 = await backend.send_turn("s1", 0, [{"role": "user", "content": "a"}])
                r2 = await backend.send_turn(
                    "s1",
                    1,
                    [
                        {"role": "user", "content": "a"},
                        {"role": "assistant", "content": "b"},
                        {"role": "user", "content": "c"},
                    ],
                )
                return r1, r2

        r1, r2 = asyncio.run(_run())
        assert r1.input_messages == 1
        assert r2.input_messages == 3


class TestMockFailureInjection:
    def test_fail_turns_specific(self):
        cfg = MockConfig(fail_turns={("s1", 1)}, error_message="injected fail")

        async def _run():
            async with MockBackend(cfg) as backend:
                r0 = await backend.send_turn("s1", 0, [])
                r1 = await backend.send_turn("s1", 1, [])
                return r0, r1

        r0, r1 = asyncio.run(_run())
        assert r0.completed is True
        assert r1.completed is False and r1.error == "injected fail"

    def test_fail_after_threshold(self):
        cfg = MockConfig(fail_after=2)

        async def _run():
            async with MockBackend(cfg) as backend:
                return [await backend.send_turn("s1", i, []) for i in range(4)]

        results = asyncio.run(_run())
        assert results[0].completed and results[1].completed
        assert not results[2].completed and not results[3].completed


def test_mock_health_check():
    assert asyncio.run(MockBackend(MockConfig(healthy=True)).health_check()) is True
    assert asyncio.run(MockBackend(MockConfig(healthy=False)).health_check()) is False


def test_mock_call_log():
    async def _run():
        backend = MockBackend()
        async with backend:
            await backend.send_turn("s1", 0, [{"role": "user", "content": "a"}])
            await backend.send_turn("s2", 1, [{"role": "user", "content": "b"}])
        return backend

    backend = asyncio.run(_run())
    assert len(backend.call_log) == 2
    assert backend.call_log[0]["session_id"] == "s1"


def test_mock_callback_override():
    def custom_cb(session_id, turn_index, messages, config):
        return TurnResult(
            session_id=session_id,
            turn_index=turn_index,
            completed=True,
            ttft_ms=99.9,
            output_tokens=999,
            input_messages=len(messages),
        )

    cfg = MockConfig(on_send_turn=custom_cb)

    async def _run():
        async with MockBackend(cfg) as backend:
            return await backend.send_turn("s1", 0, [{"role": "user", "content": "x"}])

    result = asyncio.run(_run())
    assert result.output_tokens == 999


def test_mock_concurrent_sends():
    cfg = MockConfig(latency_sec=0.01)

    async def _run():
        async with MockBackend(cfg) as backend:
            tasks = [backend.send_turn(f"s{i}", 0, []) for i in range(20)]
            results = await asyncio.gather(*tasks)
            return backend, results

    backend, results = asyncio.run(_run())
    assert len(results) == 20 and all(r.completed for r in results)
    assert all(r.output_tokens > 0 for r in results)


# ===========================================================================
# VllmConfig / VllmBackend
# ===========================================================================


# VllmConfig and VllmBackend basics are tested in test_backend_adapters.py


# ===========================================================================
# SSE parsing helpers
# ===========================================================================


class _MockLineIterator:
    def __init__(self, lines: list[str]):
        self._lines = [line.encode() for line in lines]
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if self._idx >= len(self._lines):
            raise StopAsyncIteration
        line = self._lines[self._idx]
        self._idx += 1
        return line


def _chat_sse_lines(content_chunks, usage_tokens=None):
    lines = []
    for chunk_text in content_chunks:
        chunk = {"choices": [{"delta": {"content": chunk_text}}]}
        lines.append(f"data: {json.dumps(chunk)}\n")
    if usage_tokens is not None:
        lines.append(f"data: {json.dumps({'usage': {'completion_tokens': usage_tokens}})}\n")
    lines.append("data: [DONE]\n")
    return lines


class TestStreamSSETTFT:
    def test_sse_chat_stream_returns_token_count(self):
        """stream_sse_ttft returns (ttft_ms|None, output_tokens, usage_tokens|None, prompt_tokens)."""
        lines = _chat_sse_lines(["hello", " world"], usage_tokens=2)

        async def _run():
            content = _MockLineIterator(lines)
            return await stream_sse_ttft(content, api_type="chat")

        ttft_ms, output_tokens, usage_tokens, prompt_tokens = asyncio.run(_run())
        assert output_tokens == 2
        assert usage_tokens == 2

    def test_chat_stream_done_only(self):
        lines = ["data: [DONE]\n"]

        async def _run():
            content = _MockLineIterator(lines)
            return await stream_sse_ttft(content, api_type="chat")

        ttft_ms, output_tokens, usage_tokens, prompt_tokens = asyncio.run(_run())
        assert output_tokens == 0


# ===========================================================================
# Backend registry
# ===========================================================================


def test_backend_registry_get_mock():
    cls = backend_registry.get("mock")
    assert cls is MockBackend


def test_backend_registry_get_vllm():
    cls = backend_registry.get("vllm")
    assert cls is VllmBackend


def test_backend_registry_unknown_raises():
    with pytest.raises(KeyError):
        backend_registry.get("nonexistent")


# ===========================================================================
# RequestAdapter protocol
# ===========================================================================


def test_default_request_adapter():
    adapter = DefaultRequestAdapter()
    cfg = BackendConfig(model="m", max_tokens=128, temperature=0.5)
    messages = [{"role": "user", "content": "hi"}]
    payload = adapter.adapt(messages, cfg)
    assert payload["model"] == "m"
    assert payload["messages"] == messages
    assert payload["max_tokens"] == 128
