# SPDX-License-Identifier: MIT
"""Tests for agentsurge.backends._http module."""

import asyncio
import json

import pytest

from agentsurge.backends._http import HttpSessionMixin, send_sse_request
from tests.conftest import make_mock_aiohttp_response, make_mock_aiohttp_session


def _chat_sse_lines(content_chunks: list[str], *, done: bool = True) -> list[bytes]:
    """Build SSE byte lines for a chat-completions stream."""
    lines: list[bytes] = []
    for text in content_chunks:
        chunk = {
            "choices": [{"delta": {"content": text}}],
        }
        lines.append(f"data: {json.dumps(chunk)}\n".encode())
    if done:
        lines.append(b"data: [DONE]\n")
    return lines


# ------------------------------------------------------------------
# send_sse_request
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_sse_request_success():
    sse_lines = _chat_sse_lines(["Hello", " world"])
    resp = make_mock_aiohttp_response(status=200, sse_lines=sse_lines)
    session = make_mock_aiohttp_session(response=resp)

    result = await send_sse_request(
        http=session,
        session_id="s1",
        turn_index=0,
        messages=[{"role": "user", "content": "hi"}],
        endpoint="http://localhost:8000/v1/chat/completions",
        payload={"model": "test"},
    )

    assert result.completed is True
    assert result.session_id == "s1"
    assert result.turn_index == 0
    assert result.output_tokens >= 1
    assert result.total_ms > 0
    assert result.input_messages == 1


@pytest.mark.asyncio
async def test_send_sse_request_http_error():
    resp = make_mock_aiohttp_response(status=500, body_text="Internal Server Error")
    session = make_mock_aiohttp_session(response=resp)

    result = await send_sse_request(
        http=session,
        session_id="s2",
        turn_index=1,
        messages=[],
        endpoint="http://localhost:8000/v1/chat/completions",
        payload={"model": "test"},
    )

    assert result.completed is False
    assert "Internal Server Error" in result.error
    assert result.session_id == "s2"
    assert result.turn_index == 1


@pytest.mark.asyncio
async def test_send_sse_request_timeout():
    from unittest.mock import MagicMock

    session = MagicMock()
    ctx = MagicMock()
    ctx.__aenter__ = MagicMock(side_effect=asyncio.TimeoutError)
    ctx.__aexit__ = MagicMock()
    session.post = MagicMock(return_value=ctx)

    result = await send_sse_request(
        http=session,
        session_id="s3",
        turn_index=0,
        messages=[{"role": "user", "content": "hi"}],
        endpoint="http://localhost:8000/v1/chat/completions",
        payload={"model": "test"},
    )

    assert result.completed is False
    assert result.total_ms >= 0


@pytest.mark.asyncio
async def test_send_sse_request_error_truncated():
    long_error = "x" * 500
    resp = make_mock_aiohttp_response(status=502, body_text=long_error)
    session = make_mock_aiohttp_session(response=resp)

    result = await send_sse_request(
        http=session,
        session_id="s4",
        turn_index=0,
        messages=[],
        endpoint="http://localhost:8000/v1/chat/completions",
        payload={},
    )

    assert result.completed is False
    # Error is prefixed with "HTTP <status>: " (~10 extra chars) so the cap is
    # slightly above 200. The intent of this test is "the server body is not
    # dumped unbounded into error" - 220 preserves that.
    assert len(result.error) <= 220
    assert result.error.startswith("HTTP 502:")


# ------------------------------------------------------------------
# HttpSessionMixin
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_sse_request_tool_calls_propagated():
    """send_sse_request preserves streamed raw tool calls in TurnResult."""
    chunk_with_tool = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"SF"}'},
                        }
                    ]
                }
            }
        ],
    }
    sse_lines = [
        f"data: {json.dumps(chunk_with_tool)}\n".encode(),
        b"data: [DONE]\n",
    ]
    resp = make_mock_aiohttp_response(status=200, sse_lines=sse_lines)
    session = make_mock_aiohttp_session(response=resp)

    result = await send_sse_request(
        http=session,
        session_id="s-tool",
        turn_index=0,
        messages=[{"role": "user", "content": "weather?"}],
        endpoint="http://localhost:8000/v1/chat/completions",
        payload={"model": "test"},
    )

    assert result.completed is True
    assert result.tool_calls == ["get_weather"]
    assert result.tool_calls_detail == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"SF"}'},
        }
    ]


@pytest.mark.asyncio
async def test_send_sse_request_populates_finish_reason():
    """send_sse_request forwards finish_reason from the SSE stream into TurnResult."""
    chunk = {
        "choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}],
    }
    sse_lines = [
        f"data: {json.dumps(chunk)}\n".encode(),
        b"data: [DONE]\n",
    ]
    resp = make_mock_aiohttp_response(status=200, sse_lines=sse_lines)
    session = make_mock_aiohttp_session(response=resp)

    result = await send_sse_request(
        http=session,
        session_id="s-fr",
        turn_index=0,
        messages=[{"role": "user", "content": "say hi"}],
        endpoint="http://localhost:8000/v1/chat/completions",
        payload={"model": "test"},
    )

    assert result.completed is True
    assert result.finish_reason == "stop"


# ------------------------------------------------------------------
# HttpSessionMixin
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_session_mixin_creates_session():
    mixin = HttpSessionMixin()
    async with mixin as ctx:
        assert ctx._http is not None
        session = ctx._ensure_http()
        assert session is ctx._http


@pytest.mark.asyncio
async def test_http_session_mixin_closes_session():
    mixin = HttpSessionMixin()
    async with mixin:
        assert mixin._http is not None
    assert mixin._http is None


@pytest.mark.asyncio
async def test_http_session_mixin_ensure_http_raises_outside_context():
    mixin = HttpSessionMixin()
    with pytest.raises(RuntimeError, match="must be used as an async context manager"):
        mixin._ensure_http()


@pytest.mark.asyncio
async def test_http_session_mixin_ssl_disabled_for_http():
    class FakeConfig:
        base_url = "http://localhost:8000"

    mixin = HttpSessionMixin()
    mixin.config = FakeConfig()
    async with mixin:
        assert mixin._http is not None
    assert mixin._http is None


@pytest.mark.asyncio
async def test_http_session_mixin_ssl_enabled_for_https():
    class FakeConfig:
        base_url = "https://api.example.com"

    mixin = HttpSessionMixin()
    mixin.config = FakeConfig()
    async with mixin:
        assert mixin._http is not None
    assert mixin._http is None


# ------------------------------------------------------------------
# stream_idle_timeout wire-through
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_sse_request_stream_idle_timeout_forwarded():
    """stream_idle_timeout is passed as sock_read to aiohttp.ClientTimeout."""
    from unittest.mock import MagicMock, patch

    import aiohttp

    captured: list[aiohttp.ClientTimeout] = []

    sse_lines = _chat_sse_lines(["hi"])
    resp = make_mock_aiohttp_response(status=200, sse_lines=sse_lines)

    session = MagicMock()
    session.post = MagicMock(return_value=resp)

    original_timeout = aiohttp.ClientTimeout

    def capturing_timeout(*args, **kwargs):
        t = original_timeout(*args, **kwargs)
        captured.append(t)
        return t

    with patch("agentsurge.backends._http.aiohttp.ClientTimeout", side_effect=capturing_timeout):
        await send_sse_request(
            http=session,
            session_id="s-idle",
            turn_index=0,
            messages=[{"role": "user", "content": "hi"}],
            endpoint="http://localhost:8000/v1/chat/completions",
            payload={"model": "test"},
            stream_idle_timeout=5.0,
        )

    assert captured, "ClientTimeout was never called"
    assert captured[0].sock_read == 5.0, f"expected sock_read=5.0, got {captured[0].sock_read}"


@pytest.mark.asyncio
async def test_send_sse_request_stream_idle_timeout_zero_means_none():
    """stream_idle_timeout=0.0 (default) disables sock_read (passes None)."""
    from unittest.mock import MagicMock, patch

    import aiohttp

    captured: list[aiohttp.ClientTimeout] = []

    sse_lines = _chat_sse_lines(["hi"])
    resp = make_mock_aiohttp_response(status=200, sse_lines=sse_lines)

    session = MagicMock()
    session.post = MagicMock(return_value=resp)

    original_timeout = aiohttp.ClientTimeout

    def capturing_timeout(*args, **kwargs):
        t = original_timeout(*args, **kwargs)
        captured.append(t)
        return t

    with patch("agentsurge.backends._http.aiohttp.ClientTimeout", side_effect=capturing_timeout):
        await send_sse_request(
            http=session,
            session_id="s-idle-zero",
            turn_index=0,
            messages=[{"role": "user", "content": "hi"}],
            endpoint="http://localhost:8000/v1/chat/completions",
            payload={"model": "test"},
            stream_idle_timeout=0.0,
        )

    assert captured, "ClientTimeout was never called"
    assert captured[0].sock_read is None, (
        f"expected sock_read=None for 0.0, got {captured[0].sock_read}"
    )


@pytest.mark.asyncio
async def test_vllm_backend_wires_stream_idle_timeout():
    """VllmBackend reads stream_idle_timeout from config and passes to send_sse_request."""
    from unittest.mock import AsyncMock, MagicMock, patch

    import aiohttp

    from agentsurge.backends.vllm import VllmBackend, VllmConfig

    captured: list[aiohttp.ClientTimeout] = []
    original_timeout = aiohttp.ClientTimeout

    def capturing_timeout(*args, **kwargs):
        t = original_timeout(*args, **kwargs)
        captured.append(t)
        return t

    sse_lines = _chat_sse_lines(["hello"])
    resp = MagicMock()
    resp.status = 200
    resp.content = __import__(
        "tests.conftest", fromlist=["MockAsyncLineIterator"]
    ).MockAsyncLineIterator(sse_lines)
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)

    cfg = VllmConfig(
        base_url="http://localhost:8000",
        model="test-model",
        stream_idle_timeout=7.0,
    )
    backend = VllmBackend(cfg)
    backend._http = MagicMock()
    backend._http.post = MagicMock(return_value=resp)

    with patch("agentsurge.backends._http.aiohttp.ClientTimeout", side_effect=capturing_timeout):
        await backend._send_chat("s1", 0, [{"role": "user", "content": "hi"}])

    assert captured, "ClientTimeout was never called"
    assert captured[0].sock_read == 7.0, f"expected sock_read=7.0, got {captured[0].sock_read}"
