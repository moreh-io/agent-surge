"""Tests for vLLM and OpenAI-compatible backend adapters - mock HTTP, test adapter logic."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentsurge.backends.base import BackendConfig
from agentsurge.backends.openai import OpenAiBackend
from agentsurge.backends.vllm import VllmBackend, VllmConfig
from tests.conftest import make_mock_aiohttp_response, make_mock_aiohttp_session


def _chat_sse_success(content_chunks: list[str], usage_tokens: int | None = None) -> list[bytes]:
    """Generate SSE byte lines for a successful chat completion response."""
    lines = []
    for chunk_text in content_chunks:
        chunk = {"choices": [{"delta": {"content": chunk_text}}]}
        lines.append(f"data: {json.dumps(chunk)}\n".encode())
    if usage_tokens is not None:
        lines.append(
            f"data: {json.dumps({'usage': {'completion_tokens': usage_tokens}})}\n".encode()
        )
    lines.append(b"data: [DONE]\n")
    return lines


def _responses_sse_success(output_tokens: int = 5) -> list[bytes]:
    """Generate SSE byte lines for a successful responses API."""
    return [
        b"event: response.completed\n",
        f'data: {{"type":"response.completed","response":{{"usage":{{"output_tokens":{output_tokens}}}}}}}\n\n'.encode(),
    ]


# ===========================================================================
# VllmConfig validation
# ===========================================================================


class TestVllmConfig:
    def test_defaults(self):
        cfg = VllmConfig()
        assert cfg.api_type == "chat"
        assert cfg.chat_template_kwargs is None

    def test_responses_api_type(self):
        cfg = VllmConfig(api_type="responses")
        assert cfg.api_type == "responses"

    def test_invalid_api_type_raises(self):
        with pytest.raises(ValueError, match="api_type must be"):
            VllmConfig(api_type="invalid")

    def test_inherits_backend_config_validation(self):
        with pytest.raises(ValueError, match="request_timeout must be positive"):
            VllmConfig(request_timeout=-1)


# ===========================================================================
# VllmBackend lifecycle
# ===========================================================================


class TestVllmBackendLifecycle:
    def test_ensure_http_raises_outside_context(self):
        backend = VllmBackend(VllmConfig(base_url="http://test:8000", model="m"))
        with pytest.raises(RuntimeError, match="async context manager"):
            backend._ensure_http()

    def test_context_manager_creates_and_closes_session(self):
        async def _run():
            with patch("agentsurge.backends._http.aiohttp") as mock_aiohttp:
                mock_session = MagicMock()
                mock_session.close = AsyncMock()
                mock_aiohttp.ClientSession.return_value = mock_session
                mock_aiohttp.TCPConnector.return_value = MagicMock()

                cfg = VllmConfig(base_url="http://test:8000", model="m")
                backend = VllmBackend(cfg)
                async with backend as b:
                    assert b._http is mock_session
                assert b._http is None
                mock_session.close.assert_awaited_once()

        asyncio.run(_run())


# ===========================================================================
# VllmBackend.send_turn - chat API
# ===========================================================================


class TestVllmSendTurnChat:
    def _make_backend(self) -> VllmBackend:
        cfg = VllmConfig(
            base_url="http://test:8000", model="test-model", max_tokens=64, temperature=0.5
        )
        backend = VllmBackend(cfg)
        return backend

    def test_success_with_usage(self):
        backend = self._make_backend()
        sse = _chat_sse_success(["hello", " world"], usage_tokens=42)
        mock_resp = make_mock_aiohttp_response(status=200, sse_lines=sse)
        backend._http = make_mock_aiohttp_session(response=mock_resp)

        result = asyncio.run(backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}]))
        assert result.completed is True
        assert result.session_id == "s1"
        assert result.turn_index == 0
        assert result.output_tokens == 42  # usage_tokens preferred over chunk count
        assert result.input_messages == 1
        assert result.total_ms > 0

    def test_success_without_usage_falls_back_to_chunk_count(self):
        backend = self._make_backend()
        sse = _chat_sse_success(["a", "b", "c"])  # no usage field
        mock_resp = make_mock_aiohttp_response(status=200, sse_lines=sse)
        backend._http = make_mock_aiohttp_session(response=mock_resp)

        result = asyncio.run(backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}]))
        assert result.completed is True
        assert result.output_tokens == 3

    def test_per_turn_max_tokens_and_tools_override_payload(self):
        backend = self._make_backend()
        sse = _chat_sse_success(["ok"], usage_tokens=1)
        mock_resp = make_mock_aiohttp_response(status=200, sse_lines=sse)
        mock_session = make_mock_aiohttp_session(response=mock_resp)
        backend._http = mock_session
        tools = [{"type": "function", "function": {"name": "finish", "parameters": {}}}]

        asyncio.run(
            backend.send_turn(
                "s1",
                0,
                [{"role": "user", "content": "hi"}],
                tools=tools,
                max_tokens=11,
            )
        )
        payload = (
            mock_session.post.call_args.kwargs.get("json") or mock_session.post.call_args[1]["json"]
        )
        assert payload["max_tokens"] == 11
        assert payload["tools"] == tools
        assert payload["tool_choice"] == "auto"

    def test_http_error_returns_failure(self):
        backend = self._make_backend()
        mock_resp = make_mock_aiohttp_response(status=500, body_text="Internal Server Error")
        backend._http = make_mock_aiohttp_session(response=mock_resp)

        result = asyncio.run(backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}]))
        assert result.completed is False
        assert "Internal Server Error" in result.error
        assert result.total_ms > 0

    def test_error_truncated_to_200_chars(self):
        backend = self._make_backend()
        long_error = "x" * 500
        mock_resp = make_mock_aiohttp_response(status=400, body_text=long_error)
        backend._http = make_mock_aiohttp_session(response=mock_resp)

        result = asyncio.run(backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}]))
        assert result.completed is False
        # Error is prefixed with "HTTP <status>: " (~10 extra chars) so the cap
        # is slightly above 200. Intent is "body not dumped unbounded".
        assert len(result.error) <= 220
        assert result.error.startswith("HTTP 400:")

    def test_exception_returns_failure(self):
        backend = self._make_backend()
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.__aenter__ = AsyncMock(side_effect=ConnectionError("refused"))
        mock_session.post = MagicMock(return_value=mock_resp)
        backend._http = mock_session

        result = asyncio.run(backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}]))
        assert result.completed is False
        assert "refused" in result.error


# ===========================================================================
# VllmBackend.send_turn - responses API
# ===========================================================================


class TestVllmSendTurnResponses:
    def test_responses_api_payload(self):
        cfg = VllmConfig(
            base_url="http://test:8000", model="m", api_type="responses", max_tokens=64
        )
        backend = VllmBackend(cfg)
        sse = _responses_sse_success(output_tokens=5)
        mock_resp = make_mock_aiohttp_response(status=200, sse_lines=sse)
        mock_session = make_mock_aiohttp_session(response=mock_resp)
        backend._http = mock_session

        asyncio.run(backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}]))
        payload = (
            mock_session.post.call_args.kwargs.get("json") or mock_session.post.call_args[1]["json"]
        )
        assert "input" in payload
        assert "messages" not in payload
        assert payload["max_output_tokens"] == 64
        url = mock_session.post.call_args[0][0]
        assert url == "http://test:8000/v1/responses"

    def test_responses_api_per_turn_max_tokens_override(self):
        cfg = VllmConfig(
            base_url="http://test:8000", model="m", api_type="responses", max_tokens=64
        )
        backend = VllmBackend(cfg)
        sse = _responses_sse_success(output_tokens=5)
        mock_resp = make_mock_aiohttp_response(status=200, sse_lines=sse)
        mock_session = make_mock_aiohttp_session(response=mock_resp)
        backend._http = mock_session

        asyncio.run(backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}], max_tokens=7))
        payload = (
            mock_session.post.call_args.kwargs.get("json") or mock_session.post.call_args[1]["json"]
        )
        assert payload["max_output_tokens"] == 7


# ===========================================================================
# OpenAiBackend
# ===========================================================================


class TestOpenAiBackendSendTurn:
    def _make_backend(self) -> OpenAiBackend:
        cfg = BackendConfig(
            base_url="http://test:8000", model="gpt-4o", max_tokens=128, temperature=0.7
        )
        backend = OpenAiBackend(cfg)
        return backend

    def test_success(self):
        backend = self._make_backend()
        sse = _chat_sse_success(["hello"], usage_tokens=10)
        mock_resp = make_mock_aiohttp_response(status=200, sse_lines=sse)
        backend._http = make_mock_aiohttp_session(response=mock_resp)

        result = asyncio.run(backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}]))
        assert result.completed is True
        assert result.output_tokens == 10

    def test_stream_options_include_usage_in_payload(self):
        """OpenAI-compatible backend must include stream_options.include_usage so the API
        returns a final usage chunk; without it usage_tokens is always None and
        send_sse_request falls back to counting SSE delta events, which diverges
        from server-reported completion tokens for multi-token strings."""
        backend = self._make_backend()
        sse = _chat_sse_success(["ok"], usage_tokens=1)
        mock_resp = make_mock_aiohttp_response(status=200, sse_lines=sse)
        mock_session = make_mock_aiohttp_session(response=mock_resp)
        backend._http = mock_session

        asyncio.run(backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}]))
        payload = (
            mock_session.post.call_args.kwargs.get("json") or mock_session.post.call_args[1]["json"]
        )
        assert payload.get("stream_options") == {"include_usage": True}
        assert "chat_template_kwargs" not in payload
        assert payload["stream"] is True

    def test_stream_options_not_overridden_by_extra_body(self):
        """extra_body must not overwrite stream_options already in the payload."""
        from agentsurge.backends.base import BackendConfig

        cfg = BackendConfig(
            base_url="http://test:8000",
            model="gpt-4o",
            max_tokens=128,
            temperature=0.7,
            extra_body={"stream_options": {"include_usage": False}},
        )
        backend = OpenAiBackend(cfg)
        sse = _chat_sse_success(["ok"], usage_tokens=1)
        mock_resp = make_mock_aiohttp_response(status=200, sse_lines=sse)
        mock_session = make_mock_aiohttp_session(response=mock_resp)
        backend._http = mock_session

        asyncio.run(backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}]))
        payload = (
            mock_session.post.call_args.kwargs.get("json") or mock_session.post.call_args[1]["json"]
        )
        assert payload.get("stream_options") == {"include_usage": True}

    def test_endpoint_strips_trailing_slash(self):
        cfg = BackendConfig(base_url="http://test:8000/", model="m")
        backend = OpenAiBackend(cfg)
        sse = _chat_sse_success(["ok"], usage_tokens=1)
        mock_resp = make_mock_aiohttp_response(status=200, sse_lines=sse)
        mock_session = make_mock_aiohttp_session(response=mock_resp)
        backend._http = mock_session

        asyncio.run(backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}]))
        url = mock_session.post.call_args[0][0]
        assert url == "http://test:8000/v1/chat/completions"

    def test_http_error(self):
        backend = self._make_backend()
        mock_resp = make_mock_aiohttp_response(status=429, body_text="Rate limited")
        backend._http = make_mock_aiohttp_session(response=mock_resp)

        result = asyncio.run(backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}]))
        assert result.completed is False
        assert "Rate limited" in result.error

    def test_extra_body_merged(self):
        cfg = BackendConfig(
            base_url="http://test:8000", model="m", extra_body={"frequency_penalty": 0.5}
        )
        backend = OpenAiBackend(cfg)
        sse = _chat_sse_success(["ok"], usage_tokens=1)
        mock_resp = make_mock_aiohttp_response(status=200, sse_lines=sse)
        mock_session = make_mock_aiohttp_session(response=mock_resp)
        backend._http = mock_session

        asyncio.run(backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}]))
        payload = (
            mock_session.post.call_args.kwargs.get("json") or mock_session.post.call_args[1]["json"]
        )
        assert payload["frequency_penalty"] == 0.5


class TestOpenAiHealthCheck:
    def test_healthy(self):
        cfg = BackendConfig(base_url="http://test:8000", model="m")
        backend = OpenAiBackend(cfg)
        mock_resp = make_mock_aiohttp_response(status=200)
        mock_session = make_mock_aiohttp_session(response=mock_resp)
        backend._http = mock_session

        assert asyncio.run(backend.health_check()) is True
        url = mock_session.get.call_args[0][0]
        assert "/v1/models" in url

    def test_exception_returns_false(self):
        """OpenAI-compatible backend returns False on health check exception."""
        cfg = BackendConfig(base_url="http://test:8000", model="m")
        backend = OpenAiBackend(cfg)
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=ConnectionError("refused"))
        backend._http = mock_session

        assert asyncio.run(backend.health_check()) is False

    def test_no_session_returns_false(self):
        backend = OpenAiBackend(BackendConfig(base_url="http://test:8000", model="m"))
        assert asyncio.run(backend.health_check()) is False


# ===========================================================================
# Registry
# ===========================================================================


def test_openai_backend_in_registry():
    from agentsurge.backends.registry import backend_registry

    cls = backend_registry.get("openai")
    assert cls is OpenAiBackend
