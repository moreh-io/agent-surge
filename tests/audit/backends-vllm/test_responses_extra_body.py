"""H3: vLLM responses-API path must merge cfg.extra_body into the payload."""

from __future__ import annotations

import asyncio

from agentsurge.backends.vllm import VllmBackend, VllmConfig
from tests.conftest import make_mock_aiohttp_response, make_mock_aiohttp_session


def _responses_sse_success(output_tokens: int = 1) -> list[bytes]:
    return [
        b"event: response.completed\n",
        f'data: {{"type":"response.completed","response":{{"usage":{{"output_tokens":{output_tokens}}}}}}}\n\n'.encode(),
    ]


def test_responses_extra_body_merged():
    cfg = VllmConfig(
        base_url="http://test:8000",
        model="m",
        api_type="responses",
        max_tokens=64,
        extra_body={"top_k": 50, "guided_json": {"type": "object"}},
    )
    backend = VllmBackend(cfg)
    sse = _responses_sse_success(output_tokens=1)
    mock_resp = make_mock_aiohttp_response(status=200, sse_lines=sse)
    mock_session = make_mock_aiohttp_session(response=mock_resp)
    backend._http = mock_session

    asyncio.run(backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}]))
    payload = (
        mock_session.post.call_args.kwargs.get("json") or mock_session.post.call_args[1]["json"]
    )
    assert payload["top_k"] == 50
    assert payload["guided_json"] == {"type": "object"}


def test_responses_extra_body_does_not_override_canonical_keys():
    cfg = VllmConfig(
        base_url="http://test:8000",
        model="m",
        api_type="responses",
        max_tokens=64,
        extra_body={"model": "evil-override", "stream": False},
    )
    backend = VllmBackend(cfg)
    sse = _responses_sse_success(output_tokens=1)
    mock_resp = make_mock_aiohttp_response(status=200, sse_lines=sse)
    mock_session = make_mock_aiohttp_session(response=mock_resp)
    backend._http = mock_session

    asyncio.run(backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}]))
    payload = (
        mock_session.post.call_args.kwargs.get("json") or mock_session.post.call_args[1]["json"]
    )
    assert payload["model"] == "m"
    assert payload["stream"] is True
