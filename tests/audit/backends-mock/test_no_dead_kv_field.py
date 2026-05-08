"""M9: MockSSEStream must not expose the dead kv_cache_usage_perc surface.

The original ``kv_cache_usage_perc`` kwarg injected ``vllm_kv_cache_usage_perc``
into the SSE byte stream, but ``parse_sse`` never reads that field -- it was
silently discarded.  Real vLLM emits KV utilisation via /metrics (Prometheus),
not SSE.  Drop the kwarg, the byte injection, and the misleading
"vLLM-extended stats format" claim from the docstring.
"""

from __future__ import annotations

import asyncio
import inspect

from agentsurge.backends.mock._sse import MockSSEStream


def test_mock_sse_has_no_kv_cache_usage_perc_kwarg():
    sig = inspect.signature(MockSSEStream)
    assert "kv_cache_usage_perc" not in sig.parameters, (
        "MockSSEStream still exposes the dead kv_cache_usage_perc kwarg"
    )


def test_mock_sse_chat_does_not_emit_kv_cache_usage_perc():
    stream = MockSSEStream(
        output_tokens=2,
        ttft_ms=0.0,
        inter_token_ms=0.0,
    )

    async def _collect():
        return [chunk async for chunk in stream]

    chunks = asyncio.run(_collect())
    blob = b"".join(chunks)
    assert b"vllm_kv_cache_usage_perc" not in blob


def test_mock_sse_responses_does_not_emit_kv_cache_usage_perc():
    stream = MockSSEStream(
        output_tokens=2,
        ttft_ms=0.0,
        inter_token_ms=0.0,
        api_type="responses",
    )

    async def _collect():
        return [chunk async for chunk in stream]

    chunks = asyncio.run(_collect())
    blob = b"".join(chunks)
    assert b"vllm_kv_cache_usage_perc" not in blob


def test_mock_sse_docstring_no_longer_claims_vllm_extended_stats():
    doc = MockSSEStream.__doc__ or ""
    assert "vLLM-extended stats" not in doc
