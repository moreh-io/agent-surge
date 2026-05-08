# SPDX-License-Identifier: MIT
"""Synthetic SSE (Server-Sent Events) stream generator for the mock backend.

Produces OpenAI-compatible chat-completions SSE chunks with configurable
inter-token timing.  The output can be consumed by
:func:`agentsurge.backends.vllm._sse.stream_sse_ttft` for integration testing
of the SSE parser against a fully in-process data source.

Usage::

    from agentsurge.backends.mock._sse import MockSSEStream

    stream = MockSSEStream(
        output_tokens=20,
        ttft_ms=50.0,
        inter_token_ms=10.0,
    )

    async for line in stream:
        process(line)  # b'data: {"id":"mock-0", ...}\\n'
"""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass
class MockSSEStream:
    """Async iterator yielding synthetic SSE lines in OpenAI chat-completions format.

    Simulates realistic streaming behavior:

    * A configurable delay before the first token (TTFT).
    * A configurable inter-token delay between subsequent chunks.
    * A final ``data: [DONE]`` sentinel.
    * An optional ``usage`` object in the last content chunk.

    Prefix-cache hits are reported via ``prompt_tokens_details.cached_tokens``
    in the usage object, matching the OpenAI extension format that vLLM
    exposes.  KV utilisation is **not** emitted via SSE -- real vLLM exposes
    it via ``/metrics`` (Prometheus), not the streaming response.

    The generated lines are raw ``bytes`` matching the format expected by
    :func:`agentsurge.backends.vllm._sse.stream_sse_ttft`, so the mock stream
    can be used as a drop-in replacement for ``aiohttp.ClientResponse.content``
    in tests.

    Parameters
    ----------
    output_tokens : int
        Number of content-bearing SSE chunks to emit (each counts as one token).
    ttft_ms : float
        Simulated time-to-first-token delay in milliseconds.
    inter_token_ms : float
        Delay between successive token chunks in milliseconds.
    model : str
        Model name included in the SSE payloads.
    request_id : str
        Request identifier included in the SSE payloads.
    include_usage : bool
        Whether to include a ``usage`` object in the final chunk.
    api_type : str
        ``"chat"`` for chat completions format, ``"responses"`` for the
        responses API format.
    token_text : str
        Text content for each token chunk.  Defaults to a single space.
    prompt_tokens : int
        Number of input (prompt) tokens reported in the usage block.
        Defaults to 10.
    cached_tokens : int | None
        Number of prompt tokens served from the KV prefix cache.  When set,
        ``prompt_tokens_details.cached_tokens`` is included in the usage
        object, mirroring the vLLM / OpenAI prefix-cache reporting format.
        ``None`` omits the ``prompt_tokens_details`` field entirely.
    emit_tool_calls : list[dict] | None
        When set, the chat stream emits ``delta.tool_calls`` chunks (one per
        ``output_token``) accumulating the supplied tool-call records.  Each
        record must follow the OpenAI shape
        ``{"id": str, "function": {"name": str, "arguments": str}}``; the
        ``arguments`` string is split across the ``output_tokens`` chunks so
        the parser exercises its delta-accumulation path.  ``None`` (default)
        disables tool-call emission.
    emit_reasoning : bool
        When ``True``, the chat stream emits ``delta.reasoning_content`` chunks
        instead of ``delta.content`` chunks, exercising the parser's
        reasoning_content / content priority logic.
    """

    output_tokens: int = 10
    ttft_ms: float = 50.0
    inter_token_ms: float = 5.0
    model: str = "mock-model"
    request_id: str = "mock-req-0"
    include_usage: bool = True
    api_type: str = "chat"
    token_text: str = " tok"
    prompt_tokens: int = 10
    cached_tokens: int | None = None
    emit_tool_calls: list[dict] | None = None
    emit_reasoning: bool = False

    def __aiter__(self) -> AsyncIterator[bytes]:
        if self.api_type == "responses":
            return self._iter_responses()
        return self._iter_chat()

    def _build_chat_delta(self, i: int) -> dict:
        """Build the ``choices[0].delta`` dict for chunk index *i*.

        Honours ``emit_tool_calls`` and ``emit_reasoning`` so the mock can
        drive the corresponding parser branches.  Default is content-only.
        """
        if self.emit_tool_calls:
            tool_call_deltas = []
            for idx, tc in enumerate(self.emit_tool_calls):
                args = tc.get("function", {}).get("arguments", "") or ""
                # Split arguments across output_tokens chunks to exercise
                # the parser's delta-accumulation path.
                n = max(1, self.output_tokens)
                chunk_size = max(1, (len(args) + n - 1) // n)
                start = i * chunk_size
                args_slice = args[start : start + chunk_size]
                entry: dict = {"index": idx}
                if i == 0:
                    entry["id"] = tc.get("id", f"call_{idx}")
                    entry["type"] = tc.get("type", "function")
                    fn: dict = {}
                    if name := tc.get("function", {}).get("name"):
                        fn["name"] = name
                    if args_slice:
                        fn["arguments"] = args_slice
                    if fn:
                        entry["function"] = fn
                else:
                    if args_slice:
                        entry["function"] = {"arguments": args_slice}
                tool_call_deltas.append(entry)
            return {"tool_calls": tool_call_deltas}
        if self.emit_reasoning:
            return {"reasoning_content": self.token_text}
        return {"content": self.token_text}

    async def _iter_chat(self) -> AsyncIterator[bytes]:
        """Yield SSE lines in OpenAI chat-completions format."""
        if self.ttft_ms > 0:
            await asyncio.sleep(self.ttft_ms / 1000.0)

        for i in range(self.output_tokens):
            delta = self._build_chat_delta(i)
            chunk: dict = {
                "id": self.request_id,
                "object": "chat.completion.chunk",
                "model": self.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None,
                    }
                ],
            }

            if self.include_usage and i == self.output_tokens - 1:
                usage: dict = {
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.output_tokens,
                    "total_tokens": self.prompt_tokens + self.output_tokens,
                }
                # vLLM-style prefix-cache hit reporting:
                # prompt_tokens_details.cached_tokens mirrors the OpenAI
                # extension format that vLLM uses for prefix-cache stats.
                if self.cached_tokens is not None:
                    usage["prompt_tokens_details"] = {
                        "cached_tokens": self.cached_tokens,
                    }
                chunk["usage"] = usage

            yield (b"data: " + json.dumps(chunk).encode() + b"\n")
            yield b"\n"

            if i < self.output_tokens - 1 and self.inter_token_ms > 0:
                await asyncio.sleep(self.inter_token_ms / 1000.0)

        yield b"data: [DONE]\n"
        yield b"\n"

    async def _iter_responses(self) -> AsyncIterator[bytes]:
        """Yield SSE lines in OpenAI responses API format."""
        if self.ttft_ms > 0:
            await asyncio.sleep(self.ttft_ms / 1000.0)

        for i in range(self.output_tokens):
            event = {
                "type": "response.output_text.delta",
                "delta": self.token_text,
            }
            yield b"event: response.output_text.delta\n"
            yield (b"data: " + json.dumps(event).encode() + b"\n")
            yield b"\n"

            if i < self.output_tokens - 1 and self.inter_token_ms > 0:
                await asyncio.sleep(self.inter_token_ms / 1000.0)

        usage: dict = {
            "input_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
        }
        if self.cached_tokens is not None:
            usage["input_tokens_details"] = {
                "cached_tokens": self.cached_tokens,
            }

        completed: dict = {
            "type": "response.completed",
            "response": {
                "usage": usage,
            },
        }

        yield b"event: response.completed\n"
        yield (b"data: " + json.dumps(completed).encode() + b"\n")
        yield b"\n"
