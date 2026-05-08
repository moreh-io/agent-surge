# SPDX-License-Identifier: MIT
"""SSE (Server-Sent Events) stream parser for vLLM endpoints.

Canonical SSE parser shared by all backends and the runner.  Handles both the
OpenAI-compatible chat-completions and responses API streaming formats.
"""

import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

_log = logging.getLogger(__name__)


class _AsyncLineIterator(Protocol):
    """Minimal protocol for an async iterable of bytes lines."""

    def __aiter__(self) -> AsyncIterator[bytes]: ...


@dataclass(slots=True)
class SSEResult:
    """Parsed results from an SSE stream."""

    ttft: float | None = None
    tokens: int = 0
    reasoning_tokens: int = 0
    usage_tokens: int | None = None
    prompt_tokens: int | None = None  # server-reported input token count
    cached_tokens: int | None = None  # per-request prefix cache hits
    parse_errors: int = 0
    response_text: str = ""
    tool_calls: list[dict] | None = None  # accumulated tool_call deltas
    finish_reason: str = ""


async def parse_sse(
    line_iter: _AsyncLineIterator,
    api_type: str = "chat",
    capture_text: bool = False,
) -> SSEResult:
    """Parse an SSE stream and extract TTFT + token counts.

    This is the **canonical** SSE parser.  All backends and the runner
    delegate to this single implementation.

    Parameters
    ----------
    line_iter
        Async iterator yielding raw ``bytes`` lines (e.g.
        ``aiohttp.ClientResponse.content``).
    api_type
        ``"chat"`` for ``/v1/chat/completions`` or ``"responses"`` for
        ``/v1/responses``.
    capture_text
        When ``True``, accumulates the full response text from content
        deltas (needed for ``use_model_reply_in_next_turn`` injection).

    Returns
    -------
    SSEResult
        Structured result with TTFT, token counts, usage details, and
        optionally the captured response text.
    """
    t0 = time.monotonic()
    r = SSEResult()
    text_parts: list[str] | None = [] if capture_text else None
    tool_call_acc: dict[int, dict] = {}  # index → accumulated tool_call

    async for line in line_iter:
        decoded = line.decode("utf-8", errors="replace").strip()
        if not decoded:
            continue
        if api_type == "responses":
            if decoded.startswith("event:"):
                continue
            # Accept both `data: {...}` and `data:{...}` per SSE spec, matching
            # the chat branch below.
            if not decoded.startswith("data:"):
                continue
            payload = decoded[5:].lstrip(" ")
            try:
                chunk = json.loads(payload)
                event_type = chunk.get("type", "")
                if event_type in (
                    "response.output_text.delta",
                    "response.reasoning_text.delta",
                ):
                    if r.ttft is None:
                        r.ttft = (time.monotonic() - t0) * 1000
                    r.tokens += 1
                    if text_parts is not None:
                        delta_text = chunk.get("delta", "")
                        if delta_text:
                            text_parts.append(delta_text)
                elif event_type == "response.completed":
                    usage = chunk.get("response", {}).get("usage", {})
                    if usage:
                        r.usage_tokens = usage.get("output_tokens", 0)
                        r.prompt_tokens = usage.get("input_tokens")
                        details = usage.get("input_tokens_details") or {}
                        if details.get("cached_tokens") is not None:
                            r.cached_tokens = details["cached_tokens"]
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                # Structural chunk failures (malformed JSON, null fields, missing
                # keys) must not abort the whole stream -- bump parse_errors and
                # keep parsing remaining chunks.
                r.parse_errors += 1
        else:
            # SSE spec permits optional whitespace after the colon, so vendors
            # legally emit either `data: {...}` or `data:{...}` / `data:[DONE]`.
            if not decoded.startswith("data:"):
                continue
            payload = decoded[5:].lstrip(" ")
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                usage = chunk.get("usage")
                if usage:
                    r.usage_tokens = usage.get("completion_tokens", 0)
                    r.prompt_tokens = usage.get("prompt_tokens")
                    details = usage.get("prompt_tokens_details") or {}
                    if details.get("cached_tokens") is not None:
                        r.cached_tokens = details["cached_tokens"]
                choices = chunk.get("choices", [])
                if choices:
                    fr = choices[0].get("finish_reason")
                    if fr:
                        r.finish_reason = fr
                    delta = choices[0].get("delta", {})
                    # Use `in` check rather than truthiness: vLLM emits
                    # delta.content="" (empty string) when Qwen3.5 runs with
                    # enable_thinking=False - bool("") is False and would miss
                    # the first token.  Also accept reasoning_content for
                    # thinking-mode models where content stays empty throughout.
                    has_content = "content" in delta and delta["content"] is not None
                    has_reasoning = (
                        "reasoning_content" in delta and delta["reasoning_content"] is not None
                    )
                    # TTFT semantics: fires on the first delta carrying any
                    # meaningful payload, *including* tool-call deltas. A
                    # tool-only stream is still useful first-byte payload from
                    # the user's perspective (they receive a tool_call action
                    # to dispatch), so we treat tool-call deltas equivalently
                    # to content/reasoning for first-token latency.
                    tool_call_count = len(delta.get("tool_calls") or [])
                    has_tool_calls = tool_call_count > 0
                    if has_content or has_reasoning or has_tool_calls:
                        if r.ttft is None:
                            r.ttft = (time.monotonic() - t0) * 1000
                        # Approximate streamed tokens. For content/reasoning
                        # this is one increment per chunk; for tool-call
                        # deltas we count fan-out (a chunk carrying N parallel
                        # tool-call fragments contributes N units) since each
                        # parallel call is a distinct payload the model
                        # emitted. usage.completion_tokens (when present)
                        # remains the authoritative count via r.usage_tokens.
                        if has_content or has_reasoning:
                            r.tokens += 1
                        if has_tool_calls:
                            r.tokens += tool_call_count
                        if has_reasoning:
                            r.reasoning_tokens += 1
                        if text_parts is not None:
                            c = delta.get("content")
                            if c is None:
                                c = delta.get("reasoning_content", "")
                            text_parts.append(c or "")
                    if has_tool_calls:
                        for tc_delta in delta["tool_calls"]:
                            idx = tc_delta.get("index", 0)
                            if idx not in tool_call_acc:
                                tool_call_acc[idx] = {
                                    "id": tc_delta.get("id", ""),
                                    "type": tc_delta.get("type", "function"),
                                    "function": {"name": "", "arguments": ""},
                                }
                            acc = tool_call_acc[idx]
                            # Some servers (e.g. vendors that split the tool-call
                            # header across deltas) emit `index` + partial name
                            # in the first delta and the `id`/`type` in a later
                            # delta. Merge late-arriving id/type with
                            # last-non-empty-wins so the assembled tool_call has
                            # a usable `tool_call_id` for the role:"tool" reply.
                            late_id = tc_delta.get("id")
                            if late_id and not acc["id"]:
                                acc["id"] = late_id
                            late_type = tc_delta.get("type")
                            if late_type and acc["type"] != late_type:
                                acc["type"] = late_type
                            fn = tc_delta.get("function", {})
                            if fn.get("name"):
                                acc["function"]["name"] = fn["name"]
                            if fn.get("arguments"):
                                acc["function"]["arguments"] += fn["arguments"]
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                # Structural chunk failures (malformed JSON, null fields, missing
                # keys) must not abort the whole stream -- bump parse_errors and
                # keep parsing remaining chunks.
                r.parse_errors += 1

    r.response_text = "".join(text_parts) if text_parts is not None else ""
    if tool_call_acc:
        # Finalize empty arguments to "{}". Some vendors emit
        # `{"function": {"arguments": ""}}` as the only arguments fragment for
        # a no-arg tool call; the falsy `if fn.get("arguments")` guard above
        # never concatenates an empty string, so the final value stays "" and
        # `json.loads("")` then explodes in validate_tool_calls. "{}" is the
        # canonical no-arg representation downstream code already handles.
        for tc in tool_call_acc.values():
            if tc["function"]["arguments"] == "":
                tc["function"]["arguments"] = "{}"
        r.tool_calls = [tool_call_acc[i] for i in sorted(tool_call_acc)]
    return r


async def stream_sse_ttft(
    line_iter: _AsyncLineIterator,
    api_type: str = "chat",
) -> tuple[float | None, int, int | None, int]:
    """Parse an SSE stream and return a backward-compatible 4-tuple.

    This is a thin wrapper around :func:`parse_sse` that preserves the
    original return signature for callers that only need the basic fields.

    Returns
    -------
    tuple[float | None, int, int | None, int]
        ``(ttft_ms, chunk_token_count, usage_tokens, parse_errors)``
    """
    r = await parse_sse(line_iter, api_type, capture_text=False)
    return r.ttft, r.tokens, r.usage_tokens, r.parse_errors
