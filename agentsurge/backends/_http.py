# SPDX-License-Identifier: MIT
"""Shared HTTP helpers for aiohttp-based backends.

Provides :func:`send_sse_request`, the canonical POST -> error check -> SSE
parse -> TurnResult pipeline shared by all HTTP backends (vLLM,
OpenAI-compatible endpoints, etc.).

Also provides the :class:`HttpSessionMixin` that supplies ``_ensure_http``
and the ``__aenter__``/``__aexit__`` lifecycle for backends that use
``aiohttp.ClientSession``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import aiohttp

from agentsurge.backends._sse import parse_sse
from agentsurge.types import TurnResult
from agentsurge.types.results import _extract_last_input

_log = logging.getLogger(__name__)


async def send_sse_request(
    http: aiohttp.ClientSession,
    session_id: str,
    turn_index: int,
    messages: list[dict[str, Any]],
    endpoint: str,
    payload: dict[str, Any],
    api_type: str = "chat",
    request_timeout: int = 7200,
    stream_idle_timeout: float = 0.0,
    capture_text: bool = False,
) -> TurnResult:
    """POST a streaming request and parse the SSE response into a TurnResult.

    This is the shared implementation used by all aiohttp-based backends.
    It handles:
    - HTTP POST with the given payload
    - Non-200 status -> TurnResult(completed=False)
    - SSE stream parsing via :func:`parse_sse`
    - Exception handling -> TurnResult(completed=False)

    Parameters
    ----------
    http
        Active ``aiohttp.ClientSession``.
    session_id
        Session identifier for the result.
    turn_index
        Turn index for the result.
    messages
        The message list (used only for ``input_messages`` count).
    endpoint
        Full URL to POST to.
    payload
        JSON-serializable request body.
    api_type
        ``"chat"`` or ``"responses"`` -- passed to the SSE parser.
    request_timeout
        Per-request timeout in seconds.
    """
    t0 = time.monotonic()
    try:
        async with http.post(
            endpoint,
            json=payload,
            timeout=aiohttp.ClientTimeout(
                total=request_timeout,
                sock_read=stream_idle_timeout or None,
            ),
        ) as resp:
            if resp.status != 200:
                err = await resp.text()
                return TurnResult(
                    session_id=session_id,
                    turn_index=turn_index,
                    completed=False,
                    total_ms=(time.monotonic() - t0) * 1000,
                    input_messages=len(messages),
                    # Prefix the HTTP status so logs distinguish 429/500/400 at
                    # a glance without re-reading raw server bodies.
                    error=f"HTTP {resp.status}: {err[:200]}",
                    input_text=_extract_last_input(messages),
                )
            sse = await parse_sse(resp.content, api_type, capture_text=capture_text)
            if sse.parse_errors > 0:
                _log.warning(
                    "session=%s turn=%d: %d SSE chunk(s) failed JSON decode",
                    session_id,
                    turn_index,
                    sse.parse_errors,
                )
        total_ms = (time.monotonic() - t0) * 1000
        final_tokens = sse.usage_tokens if sse.usage_tokens is not None else sse.tokens
        empty_response = final_tokens == 0 and sse.ttft is None
        tool_names = None
        if sse.tool_calls:
            tool_names = [
                name for tc in sse.tool_calls if (name := (tc.get("function") or {}).get("name"))
            ]
        return TurnResult(
            session_id=session_id,
            turn_index=turn_index,
            completed=not empty_response,
            ttft_ms=sse.ttft if sse.ttft is not None else 0.0,
            total_ms=total_ms,
            output_tokens=final_tokens,
            input_messages=len(messages),
            reasoning_tokens=sse.reasoning_tokens,
            prompt_tokens_server=sse.prompt_tokens,
            cached_tokens=sse.cached_tokens,
            response_text=sse.response_text,
            tool_calls=tool_names or None,
            tool_calls_detail=sse.tool_calls,
            error="empty response: 0 output tokens" if empty_response else "",
            input_text=_extract_last_input(messages),
            finish_reason=sse.finish_reason or "",
        )
    except Exception as e:
        return TurnResult(
            session_id=session_id,
            turn_index=turn_index,
            completed=False,
            total_ms=(time.monotonic() - t0) * 1000,
            input_messages=len(messages),
            # Prefix the exception class so operators can distinguish
            # TimeoutError / ClientConnectorError / JSONDecodeError etc.
            # from the truncated message body.
            error=f"{type(e).__name__}: {e}"[:200],
            input_text=_extract_last_input(messages),
        )


class HttpSessionMixin:
    """Mixin providing ``_http`` session management and ``_ensure_http``.

    Subclasses get ``__aenter__`` / ``__aexit__`` that create and close an
    ``aiohttp.ClientSession``, and ``_ensure_http()`` that returns the
    session or raises ``RuntimeError`` if not inside the context manager.

    The mixin reads ``self.config.base_url`` (from :class:`BackendBase`)
    to decide SSL settings.  The ``_backend_display_name`` attribute is
    used in the error message and defaults to the class name.
    """

    _http: aiohttp.ClientSession | None = None
    _backend_display_name: str = ""

    async def __aenter__(self) -> HttpSessionMixin:
        cfg = getattr(self, "config", None)
        base_url = getattr(cfg, "base_url", "http://")
        ssl_kwargs: dict = {"ssl": False} if base_url.startswith("http://") else {}
        # Default to a finite connection cap so high-concurrency runners do
        # not exhaust file descriptors / ephemeral ports.  ``0`` on either
        # field disables the corresponding cap (aiohttp semantics).
        limit = getattr(cfg, "connection_limit", 1024)
        limit_per_host = getattr(cfg, "connection_limit_per_host", 0)
        connector = aiohttp.TCPConnector(
            limit=limit,
            limit_per_host=limit_per_host,
            **ssl_kwargs,
        )
        self._http = aiohttp.ClientSession(connector=connector)
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object
    ) -> None:
        if self._http is not None:
            await self._http.close()
            self._http = None

    def _ensure_http(self) -> aiohttp.ClientSession:
        """Return the active HTTP session, raising if not entered."""
        if self._http is None:
            name = self._backend_display_name or type(self).__name__
            raise RuntimeError(
                f"{name} must be used as an async context manager: "
                f"`async with {name}(cfg) as backend: ...`"
            )
        return self._http
