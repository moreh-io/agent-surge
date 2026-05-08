# SPDX-License-Identifier: MIT
"""OpenAI-compatible backend adapter for agentsurge.

Implements :class:`~agentsurge.backends.base.BackendBase` by talking to any
OpenAI-compatible API server over HTTP (``aiohttp``).  Supports SSE
streaming via ``/v1/chat/completions``.

The ``openai`` backend name refers to the API shape, not official hosted
OpenAI API auth support.  It is intended for compatible endpoints that do not
require AgentSurge to add provider-specific authentication headers.

Key difference from the vLLM backend:

* No ``/metrics`` endpoint assumed -- latency is measured from SSE streaming
  only.  When used, ``--no-metrics`` is auto-enabled unless the user
  explicitly provides a metrics URL.
* No vLLM-specific fields (``chat_template_kwargs``).
* The payload is a standard OpenAI-compatible chat completions request.
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from agentsurge.backends._http import HttpSessionMixin, send_sse_request
from agentsurge.backends.base import BackendBase, BackendConfig
from agentsurge.types import TurnResult

_log = logging.getLogger(__name__)


class OpenAiBackend(HttpSessionMixin, BackendBase):
    """Generic OpenAI-compatible backend using ``aiohttp`` for streaming HTTP.

    Usage::

        cfg = BackendConfig(base_url="http://localhost:8000", model="my-model")
        async with OpenAiBackend(cfg) as backend:
            result = await backend.send_turn("s1", 0, messages)

    The backend acquires an ``aiohttp.ClientSession`` on enter and releases
    it on exit.  ``send_turn`` is safe to call concurrently from multiple
    asyncio tasks (the underlying connection pool is shared).
    """

    name = "openai"
    _backend_display_name = "OpenAiBackend"

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        self._http: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> OpenAiBackend:
        await HttpSessionMixin.__aenter__(self)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        await HttpSessionMixin.__aexit__(self, exc_type, exc_val, exc_tb)

    async def send_turn(
        self,
        session_id: str,
        turn_index: int,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        capture_text: bool = False,
        session_meta: dict[str, Any] | None = None,
    ) -> TurnResult:
        """Send a single conversation turn to the OpenAI-compatible endpoint.

        Builds a standard chat completions payload and streams the SSE
        response to measure TTFT.  No vLLM-specific fields are included
        (``chat_template_kwargs`` is omitted; ``stream_options`` follows the
        OpenAI-compatible chat completions API and is always set to request a
        final usage chunk).
        """
        cfg = self.config
        payload: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else cfg.max_tokens,
            "temperature": cfg.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if cfg.extra_body:
            for k, v in cfg.extra_body.items():
                if k not in payload:
                    payload[k] = v

        endpoint = f"{cfg.base_url.rstrip('/')}/v1/chat/completions"
        return await send_sse_request(
            self._ensure_http(),
            session_id,
            turn_index,
            messages,
            endpoint,
            payload,
            "chat",
            cfg.request_timeout,
            cfg.stream_idle_timeout,
            capture_text,
        )

    async def health_check(self) -> bool:
        """Strict health check via ``GET /v1/models``.

        Returns ``True`` only if the endpoint replies with HTTP 200.  Returns
        ``False`` on any non-200 status, on any transport exception, and when
        the backend is not inside its async context manager (``self._http is
        None``).  Callers that gate startup on this check should be prepared
        for hosted endpoints that 404 ``/v1/models`` to report ``False``.
        """
        if self._http is None:
            return False
        try:
            async with self._http.get(
                f"{self.config.base_url.rstrip('/')}/v1/models",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                return resp.status == 200
        except Exception as e:
            _log.debug("health check failed: %s", e)
            return False
