# SPDX-License-Identifier: MIT
"""vLLM backend adapter for agentsurge.

Implements :class:`~agentsurge.backends.base.BackendBase` by talking to a
vLLM-compatible OpenAI API server over HTTP (``aiohttp``).  Supports both
the **chat completions** (``/v1/chat/completions``) and **responses**
(``/v1/responses``) streaming endpoints.

This is the **default** backend - it is loaded automatically when no
explicit backend is specified.
"""

import logging
from dataclasses import dataclass
from typing import Any

import aiohttp

from agentsurge.backends._http import HttpSessionMixin, send_sse_request
from agentsurge.backends.base import BackendBase, BackendConfig
from agentsurge.types import TurnResult

_log = logging.getLogger(__name__)


@dataclass
class VllmConfig(BackendConfig):
    """vLLM-specific configuration extending :class:`BackendConfig`.

    Parameters
    ----------
    api_type : str
        Which vLLM API to use: ``"chat"`` for ``/v1/chat/completions``
        or ``"responses"`` for ``/v1/responses``.
    chat_template_kwargs : dict | None
        Extra kwargs passed in the ``chat_template_kwargs`` field of the
        chat completions payload (e.g. ``{"enable_thinking": False}`` for
        Qwen3.5).
    """

    api_type: str = "chat"
    chat_template_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.api_type not in ("chat", "responses"):
            raise ValueError(f"api_type must be 'chat' or 'responses', got {self.api_type!r}")


class VllmBackend(HttpSessionMixin, BackendBase):
    """vLLM backend adapter using ``aiohttp`` for streaming HTTP requests.

    Usage::

        cfg = VllmConfig(base_url="http://localhost:8000", model="my-model")
        async with VllmBackend(cfg) as backend:
            result = await backend.send_turn("s1", 0, messages)

    The backend acquires an ``aiohttp.ClientSession`` on enter and releases
    it on exit.  ``send_turn`` is safe to call concurrently from multiple
    asyncio tasks (the underlying connection pool is shared).
    """

    name = "vllm"
    _backend_display_name = "VllmBackend"

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        self._http: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "VllmBackend":
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
        """Send a single conversation turn to the vLLM endpoint.

        Builds the appropriate payload for the configured ``api_type``
        (chat completions or responses API) and streams the SSE response
        to measure TTFT.
        """
        cfg = self.config
        vllm_cfg = cfg if isinstance(cfg, VllmConfig) else None
        api_type = vllm_cfg.api_type if vllm_cfg else "chat"

        if api_type == "responses":
            if tools:
                # Responses-API SSE parser has no tool-call delta handling
                # (see agentsurge/backends/_sse.py: only output_text /
                # reasoning_text / completed events are decoded).  Silently
                # dropping tools would produce TurnResult.tool_calls=None even
                # when the model emitted tool calls, so fail loudly instead.
                raise NotImplementedError(
                    "tools= is not supported with api_type='responses'; "
                    "use api_type='chat' for tool-mode workloads"
                )
            return await self._send_responses(
                session_id,
                turn_index,
                messages,
                max_tokens=max_tokens,
                capture_text=capture_text,
            )
        return await self._send_chat(
            session_id,
            turn_index,
            messages,
            tools=tools,
            max_tokens=max_tokens,
            capture_text=capture_text,
        )

    async def health_check(self) -> bool:
        """Check vLLM server health via ``GET /health``."""
        if self._http is None:
            return False
        try:
            async with self._http.get(
                f"{self.config.base_url.rstrip('/')}/health",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                return resp.status == 200
        except Exception as e:
            _log.debug("health check failed: %s", e)
            return False

    async def _send_chat(
        self,
        session_id: str,
        turn_index: int,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        capture_text: bool = False,
    ) -> TurnResult:
        """Send via ``/v1/chat/completions`` with SSE streaming."""
        cfg = self.config
        vllm_cfg = cfg if isinstance(cfg, VllmConfig) else None

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

        if vllm_cfg and vllm_cfg.chat_template_kwargs:
            payload["chat_template_kwargs"] = vllm_cfg.chat_template_kwargs

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

    async def _send_responses(
        self,
        session_id: str,
        turn_index: int,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        capture_text: bool = False,
    ) -> TurnResult:
        """Send via ``/v1/responses`` with SSE streaming."""
        cfg = self.config
        payload: dict[str, Any] = {
            "model": cfg.model,
            "input": messages,
            "max_output_tokens": max_tokens if max_tokens is not None else cfg.max_tokens,
            "temperature": cfg.temperature,
            "stream": True,
        }
        if cfg.extra_body:
            for k, v in cfg.extra_body.items():
                if k not in payload:
                    payload[k] = v
        endpoint = f"{cfg.base_url.rstrip('/')}/v1/responses"
        return await send_sse_request(
            self._ensure_http(),
            session_id,
            turn_index,
            messages,
            endpoint,
            payload,
            "responses",
            cfg.request_timeout,
            cfg.stream_idle_timeout,
            capture_text,
        )
