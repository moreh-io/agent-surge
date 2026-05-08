# SPDX-License-Identifier: MIT
"""Mock backend adapter for testing and CI.

Implements :class:`~agentsurge.backends.base.BackendBase` with configurable fake
responses, latency simulation, KV-cache simulation, and programmable failure
injection.  No network calls are made -- everything runs in-process.

Features:

* **Synthetic SSE streaming**: When ``simulate_sse=True``, ``send_turn`` internally
  generates an SSE stream (using :class:`~agentsurge.backends.mock._sse.MockSSEStream`)
  and parses it via :func:`~agentsurge.backends.vllm._sse.stream_sse_ttft`, producing
  realistic timing from actual async delays rather than hard-coded values.
* **KV-cache simulation**: When a :class:`KVSimConfig` is supplied, each turn
  allocates KV-cache blocks proportional to the input token count, tracks
  aggregate usage, and includes KV metrics in ``TurnResult`` timing.

Quick start::

    from agentsurge.backends.mock import MockBackend, MockConfig

    cfg = MockConfig(output_tokens=42, ttft_ms=5.0)
    async with MockBackend(cfg) as backend:
        result = await backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}])
        assert result.completed and result.output_tokens == 42

SSE mode::

    from agentsurge.backends.mock import MockBackend, MockConfig

    cfg = MockConfig(output_tokens=20, ttft_ms=50.0, simulate_sse=True)
    async with MockBackend(cfg) as backend:
        result = await backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}])
        # result.ttft_ms reflects actual measured SSE timing

KV simulation::

    from agentsurge.backends.mock import MockBackend, MockConfig
    from agentsurge.backends.mock._kv_sim import KVSimConfig

    kv_cfg = KVSimConfig(total_blocks=100, block_size=16)
    cfg = MockConfig(output_tokens=10, kv_sim=kv_cfg)
    async with MockBackend(cfg) as backend:
        result = await backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}])
        kv_snap = backend.kv_snapshot()
        kv_usage = kv_snap["kv_usage_perc"]
"""

import asyncio
import copy
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agentsurge.backends.base import BackendBase, BackendConfig
from agentsurge.types import TurnResult
from agentsurge.types.results import _extract_last_input

_log = logging.getLogger(__name__)

#: Signature for ``on_send_turn`` callbacks.
#: Receives ``(session_id, turn_index, messages, config)`` and returns a
#: :class:`TurnResult` **or** ``None`` (to fall through to the default logic).
SendTurnCallback = Callable[
    ...,
    Any,
]


@dataclass
class MockConfig(BackendConfig):
    """Configuration for the mock backend.

    All fields have sensible defaults so a bare ``MockConfig()`` is valid for
    tests that only care about the happy path.

    Parameters
    ----------
    output_tokens : int
        Number of tokens reported in :pyattr:`TurnResult.output_tokens`.
    ttft_ms : float
        Simulated time-to-first-token in milliseconds (injected into
        :pyattr:`TurnResult.ttft_ms`).  Set to ``0`` for instant responses.
    total_ms : float
        Simulated total latency in milliseconds.  Set to ``0`` for instant.
    latency_sec : float
        Actual ``asyncio.sleep`` delay *before* returning the response.  Use
        this to simulate realistic async timing in integration tests.
    fail_turns : set[tuple[str, int]]
        Set of ``(session_id, turn_index)`` pairs that should return
        ``completed=False`` with a synthetic error message.
    fail_after : int
        If > 0, every turn *after* this many total calls returns ``completed=False``.
        Useful for simulating backend degradation.
    error_message : str
        Error string placed in :pyattr:`TurnResult.error` for failed turns.
    healthy : bool
        Value returned by :meth:`health_check`.
    on_send_turn : SendTurnCallback | None
        Optional callback invoked on each :meth:`send_turn`.  If the callback
        returns a :class:`TurnResult`, that result is used directly.  If it
        returns ``None``, the default mock logic runs.
    simulate_sse : bool
        When ``True``, ``send_turn`` generates a synthetic SSE stream and
        parses it to derive realistic timing values (TTFT measured from
        actual async delays).  The ``ttft_ms`` config field controls the
        simulated TTFT delay and ``inter_token_ms`` controls spacing between
        chunks.  When ``False`` (default), the configured ``ttft_ms`` /
        ``total_ms`` values are returned directly (original behavior).
    inter_token_ms : float
        Delay between successive SSE token chunks in milliseconds.  Only
        used when ``simulate_sse=True``.
    api_type : str
        SSE format: ``"chat"`` for chat completions, ``"responses"`` for
        responses API.  Only used when ``simulate_sse=True``.
    kv_sim : object | None
        Optional :class:`~agentsurge.backends.mock._kv_sim.KVSimConfig` to
        enable KV-cache simulation.  When set, each ``send_turn`` call
        allocates KV blocks proportional to the input token count.
    tokens_per_message : int
        Approximate tokens per message for KV simulation (used to estimate
        input tokens from the message count).
    """

    output_tokens: int = 10
    ttft_ms: float = 1.0
    total_ms: float = 5.0
    latency_sec: float = 0.0
    fail_turns: set[tuple[str, int]] = field(default_factory=set)
    fail_after: int = 0
    error_message: str = "mock error"
    healthy: bool = True
    on_send_turn: SendTurnCallback | None = field(default=None, repr=False)
    simulate_sse: bool = False
    inter_token_ms: float = 5.0
    api_type: str = "chat"
    kv_sim: Any = None
    tokens_per_message: int = 50


class MockBackend(BackendBase):
    """In-process mock backend for testing.

    Records all calls in :attr:`call_log` and returns configurable fake
    :class:`TurnResult` instances.  Supports failure injection, latency
    simulation, synthetic SSE streaming, KV-cache simulation, and fully
    programmable responses via callbacks.

    Thread / task safety: ``send_turn`` is safe for concurrent calls -- the
    internal counter uses a simple ``int`` increment which is atomic under
    CPython's GIL and each call reads its own snapshot.
    """

    name: str = "mock"

    def __init__(self, config: MockConfig | None = None) -> None:
        super().__init__(config or MockConfig())
        self.config: MockConfig  # narrow the type for IDE support
        self.call_log: list[dict[str, Any]] = []
        self._call_count: int = 0
        self._kv_sim: Any = None
        if self.config.kv_sim is not None:
            from agentsurge.backends.mock._kv_sim import KVSimulator

            self._kv_sim = KVSimulator(self.config.kv_sim)

    async def __aenter__(self) -> "MockBackend":
        _log.debug("MockBackend entered")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        _log.debug("MockBackend exited")

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
        """Return a fake :class:`TurnResult`.

        The response is determined by (in priority order):

        1. ``config.on_send_turn`` callback (if it returns non-``None``).
        2. ``config.fail_turns`` set membership.
        3. ``config.fail_after`` threshold.
        4. SSE-simulated or default success response.

        When ``simulate_sse=True``, the method generates a synthetic SSE stream
        and measures TTFT from actual async timing.  When KV simulation is
        enabled, blocks are allocated proportional to input token count.
        """
        cfg = self.config

        if cfg.latency_sec > 0:
            await asyncio.sleep(cfg.latency_sec)

        if cfg.on_send_turn is not None:
            cb_result: TurnResult | None = cfg.on_send_turn(session_id, turn_index, messages, cfg)
            if asyncio.iscoroutine(cb_result):
                cb_result = await cb_result
            if cb_result is not None:
                # Callback-overridden calls do not consume fail_after budget
                # and are not appended to call_log -- this lets tests use
                # call_log to count "default-path" calls only.
                return cb_result

        # Increment counter / append to call_log only for calls that fall
        # through to the default mock logic (or to the failure path).
        self._call_count += 1
        call_number = self._call_count

        self.call_log.append(
            {
                "session_id": session_id,
                "turn_index": turn_index,
                "messages": copy.deepcopy(messages),
                "call_number": call_number,
                "tools": copy.deepcopy(tools),
                "max_tokens": max_tokens,
                "capture_text": capture_text,
            }
        )

        _log.debug(
            "MockBackend.send_turn #%d  session=%s turn=%d  msgs=%d",
            call_number,
            session_id,
            turn_index,
            len(messages),
        )

        if (session_id, turn_index) in cfg.fail_turns:
            return self._make_fail(session_id, turn_index, messages)

        if cfg.fail_after > 0 and call_number > cfg.fail_after:
            return self._make_fail(session_id, turn_index, messages)

        kv_alloc = None
        if self._kv_sim is not None:
            est_tokens = len(messages) * cfg.tokens_per_message
            kv_alloc = self._kv_sim.allocate(session_id, est_tokens)

        if cfg.simulate_sse:
            return await self._send_sse(session_id, turn_index, messages, kv_alloc=kv_alloc)

        return TurnResult(
            session_id=session_id,
            turn_index=turn_index,
            completed=True,
            ttft_ms=cfg.ttft_ms,
            total_ms=cfg.total_ms,
            output_tokens=cfg.output_tokens,
            input_messages=len(messages),
            input_tokens=len(messages) * cfg.tokens_per_message,
            input_text=_extract_last_input(messages),
        )

    async def health_check(self) -> bool:
        """Return the configured :pyattr:`MockConfig.healthy` value."""
        return self.config.healthy

    async def _send_sse(
        self,
        session_id: str,
        turn_index: int,
        messages: list[dict[str, Any]],
        kv_alloc: Any = None,
    ) -> TurnResult:
        """Generate and parse a synthetic SSE stream for realistic timing.

        Creates a :class:`MockSSEStream`, feeds it through the same
        :func:`stream_sse_ttft` parser used by the real vLLM backend, and
        returns a :class:`TurnResult` with measured timing values.

        When *kv_alloc* is provided (a :class:`KVAllocation` result), its
        ``prefix_hit_tokens`` is injected as ``prompt_tokens_details.cached_tokens``
        in the SSE usage payload.  ``kv_usage_perc`` is **not** emitted via SSE --
        real vLLM exposes KV utilisation through ``/metrics`` (Prometheus); query
        :meth:`kv_snapshot` instead for in-process reads.
        """
        from agentsurge.backends._sse import parse_sse
        from agentsurge.backends.mock._sse import MockSSEStream

        cfg = self.config

        cached_tokens: int | None = None
        prompt_tokens: int = len(messages) * cfg.tokens_per_message
        if kv_alloc is not None:
            cached_tokens = kv_alloc.prefix_hit_tokens

        stream = MockSSEStream(
            output_tokens=cfg.output_tokens,
            ttft_ms=cfg.ttft_ms,
            inter_token_ms=cfg.inter_token_ms,
            model=cfg.model or "mock-model",
            request_id=f"mock-{session_id}-{turn_index}",
            include_usage=True,
            api_type=cfg.api_type,
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
        )

        t0 = time.monotonic()
        sse = await parse_sse(stream, cfg.api_type)
        total_ms = (time.monotonic() - t0) * 1000

        if sse.parse_errors > 0:
            _log.warning(
                "MockBackend SSE: session=%s turn=%d parse_errors=%d",
                session_id,
                turn_index,
                sse.parse_errors,
            )

        final_tokens = sse.usage_tokens if sse.usage_tokens is not None else sse.tokens
        return TurnResult(
            session_id=session_id,
            turn_index=turn_index,
            completed=True,
            ttft_ms=sse.ttft if sse.ttft is not None else 0.0,
            total_ms=total_ms,
            output_tokens=final_tokens,
            input_messages=len(messages),
            input_tokens=len(messages) * cfg.tokens_per_message,
            input_text=_extract_last_input(messages),
            reasoning_tokens=sse.reasoning_tokens,
            prompt_tokens_server=sse.prompt_tokens,
            cached_tokens=sse.cached_tokens,
        )

    def kv_snapshot(self) -> dict[str, Any]:
        """Return a snapshot of the KV simulator state.

        Returns
        -------
        dict
            Contains ``kv_usage_perc``, ``blocks_used``, ``blocks_free``,
            ``total_blocks``, ``num_sessions``, ``total_evictions``.
            Returns an empty dict if KV simulation is not enabled.
        """
        if self._kv_sim is None:
            return {}
        result: dict[str, Any] = self._kv_sim.snapshot()
        return result

    def _make_fail(
        self,
        session_id: str,
        turn_index: int,
        messages: list[dict[str, Any]],
    ) -> TurnResult:
        return TurnResult(
            session_id=session_id,
            turn_index=turn_index,
            completed=False,
            ttft_ms=0.0,
            total_ms=0.0,
            output_tokens=0,
            input_messages=len(messages),
            error=self.config.error_message,
            input_text=_extract_last_input(messages),
        )

    def reset(self) -> None:
        """Clear call log, counters, and KV simulator state."""
        self.call_log.clear()
        self._call_count = 0
        if self._kv_sim is not None:
            self._kv_sim.reset()
