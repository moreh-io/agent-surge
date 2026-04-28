# SPDX-License-Identifier: MIT
"""Async benchmark runner - send sessions to vLLM and collect results."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import random
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import aiohttp

from agentsurge.backends._sse import parse_sse

_log = logging.getLogger(__name__)

_tokenizer_cache: dict[tuple[str, bool], object] = {}


def _sanitize_tool_schema(tools: list[dict]) -> list[dict]:
    """Strip null values from tool parameter schemas.

    OpenHands traces contain null enum/property values that break
    vLLM's constrained decoding grammar (tool_choice=required).
    """

    def _clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if v is not None}
        if isinstance(obj, list):
            return [_clean(item) for item in obj]
        return obj

    result: list[dict] = _clean(tools)
    return result


def _load_tokenizer(model: str, *, trust_remote_code: bool = False) -> Any:
    """Load a HuggingFace tokenizer for accurate token counting.

    Returns:
        The tokenizer on success.
        ``None`` if *model* is an HF repo-id and loading fails (soft-fail path).
    Raises:
        RuntimeError: if *model* is a local filesystem path and loading fails.

    Results are cached by (model, trust_remote_code).
    """
    key = (model, trust_remote_code)
    if key in _tokenizer_cache:
        return _tokenizer_cache[key]
    is_local_path = os.path.isabs(model) or model.startswith("./") or model.startswith("../")
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        _tokenizer_cache[key] = tok
        _log.info("Loaded tokenizer for %s", model)
        return tok
    except Exception as e:
        if is_local_path:
            raise RuntimeError(
                f"Failed to load tokenizer from local path: {model!r}\n"
                f"  Underlying error: {e}\n"
                "The path must contain HuggingFace tokenizer files (tokenizer.json or\n"
                "tokenizer.model + tokenizer_config.json).\n"
                "Pass --tokenizer <HF repo-id> to load a tokenizer from the Hub instead,\n"
                "or --ignore-replay-output-length to skip tokenizer-based per-turn sizing\n"
                "(synthetic workloads only)."
            ) from e
        _log.warning("Could not load tokenizer for %s: %s -- pass --tokenizer to specify", model, e)
        # Do NOT cache the None outcome. A transient HF Hub outage that
        # causes one soft-fail must not permanently disable the
        # tokenizer for the rest of the process; subsequent calls
        # should retry and recover once connectivity returns.
        return None


def _stringify_content(m: dict) -> str:
    """Coerce a chat message to a single string for token counting.

    Handles three shapes:
      * ``content`` is a plain string (legacy / vLLM chat completions),
      * ``content`` is OpenAI multipart (list of dicts; sum the ``text`` parts),
      * assistant turn carries only ``tool_calls`` (serialise as JSON).

    Anything else falls back to ``str(content)`` so the chars/4 estimate
    at least reflects payload size rather than silently zeroing.
    """
    content = m.get("content")
    parts: list[str] = []
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            elif isinstance(item, str):
                parts.append(item)
    elif content is not None:
        parts.append(str(content))
    tool_calls = m.get("tool_calls")
    if tool_calls:
        parts.append(json.dumps(tool_calls, ensure_ascii=False))
    return "".join(parts)


def _count_input_tokens(messages: list[dict], tokenizer=None) -> int:
    """Count input tokens using tokenizer if available, else chars÷4.

    Multipart content (list of dicts with ``text`` parts) and assistant
    turns whose payload sits in ``tool_calls`` both contribute to the
    count — silently zeroing them deflates the headroom check and ISL
    totals on tool-loop sessions.
    """
    serialised = [_stringify_content(m) for m in messages]
    total_chars = sum(len(s) for s in serialised)
    if tokenizer is None:
        return total_chars // 4
    try:
        return sum(len(tokenizer.encode(s)) for s in serialised)
    except Exception as e:
        _log.warning("tokenizer encoding failed, falling back to chars/4: %s", e)
        return total_chars // 4


from agentsurge.callbacks import (
    OnSessionCallback,
    OnTurnCallback,
    invoke_on_session,
    invoke_on_turn,
)
from agentsurge.generators.trace_replay import TOOL_OUTPUT_PREFIX
from agentsurge.metrics import MetricsCollector
from agentsurge.types.results import _extract_last_input

_PLACEHOLDER_LINE = "    v = process(data, key=cfg)  # placeholder\n"


def _sample_multitoolcall_delay(tool_names: list[str] | None, rng: random.Random) -> float:
    """Sample inter-turn delay for a turn that may have issued multiple tools.

    The previous implementation only inspected ``tool_calls[0]``: a turn that
    issued ``[think, bash]`` was charged the ``think`` delay (~0.1s) instead
    of the dominating ``bash`` delay (~2s). We aggregate by taking the max
    of per-call samples — the slowest tool dominates the inter-turn gap.
    Other reasonable choices (sum, slowest-category) are documented in the
    audit; ``max`` matches the most common Continuum-style assumption that
    multi-tool calls are dispatched concurrently and gated on the slowest.
    """
    from agentsurge.tool_timing import sample_tool_delay

    if not tool_names:
        return 0.0
    return max(sample_tool_delay(name, rng) for name in tool_names)


def _tool_calls_detail(tool_calls: list) -> list[dict] | None:
    """Build detailed tool call payloads when only validated calls exist."""
    if not tool_calls:
        return None
    return [
        {
            "id": tc.tool_call_id,
            "name": tc.name,
            "arguments": tc.arguments_parsed,
            "arguments_raw": tc.arguments_raw,
            "valid": tc.valid,
            "errors": tc.errors,
        }
        for tc in tool_calls
    ]


def _raw_tool_calls_detail(response: dict) -> list[dict] | None:
    """Return raw OpenAI tool_call payloads from a chat-completions response."""
    choices = response.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    tool_calls = message.get("tool_calls")
    return copy.deepcopy(tool_calls) if tool_calls else None


def _tool_call_names_from_detail(tool_calls: list[dict] | None) -> list[str] | None:
    if not tool_calls:
        return None
    names = []
    for tc in tool_calls:
        func = tc.get("function") if isinstance(tc, dict) else None
        if isinstance(func, dict) and func.get("name"):
            names.append(func["name"])
        elif isinstance(tc, dict) and tc.get("name"):
            names.append(tc["name"])
    return names or None


def _detail_to_raw_tool_call(tc: dict, idx: int) -> dict:
    """Normalize stored raw-or-legacy tool-call detail into OpenAI shape."""
    if isinstance(tc.get("function"), dict):
        return copy.deepcopy(tc)
    name = tc.get("name", "")
    args_raw = tc.get("arguments_raw")
    if args_raw is None:
        args = tc.get("arguments")
        args_raw = json.dumps(args if args is not None else {})
    return {
        "id": tc.get("id") or f"call_{idx}",
        "type": "function",
        "function": {"name": name, "arguments": args_raw},
    }


def _turn_result_to_chat_response(turn_result: TurnResult) -> dict:
    """Build a minimal chat-completions response for backend tool-loop validation."""
    message: dict = {
        "role": "assistant",
        "content": turn_result.response_text or "",
    }
    details = turn_result.tool_calls_detail
    if details:
        message["tool_calls"] = [_detail_to_raw_tool_call(tc, i) for i, tc in enumerate(details)]
    elif turn_result.tool_calls:
        message["tool_calls"] = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
            for i, name in enumerate(turn_result.tool_calls)
        ]

    return {
        "choices": [
            {
                "finish_reason": turn_result.finish_reason
                or ("tool_calls" if message.get("tool_calls") else "stop"),
                "message": message,
            }
        ],
        "usage": {
            "prompt_tokens": turn_result.prompt_tokens_server or turn_result.input_tokens,
            "completion_tokens": turn_result.output_tokens,
        },
    }


def _populate_trace_tool_names(sessions: list) -> None:
    """Extract tool names from trace messages and store in metadata.

    When a loader doesn't set metadata["tools"], the runner needs to know
    which tool names appear in the trace so they can be added to the
    allowed set. This scans once at startup instead of per-session.
    """
    for session in sessions:
        if session.metadata.get("tools") or session.metadata.get("trace_tool_names"):
            continue
        names: set[str] = set()
        for turn_msgs in session.turn_messages:
            for msg in turn_msgs:
                for tc in msg.get("tool_calls", []):
                    name = tc.get("function", {}).get("name", "")
                    if name:
                        names.add(name)
        if names:
            session.metadata["trace_tool_names"] = sorted(names)


def _ensure_tool_calls(message: dict, tool_calls: list) -> dict:
    """Ensure assistant message has tool_calls (for synthesized fallback calls)."""
    if message.get("tool_calls"):
        return message
    message["tool_calls"] = [
        {
            "id": tc.tool_call_id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": tc.arguments_raw,
            },
        }
        for tc in tool_calls
        if tc.valid
    ]
    return message


def _placeholderize_tool_outputs(messages: list[dict]) -> list[dict]:
    """Replace tool output content with char-count-matched placeholder filler.

    Preserves the ``[Tool output: name]`` prefix so that
    :meth:`BenchmarkRunner._is_tool_turn` still detects these as tool turns.
    The body is replaced with deterministic code-like filler of approximately
    the same character count (≈ token count) as the original.
    """
    result: list[dict] = []
    for msg in messages:
        content = msg.get("content", "")
        is_tool = msg.get("role") == "tool" or TOOL_OUTPUT_PREFIX in content
        if not is_tool:
            result.append(msg)
            continue
        # Preserve [Tool output: ...] prefix
        prefix = ""
        body_start = 0
        if TOOL_OUTPUT_PREFIX in content:
            bracket_end = content.find("]", content.index(TOOL_OUTPUT_PREFIX))
            if bracket_end >= 0:
                prefix = content[: bracket_end + 2]  # include "] "
                body_start = bracket_end + 2
        body_len = max(0, len(content) - body_start)
        repeats = body_len // len(_PLACEHOLDER_LINE) + 1
        filler = (_PLACEHOLDER_LINE * repeats)[:body_len]
        result.append({**msg, "content": prefix + filler})
    return result


def _append_followup_user_turn(
    messages: list[dict], assistant_message: dict, user_content: str
) -> None:
    """Append assistant text plus the next user message for a continued turn."""
    messages.append(
        {
            "role": assistant_message.get("role", "assistant"),
            "content": assistant_message.get("content") or "",
        }
    )
    messages.append({"role": "user", "content": user_content})


# Canonical definitions live in agentsurge.types; re-exported here for backward
# compatibility so that ``from agentsurge.runner import BenchmarkConfig`` still works.
from agentsurge.types import (  # noqa: F401
    BenchmarkConfig,
    ReplaySession,
    RunResult,
    SessionResult,
    TurnResult,
)

if TYPE_CHECKING:
    from agentsurge.backends.base import BackendBase, RequestAdapter


from agentsurge.capacity.wasted_store import compute_external_reuse as _compute_external_reuse


def _make_ssl_kwargs(url: str) -> dict:
    """Return TCPConnector kwargs: ``ssl=False`` for HTTP, empty for HTTPS."""
    if url.startswith("http://"):
        return {"ssl": False}
    return {}


class BenchmarkRunner:
    """Run replay sessions against a vLLM endpoint with controlled concurrency.

    Supports two execution modes selected by the *backend* parameter:

    * **HTTP mode** (default, *backend* = ``None``): sends requests directly
      via ``aiohttp`` to the URL in :attr:`BenchmarkConfig.vllm_url`.  This
      is the legacy path and is always available as long as ``aiohttp`` is
      installed.

    * **Backend mode** (*backend* = a :class:`~agentsurge.backends.base.BackendBase`
      instance): routes every ``send_turn`` call through the injected backend
      object.  The backend is entered as an async context manager for the
      duration of the run.  This enables custom inference engines, in-process
      mock backends (for CI), and third-party serving frameworks to be used
      without modifying the runner.

    Parameters
    ----------
    config : BenchmarkConfig
        Run configuration (endpoints, concurrency, timing, …).
    on_turn : callable or None
        Callback invoked after each turn; receives :class:`TurnResult`.
    on_session : callable or None
        Callback invoked after each session; receives :class:`SessionResult`.
    backend : BackendBase or None
        Optional backend adapter.  When provided, all turn requests are
        routed through :meth:`backend.send_turn` instead of raw HTTP.
        ``None`` (default) uses the existing ``aiohttp``-based HTTP path.
    request_adapter : RequestAdapter or None
        Optional request adapter for the **HTTP mode** path.  When set, the
        adapter's :meth:`~agentsurge.backends.base.RequestAdapter.adapt` method
        is called to build the request payload instead of the hardcoded
        OpenAI-compatible payload.  Ignored when *backend* is provided
        (adapters for backend mode are configured on the backend itself).
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        *,
        on_turn: OnTurnCallback | None = None,
        on_session: OnSessionCallback | None = None,
        backend: BackendBase | None = None,
        request_adapter: RequestAdapter | None = None,
    ) -> None:
        self.config = config
        self._rng = random.Random(config.seed)
        self._on_turn = on_turn
        self._on_session = on_session
        self._injected_backend: BackendBase | None = backend
        self._active_backend: BackendBase | None = None
        self._request_adapter: RequestAdapter | None = request_adapter
        if config.skip_tokenizer_load and not config.ignore_replay_output_length:
            raise ValueError(
                "Explicit --tokenizer none requires --ignore-replay-output-length. "
                "Per-turn max_tokens sizing from the replay needs a tokenizer; "
                "omit --tokenizer to use 'auto' (load from --model), pass a tokenizer id/path, "
                "or add --ignore-replay-output-length to use --max-tokens for every turn."
            )
        if config.ignore_replay_output_length or config.skip_tokenizer_load:
            self._tokenizer = None
        else:
            self._tokenizer = _load_tokenizer(
                config.tokenizer_model or config.model,
                trust_remote_code=config.trust_remote_code,
            )
            if self._tokenizer is None:
                raise ValueError(
                    "Per-turn max_tokens sizing from the replay requires a tokenizer. "
                    "Pass --tokenizer <HF model ID>, or "
                    "--ignore-replay-output-length to use --max-tokens for every turn."
                )
        self._run_start: float = 0.0

    async def run(self, sessions: list[ReplaySession]) -> RunResult:
        """Execute all sessions and collect results.

        Automatically selects the execution path based on whether a backend
        was injected at construction time:

        * **Backend mode** (``self._injected_backend is not None``): enters the
          backend as an async context manager, sets ``self._active_backend``,
          then dispatches all ``send_turn`` calls through it.  The ``aiohttp``
          session is not created in this mode.

        * **HTTP mode** (``self._injected_backend is None``): creates an
          ``aiohttp.ClientSession`` and dispatches requests directly to the
          vLLM endpoint.  This is the original behavior and is always used
          when no backend is injected.

        In both modes the :class:`MetricsCollector` runs concurrently to poll
        the vLLM ``/metrics`` endpoint.  For non-vLLM backends the collector
        will silently produce empty snapshots if the endpoint is unreachable.
        """
        cfg = self.config

        if self._injected_backend is None and not cfg.no_metrics:
            await self._preflight_check(cfg)

        original_count = len(sessions)
        if cfg.single_turn_ratio > 0:
            from agentsurge.generators.synthetic import SingleTurnGenerator

            if cfg.single_turn_ratio >= 1.0:
                raise ValueError(
                    "single_turn_ratio must be < 1.0 (use synthetic generator for 100% single-turn)"
                )
            n_single = round(len(sessions) * cfg.single_turn_ratio / (1 - cfg.single_turn_ratio))
            single_gen = SingleTurnGenerator(
                max_model_len=cfg.max_model_len,
                seed=cfg.seed + 1,
                distribution=cfg.context_distribution,
            )
            singles = single_gen.generate(n_single)
            sessions = list(sessions) + singles
            self._rng.shuffle(sessions)
            if len(sessions) != original_count:
                _log.info(
                    "Session count adjusted: %d original + %d single-turn = %d total",
                    original_count,
                    len(sessions) - original_count,
                    len(sessions),
                )

        if cfg.tool_mode in ("real", "replay"):
            if cfg.tool_mode == "real":
                self._maybe_warmstart(sessions, cfg)
            _populate_trace_tool_names(sessions)

        collector = MetricsCollector(
            cfg.vllm_url,
            interval=0.3,
            extra_metrics_urls=cfg.extra_metrics_urls,
            enabled=not cfg.no_metrics,
        )
        semaphore = asyncio.Semaphore(cfg.max_concurrency)
        t_start = time.monotonic()
        self._run_start = t_start

        # Duration mode: recycle sessions until time limit
        if cfg.duration > 0:
            if not sessions:
                raise ValueError("duration mode requires at least one session to recycle")
            _log.info(
                "Duration mode: running for %.0fs (recycling %d sessions)",
                cfg.duration,
                len(sessions),
            )
            recycled: list[ReplaySession] = []
            idx = 0
            avg_turns = sum(s.n_turns for s in sessions) / max(len(sessions), 1)
            while len(recycled) * avg_turns < 100_000:  # safety cap: use avg, not sessions[0]
                s = sessions[idx % len(sessions)]
                # Deep-copy turn_messages and metadata so concurrent recycled
                # clones that call `inject_response` (which mutates message
                # dicts in place) cannot corrupt each other's history.
                recycled.append(
                    ReplaySession(
                        session_id=f"{s.session_id}_r{idx // len(sessions)}",
                        turn_messages=copy.deepcopy(s.turn_messages),
                        metadata=copy.deepcopy(s.metadata),
                        fan_out=s.fan_out,
                    )
                )
                idx += 1
                # Estimate if we have enough sessions for the duration
                est_per_session = max(cfg.ramp_duration, 1.0) / max(cfg.arrival_rate, 1.0)
                if idx * est_per_session > cfg.duration * 1.5:
                    break
            sessions = recycled

        delays = self._compute_arrival_delays(len(sessions))

        if cfg.warm_up > 0 and self._injected_backend is None:
            await self._run_warmup(cfg)

        async with collector:
            if self._injected_backend is not None:
                # Enter the backend as an async context manager so it can
                # acquire its own resources (HTTP session, gRPC channel, ...).
                # _active_backend is visible to _send_with_retry throughout.
                async with self._injected_backend as _backend:
                    self._active_backend = _backend
                    _log.debug(
                        "BenchmarkRunner: using injected backend %r for %d sessions",
                        _backend,
                        len(sessions),
                    )
                    try:
                        session_results = await self._dispatch_sessions(
                            None,
                            semaphore,
                            sessions,
                            delays,
                        )
                    finally:
                        # Always clear even if gather() raised
                        self._active_backend = None
            else:
                connector = aiohttp.TCPConnector(limit=0, **_make_ssl_kwargs(cfg.vllm_url))
                async with aiohttp.ClientSession(connector=connector) as http:
                    session_results = await self._dispatch_sessions(
                        http,
                        semaphore,
                        sessions,
                        delays,
                    )

        elapsed = time.monotonic() - t_start
        t0 = collector.snapshots[0].timestamp if collector.snapshots else 0
        from agentsurge.types import LmcacheV1Metrics

        metrics_timeseries: list[dict[str, object]] = []
        for snap in collector.snapshots:
            preemptions = snap.backend_metrics.get("num_preemptions_total", 0.0)
            entry: dict[str, object] = {
                "t": round(snap.timestamp - t0, 3),
                "kv": round(snap.kv_cache_usage_perc, 4),
                "running": snap.num_requests_running,
                "waiting": snap.num_requests_waiting,
                "preemptions": float(preemptions) if isinstance(preemptions, (int, float)) else 0.0,
            }
            lmc_hits = snap.backend_metrics.get("lmcache_hits", {})
            lmc_evictions = snap.backend_metrics.get("lmcache_evictions", {})
            lmc_mem = snap.backend_metrics.get("lmcache_memory_bytes", {})
            if isinstance(lmc_hits, dict) and lmc_hits:
                entry["lmcache_hits"] = dict(lmc_hits)
            if isinstance(lmc_evictions, dict) and lmc_evictions:
                entry["lmcache_evictions"] = dict(lmc_evictions)
            if isinstance(lmc_mem, dict) and lmc_mem:
                entry["lmcache_memory_bytes"] = dict(lmc_mem)
            v1_raw = snap.backend_metrics.get("lmcache_v1")
            v1: LmcacheV1Metrics = (
                v1_raw if isinstance(v1_raw, LmcacheV1Metrics) else LmcacheV1Metrics()
            )
            if v1.hit_rate > 0:
                entry["lmcache_v1_hit_rate"] = v1.hit_rate
            if v1.local_cache_bytes > 0:
                entry["lmcache_v1_local_cache_bytes"] = v1.local_cache_bytes
            if v1.local_storage_bytes > 0:
                entry["lmcache_v1_local_storage_bytes"] = v1.local_storage_bytes
            if v1.remote_cache_bytes > 0:
                entry["lmcache_v1_remote_cache_bytes"] = v1.remote_cache_bytes
            if v1.cpu_evictions > 0:
                entry["lmcache_v1_cpu_evictions"] = v1.cpu_evictions
            metrics_timeseries.append(entry)
        hit_delta, query_delta = collector.prefix_cache_delta()
        ext_hit_delta, ext_query_delta = collector.external_prefix_cache_delta()
        tier_deltas = collector.lmcache_tier_deltas()
        v1_deltas = collector.lmcache_v1_deltas()
        residual = _compute_external_reuse(collector)

        # Get server-side token totals from last snapshot
        first_snap = collector.snapshots[0] if collector.snapshots else None
        last_snap = collector.snapshots[-1] if collector.snapshots else None

        def _snap_delta(attr: str) -> float:
            if first_snap and last_snap:
                return float(max(0.0, getattr(last_snap, attr) - getattr(first_snap, attr)))
            return 0.0

        isl = _snap_delta("isl_total")
        osl = _snap_delta("osl_total")
        if isl == 0.0 and osl == 0.0:
            all_turns = [t for s in session_results for t in s.turns]
            isl = float(sum(t.input_tokens for t in all_turns))
            osl = float(sum(t.output_tokens for t in all_turns))

        return RunResult(
            sessions=list(session_results),
            kv_util_peak=collector.kv_util_peak,
            running_peak=collector.running_peak,
            waiting_peak=collector.waiting_peak,
            prefix_cache_hit_delta=hit_delta,
            prefix_cache_query_delta=query_delta,
            external_prefix_cache_hit_delta=ext_hit_delta,
            external_prefix_cache_query_delta=ext_query_delta,
            preemption_total=collector.preemption_delta(),
            isl_total=isl,
            osl_total=osl,
            total_elapsed_s=elapsed,
            server_ttft_sum=_snap_delta("server_ttft_sum"),
            server_ttft_count=_snap_delta("server_ttft_count"),
            prefill_time_sum=_snap_delta("prefill_time_sum"),
            prefill_time_count=_snap_delta("prefill_time_count"),
            queue_time_sum=_snap_delta("queue_time_sum"),
            queue_time_count=_snap_delta("queue_time_count"),
            lmcache_tier_deltas=tier_deltas,
            lmcache_v1_deltas=v1_deltas,
            metrics_timeseries=metrics_timeseries,
            external_reuse=residual,
            config={
                **cfg.to_run_result_dict(),
                "n_sessions": len(sessions),
                "env_pythonhashseed": os.environ.get("PYTHONHASHSEED", "not_set"),
            },
        )

    def _compute_arrival_delays(self, n: int) -> list[float]:
        """Pre-compute cumulative arrival delays for all sessions.

        For Poisson arrivals the delays are cumulative sums of exponential
        inter-arrival times, so sessions are spread over time rather than
        all sleeping an independent (non-cumulative) amount and arriving
        in a burst.
        """
        cfg = self.config
        if cfg.arrival_pattern == "burst":
            return [0.0] * n
        elif cfg.arrival_pattern == "poisson":
            delays: list[float] = []
            cumulative = 0.0
            for _ in range(n):
                cumulative += self._rng.expovariate(cfg.arrival_rate)
                delays.append(cumulative)
            return delays
        elif cfg.arrival_pattern == "constant":
            return [i / max(cfg.arrival_rate, 1.0) for i in range(n)]
        elif cfg.arrival_pattern == "gamma":
            # Gamma inter-arrival times with CV>1 for bursty arrivals
            # (ServeGen, arXiv:2505.09999)
            # shape = 1/CV^2, scale = CV^2/rate
            shape = 1.0 / (cfg.gamma_cv**2)
            scale = (cfg.gamma_cv**2) / max(cfg.arrival_rate, 0.1)
            gamma_delays: list[float] = []
            cumulative = 0.0
            for _ in range(n):
                cumulative += self._rng.gammavariate(shape, scale)
                gamma_delays.append(cumulative)
            return gamma_delays
        elif cfg.arrival_pattern == "ramp":
            # Linearly increasing rate from 0 to arrival_rate over ramp_duration.
            # After ramp, continues at arrival_rate. Models gradual load increase
            # for finding the throughput knee.
            ramp_dur = max(cfg.ramp_duration, 1.0)
            target_rate = max(cfg.arrival_rate, 0.1)
            ramp_delays: list[float] = []
            cumulative = 0.0
            for _i in range(n):
                # Current rate increases linearly: rate(t) = target_rate * min(t / ramp_dur, 1.0)
                progress = min(cumulative / ramp_dur, 1.0)
                current_rate = max(target_rate * progress, 0.1)
                cumulative += self._rng.expovariate(current_rate)
                ramp_delays.append(cumulative)
            return ramp_delays
        return [0.0] * n

    async def _dispatch_sessions(
        self,
        http: aiohttp.ClientSession | None,
        semaphore: asyncio.Semaphore,
        sessions: list[ReplaySession],
        delays: list[float],
    ) -> tuple[SessionResult, ...]:
        """Create tasks for all sessions, gather and return their results.

        Both the backend and HTTP execution paths share this logic - the only
        difference is the *http* argument passed to :meth:`_run_session`
        (``None`` for the backend path, a live :class:`aiohttp.ClientSession`
        for the HTTP path).
        """
        mode = self.config.tool_mode
        if mode == "real":
            run_fn = self._run_tool_session
        elif mode == "replay":
            run_fn = self._run_replay_session
        else:
            run_fn = self._run_session
        # Frontend dispatch: direct mode uses run_fn above; any other frontend
        # routes through the frontend execution path. Task D wires real providers.
        if self.config.frontend_name != "direct":
            run_fn = self._run_session_frontend
        tasks = [
            asyncio.create_task(run_fn(http, semaphore, session, delays[i], session_index=i))
            for i, session in enumerate(sessions)
        ]
        # return_exceptions=True so one crashing session does not cancel its
        # in-flight siblings; we still fire on_session for the failed ones so
        # downstream bookkeeping (inflight dump, metrics) sees every session.
        raw = await asyncio.gather(*tasks, return_exceptions=True)
        results: list[SessionResult] = []
        for session, item in zip(sessions, raw, strict=True):
            if isinstance(item, asyncio.CancelledError):
                raise item
            if isinstance(item, BaseException):
                _log.exception(
                    "Session %s crashed with %s; emitting placeholder result",
                    session.session_id,
                    type(item).__name__,
                    exc_info=item,
                )
                failed = SessionResult(
                    session_id=session.session_id,
                    metadata={
                        **session.metadata,
                        "error": f"{type(item).__name__}: {item}",
                        "failed": True,
                    },
                )
                await invoke_on_session(self._on_session, failed)
                results.append(failed)
            else:
                results.append(item)
        return tuple(results)

    async def _preflight_check(self, cfg: BenchmarkConfig) -> None:
        """Verify server is reachable and model is available before measurement."""
        url = f"{cfg.vllm_url}/v1/models"
        try:
            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(**_make_ssl_kwargs(cfg.vllm_url))
            ) as http:
                async with http.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        raise ConnectionError(
                            f"Preflight: GET {url} returned {resp.status}. "
                            "Check --vllm-url and that the server is running."
                        )
                    data = await resp.json()
                    models = [m.get("id", "") for m in data.get("data", [])]
                    if cfg.model not in models:
                        raise ValueError(
                            f"Preflight: model {cfg.model!r} not found on server. "
                            f"Available: {', '.join(models) or '(none)'}. "
                            "Check --model matches the served model name."
                        )
                    _log.info("Preflight OK: model %s available", cfg.model)
        except (ConnectionError, ValueError):
            raise
        except Exception as e:
            raise ConnectionError(
                f"Server unreachable at {cfg.vllm_url}: {e}. "
                "Check --vllm-url and that the server is running."
            ) from e

    async def _run_warmup(self, cfg: BenchmarkConfig) -> None:
        """Send warm-up requests to prime JIT/CUDA graphs before measurement."""
        _log.info("Sending %d warm-up request(s)...", cfg.warm_up)
        connector = aiohttp.TCPConnector(limit=0, **_make_ssl_kwargs(cfg.vllm_url))
        async with aiohttp.ClientSession(connector=connector) as http:
            warmup_msgs = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say OK."},
            ]
            payload = {
                "model": cfg.model,
                "messages": warmup_msgs,
                "max_tokens": 8,
                "temperature": 0.0,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            endpoint = f"{cfg.vllm_url}/v1/chat/completions"
            for i in range(cfg.warm_up):
                try:
                    async with http.post(
                        endpoint,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as resp:
                        async for _ in resp.content:
                            pass
                except Exception as e:
                    _log.warning("Warm-up request %d failed: %s", i, e)
        _log.info("Warm-up complete")

    async def _run_session(
        self,
        http: aiohttp.ClientSession | None,
        semaphore: asyncio.Semaphore,
        session: ReplaySession,
        delay: float,
        session_index: int = 0,
    ) -> SessionResult:
        """Run all turns of a session sequentially."""
        if delay > 0:
            await asyncio.sleep(delay)

        result = SessionResult(session_id=session.session_id, metadata=session.metadata)
        result.expected_turns = len(session.turn_messages)
        result.start_time = time.monotonic() - self._run_start
        t_start = time.monotonic()
        llm_total = 0.0
        # Per-session RNG for deterministic delay regardless of scheduling
        sess_rng = random.Random(self.config.seed + session_index)

        cfg = self.config
        use_placeholder = cfg.tool_output_mode == "placeholder"
        for turn_idx, messages in enumerate(session.turn_messages):
            if use_placeholder:
                messages = _placeholderize_tool_outputs(messages)
            # Layered delay: precomputed > tool-type > think_time
            if turn_idx > 0:
                delay = self._compute_turn_delay(session, turn_idx, sess_rng)
                if delay > 0:
                    await asyncio.sleep(delay)

            fan_out = getattr(session, "fan_out", 1)
            if fan_out > 1:
                # Fire fan_out parallel requests, each acquiring the semaphore independently.
                # Each call is recorded as a separate TurnResult for full observability.
                async def _fan_call(fi: int, _msgs=messages, _tidx=turn_idx) -> TurnResult:
                    async with semaphore:
                        return await self._send_with_retry(
                            http,
                            f"{session.session_id}_fan{fi}",
                            _tidx,
                            _msgs,
                            session_meta=session.metadata,
                        )

                sub_results = await asyncio.gather(*[_fan_call(fi) for fi in range(fan_out)])
                for sr in sub_results:
                    sr.fan_out_count = fan_out
                    result.turns.append(sr)
                    llm_total += sr.total_ms
                    await invoke_on_turn(self._on_turn, sr)
                # Session continues if any fan-out call succeeded
                completed_results = [r for r in sub_results if r.completed]
                if completed_results:
                    turn_result = completed_results[
                        0
                    ]  # use first-ok for use_model_reply_in_next_turn
                else:
                    turn_result = sub_results[0]
            else:
                async with semaphore:
                    turn_result = await self._send_with_retry(
                        http,
                        session.session_id,
                        turn_idx,
                        messages,
                        session_meta=session.metadata,
                    )
                result.turns.append(turn_result)
                llm_total += turn_result.total_ms
                await invoke_on_turn(self._on_turn, turn_result)

            if not turn_result.completed:
                break

            # Interruption: discard result and re-request (R3: acquire semaphore)
            if cfg.interruption_rate > 0:
                if sess_rng.random() < cfg.interruption_rate:
                    turn_result.interrupted = True
                    async with semaphore:
                        turn_result_2 = await self._send_with_retry(
                            http,
                            session.session_id,
                            turn_idx,
                            messages,
                            session_meta=session.metadata,
                        )
                    result.turns.append(turn_result_2)
                    llm_total += turn_result_2.total_ms
                    await invoke_on_turn(self._on_turn, turn_result_2)
                    turn_result = turn_result_2

            # Inject actual LLM response into next turn so intra-session GPU
            # prefix cache hits are measured correctly.
            if (
                cfg.use_model_reply_in_next_turn
                and turn_result.completed
                and turn_result.response_text
            ):
                session.inject_response(turn_idx, turn_result.response_text)

        result.total_ms = (time.monotonic() - t_start) * 1000
        result.llm_ms = llm_total
        result.end_time = time.monotonic() - self._run_start
        await invoke_on_session(self._on_session, result)
        return result

    async def _run_session_frontend(
        self,
        http: aiohttp.ClientSession | None,
        semaphore: asyncio.Semaphore,
        session: ReplaySession,
        delay: float,
        session_index: int = 0,
    ) -> SessionResult:
        """Frontend-mode session execution: spawn provider subprocess and parse events."""
        from agentsurge.frontends.runner import FrontendSessionRenderer

        if delay > 0:
            await asyncio.sleep(delay)

        renderer = getattr(self, "_frontend_renderer", None)
        if renderer is None:
            fr = self.config.frontend
            ws = fr.workspace_dir if fr is not None else None
            if ws:
                workspace_root = Path(ws)
            else:
                workspace_root = (
                    Path("frontend-runs") / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
                )
            workspace = workspace_root / "sessions"
            workspace.mkdir(parents=True, exist_ok=True)
            renderer = FrontendSessionRenderer(self.config, workspace)
            self._frontend_renderer = renderer
            self._frontend_workspace = workspace

        async with semaphore:
            result = await renderer.run(session, session_index=session_index)
        await invoke_on_session(self._on_session, result)
        return result

    async def _execute_single_tool_turn(
        self,
        http: aiohttp.ClientSession | None,
        semaphore: asyncio.Semaphore,
        session_id: str,
        turn_idx: int,
        messages: list[dict],
        endpoint: str,
        coding_tools: list[dict],
        validate_tool_calls,
        *,
        session_meta: dict | None = None,
    ) -> tuple[TurnResult, dict | None]:
        """Execute one LLM turn in a tool-calling session.

        Returns (turn_result, response_json).  response_json is None when
        the request failed.
        """
        cfg = self.config
        effective_max_tokens = self._effective_max_tokens(turn_idx, session_meta)
        if cfg.max_model_len > 0:
            input_tok = _count_input_tokens(messages, self._tokenizer)
            headroom = cfg.max_model_len - input_tok
            if headroom < effective_max_tokens:
                effective_max_tokens = max(1, headroom)

        if self._active_backend is not None:
            async with semaphore:
                tr = await self._send_with_retry(
                    None,
                    session_id,
                    turn_idx,
                    messages,
                    tools=coding_tools,
                    session_meta=session_meta,
                    max_tokens=effective_max_tokens,
                    capture_text=True,
                )
            response_json = _turn_result_to_chat_response(tr) if tr.completed else None
            if response_json is not None:
                validation = validate_tool_calls(response_json, turn_idx)
                raw_detail = _raw_tool_calls_detail(response_json)
                tr.tool_calls = _tool_call_names_from_detail(raw_detail)
                tr.tool_calls_detail = raw_detail
                tr.tool_valid = validation.valid
            return tr, response_json

        use_stream = not cfg.no_stream
        payload = {
            "model": cfg.model,
            "messages": messages,
            "tools": coding_tools,
            "tool_choice": "auto",
            "max_tokens": effective_max_tokens,
            "temperature": cfg.temperature,
            "stream": use_stream,
        }
        if cfg.enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": True}
        if use_stream:
            payload["stream_options"] = {"include_usage": True}
        if cfg.extra_body:
            payload.update(cfg.extra_body)

        if http is None:
            raise TypeError("_execute_single_tool_turn requires an HTTP session")
        t0 = time.monotonic()
        _idle = self.config.stream_idle_timeout or None
        async with semaphore:
            try:
                async with http.post(
                    endpoint,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=cfg.request_timeout, sock_read=_idle),
                ) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        tr = TurnResult(
                            session_id=session_id,
                            turn_index=turn_idx,
                            completed=False,
                            total_ms=(time.monotonic() - t0) * 1000,
                            input_messages=len(messages),
                            input_tokens=_count_input_tokens(messages, self._tokenizer),
                            error=f"HTTP {resp.status}: {err[:200]}",
                            input_text=_extract_last_input(messages),
                        )
                        return tr, None
                    if use_stream:
                        sse = await parse_sse(resp.content, "chat", capture_text=True)
                    else:
                        response_json = cast(dict[str, Any], await resp.json())
            except Exception as e:
                tr = TurnResult(
                    session_id=session_id,
                    turn_index=turn_idx,
                    completed=False,
                    total_ms=(time.monotonic() - t0) * 1000,
                    input_messages=len(messages),
                    input_tokens=_count_input_tokens(messages, self._tokenizer),
                    error=f"{type(e).__name__}: {e}"[:200],
                    input_text=_extract_last_input(messages),
                )
                return tr, None

        total_ms = (time.monotonic() - t0) * 1000

        if use_stream:
            final_tokens = sse.usage_tokens if sse.usage_tokens is not None else sse.tokens
            input_tok = (
                sse.prompt_tokens
                if sse.prompt_tokens is not None
                else _count_input_tokens(messages, self._tokenizer)
            )
            response_json = {
                "choices": [
                    {
                        "finish_reason": sse.finish_reason or "stop",
                        "message": {
                            "content": sse.response_text or None,
                            "tool_calls": sse.tool_calls,
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": sse.prompt_tokens or 0,
                    "completion_tokens": final_tokens,
                    **(
                        {"prompt_tokens_details": {"cached_tokens": sse.cached_tokens}}
                        if sse.cached_tokens is not None
                        else {}
                    ),
                },
            }
            ttft_ms = sse.ttft if sse.ttft is not None else 0.0
            response_text = sse.response_text
            prompt_tokens_server = sse.prompt_tokens
            cached_tokens = sse.cached_tokens
        else:
            assert response_json is not None
            usage = response_json.get("usage", {})
            final_tokens = usage.get("completion_tokens", 0)
            input_tok = usage.get("prompt_tokens", _count_input_tokens(messages, self._tokenizer))
            ttft_ms = total_ms
            response_text = (response_json.get("choices") or [{}])[0].get("message", {}).get(
                "content"
            ) or ""
            prompt_tokens_server = usage.get("prompt_tokens")
            cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")

        validation = validate_tool_calls(response_json, turn_idx)
        empty_response = (final_tokens == 0 and sse.ttft is None) if use_stream else False

        tr = TurnResult(
            session_id=session_id,
            turn_index=turn_idx,
            completed=not empty_response,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            output_tokens=final_tokens,
            reasoning_tokens=sse.reasoning_tokens
            if use_stream
            else (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
            input_messages=len(messages),
            input_tokens=input_tok,
            prompt_tokens_server=prompt_tokens_server,
            cached_tokens=cached_tokens,
            tool_calls=[tc.name for tc in validation.tool_calls] if validation.tool_calls else None,
            tool_calls_detail=_raw_tool_calls_detail(response_json),
            tool_valid=validation.valid,
            response_text=response_text,
            input_text=_extract_last_input(messages),
            finish_reason=(
                sse.finish_reason
                if use_stream
                else ((response_json.get("choices") or [{}])[0].get("finish_reason") or "")
            )
            or "",
        )
        return tr, response_json

    @staticmethod
    def _run_sandbox_eval(sandbox, session_metadata: dict, result_metadata: dict) -> None:
        """Run eval checks in the execution workspace and record results."""
        result_metadata["workspace_path"] = str(sandbox.root)
        result_metadata["sandbox_path"] = str(sandbox.root)
        result_metadata["files_created"] = sandbox.list_files()

        check_cmds = session_metadata.get("check_commands", [])
        if check_cmds:
            check_results = []
            all_passed = True
            for cmd in check_cmds:
                out = sandbox.run_command(f"{cmd} && echo __PASS__", timeout=30)
                passed = "__PASS__" in out
                check_results.append({"cmd": cmd, "passed": passed, "output": out.strip()})
                if not passed:
                    all_passed = False
            result_metadata["check_results"] = check_results
            result_metadata["task_completed"] = all_passed

    @staticmethod
    def _maybe_warmstart(
        sessions: list[ReplaySession],
        cfg: BenchmarkConfig,
    ) -> None:
        """Clone repos into a shared execution workspace before the timed run begins."""
        raw_workspace_dir = getattr(cfg, "workspace_dir", None) or getattr(cfg, "sandbox_dir", None)
        workspace_dir = Path(raw_workspace_dir) if raw_workspace_dir else None
        if workspace_dir and workspace_dir.exists() and any(workspace_dir.iterdir()):
            _log.info(
                "Warmstart: workspace_dir %s already populated, skipping.",
                workspace_dir,
            )
            return

        _log.warning(
            "Warmstart: cloning repos from GitHub for %d sessions. "
            "Pass --tool-workspace-dir with pre-populated workspaces to skip.",
            len(sessions),
        )

        from agentsurge.warmstart import prepare_execution_workspace

        manifest = prepare_execution_workspace(sessions, workspace_dir=workspace_dir)

        if workspace_dir is None:
            cfg.workspace_dir = manifest["workspace_dir"]
            cfg.sandbox_dir = manifest["workspace_dir"]

    @staticmethod
    def _setup_sandbox(cfg, session):
        """Create and initialize an execution workspace for a tool session.

        Returns (workspace, tmp_dir_obj).  Raises RuntimeError on clone failure.
        """
        from agentsurge.tool_call import ExecutionWorkspace

        sandbox = None
        tmp_dir_obj = None
        tool_env = getattr(cfg, "tool_env", "safe")
        workspace_dir = getattr(cfg, "workspace_dir", None) or getattr(cfg, "sandbox_dir", None)
        if workspace_dir:
            session_dir = os.path.join(workspace_dir, session.session_id)
            if os.path.isdir(session_dir):
                sandbox = ExecutionWorkspace.from_directory(session_dir, tool_env=tool_env)

        if sandbox is None:
            import tempfile

            tmp_dir_obj = tempfile.TemporaryDirectory(prefix=f"agentsurge_{session.session_id}_")
            sandbox = ExecutionWorkspace(tmp_dir_obj.name, tool_env=tool_env)
            instance_id = session.metadata.get("instance_id", "")
            if instance_id and not session.metadata.get("setup_files"):
                sandbox.setup_from_instance_id(instance_id, session.metadata)

        setup_files = session.metadata.get("setup_files")
        if setup_files:
            sandbox.setup_files(setup_files)
        setup_cmds = session.metadata.get("setup_commands")
        if setup_cmds:
            for cmd in setup_cmds:
                sandbox.run_command(cmd)

        return sandbox, tmp_dir_obj

    async def _run_tool_session(
        self,
        http: aiohttp.ClientSession | None,
        semaphore: asyncio.Semaphore,
        session: ReplaySession,
        delay: float,
        session_index: int = 0,
    ) -> SessionResult:
        """Run a dynamic tool-calling agent session.

        Unlike _run_session which replays pre-built turn_messages, this method
        builds messages dynamically: send to LLM → parse tool_calls → execute
        tool → append results → repeat until finish or max turns.
        """
        from agentsurge.tool_call import (
            CODING_TOOLS,
            build_tool_response_messages_async,
            looks_like_unparsed_tool_call,
            synthesize_tool_calls_from_text,
            validate_tool_calls,
        )

        if delay > 0:
            await asyncio.sleep(delay)

        cfg = self.config
        result = SessionResult(session_id=session.session_id, metadata=session.metadata)
        result.expected_turns = len(session.turn_messages)
        result.start_time = time.monotonic() - self._run_start
        t_start = time.monotonic()
        llm_total = 0.0

        messages: list[dict] = list(session.turn_messages[0]) if session.turn_messages else []

        sandbox = None
        _tmp_dir_obj = None
        setup_ms = 0.0
        if cfg.tool_mode == "real":
            try:
                t_setup = time.monotonic()
                sandbox, _tmp_dir_obj = self._setup_sandbox(cfg, session)
                setup_ms = (time.monotonic() - t_setup) * 1000
            except RuntimeError as exc:
                _log.error(
                    "session=%s workspace setup failed: %s",
                    session.session_id,
                    exc,
                    exc_info=exc,
                )
                _failure_turn = TurnResult(
                    session_id=session.session_id,
                    turn_index=0,
                    completed=False,
                    total_ms=0.0,
                    error=f"{type(exc).__name__}: {exc}"[:500],
                    input_text=_extract_last_input(messages),
                )
                result.turns.append(_failure_turn)
                await invoke_on_turn(self._on_turn, _failure_turn)
                result.total_ms = 0.0
                result.setup_ms = (time.monotonic() - t_setup) * 1000
                result.end_time = time.monotonic() - self._run_start
                await invoke_on_session(self._on_session, result)
                return result

        # Reset t_start AFTER setup - total_ms measures only the agent loop
        t_start = time.monotonic()

        session_tools = _sanitize_tool_schema(session.metadata.get("tools") or CODING_TOOLS)
        allowed_tool_names = {
            t.get("function", {}).get("name")
            for t in session_tools
            if isinstance(t, dict) and isinstance(t.get("function"), dict)
        }
        allowed_tool_names.discard(None)

        trace_names = session.metadata.get("trace_tool_names")
        if trace_names:
            allowed_tool_names.update(trace_names)

        max_turns = session.metadata.get("max_turns", len(session.turn_messages))
        pending_queue: list[str] = list(session.pending_user_messages or [])
        max_turns = max_turns + len(pending_queue)
        endpoint = f"{cfg.vllm_url}/v1/chat/completions"
        sess_rng = random.Random(cfg.seed + session_index)
        recent_text_only: list[str] = []
        consecutive_invalid = 0
        exhaust_reason: str | None = None
        accumulated_cost_usd = 0.0

        for turn_idx in range(max_turns):
            if cfg.max_session_time > 0 and (time.monotonic() - t_start) >= cfg.max_session_time:
                exhaust_reason = "max_session_time"
                if result.turns:
                    result.turns[-1].error = result.turns[-1].error or "max_session_time"
                break
            if cfg.max_budget_usd > 0 and accumulated_cost_usd >= cfg.max_budget_usd:
                exhaust_reason = "max_budget_usd"
                if result.turns:
                    result.turns[-1].error = result.turns[-1].error or "max_budget_usd"
                break
            _turn_wall_start = (time.monotonic() - t_start) * 1000
            if cfg.tool_delay and turn_idx > 0:
                prior_tool_names = (
                    result.turns[-1].tool_calls
                    if (result.turns and result.turns[-1].tool_calls)
                    else None
                )
                d = _sample_multitoolcall_delay(prior_tool_names, sess_rng)
                if d > 0:
                    await asyncio.sleep(d)

            turn_result, response_json = await self._execute_single_tool_turn(
                http,
                semaphore,
                session.session_id,
                turn_idx,
                messages,
                endpoint,
                session_tools,
                lambda response, idx: validate_tool_calls(
                    response,
                    turn_index=idx,
                    allowed_tool_names=allowed_tool_names,
                    sanitize_truncated=cfg.sanitize_truncated_tool_calls,
                ),
                session_meta=session.metadata,
            )
            turn_result.wall_start_ms = _turn_wall_start
            result.turns.append(turn_result)
            llm_total += turn_result.total_ms
            if cfg.max_budget_usd > 0:
                accumulated_cost_usd += (
                    turn_result.input_tokens * cfg.input_price_per_mtok
                    + turn_result.output_tokens * cfg.output_price_per_mtok
                ) / 1_000_000
            await invoke_on_turn(self._on_turn, turn_result)

            if not turn_result.completed:
                turn_result.wall_end_ms = (time.monotonic() - t_start) * 1000
                break

            if response_json is None:
                raise RuntimeError(
                    f"session={session.session_id} turn={turn_idx}: completed but no response"
                )
            validation = validate_tool_calls(
                response_json,
                turn_index=turn_idx,
                allowed_tool_names=allowed_tool_names,
                sanitize_truncated=cfg.sanitize_truncated_tool_calls,
            )

            if not validation.has_tool_calls:
                synthetic_tool_calls = synthesize_tool_calls_from_text(
                    turn_result.response_text or "",
                    allowed_tool_names=allowed_tool_names,
                    mode=cfg.tool_call_parser_fallback,
                )
                if synthetic_tool_calls:
                    _log.info(
                        "session=%s turn=%d: synthesized %d tool call(s) from text fallback",
                        session.session_id,
                        turn_idx,
                        len(synthetic_tool_calls),
                    )
                    validation.has_tool_calls = True
                    validation.tool_calls = synthetic_tool_calls
                    validation.errors = []
                    turn_result.tool_calls = [tc.name for tc in synthetic_tool_calls]
                    turn_result.tool_calls_detail = _tool_calls_detail(synthetic_tool_calls)
                    turn_result.tool_valid = True

            if turn_idx == 0 and not validation.has_tool_calls:
                resp_text = turn_result.response_text or ""
                if looks_like_unparsed_tool_call(resp_text):
                    _log.warning(
                        "session=%s: first turn has no tool_calls but response "
                        "contains tool-call-like text. The vLLM "
                        "--tool-call-parser may not match this model. "
                        "Check server logs or try a different parser "
                        "(e.g. hermes, qwen3_xml, llama3_json).\n"
                        "  Response preview: %.200s",
                        session.session_id,
                        resp_text,
                    )

            # Sanitized calls flow through build_tool_response_messages_async
            # so the tool reports the truncation error back to the model; they
            # must not short-circuit into early termination.
            finish_called = any(
                tc.name == "finish" for tc in validation.tool_calls if tc.valid and not tc.sanitized
            )
            text_only = not validation.has_tool_calls and validation.has_content

            if finish_called:
                if pending_queue:
                    # Multi-turn feeding: model finished but the trace has more
                    # user turns queued. Append assistant text (no tool_calls)
                    # plus the next user msg and continue the loop.
                    raw_msg = response_json["choices"][0]["message"]
                    _append_followup_user_turn(messages, raw_msg, pending_queue.pop(0))
                    turn_result.wall_end_ms = (time.monotonic() - t_start) * 1000
                    continue
                turn_result.wall_end_ms = (time.monotonic() - t_start) * 1000
                break

            if text_only and cfg.continue_turn_on == "never":
                if pending_queue:
                    raw_msg = response_json["choices"][0]["message"]
                    _append_followup_user_turn(messages, raw_msg, pending_queue.pop(0))
                    turn_result.wall_end_ms = (time.monotonic() - t_start) * 1000
                    continue
                turn_result.wall_end_ms = (time.monotonic() - t_start) * 1000
                break

            if text_only:
                text_content = (turn_result.response_text or "").strip()
                recent_text_only.append(text_content)
                if len(recent_text_only) > 3:
                    recent_text_only.pop(0)
                if len(recent_text_only) == 3 and len(set(recent_text_only)) == 1:
                    turn_result.completed = False
                    turn_result.error = "stuck_detection"
                    exhaust_reason = "stuck_detection"
                    _log.warning(
                        "session=%s turn=%d: stuck detection (3 identical text-only responses)",
                        session.session_id,
                        turn_idx,
                    )
                    turn_result.wall_end_ms = (time.monotonic() - t_start) * 1000
                    break
                next_user_content = pending_queue.pop(0) if pending_queue else cfg.continue_prompt
                raw_msg = response_json["choices"][0]["message"]
                _append_followup_user_turn(messages, raw_msg, next_user_content)
                turn_result.wall_end_ms = (time.monotonic() - t_start) * 1000
                continue

            recent_text_only.clear()

            if not validation.valid:
                _log.warning("session=%s turn=%d: invalid tool call", session.session_id, turn_idx)
                consecutive_invalid += 1
                if cfg.max_retry_turns > 0 and consecutive_invalid >= cfg.max_retry_turns:
                    turn_result.completed = False
                    turn_result.error = "max_retry_turns"
                    exhaust_reason = "max_retry_turns"
                    turn_result.wall_end_ms = (time.monotonic() - t_start) * 1000
                    break
            else:
                consecutive_invalid = 0

            if validation.has_tool_calls:
                turn_result.tool_calls = [tc.name for tc in validation.tool_calls]
                turn_result.tool_calls_detail = _raw_tool_calls_detail(response_json)
                turn_result.tool_valid = validation.valid
                choice = response_json["choices"][0]
                raw_msg = choice["message"]
                assistant_msg = {
                    "role": raw_msg.get("role", "assistant"),
                    "content": raw_msg.get("content") or "",
                }
                # Use sanitized tool_calls from validation (not raw_msg)
                # to avoid forwarding truncated JSON arguments to the next turn.
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments_raw,
                        },
                    }
                    for tc in validation.tool_calls
                ]
                new_msgs = await build_tool_response_messages_async(
                    assistant_msg,
                    validation.tool_calls,
                    sandbox=sandbox,
                )
                messages.extend(new_msgs)
            else:
                if not validation.has_content:
                    turn_result.completed = False
                    turn_result.error = "empty response (no content, no tool_calls)"
                    _log.warning(
                        "session=%s turn=%d: empty response",
                        session.session_id,
                        turn_idx,
                    )
                turn_result.wall_end_ms = (time.monotonic() - t_start) * 1000
                break
            turn_result.wall_end_ms = (time.monotonic() - t_start) * 1000
        else:
            if len(result.turns) >= max_turns and not any(
                (t.error or "").startswith(("max_retry", "stuck", "empty"))
                for t in result.turns[-1:]
            ):
                exhaust_reason = "max_turns"

        if cfg.auto_finish_on_exhaust and exhaust_reason is not None:
            result.metadata["exhaust_reason"] = exhaust_reason

        result.total_ms = (time.monotonic() - t_start) * 1000
        result.llm_ms = llm_total
        result.setup_ms = setup_ms
        result.end_time = time.monotonic() - self._run_start

        if sandbox:
            self._run_sandbox_eval(sandbox, session.metadata, result.metadata)

        await invoke_on_session(self._on_session, result)
        if _tmp_dir_obj is not None:
            _tmp_dir_obj.cleanup()
        return result

    async def _run_replay_session(
        self,
        http: aiohttp.ClientSession | None,
        semaphore: asyncio.Semaphore,
        session: ReplaySession,
        delay: float,
        session_index: int = 0,
    ) -> SessionResult:
        """Replay a recorded tool session by prefilling original trace context.

        Each turn receives the exact conversation prefix from the original trace
        (including prior assistant messages and tool outputs). The model generates
        a fresh response each turn (measured for TTFT), but that response is NOT
        fed back into subsequent turns - the next turn uses the original trace
        prefix instead.
        """
        from agentsurge.tool_call import CODING_TOOLS

        if delay > 0:
            await asyncio.sleep(delay)

        cfg = self.config
        result = SessionResult(session_id=session.session_id, metadata=session.metadata)
        result.expected_turns = len(session.turn_messages)
        result.start_time = time.monotonic() - self._run_start
        t_start = time.monotonic()
        llm_total = 0.0

        session_tools = _sanitize_tool_schema(session.metadata.get("tools") or CODING_TOOLS)
        sess_rng = random.Random(cfg.seed + session_index)

        for turn_idx, messages in enumerate(session.turn_messages):
            _turn_wall_start = (time.monotonic() - t_start) * 1000
            if cfg.tool_delay and turn_idx > 0:
                prior_tool_names = (
                    result.turns[-1].tool_calls
                    if (result.turns and result.turns[-1].tool_calls)
                    else None
                )
                d = _sample_multitoolcall_delay(prior_tool_names, sess_rng)
                if d > 0:
                    await asyncio.sleep(d)

            async with semaphore:
                turn_result = await self._send_with_retry(
                    http,
                    session.session_id,
                    turn_idx,
                    list(messages),
                    tools=session_tools,
                    session_meta=session.metadata,
                )
            turn_result.wall_start_ms = _turn_wall_start
            turn_result.wall_end_ms = (time.monotonic() - t_start) * 1000
            result.turns.append(turn_result)
            llm_total += turn_result.total_ms
            await invoke_on_turn(self._on_turn, turn_result)

            if not turn_result.completed:
                break

        result.total_ms = (time.monotonic() - t_start) * 1000
        result.llm_ms = llm_total
        result.end_time = time.monotonic() - self._run_start
        await invoke_on_session(self._on_session, result)
        return result

    def _get_new_messages(self, session: ReplaySession, turn_idx: int) -> list[dict]:
        """Return messages that were added between turn_idx-1 and turn_idx."""
        if turn_idx == 0:
            return []
        prev_msgs = session.turn_messages[turn_idx - 1]
        prev_prev_len = len(session.turn_messages[turn_idx - 2]) if turn_idx >= 2 else 0
        return prev_msgs[prev_prev_len:]

    def _is_tool_turn(self, session: ReplaySession, turn_idx: int) -> bool:
        """Check if the messages added in the previous turn include tool output."""
        return any(
            m.get("role") == "tool" or TOOL_OUTPUT_PREFIX in m.get("content", "")
            for m in self._get_new_messages(session, turn_idx)
        )

    def _extract_tool_name(self, session: ReplaySession, turn_idx: int) -> str | None:
        """Extract tool function name from messages added in the previous turn."""
        for m in self._get_new_messages(session, turn_idx):
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    func = tc.get("function", tc)
                    if isinstance(func, dict):
                        name = func.get("name")
                        if name:
                            return str(name)
            if m.get("role") == "tool" and m.get("name"):
                return str(m["name"])
            content = str(m.get("content", ""))
            if TOOL_OUTPUT_PREFIX in content:
                after = content.split(TOOL_OUTPUT_PREFIX, 1)[1]
                if after.startswith(": "):
                    end = after.find("]")
                    if end > 2:
                        return after[2:end].strip()
        return None

    def _compute_turn_delay(
        self, session: ReplaySession, turn_idx: int, rng: random.Random | None = None
    ) -> float:
        """Compute inter-turn delay: tool-type > think_time."""
        cfg = self.config
        rng = rng or self._rng
        is_tool = self._is_tool_turn(session, turn_idx)

        if is_tool and cfg.tool_delay:
            from agentsurge.tool_timing import sample_tool_delay

            tool_name = self._extract_tool_name(session, turn_idx)
            return sample_tool_delay(tool_name, rng)

        if not is_tool and cfg.think_time:
            return rng.lognormvariate(cfg.think_time[0], cfg.think_time[1])

        return 0.0

    async def _send_with_retry(
        self,
        http: aiohttp.ClientSession | None,
        session_id: str,
        turn_idx: int,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        session_meta: dict | None = None,
        max_tokens: int | None = None,
        capture_text: bool | None = None,
    ) -> TurnResult:
        """Send a turn with optional retry on failure.

        Routes to the injected backend when one is active, otherwise falls
        back to the HTTP path.  The *http* argument may be ``None`` when
        called from the backend execution path; it is only used by the HTTP
        path helpers (:meth:`_send_turn` / :meth:`_send_turn_responses`).
        """
        if self._active_backend is not None:
            backend = self._active_backend

            async def send_fn() -> TurnResult:
                effective_max_tokens = max_tokens
                if effective_max_tokens is None:
                    effective_max_tokens = self._effective_max_tokens(turn_idx, session_meta)
                    if self.config.max_model_len > 0:
                        input_tok = _count_input_tokens(messages, self._tokenizer)
                        headroom = self.config.max_model_len - input_tok
                        if headroom < effective_max_tokens:
                            effective_max_tokens = max(1, headroom)
                return await backend.send_turn(
                    session_id,
                    turn_idx,
                    messages,
                    tools=tools,
                    max_tokens=effective_max_tokens,
                    capture_text=(
                        capture_text
                        if capture_text is not None
                        else (
                            self.config.use_model_reply_in_next_turn or self.config.save_responses
                        )
                    ),
                    session_meta=session_meta,
                )
        elif self.config.api_type == "responses":

            async def send_fn() -> TurnResult:
                return await self._send_turn_responses(
                    http,
                    session_id,
                    turn_idx,
                    messages,
                    session_meta=session_meta,
                )
        else:

            async def send_fn() -> TurnResult:
                return await self._send_turn(
                    http,
                    session_id,
                    turn_idx,
                    messages,
                    tools=tools,
                    session_meta=session_meta,
                )

        return await self._retry_loop(send_fn)

    @staticmethod
    def _is_retryable_error(error: str) -> bool:
        """Classify *error* as retryable (transient) or terminal (deterministic).

        Retry only network-layer / 5xx outcomes that may resolve on the
        next attempt. Empty completions and 4xx are terminal — retrying
        wastes 1-4 s of backoff per affected turn and inflates TTFT.
        """
        if not error:
            return False
        e = error.lower()
        # Empty completion (deterministic decoder behaviour, e.g. boundary
        # max_tokens) — never resolves on retry.
        if "empty response" in e:
            return False
        # HTTP 5xx server errors are transient; 4xx are not.
        if e.startswith("http 5"):
            return True
        if e.startswith("http 4"):
            return False
        # Network / timeout signals from aiohttp / asyncio.
        retryable_markers = (
            "timeout",
            "connectionerror",
            "clientconnectorerror",
            "serverdisconnectederror",
            "clientpayloaderror",
        )
        return any(m in e for m in retryable_markers)

    async def _retry_loop(self, send_fn: Callable[[], Awaitable[TurnResult]]) -> TurnResult:
        """Execute *send_fn* with exponential-backoff retry.

        *send_fn* is a zero-argument coroutine factory that returns a
        :class:`TurnResult`.  The retry/backoff logic is identical for the
        HTTP and backend execution paths - only the send call differs.
        Retries are gated by :meth:`_is_retryable_error` so deterministic
        outcomes (empty response, 4xx) bail out immediately instead of
        absorbing 1-4 s of backoff into TTFT-shaped sessions.
        """
        if self.config.max_retries <= 0:
            result = await send_fn()
            result.wall_ttft_ms = result.ttft_ms
            return result

        last_result = None
        wall_start = time.monotonic()
        max_retries = self.config.max_retries
        for attempt in range(max_retries + 1):
            attempt_start = time.monotonic()
            result = await send_fn()
            if result.completed:
                result.retry_count = attempt
                # wall_ttft = time spent on failed attempts/backoff + this attempt's TTFT
                overhead_ms = (attempt_start - wall_start) * 1000
                result.wall_ttft_ms = overhead_ms + result.ttft_ms
                return result
            last_result = result
            if not self._is_retryable_error(result.error):
                # Deterministic failure — return immediately so the run
                # report reflects the real outcome.
                result.retry_count = attempt
                overhead_ms = (attempt_start - wall_start) * 1000
                result.wall_ttft_ms = overhead_ms + result.ttft_ms
                return result
            if attempt < max_retries:
                if self.config.retry_profile == "codex":
                    base_delay = 0.2 * (2**attempt)
                    jitter = self._rng.uniform(0, base_delay * 0.1)
                else:
                    base_delay = 1.0 * (2**attempt)
                    jitter = self._rng.uniform(0, base_delay * 0.5)
                await asyncio.sleep(base_delay + jitter)

        assert last_result is not None
        last_result.retry_count = max_retries
        last_result.wall_ttft_ms = (time.monotonic() - wall_start) * 1000
        return last_result

    async def _send_turn_impl(
        self,
        http: aiohttp.ClientSession,
        session_id: str,
        turn_idx: int,
        messages: list[dict],
        endpoint: str,
        payload: dict,
    ) -> TurnResult:
        """Shared implementation for sending a single turn via SSE."""
        t0 = time.monotonic()
        try:
            _idle = self.config.stream_idle_timeout or None
            async with http.post(
                endpoint,
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=self.config.request_timeout,
                    sock_read=_idle,
                ),
            ) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    return TurnResult(
                        session_id=session_id,
                        turn_index=turn_idx,
                        completed=False,
                        total_ms=(time.monotonic() - t0) * 1000,
                        input_messages=len(messages),
                        input_tokens=_count_input_tokens(messages, self._tokenizer),
                        error=f"HTTP {resp.status}: {err[:200]}",
                        input_text=_extract_last_input(messages),
                    )
                api = "responses" if "responses" in endpoint else "chat"
                capture = self.config.use_model_reply_in_next_turn or self.config.save_responses
                sse = await parse_sse(resp.content, api, capture_text=capture)
                if sse.parse_errors > 0:
                    _log.warning(
                        "session=%s turn=%d: %d SSE chunk(s) failed JSON decode",
                        session_id,
                        turn_idx,
                        sse.parse_errors,
                    )
            total_ms = (time.monotonic() - t0) * 1000
            final_tokens = sse.usage_tokens if sse.usage_tokens is not None else sse.tokens
            # Prefer server-reported prompt_tokens over client estimate
            input_tok = (
                sse.prompt_tokens
                if sse.prompt_tokens is not None
                else _count_input_tokens(messages, self._tokenizer)
            )
            empty_response = final_tokens == 0 and sse.ttft is None
            return TurnResult(
                session_id=session_id,
                turn_index=turn_idx,
                completed=not empty_response,
                ttft_ms=sse.ttft if sse.ttft is not None else 0.0,
                total_ms=total_ms,
                output_tokens=final_tokens,
                reasoning_tokens=sse.reasoning_tokens,
                input_messages=len(messages),
                input_tokens=input_tok,
                response_text=sse.response_text,
                input_text=_extract_last_input(messages),
                prompt_tokens_server=sse.prompt_tokens,
                cached_tokens=sse.cached_tokens,
                error="empty response: 0 output tokens" if empty_response else "",
                finish_reason=sse.finish_reason or "",
            )
        except Exception as e:
            return TurnResult(
                session_id=session_id,
                turn_index=turn_idx,
                completed=False,
                total_ms=(time.monotonic() - t0) * 1000,
                input_messages=len(messages),
                input_tokens=_count_input_tokens(messages, self._tokenizer),
                error=f"{type(e).__name__}: {e}"[:200],
                input_text=_extract_last_input(messages),
            )

    def _effective_max_tokens(self, turn_idx: int, session_meta: dict | None) -> int:
        """Compute per-turn max_tokens from recorded output length when available."""
        cfg = self.config
        if cfg.ignore_replay_output_length:
            return cfg.max_tokens
        if not session_meta:
            return cfg.max_tokens

        contents = session_meta.get("turn_output_contents", [])
        if turn_idx >= len(contents):
            return cfg.max_tokens

        content = contents[turn_idx]
        if not content or not content.strip():
            return cfg.max_tokens
        assert self._tokenizer is not None  # guarded by ignore_replay_output_length check above
        recorded = len(self._tokenizer.encode(content))
        return max(1, recorded + (cfg.thinking_budget if cfg.enable_thinking else 0))

    async def _send_turn(
        self,
        http,
        session_id,
        turn_idx,
        messages,
        tools=None,
        *,
        session_meta=None,
    ) -> TurnResult:
        """Send a single turn via streaming SSE (chat completions API), measuring TTFT.

        When a :attr:`_request_adapter` is set, the adapter's
        :meth:`~agentsurge.backends.base.RequestAdapter.adapt` method builds the
        request payload.  Otherwise the default OpenAI-compatible payload is
        constructed directly.
        """
        cfg = self.config
        effective_max_tokens = self._effective_max_tokens(turn_idx, session_meta)

        if self._request_adapter is not None:
            from agentsurge.backends.base import BackendConfig

            bc = BackendConfig(
                base_url=cfg.vllm_url,
                model=cfg.model,
                max_tokens=effective_max_tokens,
                temperature=cfg.temperature,
                stream=True,
                extra_body=cfg.extra_body,
            )
            self._request_adapter.validate(messages, bc)
            payload = self._request_adapter.adapt(
                messages, bc, session_id=session_id, turn_index=turn_idx
            )
            payload.setdefault("stream_options", {"include_usage": True})
            if cfg.enable_thinking:
                payload.setdefault("chat_template_kwargs", {"enable_thinking": True})
        else:
            if cfg.max_model_len > 0:
                input_tok = _count_input_tokens(messages, self._tokenizer)
                headroom = cfg.max_model_len - input_tok
                if headroom < effective_max_tokens:
                    effective_max_tokens = max(1, headroom)
            payload = {
                "model": cfg.model,
                "messages": messages,
                "max_tokens": effective_max_tokens,
                "temperature": cfg.temperature,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if cfg.enable_thinking:
                payload["chat_template_kwargs"] = {"enable_thinking": True}
            if cfg.extra_body:
                payload.update(cfg.extra_body)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return await self._send_turn_impl(
            http, session_id, turn_idx, messages, f"{cfg.vllm_url}/v1/chat/completions", payload
        )

    async def _send_turn_responses(
        self,
        http,
        session_id,
        turn_idx,
        messages,
        *,
        session_meta=None,
    ) -> TurnResult:
        """Send a single turn via Responses API streaming."""
        cfg = self.config
        effective_max_tokens = self._effective_max_tokens(turn_idx, session_meta)
        if cfg.max_model_len > 0:
            input_tok = _count_input_tokens(messages, self._tokenizer)
            headroom = cfg.max_model_len - input_tok
            if headroom < effective_max_tokens:
                effective_max_tokens = max(1, headroom)
        payload = {
            "model": cfg.model,
            "input": messages,
            "max_output_tokens": effective_max_tokens,
            "temperature": cfg.temperature,
            "stream": True,
        }
        return await self._send_turn_impl(
            http, session_id, turn_idx, messages, f"{cfg.vllm_url}/v1/responses", payload
        )
