# SPDX-License-Identifier: MIT
"""Runner result types and benchmark configuration.

Defines ``TurnResult``, ``SessionResult``, ``RunResult``,
``BenchmarkConfig``, ``ConfigProfile``, and ``SloComparisonResult``.
"""

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

_log = logging.getLogger(__name__)

_MAX_RESPONSE_TEXT = 2000


def _resolve_use_model_reply(explicit: Any, tool_mode: str) -> bool:
    """Derive use_model_reply_in_next_turn from tool_mode, with optional override.

    Default: real/off → True, replay → False. An explicit non-None value
    overrides and emits a warning if it disagrees with the tool_mode default.
    """
    default = (tool_mode or "off") != "replay"
    if explicit is None:
        return default
    override = bool(explicit)
    if override != default:
        _log.warning(
            "use_model_reply_in_next_turn=%s overrides tool_mode=%s default (%s). "
            "Confirm this is intentional.",
            override,
            tool_mode,
            default,
        )
    return override


def _extract_last_input(messages: list[dict]) -> str:
    """Extract content of the last message for debugging."""
    if not messages:
        return ""
    last = messages[-1]
    content = last.get("content", "")
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content) if content else ""


@dataclass
class TurnResult:
    """Result from a single turn (one LLM call).

    .. note::
        ``TurnResult.tool_calls`` is **names only** (``list[str]``) and is
        intentionally asymmetric with :class:`agentsurge.types.trace.Turn`'s
        ``tool_calls`` field, which carries the **full payload dicts**
        (``list[dict]``). For the full payload on the result side, use
        :pyattr:`tool_calls_detail`. Do not assume the two ``tool_calls`` fields
        are interchangeable — they share a name but differ in shape.
    """

    session_id: str
    turn_index: int
    completed: bool
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    input_messages: int = 0
    input_tokens: int = 0
    error: str = ""
    retry_count: int = 0
    wall_ttft_ms: float = 0.0
    interrupted: bool = False
    fan_out_count: int = 1
    wall_start_ms: float = 0.0
    wall_end_ms: float = 0.0
    response_text: str = ""
    input_text: str = ""
    prompt_tokens_server: int | None = None
    cached_tokens: int | None = None
    # tool_calls = names only (list[str]); see class docstring re: asymmetry
    # with Turn.tool_calls (list[dict]). Use tool_calls_detail for full payloads.
    tool_calls: list[str] | None = None
    tool_calls_detail: list[dict] | None = None  # full tool call payloads
    tool_valid: bool | None = None  # tool_call format validation result
    finish_reason: str = ""


@dataclass
class FrontendMetrics:
    provider: str
    process_exit_code: int | None = None
    process_signal: int | None = None
    process_wall_ms: float | None = None
    process_startup_to_first_event_ms: float | None = None
    streaming_text_available: bool = False
    time_to_first_assistant_text_ms: float | None = None
    time_to_final_message_ms: float | None = None
    frontend_ttft_ms: float | None = None
    visible_text_tpot_estimate_ms: float | None = None
    visible_output_tokens_estimate: int | None = None
    provider_usage: dict[str, int] | None = None
    event_count: int = 0
    artifact_dir: str | None = None
    failure_category: str | None = None


@dataclass
class ServingTraceMetrics:
    available: bool = False
    request_count: int = 0


@dataclass
class SessionResult:
    """Result from a complete session.

    Time fields:

    * ``total_ms`` / ``llm_ms`` / ``setup_ms`` – durations in milliseconds.
    * ``start_time`` / ``end_time`` – **run-relative seconds** (monotonic
      seconds since the run started, i.e. ``time.monotonic() - run_start``).
      These are *not* the same clock origin as
      :pyattr:`agentsurge.types.metrics.MetricsSnapshot.timestamp`, which is
      raw monotonic seconds. Mixing the two on one axis silently shifts
      timelines by ``run_start``.
    """

    session_id: str
    turns: list[TurnResult] = field(default_factory=list)
    total_ms: float = 0.0
    llm_ms: float = 0.0
    setup_ms: float = 0.0
    expected_turns: int = 0
    start_time: float = 0.0  # run-relative seconds (since run start)
    end_time: float = 0.0  # run-relative seconds (since run start)
    metadata: dict = field(default_factory=dict)
    frontend_metrics: FrontendMetrics | None = None

    @property
    def completed(self) -> bool:
        # A session with zero recorded turns is never "completed" - treating
        # all([]) as True would silently count crashed/placeholder sessions
        # (e.g. _dispatch_sessions' return_exceptions placeholders) as
        # successes in n_completed_sessions. Same for sessions flagged with
        # metadata["failed"]=True by the dispatcher.
        if not self.turns or self.metadata.get("failed"):
            return False
        return all(t.completed for t in self.turns)

    @property
    def n_turns(self) -> int:
        return len(self.turns)


# Backend-specific metric keys that live inside ``RunResult.backend_metrics``.
# These were previously top-level dataclass fields; transparent attribute
# access is preserved via ``__getattr__`` / ``__setattr__`` for backward
# compatibility.
_BACKEND_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "kv_util_peak",
        "running_peak",
        "waiting_peak",
        "prefix_cache_hit_delta",
        "prefix_cache_query_delta",
        "external_prefix_cache_hit_delta",
        "external_prefix_cache_query_delta",
        "preemption_total",
        "server_ttft_sum",
        "server_ttft_count",
        "prefill_time_sum",
        "prefill_time_count",
        "queue_time_sum",
        "queue_time_count",
        "lmcache_tier_deltas",
        "lmcache_v1_deltas",
        "metrics_timeseries",
        "external_reuse",
    }
)

# Default values for backend metrics (mirrors the old dataclass defaults).
_BACKEND_METRIC_DEFAULTS: dict = {
    "kv_util_peak": 0.0,
    "running_peak": 0.0,
    "waiting_peak": 0.0,
    "prefix_cache_hit_delta": 0.0,
    "prefix_cache_query_delta": 0.0,
    "external_prefix_cache_hit_delta": 0.0,
    "external_prefix_cache_query_delta": 0.0,
    "preemption_total": 0.0,
    "server_ttft_sum": 0.0,
    "server_ttft_count": 0.0,
    "prefill_time_sum": 0.0,
    "prefill_time_count": 0.0,
    "queue_time_sum": 0.0,
    "queue_time_count": 0.0,
}

# Keys whose default is a mutable container - auto-created on first access.
_BACKEND_METRIC_MUTABLE_DEFAULTS: dict[str, type] = {
    "lmcache_tier_deltas": dict,
    "lmcache_v1_deltas": dict,
    "metrics_timeseries": list,
    "external_reuse": dict,
}


def _percentile(values: list[float], pct: float) -> float:
    """Compute *pct*-th percentile (0-100) using linear interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (pct / 100) * (len(s) - 1)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + frac * (s[hi] - s[lo])


class RunResult:
    """Full benchmark run result.

    **Core fields** (generic, backend-agnostic):

    * ``sessions`` – list of :class:`SessionResult`
    * ``total_elapsed_s`` – wall-clock time for the run
    * ``config`` – dict of run configuration
    * ``isl_total`` – total ISL (input/prefill) tokens processed
    * ``osl_total`` – total OSL (output/decode) tokens produced

    **Backend metrics** (backend-specific, stored in ``backend_metrics`` dict):

    Backend-specific counters (``kv_util_peak``, ``prefix_cache_hit_delta``,
    ``lmcache_tier_deltas``, etc.) are stored inside the
    :pyattr:`backend_metrics` dictionary.  For **backward compatibility**,
    these keys are also accessible as regular attributes::

        result.kv_util_peak            # reads  result.backend_metrics["kv_util_peak"]
        result.kv_util_peak = 0.5      # writes result.backend_metrics["kv_util_peak"] = 0.5

    New code should prefer ``result.backend_metrics[key]`` for clarity.
    """

    __slots__ = (
        "sessions",
        "total_elapsed_s",
        "config",
        "isl_total",
        "osl_total",
        "backend_metrics",
        "frontend_metrics",
        "serving_trace",
    )

    # RunResult is mutable and defines __eq__: instances are intentionally
    # unhashable.  Declare it explicitly so callers can rely on the contract
    # without inferring it from Python's data-model fallback.
    __hash__ = None  # type: ignore[assignment]

    sessions: list[SessionResult]
    total_elapsed_s: float
    config: dict
    isl_total: float
    osl_total: float
    backend_metrics: dict[str, object]
    frontend_metrics: dict[str, object] | None
    serving_trace: ServingTraceMetrics | None

    def __init__(
        self,
        *,
        sessions: list[SessionResult] | None = None,
        total_elapsed_s: float = 0.0,
        config: dict | None = None,
        isl_total: float = 0.0,
        osl_total: float = 0.0,
        backend_metrics: dict | None = None,
        frontend_metrics: dict[str, object] | None = None,
        serving_trace: ServingTraceMetrics | None = None,
        **kwargs: object,
    ) -> None:
        object.__setattr__(self, "sessions", sessions if sessions is not None else [])
        object.__setattr__(self, "total_elapsed_s", total_elapsed_s)
        object.__setattr__(self, "config", config if config is not None else {})
        object.__setattr__(self, "isl_total", isl_total)
        object.__setattr__(self, "osl_total", osl_total)
        object.__setattr__(self, "frontend_metrics", frontend_metrics)
        object.__setattr__(self, "serving_trace", serving_trace)
        bm: dict = dict(backend_metrics) if backend_metrics else {}
        for key, value in kwargs.items():
            if key in _BACKEND_METRIC_KEYS:
                bm[key] = value
            else:
                raise TypeError(f"RunResult() got an unexpected keyword argument '{key}'")
        object.__setattr__(self, "backend_metrics", bm)

    def __getattr__(self, name: str) -> object:
        if name in _BACKEND_METRIC_KEYS:
            bm = object.__getattribute__(self, "backend_metrics")
            if name in bm:
                return bm[name]
            # Auto-create mutable defaults so mutations persist
            if name in _BACKEND_METRIC_MUTABLE_DEFAULTS:
                val = _BACKEND_METRIC_MUTABLE_DEFAULTS[name]()
                bm[name] = val
                return val
            return _BACKEND_METRIC_DEFAULTS.get(name, 0.0)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: object) -> None:
        if name in _BACKEND_METRIC_KEYS:
            object.__getattribute__(self, "backend_metrics")[name] = value
        else:
            object.__setattr__(self, name, value)

    def _bm_float(self, key: str) -> float:
        """Read a backend metric as a float."""
        val = self.backend_metrics.get(key, _BACKEND_METRIC_DEFAULTS.get(key, 0.0))
        return float(val) if isinstance(val, (int, float)) else 0.0

    @property
    def prefix_cache_hit_rate(self) -> float:
        query = self._bm_float("prefix_cache_query_delta")
        if query > 0:
            return self._bm_float("prefix_cache_hit_delta") / query
        return 0.0

    @property
    def isl_osl_ratio(self) -> float:
        """Ratio of total input (prefill) tokens to output (decode) tokens.

        Agent workloads are heavily prefill-dominated: coding agents can reach
        200-434:1 (SWE-Effi, arXiv:2509.09853), while chatbots are ~1:1.
        This ratio is a key indicator of workload type and scheduling needs.
        """
        if self.osl_total > 0:
            return round(self.isl_total / self.osl_total, 1)
        return 0.0

    @property
    def requests_per_s(self) -> float:
        """Request throughput: completed turns (LLM requests) per second."""
        if self.total_elapsed_s > 0:
            n_turns = sum(s.n_turns for s in self.sessions)
            return round(n_turns / self.total_elapsed_s, 2)
        return 0.0

    def all_turn_results(self) -> list[TurnResult]:
        return [t for s in self.sessions for t in s.turns]

    def ttft_values(self) -> list[float]:
        return [
            t.ttft_ms
            for t in self.all_turn_results()
            if t.completed and t.ttft_ms > 0 and not t.interrupted
        ]

    @property
    def tpot_ms_list(self) -> list[float]:
        """Time per output token (ms) for each turn with output tokens."""
        result = []
        for s in self.sessions:
            for t in s.turns:
                if (
                    t.completed
                    and t.output_tokens
                    and t.output_tokens > 0
                    and t.total_ms > t.ttft_ms
                ):
                    tpot = (t.total_ms - t.ttft_ms) / t.output_tokens
                    result.append(tpot)
        return result

    @property
    def tpot_p50(self) -> float:
        vals = self.tpot_ms_list
        return round(_percentile(vals, 50), 1) if vals else 0.0

    @property
    def tpot_p95(self) -> float:
        vals = self.tpot_ms_list
        return round(_percentile(vals, 95), 1) if vals else 0.0

    @property
    def tpot_p99(self) -> float:
        vals = self.tpot_ms_list
        return round(_percentile(vals, 99), 1) if vals else 0.0

    def _tpot_percentiles(self) -> tuple[float, float, float]:
        """Compute p50/p95/p99 for TPOT in one pass (list built once)."""
        vals = self.tpot_ms_list
        if not vals:
            return 0.0, 0.0, 0.0
        return (
            round(_percentile(vals, 50), 1),
            round(_percentile(vals, 95), 1),
            round(_percentile(vals, 99), 1),
        )

    @property
    def e2e_ms_list(self) -> list[float]:
        return [
            t.total_ms for s in self.sessions for t in s.turns if t.completed and t.total_ms > 0
        ]

    @property
    def e2e_p50(self) -> float:
        vals = self.e2e_ms_list
        return round(_percentile(vals, 50), 1) if vals else 0.0

    @property
    def e2e_p95(self) -> float:
        vals = self.e2e_ms_list
        return round(_percentile(vals, 95), 1) if vals else 0.0

    @property
    def e2e_p99(self) -> float:
        vals = self.e2e_ms_list
        return round(_percentile(vals, 99), 1) if vals else 0.0

    def _e2e_percentiles(self) -> tuple[float, float, float]:
        """Compute p50/p95/p99 for E2E latency in one pass (list built once)."""
        vals = self.e2e_ms_list
        if not vals:
            return 0.0, 0.0, 0.0
        return (
            round(_percentile(vals, 50), 1),
            round(_percentile(vals, 95), 1),
            round(_percentile(vals, 99), 1),
        )

    def to_dict(self, *, save_responses: bool = False) -> dict:
        """Serialize to a flat dictionary.

        Backend metrics are flattened into the top level for backward
        compatibility with existing JSON consumers (analyzer, CLI, etc.).
        """
        d: dict = {
            "config": self.config,
            "isl_total": self.isl_total,
            "osl_total": self.osl_total,
            "total_elapsed_s": self.total_elapsed_s,
            "isl_osl_ratio": self.isl_osl_ratio,
            "requests_per_s": self.requests_per_s,
            "prefix_cache_hit_rate": self.prefix_cache_hit_rate,
            "n_sessions": len(self.sessions),
            "n_completed_sessions": sum(1 for s in self.sessions if s.completed),
            "n_total_turns": sum(s.n_turns for s in self.sessions),
            "ttft_values": self.ttft_values(),
            "sessions": [
                {
                    "session_id": s.session_id,
                    "completed": s.completed,
                    "n_turns": s.n_turns,
                    "expected_turns": s.expected_turns,
                    "total_ms": s.total_ms,
                    "llm_ms": round(s.llm_ms, 1),
                    "setup_ms": round(s.setup_ms, 1),
                    "start_time": round(s.start_time, 3),
                    "end_time": round(s.end_time, 3),
                    "metadata": s.metadata,
                    "turns": [
                        {
                            "turn": t.turn_index,
                            "completed": t.completed,
                            "ttft_ms": round(t.ttft_ms, 1),
                            "wall_ttft_ms": round(t.wall_ttft_ms, 1),
                            "wall_start_ms": round(t.wall_start_ms, 1),
                            "wall_end_ms": round(t.wall_end_ms, 1),
                            "total_ms": round(t.total_ms, 1),
                            "input_tokens": t.input_tokens,
                            "tokens": t.output_tokens,
                            **(
                                {"reasoning_tokens": t.reasoning_tokens}
                                if t.reasoning_tokens
                                else {}
                            ),
                            **(
                                {"prompt_tokens_server": t.prompt_tokens_server}
                                if t.prompt_tokens_server is not None
                                else {}
                            ),
                            **(
                                {"cached_tokens": t.cached_tokens}
                                if t.cached_tokens is not None
                                else {}
                            ),
                            "error": t.error,
                            "retry_count": t.retry_count,
                            "interrupted": t.interrupted,
                            "fan_out_count": t.fan_out_count,
                            **({"tool_calls": t.tool_calls} if t.tool_calls is not None else {}),
                            **(
                                {"tool_calls_detail": t.tool_calls_detail}
                                if t.tool_calls_detail is not None
                                else {}
                            ),
                            **({"tool_valid": t.tool_valid} if t.tool_valid is not None else {}),
                            **(
                                {"response_text": t.response_text[:_MAX_RESPONSE_TEXT]}
                                if save_responses and t.response_text
                                else {}
                            ),
                            **(
                                {"input_text_delta": t.input_text[:_MAX_RESPONSE_TEXT]}
                                if save_responses and t.input_text
                                else {}
                            ),
                        }
                        for t in s.turns
                    ],
                }
                for s in self.sessions
            ],
        }
        # Flatten backend_metrics into the top level for backward compat.
        # Also include default-valued keys so the output schema is stable.
        for key in _BACKEND_METRIC_KEYS:
            d[key] = getattr(self, key)
        bm: dict[str, object] = self.backend_metrics
        d["backend_metrics"] = dict(bm)
        import dataclasses

        d["frontend_metrics"] = self.frontend_metrics
        d["serving_trace"] = (
            dataclasses.asdict(self.serving_trace) if self.serving_trace is not None else None
        )
        return d

    _TURN_CSV_COLUMNS = [
        "session_id",
        "turn",
        "completed",
        "ttft_ms",
        "wall_ttft_ms",
        "total_ms",
        "input_tokens",
        "output_tokens",
        "cached_tokens",
        "tpot_ms",
        "error",
        "retry_count",
        "fan_out_count",
    ]

    _SUMMARY_CSV_COLUMNS = [
        "n_sessions",
        "n_completed",
        "n_turns",
        "total_elapsed_s",
        "sessions_per_hour",
        "ttft_p50",
        "ttft_p95",
        "ttft_p99",
        "kv_util_peak",
        "prefix_cache_hit_rate",
        "isl_total",
        "osl_total",
        "isl_osl_ratio",
        "tpot_p50",
        "tpot_p95",
        "tpot_p99",
        "e2e_p50",
        "e2e_p95",
        "e2e_p99",
        "requests_per_s",
        "total_tok_per_s",
        "isl_per_s",
        "osl_per_s",
    ]

    def to_turns_csv(self, path: str) -> None:
        """Write per-turn results as CSV (one row per LLM call)."""
        import csv

        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self._TURN_CSV_COLUMNS)
            for t in self.all_turn_results():
                w.writerow(
                    [
                        t.session_id,
                        t.turn_index,
                        t.completed,
                        round(t.ttft_ms, 2),
                        round(t.wall_ttft_ms, 2),
                        round(t.total_ms, 2),
                        t.input_tokens,
                        t.output_tokens,
                        t.cached_tokens if t.cached_tokens is not None else "",
                        round((t.total_ms - t.ttft_ms) / t.output_tokens, 2)
                        if t.output_tokens and t.output_tokens > 0 and t.total_ms > t.ttft_ms
                        else "",
                        t.error,
                        t.retry_count,
                        t.fan_out_count,
                    ]
                )

    def to_summary_csv(self, path: str) -> None:
        """Write one-row summary CSV with aggregate metrics."""
        import csv
        import statistics

        ttfts = self.ttft_values()
        n_turns = sum(s.n_turns for s in self.sessions)
        elapsed = self.total_elapsed_s
        isl_tps = self.isl_total / elapsed if elapsed > 0 else 0.0
        osl_tps = self.osl_total / elapsed if elapsed > 0 else 0.0
        total_tps = isl_tps + osl_tps

        n_completed = sum(1 for s in self.sessions if s.completed)
        sessions_per_hour = n_completed / elapsed * 3600 if elapsed > 0 else 0.0

        row = {
            "n_sessions": len(self.sessions),
            "n_completed": n_completed,
            "n_turns": n_turns,
            "total_elapsed_s": round(elapsed, 2),
            "sessions_per_hour": round(sessions_per_hour, 1),
            "ttft_p50": round(statistics.median(ttfts), 1) if ttfts else 0,
            "ttft_p95": round(_percentile(ttfts, 95), 1) if ttfts else 0,
            "ttft_p99": round(_percentile(ttfts, 99), 1) if ttfts else 0,
            "kv_util_peak": round(self._bm_float("kv_util_peak"), 4),
            "prefix_cache_hit_rate": round(self.prefix_cache_hit_rate, 4),
            "isl_total": self.isl_total,
            "osl_total": self.osl_total,
            "isl_osl_ratio": self.isl_osl_ratio,
            **dict(zip(("tpot_p50", "tpot_p95", "tpot_p99"), self._tpot_percentiles())),
            **dict(zip(("e2e_p50", "e2e_p95", "e2e_p99"), self._e2e_percentiles())),
            "requests_per_s": self.requests_per_s,
            "total_tok_per_s": round(total_tps, 1),
            "isl_per_s": round(isl_tps, 1),
            "osl_per_s": round(osl_tps, 1),
        }

        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._SUMMARY_CSV_COLUMNS)
            w.writeheader()
            w.writerow(row)

    def __repr__(self) -> str:
        bm: dict[str, object] = self.backend_metrics
        bm_summary = {k: v for k, v in bm.items() if v != _BACKEND_METRIC_DEFAULTS.get(k, 0.0)}
        return (
            f"RunResult(sessions={self.sessions!r}, "
            f"total_elapsed_s={self.total_elapsed_s!r}, "
            f"config={self.config!r}, "
            f"isl_total={self.isl_total!r}, "
            f"osl_total={self.osl_total!r}, "
            f"backend_metrics={bm_summary!r})"
        )

    @classmethod
    def from_dict(cls, d: dict) -> "RunResult":
        """Reconstruct a :class:`RunResult` from a dict produced by :meth:`to_dict`.

        This is best-effort: derived fields and rounded summaries inside
        :meth:`to_dict` (e.g. ``e2e_p95``, ``isl_osl_ratio``) are recomputed
        from the per-turn data and not read back. Backend metric keys flattened
        at the top level are preferred over those in the ``backend_metrics``
        sub-dict (they share values when produced by ``to_dict``).
        """
        sessions: list[SessionResult] = []
        for sd in d.get("sessions", []) or []:
            turns: list[TurnResult] = []
            for td in sd.get("turns", []) or []:
                turns.append(
                    TurnResult(
                        session_id=sd.get("session_id", ""),
                        turn_index=td.get("turn", 0),
                        completed=td.get("completed", False),
                        ttft_ms=td.get("ttft_ms", 0.0),
                        wall_ttft_ms=td.get("wall_ttft_ms", 0.0),
                        wall_start_ms=td.get("wall_start_ms", 0.0),
                        wall_end_ms=td.get("wall_end_ms", 0.0),
                        total_ms=td.get("total_ms", 0.0),
                        input_tokens=td.get("input_tokens", 0),
                        output_tokens=td.get("tokens", 0),
                        reasoning_tokens=td.get("reasoning_tokens", 0),
                        prompt_tokens_server=td.get("prompt_tokens_server"),
                        cached_tokens=td.get("cached_tokens"),
                        error=td.get("error", ""),
                        retry_count=td.get("retry_count", 0),
                        interrupted=td.get("interrupted", False),
                        fan_out_count=td.get("fan_out_count", 1),
                        tool_calls=td.get("tool_calls"),
                        tool_calls_detail=td.get("tool_calls_detail"),
                        tool_valid=td.get("tool_valid"),
                        response_text=td.get("response_text", ""),
                        input_text=td.get("input_text_delta", ""),
                    )
                )
            sessions.append(
                SessionResult(
                    session_id=sd.get("session_id", ""),
                    turns=turns,
                    total_ms=sd.get("total_ms", 0.0),
                    llm_ms=sd.get("llm_ms", 0.0),
                    setup_ms=sd.get("setup_ms", 0.0),
                    expected_turns=sd.get("expected_turns", 0),
                    start_time=sd.get("start_time", 0.0),
                    end_time=sd.get("end_time", 0.0),
                    metadata=dict(sd.get("metadata") or {}),
                )
            )
        # Backend metrics: prefer the nested dict, falling back to flattened
        # top-level entries (both are emitted by to_dict).
        bm: dict[str, object] = dict(d.get("backend_metrics") or {})
        for key in _BACKEND_METRIC_KEYS:
            if key not in bm and key in d:
                bm[key] = d[key]
        return cls(
            sessions=sessions,
            total_elapsed_s=d.get("total_elapsed_s", 0.0),
            config=dict(d.get("config") or {}),
            isl_total=d.get("isl_total", 0.0),
            osl_total=d.get("osl_total", 0.0),
            backend_metrics=bm,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RunResult):
            return NotImplemented
        return (
            self.sessions == other.sessions
            and self.total_elapsed_s == other.total_elapsed_s
            and self.config == other.config
            and self.isl_total == other.isl_total
            and self.osl_total == other.osl_total
            and self.backend_metrics == other.backend_metrics
            and self.frontend_metrics == other.frontend_metrics
            and self.serving_trace == other.serving_trace
        )


@dataclass(frozen=True)
class FrontendRuntimeSettings:
    """Runtime settings for a frontend harness (echo/codex/claude/opencode).

    Populated by CLI flags; consumed by the runner's frontend-dispatch path.
    """

    name: Literal["direct", "echo", "codex", "claude", "opencode"]
    command_template: str | None = None
    workspace_dir: str | None = None
    prompt_mode: Literal["auto", "file", "stdin", "arg"] = "auto"
    output_format: Literal["auto", "jsonl", "stream-json", "text"] = "auto"
    model: str | None = None
    session_timeout_s: float = 7200.0
    keep_artifacts: Literal["always", "failed", "never"] = "failed"
    server_url: str | None = None
    extra_env: tuple[tuple[str, str], ...] = ()


@dataclass
class BenchmarkConfig:
    """All settings for a benchmark run."""

    vllm_url: str
    model: str
    max_concurrency: int = 100
    max_tokens: int = 256
    temperature: float = 0.3
    api_type: str = "chat"
    arrival_pattern: str = "poisson"
    arrival_rate: float = 10.0
    gamma_cv: float = 1.2
    ramp_duration: float = (
        0.0  # seconds over which to linearly ramp arrival rate from 0 to arrival_rate
    )
    duration: float = (
        0.0  # if > 0, run for this many seconds (recycling sessions) instead of fixed count
    )
    think_time: tuple[float, float] | None = None
    tool_delay: bool = False
    max_retries: int = 0
    use_model_reply_in_next_turn: bool | None = (
        None  # feed actual LLM output into next turn's prompt (vs. replay recorded assistant text); None = derive from tool_mode in __post_init__
    )
    tool_output_mode: str = (
        "recorded"  # "recorded" = original trace output, "placeholder" = token-count-matched filler
    )
    preset: str | None = None
    interruption_rate: float = 0.0
    single_turn_ratio: float = 0.0
    max_model_len: int = 16384
    context_distribution: str = "mixed"
    seed: int = 42
    extra_metrics_urls: list[str] = field(default_factory=list)
    extra_body: dict | None = None
    request_timeout: int = 7200  # per-request HTTP timeout in seconds
    no_metrics: bool = False  # disable Prometheus metrics polling (degraded mode)
    warm_up: int = 0  # number of warm-up requests before measurement
    tool_mode: str = "off"  # "off" | "real" | "replay"
    tool_call_parser: str = ""  # vLLM tool call parser (e.g. "qwen3_xml", "hermes")
    tool_call_parser_fallback: str = "off"  # "off", "deepseek_text", or "hermes_xml"
    workspace_dir: str | None = None  # pre-prepared local workspace root for real tool mode
    sandbox_dir: str | None = None  # legacy alias for workspace_dir
    tool_env: str = "safe"  # "safe" | "inherit" for real tool subprocesses
    save_responses: bool = False
    no_stream: bool = False  # disable SSE streaming (use non-streaming API for tool turns)
    ignore_replay_output_length: bool = False  # True: use cfg.max_tokens for every turn; False (default): size each turn's max_tokens to the replay's output token count on that turn
    thinking_budget: int = 8192
    enable_thinking: bool = False
    tokenizer_model: str | None = None
    skip_tokenizer_load: bool = (
        False  # True: skip _load_tokenizer, set runner._tokenizer=None directly
    )
    trust_remote_code: bool = False
    sanitize_truncated_tool_calls: bool = False  # recover truncated tool-call JSON with empty args
    continue_turn_on: str = "never"
    continue_prompt: str = (
        "Please proceed to the next step using your best judgement. If you believe you "
        "have completed the task, please call the `finish()` tool with your final answer."
    )
    max_session_time: float = 0.0
    auto_finish_on_exhaust: bool = False
    max_retry_turns: int = 0
    max_budget_usd: float = 0.0
    input_price_per_mtok: float = 0.0
    output_price_per_mtok: float = 0.0
    stream_idle_timeout: float = 0.0
    retry_profile: str = "default"
    inflight_dump: bool = False  # --enable-inflight-dump: append one JSONL line per completed turn
    frontend: FrontendRuntimeSettings | None = None

    @property
    def frontend_name(self) -> str:
        """Frontend name: 'direct' (default) or one of echo/codex/claude/opencode."""
        return self.frontend.name if self.frontend is not None else "direct"

    def to_run_result_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view of this BenchmarkConfig.

        Used by RunResult to embed run configuration. Dataclass-typed fields
        (e.g., FrontendRuntimeSettings) are converted via dataclasses.asdict;
        tuples become lists so json.dumps handles them.
        """

        def _norm(value: object) -> object:
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                return dataclasses.asdict(value)
            if isinstance(value, tuple):
                return [_norm(item) for item in value]
            return value

        return {key: _norm(val) for key, val in vars(self).items()}

    def __post_init__(self) -> None:
        # Resolve use_model_reply_in_next_turn from tool_mode unless an explicit
        # value was provided. Centralising this here ensures direct construction
        # and from_yaml produce identical semantics.
        self.use_model_reply_in_next_turn = _resolve_use_model_reply(
            self.use_model_reply_in_next_turn, self.tool_mode
        )
        if self.arrival_pattern not in ("burst", "poisson", "constant", "gamma", "ramp"):
            raise ValueError(f"Unknown arrival_pattern: {self.arrival_pattern!r}")
        if self.api_type not in ("chat", "responses"):
            raise ValueError(f"Unknown api_type: {self.api_type!r}")
        if self.tool_mode not in ("off", "real", "replay"):
            raise ValueError(f"Unknown tool_mode: {self.tool_mode!r}")
        if self.workspace_dir and self.sandbox_dir and self.workspace_dir != self.sandbox_dir:
            raise ValueError(
                "workspace_dir and legacy sandbox_dir both set to different values: "
                f"{self.workspace_dir!r} != {self.sandbox_dir!r}"
            )
        if self.workspace_dir is None and self.sandbox_dir is not None:
            self.workspace_dir = self.sandbox_dir
        if self.sandbox_dir is None and self.workspace_dir is not None:
            self.sandbox_dir = self.workspace_dir
        if self.tool_env not in ("safe", "inherit"):
            raise ValueError(f"Unknown tool_env: {self.tool_env!r}")
        if self.tool_call_parser_fallback not in ("off", "deepseek_text", "hermes_xml"):
            raise ValueError(
                f"Unknown tool_call_parser_fallback: {self.tool_call_parser_fallback!r}"
            )
        if self.continue_turn_on not in ("never", "text-only"):
            raise ValueError(f"Unknown continue_turn_on: {self.continue_turn_on!r}")
        if self.max_session_time < 0:
            raise ValueError(f"max_session_time must be >= 0, got {self.max_session_time}")
        if self.max_retry_turns < 0:
            raise ValueError(f"max_retry_turns must be >= 0, got {self.max_retry_turns}")
        if self.max_budget_usd < 0:
            raise ValueError(f"max_budget_usd must be >= 0, got {self.max_budget_usd}")
        if self.max_budget_usd > 0 and (
            self.input_price_per_mtok <= 0 or self.output_price_per_mtok <= 0
        ):
            raise ValueError(
                "max_budget_usd requires both input_price_per_mtok and "
                "output_price_per_mtok to be set (> 0)"
            )
        if self.stream_idle_timeout < 0:
            raise ValueError(f"stream_idle_timeout must be >= 0, got {self.stream_idle_timeout}")
        if self.retry_profile not in ("default", "codex"):
            raise ValueError(f"Unknown retry_profile: {self.retry_profile!r}")
        if self.request_timeout <= 0:
            raise ValueError(f"request_timeout must be positive, got {self.request_timeout}")
        if self.tool_output_mode not in ("recorded", "placeholder"):
            raise ValueError(f"Unknown tool_output_mode: {self.tool_output_mode!r}")
        if self.gamma_cv <= 0:
            raise ValueError(f"gamma_cv must be positive, got {self.gamma_cv}")
        if self.arrival_rate < 0:
            raise ValueError(f"arrival_rate must be non-negative, got {self.arrival_rate}")
        if self.max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be positive, got {self.max_concurrency}")
        if self.max_tokens < 0:
            raise ValueError(f"max_tokens must be non-negative, got {self.max_tokens}")

    @classmethod
    def from_yaml(
        cls,
        path: str | None = None,
        preset: str | None = None,
        **overrides,
    ) -> "BenchmarkConfig":
        """Create a :class:`BenchmarkConfig` from a YAML file, decoupled from argparse.

        Parameters
        ----------
        path:
            Path (or ``http(s)://`` URL) to a agentsurge YAML config.  Falls back
            to the ``AGENTSURGE_CONFIG`` env-var, then the bundled
            ``configs/default.yaml``.
        preset:
            Optional preset name (e.g. ``"stress-default"``, ``"measure"``).
            Preset values are applied on top of the base config, then
            *overrides* are applied on top of the preset.
        **overrides:
            Any :class:`BenchmarkConfig` field can be passed as a keyword
            argument.  These take highest priority (override both config
            and preset values).  ``None``-valued overrides are ignored.

        Returns
        -------
        BenchmarkConfig

        Raises
        ------
        FileNotFoundError
            If an explicit *path* is given but does not exist.
        ValueError
            If the preset name is unknown.

        Examples
        --------
        >>> cfg = BenchmarkConfig.from_yaml()                       # bundled default
        >>> cfg = BenchmarkConfig.from_yaml("my_config.yaml")       # custom file
        >>> cfg = BenchmarkConfig.from_yaml(preset="measure")       # preset
        >>> cfg = BenchmarkConfig.from_yaml(                        # overrides
        ...     preset="stress-default",
        ...     vllm_url="http://gpu-node:8000",
        ...     max_concurrency=50,
        ... )
        """
        import os

        cfg = cls._load_yaml(path)

        vllm_cfg = cfg.get("vllm", {})
        run_cfg = cfg.get("runner", {})

        preset_cfg: dict = {}
        if preset is not None:
            from agentsurge.preset import preset_to_kwargs, resolve_preset_config

            raw_preset = resolve_preset_config(preset, yaml_cfg=cfg)
            preset_cfg = preset_to_kwargs(raw_preset)

        def _pick(key: str, default: object = None) -> Any:
            v = overrides.get(key)
            if v is not None:
                return v
            v = preset_cfg.get(key)
            if v is not None:
                return v
            v = run_cfg.get(key)
            if v is not None:
                return v
            return default

        # think_time needs special handling (list → tuple)
        think_time = _pick("think_time")
        if isinstance(think_time, list):
            think_time = tuple(think_time)

        # arrival_rate "auto" handling
        arrival_rate = _pick("arrival_rate", 10.0)
        if isinstance(arrival_rate, str) and arrival_rate.lower() == "auto":
            concurrency = _pick("max_concurrency", 100)
            arrival_rate = max(1.0, int(concurrency) * 0.3)
        else:
            arrival_rate = float(arrival_rate)

        # max_model_len: check per-model config
        _picked_model = _pick("model")
        _yaml_model = vllm_cfg.get("model")  # None means key absent
        if _picked_model is not None:
            model_path = _picked_model
        elif _yaml_model is not None:
            model_path = _yaml_model
        else:
            model_path = "default"
        max_model_len = overrides.get("max_model_len")
        if max_model_len is None:
            models_cfg = cfg.get("models", {})
            model_key = os.path.basename(model_path) if model_path else ""
            max_model_len = models_cfg.get(model_key, {}).get("max_model_len", 16384)

        # LMCache extra_metrics_urls
        extra_metrics_urls = overrides.get("extra_metrics_urls")
        if extra_metrics_urls is None:
            lmc_cfg = cfg.get("lmcache", {})
            lmc_url = lmc_cfg.get("url")
            lmc_enabled = lmc_cfg.get("enabled", False)
            extra_metrics_urls = [lmc_url] if lmc_enabled and lmc_url else []

        kwargs = dict(
            vllm_url=p
            if (p := _pick("vllm_url")) is not None
            else vllm_cfg.get("url", "http://localhost:8000"),
            model=model_path,
            max_concurrency=int(_pick("max_concurrency", 100)),
            max_tokens=int(_pick("max_tokens", 256)),
            temperature=float(_pick("temperature", 0.3)),
            api_type=_pick("api_type", "chat"),
            arrival_pattern=_pick("arrival_pattern", "poisson"),
            arrival_rate=arrival_rate,
            gamma_cv=float(_pick("gamma_cv", 1.2)),
            think_time=think_time,
            tool_delay=bool(_pick("tool_delay", False)),
            max_retries=int(_pick("max_retries", 0)),
            # Pass through raw value; __post_init__ resolves the default from
            # tool_mode so direct construction and from_yaml agree.
            use_model_reply_in_next_turn=_pick("use_model_reply_in_next_turn"),
            preset=preset,
            interruption_rate=float(_pick("interruption_rate", 0.0)),
            single_turn_ratio=float(_pick("single_turn_ratio", 0.0)),
            max_model_len=int(max_model_len),
            context_distribution=_pick("context_distribution", "mixed"),
            seed=int(_pick("seed", 42)),
            extra_metrics_urls=extra_metrics_urls,
            extra_body=overrides.get("extra_body"),
            request_timeout=int(_pick("request_timeout", 7200)),
            no_metrics=bool(_pick("no_metrics", False)),
            warm_up=int(_pick("warm_up", 0)),
            tool_output_mode=_pick("tool_output_mode", "recorded"),
            ramp_duration=float(_pick("ramp_duration", 0.0)),
            duration=float(_pick("duration", 0.0)),
            tool_call_parser_fallback=_pick("tool_call_parser_fallback", "off"),
            tool_mode=_pick("tool_mode", "off"),
            tool_call_parser=_pick("tool_call_parser", ""),
            workspace_dir=_pick("workspace_dir", _pick("sandbox_dir", None)),
            sandbox_dir=_pick("sandbox_dir", None),
            tool_env=_pick("tool_env", "safe"),
            save_responses=bool(_pick("save_responses", False)),
            no_stream=bool(_pick("no_stream", False)),
            ignore_replay_output_length=bool(_pick("ignore_replay_output_length", False)),
            thinking_budget=int(_pick("thinking_budget", 8192)),
            enable_thinking=bool(_pick("enable_thinking", False)),
            tokenizer_model=_pick("tokenizer_model", None),
            skip_tokenizer_load=bool(_pick("skip_tokenizer_load", False)),
            trust_remote_code=bool(_pick("trust_remote_code", False)),
            sanitize_truncated_tool_calls=bool(_pick("sanitize_truncated_tool_calls", False)),
            continue_turn_on=_pick("continue_turn_on", "never"),
            max_session_time=float(_pick("max_session_time", 0.0)),
            auto_finish_on_exhaust=bool(_pick("auto_finish_on_exhaust", False)),
            max_retry_turns=int(_pick("max_retry_turns", 0)),
            max_budget_usd=float(_pick("max_budget_usd", 0.0)),
            input_price_per_mtok=float(_pick("input_price_per_mtok", 0.0)),
            output_price_per_mtok=float(_pick("output_price_per_mtok", 0.0)),
            stream_idle_timeout=float(_pick("stream_idle_timeout", 0.0)),
            retry_profile=_pick("retry_profile", "default"),
            inflight_dump=bool(_pick("inflight_dump", False)),
            **({"continue_prompt": p} if (p := _pick("continue_prompt")) is not None else {}),
        )

        return cls(**kwargs)

    @staticmethod
    def _load_yaml(path: str | None = None) -> dict:
        """Load a agentsurge YAML config file.

        Resolution order:
          1. Explicit *path* argument (local file or ``http(s)://`` URL)
          2. ``AGENTSURGE_CONFIG`` environment variable
          3. Bundled ``configs/default.yaml``
        """
        import os
        from pathlib import Path as _Path

        import yaml

        resolved = path or os.environ.get("AGENTSURGE_CONFIG")
        if resolved:
            if resolved.startswith(("http://", "https://")):
                import urllib.request

                with urllib.request.urlopen(resolved, timeout=10) as resp:
                    return yaml.safe_load(resp.read()) or {}
            p = _Path(resolved)
            if not p.exists():
                raise FileNotFoundError(f"Config file not found: {resolved}")
            with open(p) as f:
                return yaml.safe_load(f) or {}

        # Bundled default - walk up from agentsurge/types/results.py to repo root
        default = _Path(__file__).parent.parent / "configs" / "default.yaml"
        if default.exists():
            with open(default) as f:
                return yaml.safe_load(f) or {}

        # No config found - return empty dict so caller gets field defaults
        return {}


@dataclass
class ConfigProfile:
    """A named server/model configuration for SLO comparison."""

    name: str
    vllm_url: str
    model: str
    description: str = ""
    overrides: dict = field(default_factory=dict)


@dataclass
class SloComparisonResult:
    """Aggregated result from comparing multiple configs under a single SLO target."""

    target_slo_ms: float
    entries: list[dict] = field(default_factory=list)
    """Each entry: {"profile": ConfigProfile-as-dict, "slo_result": <per-profile result dict>}."""
