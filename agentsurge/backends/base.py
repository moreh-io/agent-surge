# SPDX-License-Identifier: MIT
"""Abstract base classes for agentsurge inference backends.

Every backend adapter must subclass :class:`BackendBase` and implement the
abstract methods defined here.  This module intentionally has **no** heavy
third-party imports so it can be imported at package-init time without
pulling in ``aiohttp``, ``openai``, or any serving-framework SDK.

Hierarchy
---------
BackendBase              – async context-manager with ``send_turn`` + ``health_check``
BackendConfig            – lightweight dataclass carrying connection / request params
RequestAdapter           – transform workload messages into engine-specific payloads
DefaultRequestAdapter    – pass-through adapter (identity transform)
MetricsAdapter           – parse engine-specific metrics into unified MetricsSnapshot
MetricsAdapterProtocol   – structural protocol (PEP 544) for duck-typed adapters
PrometheusMetricsAdapter – default adapter for Prometheus text format (vLLM / SGLang)
"""

import abc
import dataclasses as _dataclasses
import json
import typing
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agentsurge.types import TurnResult

if typing.TYPE_CHECKING:
    from agentsurge.types import MetricsSnapshot  # noqa: F811


def _auto_register(cls: type) -> None:
    from agentsurge.backends.registry import backend_registry

    backend_registry.register(cls)


@dataclass
class BackendConfig:
    """Connection and request parameters shared by all backends.

    Individual backend implementations may extend this with backend-specific
    fields (e.g. ``api_key``, ``tls_cert_path``).  The base fields cover the
    common denominator needed by every backend.

    Parameters
    ----------
    base_url : str
        Root URL of the inference server (e.g. ``"http://localhost:8000"``).
    model : str
        Model identifier accepted by the server.
    max_tokens : int
        Maximum tokens to generate per request.
    temperature : float
        Sampling temperature.
    request_timeout : int
        Per-request HTTP timeout in seconds.
    stream : bool
        Whether to request streaming (SSE) responses.  Backends that do not
        support streaming may ignore this.
    extra_body : dict | None
        Arbitrary extra fields merged into the request payload (backend-
        specific extensions like ``guided_json``, ``top_k``, etc.).
    stream_idle_timeout : float
        Per-read socket timeout in seconds for streaming responses.  ``0.0``
        (default) disables the idle timeout, matching aiohttp's behaviour when
        ``sock_read=None``.  Set to a positive value (e.g. ``300.0``) to abort
        a stalled SSE stream after that many seconds of silence.
    connection_limit : int
        Maximum number of simultaneous TCP connections opened by the
        underlying ``aiohttp.TCPConnector``.  Defaults to ``1024`` to provide
        backpressure under high concurrency; set to ``0`` only if the
        deployment has explicit kernel-level fd / ephemeral-port limits.
    connection_limit_per_host : int
        Maximum simultaneous connections per ``(host, port)`` pair.  ``0``
        (default) disables the per-host cap; set positive when one backend
        host should not absorb every connection.
    """

    base_url: str = "http://localhost:8000"
    model: str = ""
    max_tokens: int = 256
    temperature: float = 0.3
    request_timeout: int = 7200
    stream: bool = True
    extra_body: dict[str, Any] | None = None
    stream_idle_timeout: float = 0.0
    connection_limit: int = 1024
    connection_limit_per_host: int = 0

    def __post_init__(self) -> None:
        if self.request_timeout <= 0:
            raise ValueError(f"request_timeout must be positive, got {self.request_timeout}")


class RequestAdapter(abc.ABC):
    """Abstract interface for transforming workload requests into engine-specific formats.

    A :class:`RequestAdapter` sits between the runner and the backend,
    translating the generic OpenAI-style message list (plus backend config)
    into an engine-specific request payload ``dict``.  This decouples the
    workload representation from the wire format so that:

    * New serving engines can be supported by writing an adapter instead of a
      full backend.
    * Users can inject pre/post-processing (prompt rewriting, guardrails,
      token counting) without modifying either the runner or the backend.

    Subclass contract
    -----------------
    1. ``adapt`` must be a **pure** transform - no I/O, no side-effects.
    2. The returned ``dict`` must be suitable for ``json.dumps`` (JSON-serializable).
    3. Implementations must be thread-safe and re-entrant (the runner may call
       ``adapt`` concurrently from multiple asyncio tasks).
    4. ``validate`` performs input validation before ``adapt``; raise
       :class:`ValueError` for invalid inputs (default: basic message-list check).
    5. ``serialize`` converts the payload dict to a wire-ready string
       (default: ``json.dumps``); override for msgpack, protobuf, etc.

    Example
    -------
    >>> class MyAdapter(RequestAdapter):
    ...     def adapt(self, messages, config, **kwargs):
    ...         payload = {
    ...             "model": config.model,
    ...             "messages": messages,
    ...             "max_tokens": config.max_tokens,
    ...         }
    ...         payload["custom_field"] = "value"
    ...         return payload
    """

    @abc.abstractmethod
    def adapt(
        self,
        messages: list[dict[str, Any]],
        config: BackendConfig,
        *,
        session_id: str = "",
        turn_index: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Transform a workload request into an engine-specific payload.

        Parameters
        ----------
        messages : list[dict[str, Any]]
            OpenAI-compatible list of message dicts (``role`` + ``content``).
        config : BackendConfig
            The backend configuration carrying model, temperature, etc.
        session_id : str
            Identifier for the session (informational; may be empty).
        turn_index : int
            Zero-based turn index within the session (informational).
        **kwargs : Any
            Reserved for future extension.  Implementations should accept
            and ignore unknown keyword arguments for forward compatibility.

        Returns
        -------
        dict[str, Any]
            JSON-serializable request payload ready to be sent to the
            inference engine.  The exact schema depends on the target
            engine (e.g. OpenAI chat completions, vLLM responses API,
            TensorRT-LLM, etc.).
        """
        ...

    def validate(
        self,
        messages: list[dict[str, Any]],
        config: BackendConfig,
    ) -> None:
        """Validate the request inputs before transformation.

        The default implementation performs basic structural checks on
        *messages*:

        * *messages* must be a :class:`list`.
        * Each element must be a :class:`dict`.
        * Each dict must contain the key ``"role"``.

        Override this method to add engine-specific validation (e.g. maximum
        context length, required fields, role sequence constraints).

        Parameters
        ----------
        messages : list[dict[str, Any]]
            The candidate message list to validate.
        config : BackendConfig
            The backend configuration (may be inspected for limits, model
            name, etc.).

        Raises
        ------
        ValueError
            If any structural constraint is violated.

        Examples
        --------
        >>> adapter = DefaultRequestAdapter()
        >>> adapter.validate([{"role": "user", "content": "hi"}], BackendConfig())
        >>> # raises ValueError:
        >>> adapter.validate("not a list", BackendConfig())  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: messages must be a list, got str
        """
        if not isinstance(messages, list):
            raise ValueError(f"messages must be a list, got {type(messages).__name__}")
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"messages[{i}] must be a dict, got {type(msg).__name__}")
            if "role" not in msg:
                raise ValueError(f"messages[{i}] is missing required key 'role'")

    def serialize(self, payload: dict[str, Any]) -> str:
        """Serialize the engine-specific payload to a wire-ready string.

        The default implementation calls :func:`json.dumps` with no extra
        options.  Override for alternative formats (msgpack, protobuf, CBOR,
        custom JSON encoder, etc.) or to add separators/encoding options.

        Parameters
        ----------
        payload : dict[str, Any]
            The payload dict returned by :meth:`adapt`.

        Returns
        -------
        str
            A JSON (or format-specific) string representation of *payload*
            suitable for transmission over the wire.

        Raises
        ------
        TypeError
            If *payload* contains non-JSON-serializable values (default
            implementation only).

        Examples
        --------
        >>> adapter = DefaultRequestAdapter()
        >>> cfg = BackendConfig(model="m")
        >>> payload = adapter.adapt([{"role": "user", "content": "hi"}], cfg)
        >>> wire = adapter.serialize(payload)
        >>> import json; assert json.loads(wire)["model"] == "m"
        """
        return json.dumps(payload)


@typing.runtime_checkable
class RequestAdapterProtocol(typing.Protocol):
    """Structural protocol for request adapters (PEP 544 / :pep:`544`).

    Use this for type annotations when you want to accept *any* object that
    exposes the request adapter interface, without requiring inheritance from
    :class:`RequestAdapter`.  This is useful for duck-typing and for third-
    party adapters that cannot subclass the ABC.

    A class satisfies this protocol if and only if it has the three methods:
    ``adapt``, ``validate``, and ``serialize``.

    Examples
    --------
    >>> isinstance(DefaultRequestAdapter(), RequestAdapterProtocol)
    True

    >>> class DuckAdapter:
    ...     def adapt(self, messages, config, **kw): return {}
    ...     def validate(self, messages, config): pass
    ...     def serialize(self, payload): return "{}"
    >>> isinstance(DuckAdapter(), RequestAdapterProtocol)
    True
    """

    def adapt(
        self,
        messages: list[dict[str, Any]],
        config: BackendConfig,
        *,
        session_id: str = "",
        turn_index: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Transform messages to an engine-specific payload dict."""
        ...

    def validate(
        self,
        messages: list[dict[str, Any]],
        config: BackendConfig,
    ) -> None:
        """Validate request inputs; raise ``ValueError`` if invalid."""
        ...

    def serialize(
        self,
        payload: dict[str, Any],
    ) -> str:
        """Serialize payload dict to a wire-ready string."""
        ...


class DefaultRequestAdapter(RequestAdapter):
    """Pass-through request adapter (identity transform).

    Builds a standard OpenAI chat-completions-style payload from the
    message list and :class:`BackendConfig` fields.  This is the adapter
    used when no custom adapter is provided.

    The resulting payload contains:

    * ``model`` - from ``config.model``
    * ``messages`` - the input message list, unchanged
    * ``max_tokens`` - from ``config.max_tokens``
    * ``temperature`` - from ``config.temperature``
    * ``stream`` - from ``config.stream``

    If ``config.extra_body`` is set, its entries are merged into the
    payload (existing keys take precedence).
    """

    def adapt(
        self,
        messages: list[dict[str, Any]],
        config: BackendConfig,
        *,
        session_id: str = "",
        turn_index: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a standard OpenAI-compatible chat completions payload."""
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "stream": config.stream,
        }
        if config.extra_body:
            for k, v in config.extra_body.items():
                if k not in payload:
                    payload[k] = v
        return payload


class BackendBase(abc.ABC):
    """Abstract async interface that every inference backend must implement.

    A backend is an **async context manager** - callers use it as::

        async with MyBackend(config) as backend:
            result = await backend.send_turn(session_id, turn_idx, messages)

    Lifecycle
    ---------
    * ``__aenter__`` - acquire resources (HTTP session, gRPC channel, …).
    * ``__aexit__``  - release resources.
    * ``send_turn``  - fire one LLM request and return a :class:`TurnResult`.
    * ``health_check`` - verify the backend is reachable (optional best-effort).

    Subclass contract
    -----------------
    1. ``__init__`` accepts a :class:`BackendConfig` (or a subclass).
    2. ``send_turn`` must be safe to call concurrently from multiple tasks.
    3. The backend must **not** maintain mutable per-session state; the runner
       owns session-level orchestration.
    """

    # Unique, short name used in CLI flags and registry lookups.
    # Subclasses MUST override this with a class-level str, e.g. ``name = "vllm"``.
    name: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-register concrete backends with the global registry."""
        super().__init_subclass__(**kwargs)
        _auto_register(cls)

    def __init__(self, config: BackendConfig) -> None:
        self.config = config

    async def __aenter__(self) -> "BackendBase":
        """Acquire backend resources (HTTP pool, gRPC channel, …).

        The default implementation is a no-op; override in subclasses that
        need setup.
        """
        return self

    async def __aexit__(  # noqa: B027
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        """Release backend resources.

        The default implementation is a no-op; override in subclasses.
        """

    @abc.abstractmethod
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
        """Send a single conversation turn and return the result.

        Parameters
        ----------
        session_id : str
            Identifier for the session this turn belongs to.
        turn_index : int
            Zero-based index of the turn within the session.
        messages : list[dict[str, Any]]
            OpenAI-compatible list of message dicts (``role`` + ``content``).
        tools : list[dict[str, Any]] | None
            Optional OpenAI-compatible tool definitions to attach to the
            request.  Backends that do not support tools may ignore this.
        max_tokens : int | None
            Per-turn output-token cap computed by the runner.  When omitted,
            backends use their configured default.
        capture_text : bool
            Whether to retain streamed response text in ``TurnResult``.
        session_meta : dict[str, Any] | None
            Session metadata reserved for backend-specific routing or logging.

        Returns
        -------
        TurnResult
            Populated result including timing (``ttft_ms``, ``total_ms``),
            token counts, and error information.

        Raises
        ------
        agentsurge.exceptions.BackendError
            On unrecoverable backend failures.  Transient errors should be
            reflected in :pyattr:`TurnResult.completed` = ``False`` instead.
        """
        ...

    async def health_check(self) -> bool:
        """Return ``True`` if the backend is reachable and healthy.

        The default implementation always returns ``True``.  Backends that
        expose a health/readiness endpoint should override this.
        """
        return True

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} url={self.config.base_url!r}>"


class MetricsAdapter(abc.ABC):
    """Abstract interface for collecting and normalizing engine-specific metrics.

    A :class:`MetricsAdapter` transforms raw metrics output from an inference
    engine into the unified :class:`~agentsurge.types.MetricsSnapshot` format.
    This decouples the metrics collection logic from any particular engine's
    exposition format (Prometheus, JSON, custom, etc.) so that:

    * New serving engines can expose metrics in any format - only a new
      adapter is needed, not changes to the collector or analyzer.
    * The same :class:`MetricsSnapshot` schema is used throughout agentsurge
      regardless of the backend, enabling uniform analysis and comparison.

    Subclass contract
    -----------------
    1. :meth:`parse` must be a **pure** transform - no I/O, no side-effects.
       Given raw text (or bytes), it returns a populated
       :class:`MetricsSnapshot`.  Unknown / unsupported metrics should be
       silently ignored (not raise).
    2. :meth:`endpoint_url` returns the URL path to poll for metrics
       (e.g. ``"/metrics"``).  Subclasses may override to point at a
       different endpoint.
    3. :meth:`merge` combines an ``extra`` snapshot into a ``base`` snapshot
       (e.g. merging LMCache sidecar metrics into the main engine snapshot).
       The default implementation is a no-op; override when extra metrics
       sources are relevant.
    4. Implementations must be **thread-safe** and **stateless** - the
       collector may call ``parse`` concurrently from multiple asyncio tasks.

    Example
    -------
    >>> class MyEngineMetrics(MetricsAdapter):
    ...     name = "my-engine"
    ...     def parse(self, raw: str) -> MetricsSnapshot:
    ...         snap = MetricsSnapshot(timestamp=time.monotonic())
    ...         for line in raw.splitlines():
    ...             if line.startswith("kv_usage"):
    ...                 snap.kv_cache_usage_perc = float(line.split()[-1])
    ...         return snap
    """

    # Short identifier for this adapter (e.g. ``"prometheus"``, ``"json"``).
    # Subclasses should override with a descriptive class-level string.
    name: str = ""

    @abc.abstractmethod
    def parse(self, raw: str) -> "MetricsSnapshot":
        """Parse raw metrics text into a :class:`MetricsSnapshot`.

        Parameters
        ----------
        raw : str
            The raw metrics output from the engine (e.g. Prometheus text
            exposition format, JSON string, etc.).

        Returns
        -------
        MetricsSnapshot
            A fully populated snapshot.  Fields that cannot be extracted
            from the raw input are left at their default values.
        """
        ...

    def endpoint_url(self, base_url: str) -> str:
        """Return the full URL to poll for metrics.

        Parameters
        ----------
        base_url : str
            The engine's root URL (e.g. ``"http://localhost:8000"``).

        Returns
        -------
        str
            The full metrics endpoint URL.  The default implementation
            appends ``"/metrics"`` to *base_url*.
        """
        return f"{base_url.rstrip('/')}/metrics"

    def merge(
        self,
        base: "MetricsSnapshot",
        extra: "MetricsSnapshot",
    ) -> "MetricsSnapshot":
        """Merge an *extra* snapshot into *base* (in-place) and return *base*.

        This is used when metrics come from multiple endpoints (e.g. the
        main engine ``/metrics`` plus an LMCache sidecar).  The default
        implementation returns *base* unchanged; subclasses should override
        to copy relevant fields from *extra* into *base*.

        Parameters
        ----------
        base : MetricsSnapshot
            The primary snapshot (mutated in place).
        extra : MetricsSnapshot
            Additional metrics to merge.

        Returns
        -------
        MetricsSnapshot
            The *base* snapshot (same object, after mutation).
        """
        return base

    def aggregate(
        self,
        snapshots: "Sequence[MetricsSnapshot]",
    ) -> dict[str, Any]:
        """Aggregate a time-series of snapshots into summary statistics.

        Computes descriptive statistics (mean, min, max, p50, p95, p99) for
        the primary numeric metrics across the given list of snapshots.

        This method covers the **aggregation contract**: given a window of
        observations, it returns a dict that is immediately suitable for
        reporting, dashboarding, or writing into a :class:`RunResult`.

        Subclass contract
        -----------------
        * Must return an empty ``{}`` when *snapshots* is empty.
        * Returned dict must be JSON-serializable.
        * Implementations must be **pure** - no I/O, no side-effects.

        Parameters
        ----------
        snapshots : Sequence[MetricsSnapshot]
            An ordered collection of snapshots (e.g. all readings during a
            benchmark run).  May be empty.

        Returns
        -------
        dict[str, Any]
            Summary statistics dict with the following keys when *snapshots*
            is non-empty:

            * ``count`` – number of snapshots
            * ``kv_mean``, ``kv_min``, ``kv_max``,
              ``kv_p50``, ``kv_p95``, ``kv_p99``
            * ``running_mean``, ``running_max``
            * ``waiting_mean``, ``waiting_max``
            * ``duration_secs`` – wall time span covered (if timestamps > 0)

        Examples
        --------
        >>> adapter = PrometheusMetricsAdapter()
        >>> snaps = [
        ...     MetricsSnapshot(timestamp=0.0, kv_cache_usage_perc=0.2),
        ...     MetricsSnapshot(timestamp=1.0, kv_cache_usage_perc=0.8),
        ... ]
        >>> stats = adapter.aggregate(snaps)
        >>> stats["kv_mean"]
        0.5
        >>> stats["duration_secs"]
        1.0
        """
        if not snapshots:
            return {}

        def _percentile(values: list[float], p: float) -> float:
            if not values:
                return 0.0
            sorted_vals = sorted(values)
            k = (len(sorted_vals) - 1) * p / 100.0
            lo = int(k)
            hi = min(lo + 1, len(sorted_vals) - 1)
            return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)

        kv_vals = [s.kv_cache_usage_perc for s in snapshots]
        running_vals = [s.num_requests_running for s in snapshots]
        waiting_vals = [s.num_requests_waiting for s in snapshots]
        timestamps = [s.timestamp for s in snapshots if s.timestamp > 0.0]

        result: dict[str, Any] = {
            "count": len(snapshots),
            "kv_mean": sum(kv_vals) / len(kv_vals),
            "kv_min": min(kv_vals),
            "kv_max": max(kv_vals),
            "kv_p50": _percentile(kv_vals, 50),
            "kv_p95": _percentile(kv_vals, 95),
            "kv_p99": _percentile(kv_vals, 99),
            "running_mean": sum(running_vals) / len(running_vals),
            "running_max": max(running_vals),
            "waiting_mean": sum(waiting_vals) / len(waiting_vals),
            "waiting_max": max(waiting_vals),
        }
        if len(timestamps) >= 2:
            result["duration_secs"] = max(timestamps) - min(timestamps)
        return result

    def export(
        self,
        snapshot: "MetricsSnapshot",
    ) -> dict[str, Any]:
        """Export a snapshot to a serializable flat dictionary.

        This method covers the **export contract**: it converts a
        :class:`~agentsurge.types.MetricsSnapshot` into a plain Python dict
        suitable for JSON serialization, writing to disk, emitting via a
        logging sink, or passing to external monitoring systems.

        Subclass contract
        -----------------
        * Returned dict must be JSON-serializable.
        * Core generic fields (``timestamp``, ``kv_cache_usage_perc``, etc.)
          must always be present in the output, even when zero.
        * ``backend_metrics`` entries that are non-empty (dicts, non-zero
          scalars, dataclass instances) should be included; empty/zero values
          may be omitted to keep output compact.
        * Implementations must be **pure** - no I/O, no side-effects.

        Parameters
        ----------
        snapshot : MetricsSnapshot
            The snapshot to serialize.

        Returns
        -------
        dict[str, Any]
            Flat mapping of metric name → value.  All values must be JSON-
            serializable (``float``, ``int``, ``str``, ``dict``, ``list``).

        Examples
        --------
        >>> adapter = PrometheusMetricsAdapter()
        >>> snap = MetricsSnapshot(timestamp=1.0, kv_cache_usage_perc=0.42)
        >>> d = adapter.export(snap)
        >>> d["kv_cache_usage_perc"]
        0.42
        >>> "timestamp" in d
        True
        """
        result: dict[str, Any] = {
            "timestamp": snapshot.timestamp,
            "kv_cache_usage_perc": snapshot.kv_cache_usage_perc,
            "num_requests_running": snapshot.num_requests_running,
            "num_requests_waiting": snapshot.num_requests_waiting,
            "isl_total": snapshot.isl_total,
            "osl_total": snapshot.osl_total,
        }
        if snapshot.custom_metrics:
            result["custom_metrics"] = dict(snapshot.custom_metrics)
        for key, value in snapshot.backend_metrics.items():
            if isinstance(value, dict):
                if value:
                    result[key] = dict(value)
            elif _dataclasses.is_dataclass(value) and not isinstance(value, type):
                result[key] = _dataclasses.asdict(value)
            elif isinstance(value, (int, float)) and value != 0.0:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


@typing.runtime_checkable
class MetricsAdapterProtocol(typing.Protocol):
    """Structural protocol for metrics adapters (:pep:`544`).

    Use this for type annotations when you want to accept *any* object that
    exposes the metrics adapter interface without requiring inheritance from
    :class:`MetricsAdapter`.  This is useful for duck-typing and for third-
    party adapters that cannot subclass the ABC.

    A class satisfies this protocol if and only if it has all five methods:
    ``parse``, ``endpoint_url``, ``merge``, ``aggregate``, and ``export``.

    Because this protocol is decorated with :func:`typing.runtime_checkable`,
    ``isinstance`` checks work at runtime for structural compatibility.

    Examples
    --------
    >>> isinstance(PrometheusMetricsAdapter(), MetricsAdapterProtocol)
    True

    >>> class DuckAdapter:
    ...     def parse(self, raw): return MetricsSnapshot(timestamp=0.0)
    ...     def endpoint_url(self, base_url): return base_url + "/metrics"
    ...     def merge(self, base, extra): return base
    ...     def aggregate(self, snapshots): return {}
    ...     def export(self, snapshot): return {}
    >>> isinstance(DuckAdapter(), MetricsAdapterProtocol)
    True
    """

    def parse(self, raw: str) -> "MetricsSnapshot":
        """Parse raw metrics text into a :class:`~agentsurge.types.MetricsSnapshot`."""
        ...

    def endpoint_url(self, base_url: str) -> str:
        """Return the full URL to poll for metrics."""
        ...

    def merge(
        self,
        base: "MetricsSnapshot",
        extra: "MetricsSnapshot",
    ) -> "MetricsSnapshot":
        """Merge an extra snapshot into base; return base."""
        ...

    def aggregate(
        self,
        snapshots: "Sequence[MetricsSnapshot]",
    ) -> dict[str, Any]:
        """Aggregate a time-series of snapshots into summary statistics."""
        ...

    def export(
        self,
        snapshot: "MetricsSnapshot",
    ) -> dict[str, Any]:
        """Export a snapshot to a serializable flat dictionary."""
        ...


class PrometheusMetricsAdapter(MetricsAdapter):
    """Default metrics adapter for Prometheus text exposition format.

    Handles vLLM and SGLang ``/metrics`` endpoints by delegating to
    :func:`agentsurge.metrics.parse_metrics`.

    This is the adapter used when no custom metrics adapter is provided.
    It is stateless and safe for concurrent use.
    """

    name: str = "prometheus"

    def parse(self, raw: str) -> "MetricsSnapshot":
        """Parse Prometheus text format into a :class:`MetricsSnapshot`.

        Auto-detects vLLM vs SGLang by metric prefix (``vllm:`` vs
        ``sglang:``).

        Parameters
        ----------
        raw : str
            Prometheus text exposition format string.

        Returns
        -------
        MetricsSnapshot
        """
        from agentsurge.metrics import parse_metrics

        return parse_metrics(raw)
