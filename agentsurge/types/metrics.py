# SPDX-License-Identifier: MIT
"""Metrics snapshot types.

Defines ``LmcacheV1Metrics`` and ``MetricsSnapshot`` for point-in-time
server metrics readings.
"""

from dataclasses import dataclass, field


@dataclass
class LmcacheV1Metrics:
    """LMCache v1 metrics (colon-prefix format, no tier labels)."""

    retrieve_requests: float = 0
    store_requests: float = 0
    requested_tokens: float = 0
    hit_tokens: float = 0
    stored_tokens: float = 0
    hit_rate: float = 0.0
    local_cache_bytes: float = 0
    remote_cache_bytes: float = 0
    local_storage_bytes: float = 0
    cpu_evictions: float = 0
    disk_evictions: float = 0
    healthy: float = 1
    remote_read_bytes: float = 0
    remote_write_bytes: float = 0
    local_cpu_hot_cache_count: float = 0
    retrieve_latency_buckets: dict[str, float] = field(default_factory=dict)
    retrieve_latency_sum: float = 0
    retrieve_latency_count: float = 0
    store_latency_buckets: dict[str, float] = field(default_factory=dict)
    store_latency_sum: float = 0
    store_latency_count: float = 0
    remote_get_latency_buckets: dict[str, float] = field(default_factory=dict)
    remote_get_latency_sum: float = 0
    remote_get_latency_count: float = 0
    remote_put_latency_buckets: dict[str, float] = field(default_factory=dict)
    remote_put_latency_sum: float = 0
    remote_put_latency_count: float = 0
    local_hit_tokens: float = 0
    remote_hit_tokens: float = 0
    cpu_hit_tokens: float = 0
    disk_hit_tokens: float = 0
    disk_read_bytes: float = 0
    disk_write_bytes: float = 0
    disk_read_latency_buckets: dict[str, float] = field(default_factory=dict)
    disk_read_latency_sum: float = 0
    disk_read_latency_count: float = 0
    disk_write_latency_buckets: dict[str, float] = field(default_factory=dict)
    disk_write_latency_sum: float = 0
    disk_write_latency_count: float = 0
    slow_retrieval_by_time: float = 0
    slow_retrieval_by_speed: float = 0
    cache_lifespan_buckets: dict[str, float] = field(default_factory=dict)
    cache_lifespan_sum: float = 0
    cache_lifespan_count: float = 0
    cpu_evict_failed_count: float = 0
    get_blocking_failed_count: float = 0
    put_failed_count: float = 0
    retrieve_to_gpu_time_buckets: dict[str, float] = field(default_factory=dict)
    retrieve_to_gpu_time_sum: float = 0
    retrieve_to_gpu_time_count: float = 0
    store_from_gpu_time_buckets: dict[str, float] = field(default_factory=dict)
    store_from_gpu_time_sum: float = 0
    store_from_gpu_time_count: float = 0
    retrieve_speed: float = 0
    store_speed: float = 0
    tier_get_latency_buckets: dict[str, dict[str, float]] = field(default_factory=dict)
    tier_get_latency_sum: dict[str, float] = field(default_factory=dict)
    tier_get_latency_count: dict[str, float] = field(default_factory=dict)
    request_tier_served: dict[str, float] = field(default_factory=dict)
    request_tier_hit_tokens_buckets: dict[str, dict[str, float]] = field(default_factory=dict)
    request_tier_hit_tokens_sum: dict[str, float] = field(default_factory=dict)
    request_tier_hit_tokens_count: dict[str, float] = field(default_factory=dict)


# Backend-specific metric keys that live inside ``MetricsSnapshot.backend_metrics``.
# These were previously top-level dataclass fields; transparent attribute
# access is preserved via ``__getattr__`` / ``__setattr__`` for backward
# compatibility.
_SNAPSHOT_BACKEND_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "prefix_cache_hits_total",
        "prefix_cache_queries_total",
        "external_prefix_cache_hits_total",
        "external_prefix_cache_queries_total",
        "sglang_cache_hit_rate",
        "num_preemptions_total",
        "server_ttft_sum",
        "server_ttft_count",
        "prefill_time_sum",
        "prefill_time_count",
        "queue_time_sum",
        "queue_time_count",
        "lmcache_hits",
        "lmcache_latency_sum",
        "lmcache_latency_count",
        "lmcache_evictions",
        "lmcache_misses",
        "lmcache_memory_bytes",
        "lmcache_v1",
    }
)

# Default values for backend metrics (mirrors the old dataclass defaults).
_SNAPSHOT_BACKEND_METRIC_DEFAULTS: dict = {
    "prefix_cache_hits_total": 0.0,
    "prefix_cache_queries_total": 0.0,
    "external_prefix_cache_hits_total": 0.0,
    "external_prefix_cache_queries_total": 0.0,
    "sglang_cache_hit_rate": -1.0,
    "num_preemptions_total": 0.0,
    "server_ttft_sum": 0.0,
    "server_ttft_count": 0.0,
    "prefill_time_sum": 0.0,
    "prefill_time_count": 0.0,
    "queue_time_sum": 0.0,
    "queue_time_count": 0.0,
}

# Keys whose default is a mutable container - auto-created on first access.
_SNAPSHOT_BACKEND_METRIC_MUTABLE_DEFAULTS: dict[str, type] = {
    "lmcache_hits": dict,
    "lmcache_latency_sum": dict,
    "lmcache_latency_count": dict,
    "lmcache_evictions": dict,
    "lmcache_misses": dict,
    "lmcache_memory_bytes": dict,
}

# Special default: LmcacheV1Metrics sub-dataclass.
_SNAPSHOT_BACKEND_METRIC_FACTORY_DEFAULTS: dict[str, type] = {
    "lmcache_v1": LmcacheV1Metrics,
}


class MetricsSnapshot:
    """Single point-in-time metrics reading.

    **Core fields** (generic, backend-agnostic):

    * ``timestamp`` – monotonic clock time of the reading
    * ``kv_cache_usage_perc`` – KV cache usage fraction (0.0–1.0)
    * ``num_requests_running`` – currently running requests
    * ``num_requests_waiting`` – currently queued requests
    * ``isl_total`` – cumulative ISL (input/prefill) tokens processed
    * ``osl_total`` – cumulative OSL (output/decode) tokens produced
    * ``custom_metrics`` – user/engine-specific catch-all dict

    **Backend metrics** (backend-specific, stored in ``backend_metrics`` dict):

    Backend-specific counters (``prefix_cache_hits_total``,
    ``sglang_cache_hit_rate``, ``lmcache_hits``, ``lmcache_v1``, etc.) are
    stored inside the :pyattr:`backend_metrics` dictionary.  For **backward
    compatibility**, these keys are also accessible as regular attributes::

        snap.prefix_cache_hits_total      # reads  snap.backend_metrics["prefix_cache_hits_total"]
        snap.prefix_cache_hits_total = 50 # writes snap.backend_metrics["prefix_cache_hits_total"] = 50

    New code should prefer ``snap.backend_metrics[key]`` for clarity.
    """

    __slots__ = (
        "timestamp",
        "kv_cache_usage_perc",
        "num_requests_running",
        "num_requests_waiting",
        "isl_total",
        "osl_total",
        "custom_metrics",
        "backend_metrics",
    )

    # Snapshots are mutable and define __eq__: instances are intentionally
    # unhashable.  Declare it explicitly so callers can rely on the contract
    # without inferring it from Python's data-model fallback.
    __hash__ = None  # type: ignore[assignment]

    timestamp: float
    kv_cache_usage_perc: float
    num_requests_running: float
    num_requests_waiting: float
    isl_total: float
    osl_total: float
    custom_metrics: dict[str, float]
    backend_metrics: dict[str, object]

    def __init__(
        self,
        *,
        timestamp: float = 0.0,
        kv_cache_usage_perc: float = 0.0,
        num_requests_running: float = 0.0,
        num_requests_waiting: float = 0.0,
        isl_total: float = 0.0,
        osl_total: float = 0.0,
        custom_metrics: dict[str, float] | None = None,
        backend_metrics: dict | None = None,
        **kwargs: object,
    ) -> None:
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "kv_cache_usage_perc", kv_cache_usage_perc)
        object.__setattr__(self, "num_requests_running", num_requests_running)
        object.__setattr__(self, "num_requests_waiting", num_requests_waiting)
        object.__setattr__(self, "isl_total", isl_total)
        object.__setattr__(self, "osl_total", osl_total)
        object.__setattr__(
            self,
            "custom_metrics",
            dict(custom_metrics) if custom_metrics else {},
        )
        bm: dict = dict(backend_metrics) if backend_metrics else {}
        # Accept legacy keyword arguments and route them into backend_metrics
        for key, value in kwargs.items():
            if key in _SNAPSHOT_BACKEND_METRIC_KEYS:
                bm[key] = value
            else:
                raise TypeError(f"MetricsSnapshot() got an unexpected keyword argument '{key}'")
        object.__setattr__(self, "backend_metrics", bm)

    def __getattr__(self, name: str) -> object:
        if name in _SNAPSHOT_BACKEND_METRIC_KEYS:
            bm = object.__getattribute__(self, "backend_metrics")
            if name in bm:
                return bm[name]
            # Return a fresh empty container — do NOT store into backend_metrics.
            # Callers that need a persistent mutable object must use __setattr__
            # or access backend_metrics directly (e.g. via _get_dict / _get_v1).
            if name in _SNAPSHOT_BACKEND_METRIC_MUTABLE_DEFAULTS:
                return _SNAPSHOT_BACKEND_METRIC_MUTABLE_DEFAULTS[name]()
            # Auto-create factory defaults (e.g. LmcacheV1Metrics) and persist
            # them so that mutation via snap.lmcache_v1.field = x is visible on
            # the next read — this matches the established mutation-based API.
            if name in _SNAPSHOT_BACKEND_METRIC_FACTORY_DEFAULTS:
                val = _SNAPSHOT_BACKEND_METRIC_FACTORY_DEFAULTS[name]()
                bm[name] = val
                return val
            return _SNAPSHOT_BACKEND_METRIC_DEFAULTS.get(name, 0.0)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: object) -> None:
        if name in _SNAPSHOT_BACKEND_METRIC_KEYS:
            object.__getattribute__(self, "backend_metrics")[name] = value
        else:
            object.__setattr__(self, name, value)

    def _bm_float(self, key: str) -> float:
        """Read a backend metric as a float."""
        val = self.backend_metrics.get(key, _SNAPSHOT_BACKEND_METRIC_DEFAULTS.get(key, 0.0))
        return float(val) if isinstance(val, (int, float)) else 0.0

    @property
    def prefix_cache_hit_rate(self) -> float:
        queries = self._bm_float("prefix_cache_queries_total")
        if queries > 0:
            return self._bm_float("prefix_cache_hits_total") / queries
        return 0.0

    def __repr__(self) -> str:
        bm_summary: dict[str, object] = {}
        bm: dict[str, object] = self.backend_metrics
        for k, v in bm.items():
            default = _SNAPSHOT_BACKEND_METRIC_DEFAULTS.get(k, 0.0)
            if (
                isinstance(default, (int, float))
                and v != default
                or isinstance(v, dict)
                and v
                or isinstance(v, LmcacheV1Metrics)
            ):
                bm_summary[k] = v
        return (
            f"MetricsSnapshot(timestamp={self.timestamp!r}, "
            f"kv_cache_usage_perc={self.kv_cache_usage_perc!r}, "
            f"num_requests_running={self.num_requests_running!r}, "
            f"num_requests_waiting={self.num_requests_waiting!r}, "
            f"isl_total={self.isl_total!r}, "
            f"osl_total={self.osl_total!r}, "
            f"custom_metrics={self.custom_metrics!r}, "
            f"backend_metrics={bm_summary!r})"
        )

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict.

        ``backend_metrics`` is preserved as a nested dict; the special
        :class:`LmcacheV1Metrics` factory default is converted via
        :func:`dataclasses.asdict` so that the result is plain JSON-serializable.
        """
        from dataclasses import asdict, is_dataclass

        bm: dict[str, object] = {}
        for k, v in self.backend_metrics.items():
            if is_dataclass(v) and not isinstance(v, type):
                bm[k] = asdict(v)
            else:
                bm[k] = v
        return {
            "timestamp": self.timestamp,
            "kv_cache_usage_perc": self.kv_cache_usage_perc,
            "num_requests_running": self.num_requests_running,
            "num_requests_waiting": self.num_requests_waiting,
            "isl_total": self.isl_total,
            "osl_total": self.osl_total,
            "custom_metrics": dict(self.custom_metrics),
            "backend_metrics": bm,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MetricsSnapshot":
        """Inverse of :meth:`to_dict`. Converts nested lmcache_v1 dict back to
        :class:`LmcacheV1Metrics`."""
        bm = dict(d.get("backend_metrics", {}))
        lmc = bm.get("lmcache_v1")
        if isinstance(lmc, dict):
            bm["lmcache_v1"] = LmcacheV1Metrics(**lmc)
        return cls(
            timestamp=d.get("timestamp", 0.0),
            kv_cache_usage_perc=d.get("kv_cache_usage_perc", 0.0),
            num_requests_running=d.get("num_requests_running", 0.0),
            num_requests_waiting=d.get("num_requests_waiting", 0.0),
            isl_total=d.get("isl_total", 0.0),
            osl_total=d.get("osl_total", 0.0),
            custom_metrics=dict(d.get("custom_metrics", {})),
            backend_metrics=bm,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MetricsSnapshot):
            return NotImplemented
        if not (
            self.timestamp == other.timestamp
            and self.kv_cache_usage_perc == other.kv_cache_usage_perc
            and self.num_requests_running == other.num_requests_running
            and self.num_requests_waiting == other.num_requests_waiting
            and self.isl_total == other.isl_total
            and self.osl_total == other.osl_total
            and self.custom_metrics == other.custom_metrics
        ):
            return False
        # Compare effective backend metric values so that instances with default
        # values stored explicitly equal instances where the default is implicit.
        # Read via backend_metrics.get(...) — going through getattr/__getattr__
        # would route factory-default keys (e.g. "lmcache_v1") into the
        # autocreate branch and mutate backend_metrics as a side effect of ==.
        for key in _SNAPSHOT_BACKEND_METRIC_KEYS:
            if key in _SNAPSHOT_BACKEND_METRIC_FACTORY_DEFAULTS:
                default: object = _SNAPSHOT_BACKEND_METRIC_FACTORY_DEFAULTS[key]()
            elif key in _SNAPSHOT_BACKEND_METRIC_MUTABLE_DEFAULTS:
                default = _SNAPSHOT_BACKEND_METRIC_MUTABLE_DEFAULTS[key]()
            else:
                default = _SNAPSHOT_BACKEND_METRIC_DEFAULTS.get(key, 0.0)
            if self.backend_metrics.get(key, default) != other.backend_metrics.get(key, default):
                return False
        return True
