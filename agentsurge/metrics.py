# SPDX-License-Identifier: MIT
"""vLLM /metrics endpoint parser and async collector."""

import asyncio
import contextlib
import dataclasses
import logging
import re
import ssl as _ssl_mod
import time
from dataclasses import dataclass, field
from typing import cast

import aiohttp

# Canonical definitions live in agentsurge.types; re-exported here for backward
# compatibility so that ``from agentsurge.metrics import MetricsSnapshot`` still works.
from agentsurge.types import LmcacheV1Metrics, MetricsSnapshot  # noqa: F401

_GAUGE_FIELDS = {"hit_rate", "healthy", "retrieve_speed", "store_speed"}


def _bm(snap: MetricsSnapshot) -> dict[str, object]:
    """Return the ``backend_metrics`` dict bypassing ``__getattr__``."""
    return cast(dict[str, object], object.__getattribute__(snap, "backend_metrics"))


def _get_v1(snap: MetricsSnapshot) -> LmcacheV1Metrics:
    """Return the ``lmcache_v1`` sub-object with the correct static type."""
    bm = _bm(snap)
    v1 = bm.get("lmcache_v1")
    if not isinstance(v1, LmcacheV1Metrics):
        v1 = LmcacheV1Metrics()
        bm["lmcache_v1"] = v1
    return v1


def _merge_lmcache_v1(target: LmcacheV1Metrics, source: LmcacheV1Metrics) -> None:
    """Merge *source* into *target*. Gauge fields use ``max``; counters use ``sum``."""
    for f in dataclasses.fields(source):
        val = getattr(source, f.name)
        if isinstance(val, dict):
            getattr(target, f.name).update(val)
        elif isinstance(val, (int, float)) and val:
            cur = getattr(target, f.name, 0)
            if f.name in _GAUGE_FIELDS:
                setattr(target, f.name, max(cur, val))
            else:
                setattr(target, f.name, cur + val)


def _get_dict(snap: MetricsSnapshot, key: str) -> dict[str, float]:
    """Return a backend-metrics dict entry with the correct static type."""
    bm = _bm(snap)
    val = bm.get(key)
    if not isinstance(val, dict):
        val = {}
        bm[key] = val
    return cast(dict[str, float], val)


def _get_float(snap: MetricsSnapshot, key: str, default: float = 0.0) -> float:
    """Return a backend-metrics float entry with the correct static type."""
    bm = _bm(snap)
    val = bm.get(key, default)
    if isinstance(val, (int, float)):
        return float(val)
    return default


def _snap_float(snap: MetricsSnapshot, slot: str) -> float:
    """Read a core float slot from ``MetricsSnapshot`` bypassing ``__getattr__``."""
    val = object.__getattribute__(snap, slot)
    if isinstance(val, (int, float)):
        return float(val)
    return 0.0


_log = logging.getLogger(__name__)
_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')
_sglang_warned = False


def _extract_labels(line: str) -> dict[str, str]:
    """Extract Prometheus labels from a metric line."""
    brace_start = line.find("{")
    brace_end = line.find("}")
    if brace_start < 0 or brace_end < 0:
        return {}
    return dict(_LABEL_RE.findall(line[brace_start : brace_end + 1]))


def parse_metrics(text: str) -> MetricsSnapshot:
    """Parse Prometheus text format from vLLM or SGLang /metrics endpoint.

    Auto-detects backend by metric prefix (vllm: vs sglang:).

    .. warning::
       SGLang support is **experimental** and has not been validated against
       a real SGLang deployment. Metric names and semantics may differ from
       what is implemented here. Use with caution.
    """
    snap = MetricsSnapshot(timestamp=time.monotonic())
    is_sglang = "sglang:" in text

    for line in text.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        try:
            val = float(line.split()[-1])
        except (ValueError, IndexError):
            continue

        if is_sglang:
            _parse_sglang_line(line, val, snap)
        else:
            _parse_vllm_line(line, val, snap)
    return snap


def histogram_percentiles(
    buckets: dict[str, float],
    count: float,
    percentiles: tuple[int, ...] = (50, 95, 99),
) -> dict[str, float]:
    """Compute percentiles from Prometheus histogram buckets via linear interpolation."""
    if count <= 0:
        return {f"p{p}": float("nan") for p in percentiles}
    finite_buckets = {le: cnt for le, cnt in buckets.items() if le != "+Inf"}
    if any(cnt < 0 for cnt in finite_buckets.values()):
        return {f"p{p}": float("nan") for p in percentiles}
    sorted_bounds = sorted(
        ((float(le), cnt) for le, cnt in finite_buckets.items()),
        key=lambda x: x[0],
    )
    # Cumulative bucket counts must be non-decreasing along sorted bounds.
    # A counter reset or out-of-order scrape can break this after
    # _bucket_delta() subtracts two snapshots; refuse to interpolate.
    prev_cum = -1.0
    for _, cum in sorted_bounds:
        if cum < prev_cum:
            return {f"p{p}": float("nan") for p in percentiles}
        prev_cum = cum
    result = {}
    for p in percentiles:
        target = count * p / 100.0
        prev_bound, prev_count = 0.0, 0.0
        found = False
        for bound, cum_count in sorted_bounds:
            if cum_count >= target:
                if cum_count == prev_count:
                    result[f"p{p}"] = bound
                else:
                    fraction = (target - prev_count) / (cum_count - prev_count)
                    result[f"p{p}"] = prev_bound + fraction * (bound - prev_bound)
                found = True
                break
            prev_bound, prev_count = bound, cum_count
        if not found:
            # Tail sits in the +Inf bucket (count > sum of finite bucket
            # counts). Returning the largest finite bound silently saturates
            # — surface +inf so callers can see the histogram was clipped.
            result[f"p{p}"] = float("inf") if sorted_bounds else float("nan")
    return result


_V1_BUCKET_MAP: dict[str, str] = {
    "lmcache:time_to_retrieve_bucket": "retrieve_latency_buckets",
    "lmcache:time_to_store_bucket": "store_latency_buckets",
    "lmcache:remote_time_to_get_bucket": "remote_get_latency_buckets",
    "lmcache:remote_time_to_put_bucket": "remote_put_latency_buckets",
    "lmcache:local_disk_read_latency_bucket": "disk_read_latency_buckets",
    "lmcache:local_disk_write_latency_bucket": "disk_write_latency_buckets",
    "lmcache:request_cache_lifespan_bucket": "cache_lifespan_buckets",
    "lmcache:retrieve_to_gpu_time_bucket": "retrieve_to_gpu_time_buckets",
    "lmcache:store_from_gpu_time_bucket": "store_from_gpu_time_buckets",
}

_V1_TIERED_BUCKET_MAP: dict[str, str] = {
    "lmcache:tier_get_latency_bucket": "tier_get_latency_buckets",
    "lmcache:request_tier_hit_tokens_bucket": "request_tier_hit_tokens_buckets",
}

_V1_SCALAR_MAP: dict[str, str] = {
    "lmcache:time_to_retrieve_sum": "retrieve_latency_sum",
    "lmcache:time_to_retrieve_count": "retrieve_latency_count",
    "lmcache:time_to_store_sum": "store_latency_sum",
    "lmcache:time_to_store_count": "store_latency_count",
    "lmcache:remote_time_to_get_sum": "remote_get_latency_sum",
    "lmcache:remote_time_to_get_count": "remote_get_latency_count",
    "lmcache:remote_time_to_put_sum": "remote_put_latency_sum",
    "lmcache:remote_time_to_put_count": "remote_put_latency_count",
    "lmcache:local_cpu_hot_cache_count": "local_cpu_hot_cache_count",
    "lmcache:num_retrieve_requests": "retrieve_requests",
    "lmcache:num_store_requests": "store_requests",
    "lmcache:num_requested_tokens": "requested_tokens",
    "lmcache:num_stored_tokens": "stored_tokens",
    "lmcache:num_remote_read_bytes": "remote_read_bytes",
    "lmcache:num_remote_write_bytes": "remote_write_bytes",
    "lmcache:local_cpu_evict_count": "cpu_evictions",
    "lmcache:local_disk_evict_count": "disk_evictions",
    "lmcache:retrieve_hit_rate": "hit_rate",
    "lmcache:local_cache_usage": "local_cache_bytes",
    "lmcache:remote_cache_usage": "remote_cache_bytes",
    "lmcache:local_storage_usage": "local_storage_bytes",
    "lmcache:lmcache_is_healthy": "healthy",
    "lmcache:local_disk_read_bytes_total": "disk_read_bytes",
    "lmcache:local_disk_write_bytes_total": "disk_write_bytes",
    "lmcache:local_disk_read_latency_sum": "disk_read_latency_sum",
    "lmcache:local_disk_read_latency_count": "disk_read_latency_count",
    "lmcache:local_disk_write_latency_sum": "disk_write_latency_sum",
    "lmcache:local_disk_write_latency_count": "disk_write_latency_count",
    "lmcache:num_slow_retrieval_by_time": "slow_retrieval_by_time",
    "lmcache:num_slow_retrieval_by_speed": "slow_retrieval_by_speed",
    "lmcache:request_cache_lifespan_sum": "cache_lifespan_sum",
    "lmcache:request_cache_lifespan_count": "cache_lifespan_count",
    "lmcache:local_cpu_evict_failed_count": "cpu_evict_failed_count",
    "lmcache:get_blocking_failed_count": "get_blocking_failed_count",
    "lmcache:put_failed_count": "put_failed_count",
    "lmcache:retrieve_to_gpu_time_sum": "retrieve_to_gpu_time_sum",
    "lmcache:retrieve_to_gpu_time_count": "retrieve_to_gpu_time_count",
    "lmcache:store_from_gpu_time_sum": "store_from_gpu_time_sum",
    "lmcache:store_from_gpu_time_count": "store_from_gpu_time_count",
    "lmcache:retrieve_speed": "retrieve_speed",
    "lmcache:store_speed": "store_speed",
}

_V1_TIERED_SCALAR_MAP: dict[str, str] = {
    "lmcache:tier_get_latency_sum": "tier_get_latency_sum",
    "lmcache:tier_get_latency_count": "tier_get_latency_count",
    "lmcache:request_tier_served": "request_tier_served",
    "lmcache:request_tier_hit_tokens_sum": "request_tier_hit_tokens_sum",
    "lmcache:request_tier_hit_tokens_count": "request_tier_hit_tokens_count",
}

_V1_HIT_TOKENS_TIER_TO_FIELD: dict[str | None, str] = {
    "local": "local_hit_tokens",
    "cpu": "cpu_hit_tokens",
    "disk": "disk_hit_tokens",
    "remote": "remote_hit_tokens",
    None: "hit_tokens",
}


def _parse_lmcache_v1_line(line: str, val: float, snap: MetricsSnapshot) -> None:
    """Parse LMCache v1 Prometheus metrics (lmcache: colon prefix, no tier labels)."""
    v1 = snap.lmcache_v1

    if "_bucket{" in line:
        labels = _extract_labels(line)
        le = labels.get("le")
        if le is None:
            return
        for prefix, field_name in _V1_BUCKET_MAP.items():
            if prefix in line:
                getattr(v1, field_name)[le] = val
                return
        for prefix, field_name in _V1_TIERED_BUCKET_MAP.items():
            if prefix in line:
                tier = labels.get("tier")
                if tier is not None:
                    getattr(v1, field_name).setdefault(tier, {})[le] = val
                return
        return

    if "lmcache:num_hit_tokens" in line:
        tier = _extract_labels(line).get("tier")
        field_name = _V1_HIT_TOKENS_TIER_TO_FIELD.get(tier, "hit_tokens")
        setattr(v1, field_name, val)
        return

    for prefix, field_name in _V1_SCALAR_MAP.items():
        if prefix in line:
            setattr(v1, field_name, val)
            return

    for prefix, field_name in _V1_TIERED_SCALAR_MAP.items():
        if prefix in line:
            tier = _extract_labels(line).get("tier")
            if tier:
                getattr(v1, field_name)[tier] = val
            return


def _parse_lmcache_line(line: str, val: float, snap: MetricsSnapshot) -> None:
    """Parse LMCache Prometheus metrics with tier labels."""
    labels = _extract_labels(line)
    tier = labels.get("tier", "unknown")

    if "lmcache_store_hit_count" in line:
        _get_dict(snap, "lmcache_hits")[tier] = val
    elif "lmcache_store_latency_seconds_sum" in line:
        _get_dict(snap, "lmcache_latency_sum")[tier] = val
    elif "lmcache_store_latency_seconds_count" in line:
        _get_dict(snap, "lmcache_latency_count")[tier] = val
    elif "lmcache_store_evictions_total" in line:
        _get_dict(snap, "lmcache_evictions")[tier] = val
    elif "lmcache_store_miss_count" in line:
        _get_dict(snap, "lmcache_misses")[tier] = val
    elif "lmcache_memory_usage_bytes" in line:
        _get_dict(snap, "lmcache_memory_bytes")[tier] = val
    else:
        metric_name = line.split("{")[0].split()[0] if "{" in line else line.split()[0]
        key = f"{metric_name}[{tier}]" if tier != "unknown" else metric_name
        cast(dict[str, float], object.__getattribute__(snap, "custom_metrics"))[key] = val


_VLLM_METRIC_MAP: dict[str, str] = {
    "vllm:kv_cache_usage_perc": "kv_cache_usage_perc",
    "vllm:num_requests_running": "num_requests_running",
    "vllm:num_requests_waiting": "num_requests_waiting",
    "vllm:prefix_cache_hits_total": "prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total": "prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total": "external_prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total": "external_prefix_cache_queries_total",
    "vllm:num_preemptions_total": "num_preemptions_total",
    "vllm:time_to_first_token_seconds_sum": "server_ttft_sum",
    "vllm:time_to_first_token_seconds_count": "server_ttft_count",
    "vllm:request_prefill_time_seconds_sum": "prefill_time_sum",
    "vllm:request_prefill_time_seconds_count": "prefill_time_count",
    "vllm:request_queue_time_seconds_sum": "queue_time_sum",
    "vllm:request_queue_time_seconds_count": "queue_time_count",
    "vllm:prompt_tokens_total": "isl_total",
    "vllm:generation_tokens_total": "osl_total",
}


def _parse_vllm_line(line: str, val: float, snap: MetricsSnapshot) -> None:
    """Parse a single vLLM Prometheus metric line."""
    is_lmcache_v1 = "lmcache:" in line
    is_lmcache_v0 = not is_lmcache_v1 and "lmcache_" in line
    if not is_lmcache_v1 and not is_lmcache_v0 and "engine=" not in line:
        return
    if is_lmcache_v1:
        _parse_lmcache_v1_line(line, val, snap)
        return
    if is_lmcache_v0:
        _parse_lmcache_line(line, val, snap)
        return

    brace = line.find("{")
    space = line.find(" ")
    end = brace if brace >= 0 else space
    if end < 0:
        return
    metric_name = line[:end]

    if "TYPE" in line or metric_name.endswith("_created"):
        return

    attr = _VLLM_METRIC_MAP.get(metric_name)
    if attr is not None:
        setattr(snap, attr, val)


def _parse_sglang_line(line: str, val: float, snap: MetricsSnapshot) -> None:
    """Parse a single SGLang Prometheus metric line.

    Not yet validated against a real SGLang deployment.
    """
    global _sglang_warned
    if not _sglang_warned:
        _log.warning("SGLang metrics parser is unvalidated; results may be inaccurate")
        _sglang_warned = True
    if "sglang:token_usage" in line and "num_used" not in line:
        snap.kv_cache_usage_perc = val
    elif "sglang:num_running_reqs" in line:
        snap.num_requests_running = val
    elif "sglang:num_queue_reqs" in line:
        snap.num_requests_waiting = val
    elif "sglang:cache_hit_rate" in line:
        # SGLang reports point-in-time ratio (0-1), NOT a cumulative counter.
        # Store in dedicated field; delta computation on hits/queries is meaningless.
        snap.sglang_cache_hit_rate = val
    elif "sglang:prompt_tokens_total" in line:
        snap.isl_total = val
    elif "sglang:generation_tokens_total" in line:
        snap.osl_total = val


async def fetch_metrics(
    vllm_url: str,
    timeout: float = 3.0,
    session: aiohttp.ClientSession | None = None,
    extra_urls: list[str] | None = None,
) -> MetricsSnapshot | None:
    """Fetch and parse a single metrics snapshot from vLLM."""
    try:
        owns_session = session is None
        _ssl: bool | _ssl_mod.SSLContext = (
            False if vllm_url.startswith("http://") else _ssl_mod.create_default_context()
        )
        sess = (
            session
            if session is not None
            else aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=_ssl))
        )
        try:
            async with sess.get(
                f"{vllm_url}/metrics",
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                text = await resp.text()
                snap = parse_metrics(text)
            if extra_urls and snap:
                for url in extra_urls:
                    try:
                        async with sess.get(
                            url, timeout=aiohttp.ClientTimeout(total=timeout)
                        ) as resp:
                            text = await resp.text()
                            extra_snap = parse_metrics(text)
                            _get_dict(snap, "lmcache_hits").update(
                                _get_dict(extra_snap, "lmcache_hits")
                            )
                            _get_dict(snap, "lmcache_latency_sum").update(
                                _get_dict(extra_snap, "lmcache_latency_sum")
                            )
                            _get_dict(snap, "lmcache_latency_count").update(
                                _get_dict(extra_snap, "lmcache_latency_count")
                            )
                            _get_dict(snap, "lmcache_evictions").update(
                                _get_dict(extra_snap, "lmcache_evictions")
                            )
                            _get_dict(snap, "lmcache_misses").update(
                                _get_dict(extra_snap, "lmcache_misses")
                            )
                            _get_dict(snap, "lmcache_memory_bytes").update(
                                _get_dict(extra_snap, "lmcache_memory_bytes")
                            )
                            cast(
                                dict[str, float], object.__getattribute__(snap, "custom_metrics")
                            ).update(
                                cast(
                                    dict[str, float],
                                    object.__getattribute__(extra_snap, "custom_metrics"),
                                )
                            )
                            _merge_lmcache_v1(_get_v1(snap), _get_v1(extra_snap))
                    except (TimeoutError, aiohttp.ClientError, OSError):
                        pass
            return snap
        finally:
            if owns_session:
                await sess.close()
    except (TimeoutError, aiohttp.ClientError, OSError):
        return None


@dataclass
class MetricsCollector:
    """Async background metrics collector that polls vLLM at a fixed interval.

    Usage:
        collector = MetricsCollector("http://localhost:8000", interval=0.3)
        async with collector:
            # ... run workload ...
            pass
        print(collector.kv_util_peak, collector.snapshots)
    """

    vllm_url: str
    interval: float = 0.3
    extra_metrics_urls: list[str] = field(default_factory=list)
    enabled: bool = True
    #: Maximum number of snapshots retained in :attr:`snapshots`. ``0``
    #: disables the cap (legacy behaviour). When the cap is hit, the
    #: FIRST snapshot is preserved (delta calculations need it) and the
    #: oldest middle snapshot is evicted. Long ``--duration`` runs at
    #: the default 0.3 s interval grow this list ~12 k entries / hour;
    #: cap to keep memory bounded.
    max_snapshots: int = 0
    snapshots: list[MetricsSnapshot] = field(default_factory=list)
    kv_util_peak: float = 0.0
    running_peak: float = 0.0
    waiting_peak: float = 0.0
    preemption_total: float = 0.0
    _kv_samples: list[float] = field(default_factory=list, repr=False)
    _task: asyncio.Task | None = field(default=None, repr=False)
    _running: bool = field(default=False, repr=False)
    _session: aiohttp.ClientSession | None = field(default=None, repr=False)
    _consecutive_failures: int = field(default=0, repr=False)

    def _append_snapshot(self, snap: MetricsSnapshot) -> None:
        """Append *snap* to :attr:`snapshots`, honouring :attr:`max_snapshots`.

        When the cap is hit, evict the OLDEST non-first snapshot. The first
        entry is preserved because cumulative-delta helpers (preemption,
        prefix-cache, bucket deltas, etc.) all subtract ``snapshots[0]``.
        """
        cap = self.max_snapshots
        if cap and len(self.snapshots) >= cap:
            if cap == 1:
                self.snapshots.clear()
            else:
                # Drop the oldest middle entry, keep snapshots[0] for deltas.
                del self.snapshots[1]
        self.snapshots.append(snap)

    async def start(self) -> None:
        if not self.enabled:
            return
        _collector_ssl: bool | _ssl_mod.SSLContext = (
            False if self.vllm_url.startswith("http://") else _ssl_mod.create_default_context()
        )
        self._session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=_collector_ssl))
        # Take an immediate pre-flight snapshot for accurate deltas
        snap = await fetch_metrics(
            self.vllm_url, session=self._session, extra_urls=self.extra_metrics_urls or None
        )
        if snap:
            self._append_snapshot(snap)
            self.kv_util_peak = max(self.kv_util_peak, _snap_float(snap, "kv_cache_usage_perc"))
            self.running_peak = max(self.running_peak, _snap_float(snap, "num_requests_running"))
            self.waiting_peak = max(self.waiting_peak, _snap_float(snap, "num_requests_waiting"))
            self._kv_samples.append(_snap_float(snap, "kv_cache_usage_perc"))
        else:
            # Surface the preflight failure so backend-mode runs do not
            # silently produce empty kv_util/timeseries fields. The poll
            # loop still gets a chance to recover, but the user has a
            # breadcrumb to correlate the empty fields with a bad URL.
            logging.getLogger(__name__).warning(
                "MetricsCollector: initial fetch failed for %s; "
                "subsequent samples may be empty. Check that the metrics "
                "endpoint is reachable.",
                self.vllm_url,
            )
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        if not self.enabled:
            return
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._session:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "MetricsCollector":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    async def _poll_loop(self) -> None:
        while self._running:
            snap = await fetch_metrics(
                self.vllm_url, session=self._session, extra_urls=self.extra_metrics_urls or None
            )
            if snap:
                self._consecutive_failures = 0
                self._append_snapshot(snap)
                self.kv_util_peak = max(self.kv_util_peak, _snap_float(snap, "kv_cache_usage_perc"))
                self.running_peak = max(
                    self.running_peak, _snap_float(snap, "num_requests_running")
                )
                self.waiting_peak = max(
                    self.waiting_peak, _snap_float(snap, "num_requests_waiting")
                )
                self._kv_samples.append(_snap_float(snap, "kv_cache_usage_perc"))
                # Keep preemption_total for backwards compat but use preemption_delta() for accuracy
                self.preemption_total = max(
                    self.preemption_total, _get_float(snap, "num_preemptions_total")
                )
            else:
                self._consecutive_failures += 1
                if self._consecutive_failures % 5 == 0:
                    logging.getLogger(__name__).warning(
                        "MetricsCollector: 5 consecutive fetch failures for %s",
                        self.vllm_url,
                    )
            await asyncio.sleep(self.interval)

    def preemption_delta(self) -> float:
        """Return delta preemptions between first and last snapshot."""
        if len(self.snapshots) < 2:
            return 0.0
        first, last = self.snapshots[0], self.snapshots[-1]
        return max(
            0.0,
            _get_float(last, "num_preemptions_total") - _get_float(first, "num_preemptions_total"),
        )

    def prefix_cache_delta(self) -> tuple[float, float]:
        """Return (delta_hits, delta_queries) between first and last snapshot.

        Clamps to zero if vLLM restarted mid-benchmark (counters reset).
        For SGLang, returns (0.0, 0.0) since cache_hit_rate is a point-in-time
        ratio - use ``sglang_cache_hit_rate`` instead.
        """
        if len(self.snapshots) < 2:
            return (0.0, 0.0)
        first, last = self.snapshots[0], self.snapshots[-1]
        # SGLang uses point-in-time ratio, delta is meaningless
        if _get_float(last, "sglang_cache_hit_rate", -1.0) >= 0:
            return (0.0, 0.0)
        return (
            max(
                0.0,
                _get_float(last, "prefix_cache_hits_total")
                - _get_float(first, "prefix_cache_hits_total"),
            ),
            max(
                0.0,
                _get_float(last, "prefix_cache_queries_total")
                - _get_float(first, "prefix_cache_queries_total"),
            ),
        )

    def external_prefix_cache_delta(self) -> tuple[float, float]:
        """Return (delta_hits, delta_queries) for external (LMCache) prefix cache."""
        if len(self.snapshots) < 2:
            return (0.0, 0.0)
        first, last = self.snapshots[0], self.snapshots[-1]
        return (
            max(
                0.0,
                _get_float(last, "external_prefix_cache_hits_total")
                - _get_float(first, "external_prefix_cache_hits_total"),
            ),
            max(
                0.0,
                _get_float(last, "external_prefix_cache_queries_total")
                - _get_float(first, "external_prefix_cache_queries_total"),
            ),
        )

    @property
    def sglang_cache_hit_rate(self) -> float:
        """Return last SGLang cache hit rate, or -1.0 if not using SGLang."""
        if not self.snapshots:
            return -1.0
        last = self.snapshots[-1]
        return _get_float(last, "sglang_cache_hit_rate", -1.0)

    @property
    def memory_pressure(self) -> float:
        """Fraction of time KV cache was above 80% (memory-pressure zone)."""
        if not self._kv_samples:
            return 0.0
        return sum(1 for s in self._kv_samples if s > 0.80) / len(self._kv_samples)

    def lmcache_v1_deltas(self) -> dict:
        """Compute v1 LMCache metric deltas with per-tier breakdown.

        Tier info in v1 is name-encoded (not labels):
          cpu:    local_cache_usage, local_cpu_evict_count, local_cpu_hot_cache_count
          disk:   local_storage_usage
          remote: remote_cache_usage, num_remote_read/write_bytes,
                  remote_time_to_get, remote_time_to_put
        """
        if len(self.snapshots) < 2:
            return {}
        first, last = self.snapshots[0], self.snapshots[-1]
        fv1, lv1 = _get_v1(first), _get_v1(last)
        has_v1 = (
            lv1.retrieve_requests > 0
            or lv1.hit_tokens > 0
            or lv1.local_hit_tokens > 0
            or lv1.remote_hit_tokens > 0
            or lv1.cpu_hit_tokens > 0
            or lv1.disk_hit_tokens > 0
        )
        if not has_v1:
            return {}

        def _delta(attr: str) -> float:
            return float(max(0, getattr(lv1, attr) - getattr(fv1, attr)))

        def _bucket_delta(
            last_buckets: dict, first_buckets: dict, last_count: float, first_count: float
        ) -> tuple[dict, float]:
            bd = {le: cnt - first_buckets.get(le, 0) for le, cnt in last_buckets.items()}
            return bd, max(0, last_count - first_count)

        retrieve_bd, retrieve_cd = _bucket_delta(
            lv1.retrieve_latency_buckets,
            fv1.retrieve_latency_buckets,
            lv1.retrieve_latency_count,
            fv1.retrieve_latency_count,
        )
        store_bd, store_cd = _bucket_delta(
            lv1.store_latency_buckets,
            fv1.store_latency_buckets,
            lv1.store_latency_count,
            fv1.store_latency_count,
        )
        remote_get_bd, remote_get_cd = _bucket_delta(
            lv1.remote_get_latency_buckets,
            fv1.remote_get_latency_buckets,
            lv1.remote_get_latency_count,
            fv1.remote_get_latency_count,
        )
        remote_put_bd, remote_put_cd = _bucket_delta(
            lv1.remote_put_latency_buckets,
            fv1.remote_put_latency_buckets,
            lv1.remote_put_latency_count,
            fv1.remote_put_latency_count,
        )
        disk_read_bd, disk_read_cd = _bucket_delta(
            lv1.disk_read_latency_buckets,
            fv1.disk_read_latency_buckets,
            lv1.disk_read_latency_count,
            fv1.disk_read_latency_count,
        )
        disk_write_bd, disk_write_cd = _bucket_delta(
            lv1.disk_write_latency_buckets,
            fv1.disk_write_latency_buckets,
            lv1.disk_write_latency_count,
            fv1.disk_write_latency_count,
        )

        tier_get_latency = {}
        for tier in ("cpu", "disk", "remote"):
            first_buckets = fv1.tier_get_latency_buckets.get(tier, {})
            last_buckets = lv1.tier_get_latency_buckets.get(tier, {})
            first_count = fv1.tier_get_latency_count.get(tier, 0)
            last_count = lv1.tier_get_latency_count.get(tier, 0)
            bd, cd = _bucket_delta(last_buckets, first_buckets, last_count, first_count)
            tier_get_latency[tier] = histogram_percentiles(bd, cd)

        _requested = _delta("requested_tokens")
        _total_hits = _delta("hit_tokens")
        _cpu_hits = _delta("cpu_hit_tokens")
        _disk_hits = _delta("disk_hit_tokens")
        _remote_hits = _delta("remote_hit_tokens")

        def _rate(num: float, denom: float) -> float:
            return num / denom if denom > 0 else 0.0

        cpu_hit_rate = _rate(_cpu_hits, _requested)
        disk_hit_rate = _rate(_disk_hits, _requested)
        remote_hit_rate = _rate(_remote_hits, _requested)
        cpu_hit_share = _rate(_cpu_hits, _total_hits)
        disk_hit_share = _rate(_disk_hits, _total_hits)
        remote_hit_share = _rate(_remote_hits, _total_hits)

        _hits_by_tier = {"cpu": _cpu_hits, "disk": _disk_hits, "remote": _remote_hits}
        _rate_by_tier = {
            "cpu": (cpu_hit_rate, cpu_hit_share),
            "disk": (disk_hit_rate, disk_hit_share),
            "remote": (remote_hit_rate, remote_hit_share),
        }
        _tier_dicts: dict[str, dict] = {}
        for _tier in ("cpu", "disk", "remote"):
            _h = _hits_by_tier[_tier]
            _hr, _hs = _rate_by_tier[_tier]
            _served = max(
                0, lv1.request_tier_served.get(_tier, 0) - fv1.request_tier_served.get(_tier, 0)
            )
            _td: dict = {
                "hit_tokens": _h,
                "hit_rate": _hr,
                "hit_share": _hs,
                "get_latency": tier_get_latency.get(_tier, {}),
                "requests_served": _served,
            }
            if _tier == "cpu":
                _td["cache_bytes"] = lv1.local_cache_bytes
                _td["evictions"] = _delta("cpu_evictions")
                _td["hot_cache_count"] = lv1.local_cpu_hot_cache_count
            elif _tier == "disk":
                _td["storage_bytes"] = lv1.local_storage_bytes
                _td["evictions"] = _delta("disk_evictions")
                _td["read_bytes"] = _delta("disk_read_bytes")
                _td["write_bytes"] = _delta("disk_write_bytes")
                _td["read_latency"] = histogram_percentiles(disk_read_bd, disk_read_cd)
                _td["write_latency"] = histogram_percentiles(disk_write_bd, disk_write_cd)
            elif _tier == "remote":
                _td["cache_bytes"] = lv1.remote_cache_bytes
                _td["read_bytes"] = _delta("remote_read_bytes")
                _td["write_bytes"] = _delta("remote_write_bytes")
                _td["get_latency"] = histogram_percentiles(remote_get_bd, remote_get_cd)
                _td["put_latency"] = histogram_percentiles(remote_put_bd, remote_put_cd)
                _td["tier_get_latency"] = tier_get_latency.get("remote", {})
            _tier_dicts[_tier] = _td

        return {
            "version": "v1",
            # Aggregate (no tier label in v1 - upstream gap)
            "hit_tokens": _total_hits,
            "requested_tokens": _requested,
            "stored_tokens": _delta("stored_tokens"),
            "hit_rate": lv1.hit_rate,
            "retrieve_latency": histogram_percentiles(retrieve_bd, retrieve_cd),
            "store_latency": histogram_percentiles(store_bd, store_cd),
            "healthy": lv1.healthy,
            "local_hit_tokens": _delta("local_hit_tokens"),
            "remote_hit_tokens": _remote_hits,
            "tier_get_latency": tier_get_latency,
            "tiers": _tier_dicts,
            "slow_retrieval_by_time": _delta("slow_retrieval_by_time"),
            "slow_retrieval_by_speed": _delta("slow_retrieval_by_speed"),
            "cache_lifespan": histogram_percentiles(
                *_bucket_delta(
                    lv1.cache_lifespan_buckets,
                    fv1.cache_lifespan_buckets,
                    lv1.cache_lifespan_count,
                    fv1.cache_lifespan_count,
                )
            ),
            "cpu_evict_failed": _delta("cpu_evict_failed_count"),
            "get_blocking_failed": _delta("get_blocking_failed_count"),
            "put_failed": _delta("put_failed_count"),
            "retrieve_to_gpu_latency": histogram_percentiles(
                *_bucket_delta(
                    lv1.retrieve_to_gpu_time_buckets,
                    fv1.retrieve_to_gpu_time_buckets,
                    lv1.retrieve_to_gpu_time_count,
                    fv1.retrieve_to_gpu_time_count,
                )
            ),
            "store_from_gpu_latency": histogram_percentiles(
                *_bucket_delta(
                    lv1.store_from_gpu_time_buckets,
                    fv1.store_from_gpu_time_buckets,
                    lv1.store_from_gpu_time_count,
                    fv1.store_from_gpu_time_count,
                )
            ),
            "retrieve_speed": lv1.retrieve_speed,
            "store_speed": lv1.store_speed,
        }

    def lmcache_tier_deltas(self) -> dict[str, dict]:
        """Compute per-tier LMCache hit/latency deltas."""
        if len(self.snapshots) < 2:
            return {}
        first, last = self.snapshots[0], self.snapshots[-1]
        result = {}
        all_tiers = set()
        for attr in (
            "lmcache_hits",
            "lmcache_misses",
            "lmcache_latency_sum",
            "lmcache_latency_count",
            "lmcache_evictions",
            "lmcache_memory_bytes",
        ):
            all_tiers |= set(getattr(last, attr, {}).keys()) | set(getattr(first, attr, {}).keys())
        last_hits = _get_dict(last, "lmcache_hits")
        first_hits = _get_dict(first, "lmcache_hits")
        last_lat_sum = _get_dict(last, "lmcache_latency_sum")
        first_lat_sum = _get_dict(first, "lmcache_latency_sum")
        last_lat_count = _get_dict(last, "lmcache_latency_count")
        first_lat_count = _get_dict(first, "lmcache_latency_count")
        last_evictions = _get_dict(last, "lmcache_evictions")
        first_evictions = _get_dict(first, "lmcache_evictions")
        last_misses = _get_dict(last, "lmcache_misses")
        first_misses = _get_dict(first, "lmcache_misses")
        last_memory = _get_dict(last, "lmcache_memory_bytes")
        for tier in all_tiers:
            hits_delta = max(0.0, last_hits.get(tier, 0) - first_hits.get(tier, 0))
            lat_sum_delta = max(0.0, last_lat_sum.get(tier, 0) - first_lat_sum.get(tier, 0))
            lat_count_delta = max(0.0, last_lat_count.get(tier, 0) - first_lat_count.get(tier, 0))
            evict_delta = max(0.0, last_evictions.get(tier, 0) - first_evictions.get(tier, 0))
            miss_delta = max(0.0, last_misses.get(tier, 0) - first_misses.get(tier, 0))
            total_lookups = hits_delta + miss_delta
            hit_rate = (hits_delta / total_lookups) if total_lookups > 0 else 0.0
            memory_bytes = last_memory.get(tier, 0)
            avg_latency_ms = (
                (lat_sum_delta / lat_count_delta * 1000) if lat_count_delta > 0 else 0.0
            )
            result[tier] = {
                "hits": hits_delta,
                "avg_latency_ms": avg_latency_ms,
                "latency_sum_ms": lat_sum_delta * 1000,
                "latency_count": lat_count_delta,
                "evictions": evict_delta,
                "misses": miss_delta,
                "hit_rate": hit_rate,
                "memory_bytes": memory_bytes,
            }
        return result
