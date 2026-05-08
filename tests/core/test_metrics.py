"""Trimmed metrics tests: parser, collector, LMCache v0/v1, histograms."""

from __future__ import annotations

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentsurge.types import MetricsSnapshot

# ===========================================================================
# Mock metric blobs
# ===========================================================================

MOCK_METRICS = """\
vllm:kv_cache_usage_perc{engine="0"} 0.45
vllm:num_requests_running{engine="0"} 10
vllm:num_requests_waiting{engine="0"} 5
vllm:prefix_cache_hits_total{engine="0"} 1500
vllm:prefix_cache_queries_total{engine="0"} 2000
"""

MOCK_METRICS_EXTENDED = """\
vllm:kv_cache_usage_perc{engine="0"} 0.55
vllm:num_requests_running{engine="0"} 12
vllm:num_requests_waiting{engine="0"} 3
vllm:prefix_cache_hits_total{engine="0"} 800
vllm:prefix_cache_queries_total{engine="0"} 1000
vllm:num_preemptions_total{engine="0"} 7
vllm:time_to_first_token_seconds_sum{engine="0"} 45.6
vllm:time_to_first_token_seconds_count{engine="0"} 120
vllm:prompt_tokens_total{engine="0"} 50000
vllm:generation_tokens_total{engine="0"} 12000
"""

MOCK_SGLANG_METRICS = """\
sglang:num_running_reqs 8
sglang:num_queue_reqs 3
sglang:token_usage 0.62
sglang:cache_hit_rate 0.78
sglang:prompt_tokens_total 5000
sglang:generation_tokens_total 1200
"""

MOCK_LMCACHE_METRICS = """\
lmcache_store_hit_count{tier="gpu"} 500
lmcache_store_hit_count{tier="cpu"} 200
lmcache_store_hit_count{tier="disk"} 50
lmcache_store_latency_seconds_sum{tier="gpu"} 0.5
lmcache_store_latency_seconds_count{tier="gpu"} 500
lmcache_store_evictions_total{tier="gpu"} 30
lmcache_store_miss_count{tier="gpu"} 100
lmcache_memory_usage_bytes{tier="gpu"} 1073741824
"""

MOCK_LMCACHE_V1_METRICS = """\
lmcache:num_retrieve_requests 42
lmcache:num_store_requests 15
lmcache:num_requested_tokens 10000
lmcache:num_hit_tokens 7500
lmcache:num_stored_tokens 2000
lmcache:num_remote_read_bytes 1048576
lmcache:num_remote_write_bytes 524288
lmcache:local_cpu_evict_count 5
lmcache:retrieve_hit_rate 0.75
lmcache:local_cache_usage 2147483648
lmcache:remote_cache_usage 4294967296
lmcache:local_storage_usage 8589934592
lmcache:lmcache_is_healthy 1
lmcache:time_to_retrieve_bucket{le="0.005"} 10
lmcache:time_to_retrieve_bucket{le="0.01"} 25
lmcache:time_to_retrieve_bucket{le="+Inf"} 42
lmcache:time_to_retrieve_sum 0.312
lmcache:time_to_retrieve_count 42
lmcache:time_to_store_bucket{le="0.01"} 8
lmcache:time_to_store_bucket{le="+Inf"} 15
lmcache:time_to_store_sum 0.22
lmcache:time_to_store_count 15
"""


# ===========================================================================
# Parser: vLLM
# ===========================================================================


class TestParseMetrics:
    def test_parses_vllm_blob(self):
        from agentsurge.metrics import parse_metrics

        snap = parse_metrics(MOCK_METRICS)
        assert snap.kv_cache_usage_perc == pytest.approx(0.45)
        assert snap.num_requests_running == pytest.approx(10)
        assert snap.prefix_cache_hits_total == pytest.approx(1500)

    def test_empty_string(self):
        from agentsurge.metrics import parse_metrics

        snap = parse_metrics("")
        assert snap.kv_cache_usage_perc == 0.0

    def test_extended_fields(self):
        from agentsurge.metrics import parse_metrics

        snap = parse_metrics(MOCK_METRICS_EXTENDED)
        assert snap.num_preemptions_total == pytest.approx(7)
        assert snap.server_ttft_sum == pytest.approx(45.6)
        assert snap.isl_total == pytest.approx(50000)


# ===========================================================================
# Parser: SGLang
# ===========================================================================


def test_sglang_parsing():
    from agentsurge.metrics import parse_metrics

    snap = parse_metrics(MOCK_SGLANG_METRICS)
    assert snap.kv_cache_usage_perc == pytest.approx(0.62)
    assert snap.num_requests_running == pytest.approx(8)
    assert snap.sglang_cache_hit_rate == pytest.approx(0.78)


# ===========================================================================
# MetricsCollector
# ===========================================================================


class TestMetricsCollector:
    def test_prefix_cache_delta(self):
        from agentsurge.metrics import MetricsCollector

        col = MetricsCollector(vllm_url="http://localhost:8000")
        assert col.prefix_cache_delta() == (0.0, 0.0)
        col.snapshots.append(
            MetricsSnapshot(
                timestamp=0.0, prefix_cache_hits_total=100, prefix_cache_queries_total=200
            )
        )
        col.snapshots.append(
            MetricsSnapshot(
                timestamp=1.0, prefix_cache_hits_total=250, prefix_cache_queries_total=400
            )
        )
        hits, queries = col.prefix_cache_delta()
        assert hits == pytest.approx(150.0) and queries == pytest.approx(200.0)

    def test_start_stop_lifecycle(self):
        from agentsurge.metrics import MetricsCollector

        col = MetricsCollector(vllm_url="http://localhost:8000", interval=0.05)
        mock_sess = MagicMock()
        mock_sess.close = AsyncMock()

        async def _run():
            with (
                patch(
                    "agentsurge.metrics.fetch_metrics", new_callable=AsyncMock, return_value=None
                ),
                patch("agentsurge.metrics.aiohttp.ClientSession", return_value=mock_sess),
            ):
                await col.start()
                assert col._running is True
                await asyncio.sleep(0.02)
                await col.stop()
                assert col._running is False

        asyncio.run(_run())


# ===========================================================================
# fetch_metrics
# ===========================================================================


class TestFetchMetrics:
    def test_fetch_with_shared_session(self):
        from agentsurge.metrics import fetch_metrics

        mock_resp = MagicMock()
        mock_resp.text = AsyncMock(return_value=MOCK_METRICS)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        snap = asyncio.run(fetch_metrics("http://localhost:8000", session=mock_session))
        assert snap.kv_cache_usage_perc == pytest.approx(0.45)

    def test_fetch_returns_none_on_error(self):
        import aiohttp

        from agentsurge.metrics import fetch_metrics

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=aiohttp.ClientError())
        mock_session.close = AsyncMock()

        async def _run():
            with patch("agentsurge.metrics.aiohttp.ClientSession", return_value=mock_session):
                return await fetch_metrics("http://localhost:8000")

        assert asyncio.run(_run()) is None


# ===========================================================================
# LMCache v0 tier metrics
# ===========================================================================


class TestLMCacheV0:
    def test_parses_tier_hits_latency_evictions(self):
        from agentsurge.metrics import parse_metrics

        snap = parse_metrics(MOCK_LMCACHE_METRICS)
        assert snap.lmcache_hits["gpu"] == 500
        assert snap.lmcache_hits["cpu"] == 200
        assert snap.lmcache_latency_sum["gpu"] == pytest.approx(0.5)
        assert snap.lmcache_evictions["gpu"] == 30
        assert snap.lmcache_misses.get("gpu") == 100

    def test_tier_deltas(self):
        from agentsurge.metrics import MetricsCollector

        col = MetricsCollector(vllm_url="http://localhost:8000")
        s1 = MetricsSnapshot(timestamp=0.0)
        s1.lmcache_hits = {"gpu": 100, "cpu": 50}
        s1.lmcache_evictions = {"gpu": 5}
        s2 = MetricsSnapshot(timestamp=1.0)
        s2.lmcache_hits = {"gpu": 300, "cpu": 150}
        s2.lmcache_evictions = {"gpu": 15}
        col.snapshots = [s1, s2]
        d = col.lmcache_tier_deltas()
        assert d["gpu"]["hits"] == 200 and d["cpu"]["hits"] == 100
        assert d["gpu"]["evictions"] == 10


# ===========================================================================
# LMCache v1 metrics
# ===========================================================================


class TestLMCacheV1:
    def test_parses_counters_and_gauges(self):
        from agentsurge.metrics import parse_metrics

        snap = parse_metrics(MOCK_LMCACHE_V1_METRICS)
        assert snap.lmcache_v1.retrieve_requests == 42
        assert snap.lmcache_v1.hit_tokens == 7500
        assert snap.lmcache_v1.hit_rate == pytest.approx(0.75)
        assert snap.lmcache_v1.local_cache_bytes == 2147483648
        # v0 stays empty for v1 input
        assert snap.lmcache_hits == {}

    def test_parses_histograms(self):
        from agentsurge.metrics import parse_metrics

        snap = parse_metrics(MOCK_LMCACHE_V1_METRICS)
        assert snap.lmcache_v1.retrieve_latency_buckets["0.005"] == 10
        assert snap.lmcache_v1.retrieve_latency_count == 42
        assert snap.lmcache_v1.store_latency_count == 15

    def test_v1_deltas(self):
        from agentsurge.metrics import MetricsCollector

        col = MetricsCollector(vllm_url="http://localhost:8000")
        s1 = MetricsSnapshot(timestamp=0.0)
        s1.lmcache_v1.hit_tokens = 1000
        s1.lmcache_v1.requested_tokens = 5000
        s1.lmcache_v1.stored_tokens = 200
        s1.lmcache_v1.cpu_evictions = 3
        s1.lmcache_v1.retrieve_requests = 10
        s2 = MetricsSnapshot(timestamp=1.0)
        s2.lmcache_v1.hit_tokens = 8000
        s2.lmcache_v1.requested_tokens = 10000
        s2.lmcache_v1.stored_tokens = 700
        s2.lmcache_v1.cpu_evictions = 8
        s2.lmcache_v1.retrieve_requests = 42
        s2.lmcache_v1.hit_rate = 0.8
        col.snapshots = [s1, s2]
        d = col.lmcache_v1_deltas()
        assert d["version"] == "v1"
        assert d["hit_tokens"] == 7000 and d["requested_tokens"] == 5000
        assert d["hit_rate"] == pytest.approx(0.8)


# ===========================================================================
# Histogram percentiles
# ===========================================================================


class TestHistogramPercentiles:
    def test_basic(self):
        from agentsurge.metrics import histogram_percentiles

        buckets = {"0.005": 0, "0.01": 100, "0.025": 100, "+Inf": 100}
        result = histogram_percentiles(buckets, 100, percentiles=(50, 95, 99))
        assert result["p50"] == pytest.approx(0.0075)

    def test_empty_returns_nan(self):
        from agentsurge.metrics import histogram_percentiles

        result = histogram_percentiles({}, 0)
        assert all(math.isnan(v) for v in result.values())

    def test_counter_reset_returns_nan(self):
        """Negative bucket deltas from a Prometheus counter reset must yield nan, not garbage."""
        from agentsurge.metrics import histogram_percentiles

        # Simulate _bucket_delta output after a counter reset: buckets went backward
        # but the count scalar was read at a slightly different time and appears positive.
        reset_buckets = {"0.1": -95, "0.5": -190, "+Inf": -280}
        result = histogram_percentiles(reset_buckets, 5, percentiles=(50, 95, 99))
        assert all(math.isnan(v) for v in result.values()), (
            f"Expected all-nan after counter reset, got {result}"
        )


# ===========================================================================
# Per-request metrics (tier-labeled hit tokens, disk I/O)
# ===========================================================================


def test_tier_labeled_hit_tokens():
    from agentsurge.metrics import MetricsSnapshot, _parse_lmcache_v1_line

    snap = MetricsSnapshot(timestamp=0.0)
    _parse_lmcache_v1_line('lmcache:num_hit_tokens{tier="local"} 4200', 4200, snap)
    _parse_lmcache_v1_line('lmcache:num_hit_tokens{tier="remote"} 800', 800, snap)
    assert snap.lmcache_v1.local_hit_tokens == 4200
    assert snap.lmcache_v1.remote_hit_tokens == 800
    assert snap.lmcache_v1.hit_tokens == 0  # aggregate untouched


class TestGaugeFieldsMerging:
    def test_gauge_fields_merged_as_max_not_sum(self):
        """Gauge fields (hit_rate, healthy, etc.) use max(); counters use sum()."""
        from agentsurge.metrics import _merge_lmcache_v1
        from agentsurge.types import LmcacheV1Metrics

        v1_a = LmcacheV1Metrics(hit_rate=0.8, hit_tokens=100, healthy=1)
        v1_b = LmcacheV1Metrics(hit_rate=0.6, hit_tokens=200, healthy=0)

        _merge_lmcache_v1(v1_a, v1_b)

        assert v1_a.hit_rate == pytest.approx(0.8)
        assert v1_a.hit_tokens == 300
        assert v1_a.healthy == 1


def test_disk_io_parsing():
    from agentsurge.metrics import MetricsSnapshot, _parse_lmcache_v1_line

    snap = MetricsSnapshot(timestamp=0.0)
    _parse_lmcache_v1_line("lmcache:local_disk_read_bytes_total 1048576", 1048576, snap)
    _parse_lmcache_v1_line("lmcache:local_disk_write_bytes_total 2097152", 2097152, snap)
    assert snap.lmcache_v1.disk_read_bytes == 1048576
    assert snap.lmcache_v1.disk_write_bytes == 2097152


# ===========================================================================
# H11 — __eq__ asymmetry with scalar defaults
# ===========================================================================


def test_metrics_snapshot_eq_scalar_default_symmetry():
    """MetricsSnapshot() and MetricsSnapshot(sglang_cache_hit_rate=-1.0) must compare equal.

    Both return the same value for every backend metric attribute, so __eq__
    must not depend on which keys happen to be stored in backend_metrics.
    """
    a = MetricsSnapshot()
    b = MetricsSnapshot(sglang_cache_hit_rate=-1.0)
    assert a.sglang_cache_hit_rate == b.sglang_cache_hit_rate
    assert a == b


# ===========================================================================
# H12 — __getattr__ must not mutate backend_metrics for mutable defaults
# ===========================================================================


def test_metrics_snapshot_getattr_mutable_default_no_mutation():
    """Reading a mutable-default attribute must NOT store anything in backend_metrics.

    After ``_ = snap.lmcache_hits``, backend_metrics must still be empty and
    a second read must return a fresh independent dict (no shared state).
    """
    a = MetricsSnapshot()
    b = MetricsSnapshot()
    val = a.lmcache_hits
    assert "lmcache_hits" not in a.backend_metrics, (
        "__getattr__ must not store mutable default into backend_metrics"
    )
    val["injected"] = 99
    assert a.lmcache_hits == {}, "mutation of returned dict must not affect future reads"
    assert a == b


# ===========================================================================
# M1 — stop() must cancel the polling task promptly
# ===========================================================================


def test_stop_cancels_polling_task_promptly():
    """stop() must return well under 1 s even when the poll loop is blocked.

    Without task.cancel(), stop() would wait for the full slow-fetch sleep
    (5 s here) before noticing _running is False.

    We bypass start() entirely to avoid its pre-flight fetch_metrics call,
    and instead manually wire up the collector's internal task with a slow
    poll-loop coroutine that parks for 5 s.
    """
    import time

    from agentsurge.metrics import MetricsCollector

    poll_entered = asyncio.Event()

    async def slow_poll_loop(self_col):
        poll_entered.set()
        await asyncio.sleep(5.0)

    mock_sess = MagicMock()
    mock_sess.close = AsyncMock()

    async def _run():
        col = MetricsCollector(vllm_url="http://localhost:8000", interval=0.05)
        col._session = mock_sess
        col._running = True
        col._task = asyncio.create_task(slow_poll_loop(col))

        # wait until the task is parked inside the 5-second sleep
        await asyncio.wait_for(poll_entered.wait(), timeout=2.0)

        t0 = time.monotonic()
        await col.stop()
        elapsed = time.monotonic() - t0

        assert elapsed < 0.5, f"stop() took {elapsed:.2f}s — task was not cancelled"

    asyncio.run(_run())
