"""Trimmed MockBackend tests: SSE format, KV simulation, prefix cache, eviction."""

from __future__ import annotations

import asyncio
import json

import pytest

from agentsurge.backends.mock import (
    KVSimConfig,
    KVSimulator,
    MockBackend,
    MockConfig,
    MockSSEStream,
)


def run(coro):
    return asyncio.run(coro)


async def collect_sse_lines(stream: MockSSEStream) -> list[bytes]:
    lines = []
    async for line in stream:
        lines.append(line)
    return lines


async def collect_data_payloads(stream: MockSSEStream) -> list[dict]:
    payloads = []
    async for line in stream:
        decoded = line.decode("utf-8").strip()
        if decoded.startswith("data: ") and decoded != "data: [DONE]":
            payloads.append(json.loads(decoded[6:]))
    return payloads


# ===========================================================================
# SSE format compliance - chat completions (collapsed)
# ===========================================================================


class TestSSEFormatChat:
    def test_chunk_count_and_done_sentinel(self):
        n = 7
        stream = MockSSEStream(output_tokens=n, ttft_ms=0, inter_token_ms=0)
        lines = run(collect_sse_lines(stream))
        data_lines = [
            line.decode().strip() for line in lines if line.decode().strip().startswith("data: ")
        ]
        assert len(data_lines) == n + 1, f"Expected {n + 1} data lines"
        assert data_lines[-1] == "data: [DONE]"

    def test_per_chunk_fields(self):
        stream = MockSSEStream(output_tokens=4, ttft_ms=0, inter_token_ms=0)
        payloads = run(collect_data_payloads(stream))
        for p in payloads:
            assert p.get("object") == "chat.completion.chunk"
            assert "choices" in p and len(p["choices"]) == 1
            assert isinstance(p["choices"][0]["delta"]["content"], str)

    def test_usage_on_last_chunk_only(self):
        stream = MockSSEStream(output_tokens=5, ttft_ms=0, inter_token_ms=0, include_usage=True)
        payloads = run(collect_data_payloads(stream))
        for p in payloads[:-1]:
            assert "usage" not in p or p["usage"] is None
        assert payloads[-1]["usage"]["completion_tokens"] == 5


# ===========================================================================
# SSE format compliance - responses API (collapsed)
# ===========================================================================


class TestSSEFormatResponses:
    def test_delta_events_and_completed(self):
        n = 4
        stream = MockSSEStream(output_tokens=n, ttft_ms=0, inter_token_ms=0, api_type="responses")
        lines = run(collect_sse_lines(stream))
        event_lines = [
            line.decode().strip() for line in lines if line.decode().strip().startswith("event: ")
        ]
        delta_events = [e for e in event_lines if "output_text.delta" in e]
        assert len(delta_events) == n
        data_lines = [
            line.decode().strip() for line in lines if line.decode().strip().startswith("data: ")
        ]
        completed = [
            json.loads(d[6:])
            for d in data_lines
            if d != "data: [DONE]" and json.loads(d[6:]).get("type") == "response.completed"
        ]
        assert len(completed) == 1
        assert completed[0]["response"]["usage"]["output_tokens"] == n


# ===========================================================================
# KV simulation - block arithmetic
# ===========================================================================


class TestKVBlockArithmetic:
    def test_zero_tokens_gives_zero_blocks(self):
        sim = KVSimulator(KVSimConfig(total_blocks=100, block_size=16))
        alloc = sim.allocate("s1", num_tokens=0)
        assert alloc.blocks_requested == 0 and alloc.blocks_allocated == 0

    def test_ceiling_division(self):
        bs = 16
        sim = KVSimulator(KVSimConfig(total_blocks=100, block_size=bs))
        assert sim.allocate("s1", num_tokens=bs).blocks_requested == 1
        assert sim.allocate("s2", num_tokens=bs - 1).blocks_requested == 1
        assert sim.allocate("s3", num_tokens=bs + 1).blocks_requested == 2

    def test_snapshot_invariant(self):
        sim = KVSimulator(KVSimConfig(total_blocks=50, block_size=5))
        sim.allocate("s1", num_tokens=25)
        sim.allocate("s2", num_tokens=50)
        snap = sim.snapshot()
        assert snap["blocks_used"] + snap["blocks_free"] == snap["total_blocks"]

    def test_usage_perc_range(self):
        sim = KVSimulator(KVSimConfig(total_blocks=10, block_size=1))
        for i in range(15):
            sim.allocate(f"s{i}", num_tokens=1)
        assert 0.0 <= sim.usage_perc <= 100.0


# ===========================================================================
# KV simulation - prefix cache
# ===========================================================================


class TestKVPrefixCache:
    def test_zero_hit_rate(self):
        sim = KVSimulator(KVSimConfig(total_blocks=1000, block_size=16, prefix_cache_hit_rate=0.0))
        alloc = sim.allocate("s1", num_tokens=160)
        assert alloc.prefix_hit_tokens == 0

    def test_full_hit_rate(self):
        sim = KVSimulator(KVSimConfig(total_blocks=100, block_size=16, prefix_cache_hit_rate=1.0))
        alloc = sim.allocate("s1", num_tokens=160)
        assert alloc.prefix_hit_tokens == 160 and alloc.blocks_requested == 0

    def test_half_hit_rate(self):
        sim = KVSimulator(KVSimConfig(total_blocks=100, block_size=16, prefix_cache_hit_rate=0.5))
        alloc = sim.allocate("s1", num_tokens=160)
        assert alloc.prefix_hit_tokens == 80 and alloc.blocks_requested == 5

    def test_cumulative_prefix_stats(self):
        sim = KVSimulator(KVSimConfig(total_blocks=1000, block_size=16, prefix_cache_hit_rate=0.25))
        sim.allocate("s1", num_tokens=100)
        sim.allocate("s2", num_tokens=200)
        sim.allocate("s3", num_tokens=400)
        snap = sim.snapshot()
        assert snap["prefix_hit_tokens_total"] == 175
        assert snap["prefix_queries_total"] == 700

    def test_reset_clears_prefix_stats(self):
        sim = KVSimulator(KVSimConfig(total_blocks=1000, block_size=16, prefix_cache_hit_rate=0.5))
        sim.allocate("s1", num_tokens=200)
        sim.reset()
        snap = sim.snapshot()
        assert snap["prefix_hit_tokens_total"] == 0 and snap["prefix_queries_total"] == 0


# ===========================================================================
# KV simulation - LRU eviction
# ===========================================================================


class TestKVEviction:
    def test_lru_evicts_oldest(self):
        sim = KVSimulator(KVSimConfig(total_blocks=10, block_size=1))
        sim.allocate("oldest", num_tokens=5)
        sim.allocate("newer", num_tokens=5)
        alloc = sim.allocate("s3", num_tokens=3)
        assert alloc.evictions >= 3
        assert "oldest" not in sim._allocations

    def test_recent_access_moves_to_back(self):
        sim = KVSimulator(KVSimConfig(total_blocks=10, block_size=1))
        sim.allocate("s1", num_tokens=4)
        sim.allocate("s2", num_tokens=4)
        sim.allocate("s1", num_tokens=1)  # s1 becomes MRU
        sim.allocate("s3", num_tokens=3)
        assert "s1" in sim._allocations, "s1 (MRU) should survive"
        assert "s2" not in sim._allocations, "s2 (LRU) should be evicted"

    def test_eviction_count_accumulates(self):
        sim = KVSimulator(KVSimConfig(total_blocks=6, block_size=1))
        sim.allocate("s1", num_tokens=3)
        sim.allocate("s2", num_tokens=3)
        sim.allocate("s3", num_tokens=3)
        assert sim.total_evictions == 3
        sim.allocate("s4", num_tokens=3)
        assert sim.total_evictions == 6

    def test_no_eviction_policy_caps(self):
        sim = KVSimulator(KVSimConfig(total_blocks=5, block_size=1, eviction_policy="none"))
        sim.allocate("s1", num_tokens=5)
        alloc = sim.allocate("s2", num_tokens=5)
        assert alloc.blocks_allocated == 0 and alloc.evictions == 0


# ===========================================================================
# KV simulation - multi-turn
# ===========================================================================


def test_multi_turn_accumulates_blocks():
    sim = KVSimulator(KVSimConfig(total_blocks=100, block_size=1))
    sim.allocate("s1", num_tokens=5)
    snap1 = sim.snapshot()
    sim.allocate("s1", num_tokens=5)
    snap2 = sim.snapshot()
    assert snap2["blocks_used"] == snap1["blocks_used"] + 5


def test_release_and_reallocate():
    sim = KVSimulator(KVSimConfig(total_blocks=20, block_size=1))
    sim.allocate("s1", num_tokens=10)
    assert sim.usage_perc == pytest.approx(50.0)
    sim.release("s1")
    assert sim.usage_perc == 0.0
    sim.allocate("s1", num_tokens=5)
    assert sim.usage_perc == pytest.approx(25.0)


# ===========================================================================
# MockBackend._send_sse propagates cached_tokens from KV sim (H3)
# ===========================================================================


def test_send_sse_propagates_cached_tokens():
    """cached_tokens must be non-None and > 0 when KV sim has prefix hits."""
    kv_cfg = KVSimConfig(total_blocks=1000, block_size=16, prefix_cache_hit_rate=0.5)
    cfg = MockConfig(
        output_tokens=5,
        ttft_ms=0.0,
        inter_token_ms=0.0,
        simulate_sse=True,
        kv_sim=kv_cfg,
        tokens_per_message=100,
    )
    messages = [{"role": "user", "content": "hello world"}]

    async def _run():
        async with MockBackend(cfg) as backend:
            result = await backend.send_turn("s1", 0, messages)
        return result

    result = asyncio.run(_run())
    assert result.completed
    assert result.cached_tokens is not None, "cached_tokens should not be None with KV prefix hits"
    assert result.cached_tokens > 0, "cached_tokens should be > 0 with 50% prefix hit rate"


def test_send_sse_reasoning_tokens_matches_http_type():
    """Mock backend must pass reasoning_tokens as int (not None), matching HTTP."""
    cfg = MockConfig(output_tokens=3, ttft_ms=0.0, inter_token_ms=0.0, simulate_sse=True)

    async def _run():
        async with MockBackend(cfg) as backend:
            return await backend.send_turn("s1", 0, [{"role": "user", "content": "hi"}])

    result = asyncio.run(_run())
    assert isinstance(result.reasoning_tokens, int)
    assert result.reasoning_tokens == 0  # mock default SSEResult has 0, not None


# ===========================================================================
# call_log snapshot isolation (L7)
# ===========================================================================


def test_call_log_snapshots_messages_at_call_time():
    """call_log must store a snapshot; post-call mutation must not affect it."""

    async def _run():
        cfg = MockConfig(output_tokens=5, ttft_ms=0.0, total_ms=0.0)
        async with MockBackend(cfg) as backend:
            messages = [{"role": "user", "content": "hello", "tool_calls": [{"id": "tc1"}]}]
            await backend.send_turn("s1", 0, messages)

            snapshot_before_mutation = [
                {"role": "user", "content": "hello", "tool_calls": [{"id": "tc1"}]}
            ]

            # Mutate: append a new message and mutate a nested dict
            messages.append({"role": "assistant", "content": "world"})
            messages[0]["tool_calls"][0]["id"] = "MUTATED"

            logged = backend.call_log[0]["messages"]
            assert logged == snapshot_before_mutation, (
                f"call_log was aliased to caller's list; got {logged!r}"
            )

    asyncio.run(_run())
