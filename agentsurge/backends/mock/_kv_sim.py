# SPDX-License-Identifier: MIT
"""KV cache simulation for the mock backend.

Provides a lightweight, deterministic simulation of GPU KV-cache behavior
including allocation, eviction, and usage tracking.  This enables realistic
testing of workload patterns that depend on KV-cache pressure (e.g. latency
degradation experiments) without a real inference server.

Usage::

    from agentsurge.backends.mock._kv_sim import KVSimulator, KVSimConfig

    sim = KVSimulator(KVSimConfig(total_blocks=1000, block_size=16))
    alloc = sim.allocate("session-1", num_tokens=2048)
    blocks, usage = alloc.blocks_used, alloc.kv_usage_perc
    sim.release("session-1")

The simulator is intentionally simple - it tracks block-level allocation
without modeling the full vLLM block table, prefix sharing, or paging
internals.  This is sufficient for testing KV-pressure-dependent behavior.
"""

import logging
from dataclasses import dataclass
from typing import Any

_log = logging.getLogger(__name__)


@dataclass
class KVSimConfig:
    """Configuration for the KV cache simulator.

    Parameters
    ----------
    total_blocks : int
        Total number of KV cache blocks available (models GPU memory budget).
    block_size : int
        Number of tokens per block (typically 16 for vLLM).
    prefix_cache_hit_rate : float
        Simulated prefix cache hit rate (0.0–1.0).  When > 0, a fraction of
        requested tokens are served from the prefix cache and don't consume
        new blocks.
    eviction_policy : str
        Eviction strategy: ``"lru"`` (least recently used) or ``"none"``
        (allocation fails when full).
    """

    total_blocks: int = 1000
    block_size: int = 16
    prefix_cache_hit_rate: float = 0.0
    eviction_policy: str = "lru"

    def __post_init__(self) -> None:
        if self.total_blocks <= 0:
            raise ValueError(f"total_blocks must be > 0, got {self.total_blocks}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")
        if not 0.0 <= self.prefix_cache_hit_rate <= 1.0:
            raise ValueError(
                f"prefix_cache_hit_rate must be in [0.0, 1.0], got {self.prefix_cache_hit_rate}"
            )
        if self.eviction_policy not in ("lru", "none"):
            raise ValueError(
                f"eviction_policy must be 'lru' or 'none', got {self.eviction_policy!r}"
            )


@dataclass
class KVAllocation:
    """Result of a KV cache allocation request.

    Attributes
    ----------
    session_id : str
        Session that requested the allocation.
    blocks_requested : int
        Number of blocks needed for the token count.
    blocks_allocated : int
        Number of new blocks actually allocated (after prefix hits).
    blocks_used : int
        Total blocks currently in use across all sessions.
    total_blocks : int
        Total blocks available.
    kv_usage_perc : float
        Current KV cache utilization (0.0–100.0).
    prefix_hit_tokens : int
        Number of tokens served from the prefix cache.
    evictions : int
        Number of blocks evicted to make room for this allocation.
    """

    session_id: str = ""
    blocks_requested: int = 0
    blocks_allocated: int = 0
    blocks_used: int = 0
    total_blocks: int = 0
    kv_usage_perc: float = 0.0
    prefix_hit_tokens: int = 0
    evictions: int = 0


class KVSimulator:
    """Simulates GPU KV-cache allocation and eviction.

    Tracks per-session block allocations and provides aggregate usage metrics.
    Thread-safe under CPython's GIL (single-writer pattern).

    Parameters
    ----------
    config : KVSimConfig
        Simulator configuration.  Defaults to a 1000-block cache.
    """

    def __init__(self, config: KVSimConfig | None = None) -> None:
        self.config = config or KVSimConfig()
        self._allocations: dict[str, list[int]] = {}
        self._alloc_order: list[str] = []
        self._total_used: int = 0
        self._total_evictions: int = 0
        self._total_tokens_requested: int = 0
        self._total_prefix_hit_tokens: int = 0

    @property
    def usage_perc(self) -> float:
        """Current KV cache usage as a percentage (0.0–100.0)."""
        if self.config.total_blocks == 0:
            return 0.0
        return (self._total_used / self.config.total_blocks) * 100.0

    @property
    def blocks_free(self) -> int:
        """Number of free blocks."""
        return max(0, self.config.total_blocks - self._total_used)

    @property
    def total_evictions(self) -> int:
        """Cumulative number of blocks evicted."""
        return self._total_evictions

    def allocate(self, session_id: str, num_tokens: int) -> KVAllocation:
        """Allocate KV-cache blocks for a session.

        If the session already has an allocation, additional blocks are appended
        (models multi-turn growth).  Prefix cache hits reduce the number of
        new blocks needed.

        Parameters
        ----------
        session_id : str
            Session identifier.
        num_tokens : int
            Number of new tokens requiring KV storage.

        Returns
        -------
        KVAllocation
            Allocation result with usage metrics.
        """
        cfg = self.config

        self._total_tokens_requested += num_tokens
        prefix_hit_tokens = int(num_tokens * cfg.prefix_cache_hit_rate)
        tokens_needing_blocks = num_tokens - prefix_hit_tokens
        self._total_prefix_hit_tokens += prefix_hit_tokens

        blocks_needed = (tokens_needing_blocks + cfg.block_size - 1) // cfg.block_size
        blocks_needed = max(0, blocks_needed)

        evictions = 0
        if blocks_needed > self.blocks_free and cfg.eviction_policy == "lru":
            evictions = self._evict_lru(blocks_needed - self.blocks_free)
            # else: eviction_policy == "none", allocation is capped

        blocks_to_alloc = min(blocks_needed, self.blocks_free)

        if session_id not in self._allocations:
            self._allocations[session_id] = []
        self._allocations[session_id].append(blocks_to_alloc)
        self._total_used += blocks_to_alloc

        if session_id in self._alloc_order:
            self._alloc_order.remove(session_id)
        self._alloc_order.append(session_id)

        result = KVAllocation(
            session_id=session_id,
            blocks_requested=blocks_needed,
            blocks_allocated=blocks_to_alloc,
            blocks_used=self._total_used,
            total_blocks=cfg.total_blocks,
            kv_usage_perc=self.usage_perc,
            prefix_hit_tokens=prefix_hit_tokens,
            evictions=evictions,
        )

        _log.debug(
            "KVSimulator.allocate session=%s tokens=%d blocks_needed=%d "
            "allocated=%d evictions=%d usage=%.1f%%",
            session_id,
            num_tokens,
            blocks_needed,
            blocks_to_alloc,
            evictions,
            result.kv_usage_perc,
        )

        return result

    def release(self, session_id: str) -> int:
        """Release all blocks held by a session.

        Parameters
        ----------
        session_id : str
            Session whose blocks should be freed.

        Returns
        -------
        int
            Number of blocks released.
        """
        if session_id not in self._allocations:
            return 0
        blocks = self._allocations.pop(session_id)
        total_released = sum(blocks)
        self._total_used = max(0, self._total_used - total_released)
        if session_id in self._alloc_order:
            self._alloc_order.remove(session_id)
        _log.debug(
            "KVSimulator.release session=%s blocks=%d usage=%.1f%%",
            session_id,
            total_released,
            self.usage_perc,
        )
        return total_released

    def reset(self) -> None:
        """Reset all allocations and counters."""
        self._allocations.clear()
        self._alloc_order.clear()
        self._total_used = 0
        self._total_evictions = 0
        self._total_tokens_requested = 0
        self._total_prefix_hit_tokens = 0

    @property
    def prefix_hit_rate(self) -> float:
        """Cumulative prefix cache hit rate (0.0–1.0).

        Computed as total prefix-hit tokens divided by total tokens requested.
        Returns 0.0 if no allocations have been made yet.
        """
        if self._total_tokens_requested == 0:
            return 0.0
        return self._total_prefix_hit_tokens / self._total_tokens_requested

    @property
    def cache_fill_ratio(self) -> float:
        """Fraction of the total KV cache currently filled (0.0–1.0).

        Equivalent to ``usage_perc / 100``.  Provided as a convenience for
        code that prefers a 0–1 scale (e.g. for normalised metrics).
        """
        return self.usage_perc / 100.0

    def snapshot(self) -> dict[str, Any]:
        """Return a snapshot of the current simulator state.

        Returns
        -------
        dict
            Contains ``kv_usage_perc``, ``blocks_used``, ``blocks_free``,
            ``total_blocks``, ``num_sessions``, ``total_evictions``,
            ``prefix_hit_tokens_total``, ``prefix_queries_total``,
            ``prefix_hit_rate``, ``cache_fill_ratio``.
        """
        return {
            "kv_usage_perc": round(self.usage_perc, 2),
            "blocks_used": self._total_used,
            "blocks_free": self.blocks_free,
            "total_blocks": self.config.total_blocks,
            "num_sessions": len(self._allocations),
            "total_evictions": self._total_evictions,
            "prefix_hit_tokens_total": self._total_prefix_hit_tokens,
            "prefix_queries_total": self._total_tokens_requested,
            "prefix_hit_rate": round(self.prefix_hit_rate, 4),
            "cache_fill_ratio": round(self.cache_fill_ratio, 4),
        }

    def _evict_lru(self, blocks_needed: int) -> int:
        """Evict least-recently-used sessions until enough blocks are free.

        Returns the number of blocks evicted.
        """
        evicted = 0
        while blocks_needed > 0 and self._alloc_order:
            victim = self._alloc_order[0]
            released = self.release(victim)
            evicted += released
            blocks_needed -= released
            self._total_evictions += released
        return evicted

    def __repr__(self) -> str:
        return (
            f"<KVSimulator blocks={self._total_used}/{self.config.total_blocks} "
            f"({self.usage_perc:.1f}%) sessions={len(self._allocations)}>"
        )
