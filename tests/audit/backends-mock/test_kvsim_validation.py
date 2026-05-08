"""M8: KVSimConfig must validate its fields in __post_init__."""

from __future__ import annotations

import pytest

from agentsurge.backends.mock._kv_sim import KVSimConfig


def test_block_size_zero_rejected():
    with pytest.raises(ValueError, match="block_size"):
        KVSimConfig(block_size=0)


def test_total_blocks_zero_rejected():
    with pytest.raises(ValueError, match="total_blocks"):
        KVSimConfig(total_blocks=0)


def test_prefix_cache_hit_rate_above_one_rejected():
    with pytest.raises(ValueError, match="prefix_cache_hit_rate"):
        KVSimConfig(prefix_cache_hit_rate=1.5)


def test_prefix_cache_hit_rate_negative_rejected():
    with pytest.raises(ValueError, match="prefix_cache_hit_rate"):
        KVSimConfig(prefix_cache_hit_rate=-0.1)


def test_eviction_policy_invalid_rejected():
    with pytest.raises(ValueError, match="eviction_policy"):
        KVSimConfig(eviction_policy="random")


def test_valid_config_accepted():
    cfg = KVSimConfig(
        total_blocks=100,
        block_size=16,
        prefix_cache_hit_rate=0.5,
        eviction_policy="lru",
    )
    assert cfg.total_blocks == 100

    cfg2 = KVSimConfig(eviction_policy="none")
    assert cfg2.eviction_policy == "none"
