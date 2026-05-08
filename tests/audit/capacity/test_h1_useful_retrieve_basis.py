"""Audit H1: compute_external_reuse must label its denominator basis.

On SGLang backends, prefix_cache_delta() returns (0.0, 0.0) by contract,
so the apc_miss denominator silently collapses to total_prompt. Cross-backend
aggregation needs to know which definition was used, so the result dict must
carry an explicit `useful_retrieve_basis` tag.
"""

from __future__ import annotations

from agentsurge.capacity.wasted_store import compute_external_reuse


class _Snap:
    def __init__(self, isl_total: float) -> None:
        self.isl_total = isl_total


class _VLLMCollector:
    """vLLM: prefix_cache_delta returns real APC hits."""

    memory_pressure = 0.0

    def __init__(self, *, apc_hits: float, total_prompt: float, ext_hits: float, stored: float):
        self._apc_hits = apc_hits
        self._ext_hits = ext_hits
        self._stored = stored
        self.snapshots = [_Snap(0.0), _Snap(total_prompt)]

    def prefix_cache_delta(self):
        return (self._apc_hits, 0.0)

    def lmcache_v1_deltas(self):
        return {"hit_tokens": self._ext_hits, "stored_tokens": self._stored}

    def external_prefix_cache_delta(self):
        return (self._ext_hits, 0.0)


class _SGLangCollector:
    """SGLang: prefix_cache_delta clamps to (0.0, 0.0) by design."""

    memory_pressure = 0.0
    sglang_cache_hit_rate = 0.4

    def __init__(self, *, total_prompt: float, ext_hits: float, stored: float):
        self._ext_hits = ext_hits
        self._stored = stored
        self.snapshots = [_Snap(0.0), _Snap(total_prompt)]

    def prefix_cache_delta(self):
        return (0.0, 0.0)

    def lmcache_v1_deltas(self):
        return {"hit_tokens": self._ext_hits, "stored_tokens": self._stored}

    def external_prefix_cache_delta(self):
        return (self._ext_hits, 0.0)


def test_vllm_basis_is_apc_miss():
    collector = _VLLMCollector(apc_hits=40, total_prompt=100, ext_hits=30, stored=50)
    result = compute_external_reuse(collector)
    assert result["useful_retrieve_basis"] == "apc_miss"


def test_sglang_basis_is_total_prompt():
    collector = _SGLangCollector(total_prompt=100, ext_hits=30, stored=50)
    result = compute_external_reuse(collector)
    assert result["useful_retrieve_basis"] == "total_prompt"
