# SPDX-License-Identifier: MIT
"""H1: MetricsSnapshot.__eq__ must not mutate backend_metrics via factory autocreation."""

from agentsurge.types.metrics import MetricsSnapshot


def test_eq_does_not_create_lmcache_v1_key():
    s1 = MetricsSnapshot(timestamp=1.0)
    s2 = MetricsSnapshot(timestamp=1.0)
    before_s1 = "lmcache_v1" in s1.backend_metrics
    before_s2 = "lmcache_v1" in s2.backend_metrics
    _ = s1 == s2
    after_s1 = "lmcache_v1" in s1.backend_metrics
    after_s2 = "lmcache_v1" in s2.backend_metrics
    assert before_s1 == after_s1, "s1.backend_metrics mutated by ==: lmcache_v1 inserted"
    assert before_s2 == after_s2, "s2.backend_metrics mutated by ==: lmcache_v1 inserted"


def test_eq_does_not_create_mutable_default_keys():
    s1 = MetricsSnapshot()
    s2 = MetricsSnapshot()
    before = dict(s1.backend_metrics), dict(s2.backend_metrics)
    _ = s1 == s2
    after = dict(s1.backend_metrics), dict(s2.backend_metrics)
    assert before == after, "Equality comparison mutated backend_metrics"


def test_repr_unaffected_by_eq():
    """repr should be stable across __eq__ invocations."""
    s = MetricsSnapshot(timestamp=2.0)
    other = MetricsSnapshot(timestamp=2.0)
    repr_before = repr(s)
    _ = s == other
    repr_after = repr(s)
    assert repr_before == repr_after
