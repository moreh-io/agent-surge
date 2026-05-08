"""Tests for agentsurge.tool_timing module."""

from __future__ import annotations

import random

import pytest

from agentsurge.tool_timing import (
    classify_tool,
    sample_delay,
    sample_tool_delay,
)

# ---------------------------------------------------------------------------
# classify_tool
# ---------------------------------------------------------------------------


def test_classify_known_tools():
    assert classify_tool("bash") == "exec"
    assert classify_tool("read_file") == "file_io"
    assert classify_tool("finish") == "fast"
    assert classify_tool("curl") == "network"


def test_classify_unknown():
    assert classify_tool("totally_unknown") == "default"


# ---------------------------------------------------------------------------
# sample_delay
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return random.Random(42)


def test_sample_delay_fast_positive(rng):
    for _ in range(50):
        d = sample_delay("fast", rng)
        assert d > 0


def test_sample_delay_exec_capped(rng):
    for _ in range(200):
        d = sample_delay("exec", rng)
        assert d <= 60.0


def test_sample_delay_fast_has_low_mean(rng):
    samples = [sample_delay("fast", rng) for _ in range(1000)]
    mean = sum(samples) / len(samples)
    assert mean < 0.5


def test_sample_delay_exec_has_higher_mean_than_fast(rng):
    fast_samples = [sample_delay("fast", rng) for _ in range(1000)]
    exec_samples = [sample_delay("exec", rng) for _ in range(1000)]
    assert sum(exec_samples) / len(exec_samples) > sum(fast_samples) / len(fast_samples)


def test_sample_delay_deterministic():
    r1 = random.Random(99)
    r2 = random.Random(99)
    vals1 = [sample_delay("exec", r1) for _ in range(10)]
    vals2 = [sample_delay("exec", r2) for _ in range(10)]
    assert vals1 == vals2


# ---------------------------------------------------------------------------
# sample_tool_delay (convenience)
# ---------------------------------------------------------------------------


def test_sample_tool_delay_returns_float(rng):
    d = sample_tool_delay("bash", rng)
    assert d > 0
    assert d <= 60.0


def test_sample_tool_delay_none(rng):
    d = sample_tool_delay(None, rng)
    assert d > 0
    assert d <= 30.0
