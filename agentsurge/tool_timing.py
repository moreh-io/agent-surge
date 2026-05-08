# SPDX-License-Identifier: MIT
"""Per-tool-type delay distributions for realistic inter-turn timing."""

import random

_TOOL_CATEGORIES: dict[str, str] = {
    # Fast compute (~0.1s)
    "think": "fast",
    "finish": "fast",
    "task_tracker": "fast",
    # File I/O (~0.6s)
    "str_replace_editor": "file_io",
    "read_file": "file_io",
    "write_file": "file_io",
    "edit_file": "file_io",
    "view": "file_io",
    # Shell execution (~2.0s, high variance)
    "execute_bash": "exec",
    "run_tests": "exec",
    "bash": "exec",
    "terminal": "exec",
    "search_code": "exec",
    "grep": "exec",
    # Network (~2.7s)
    "curl": "network",
    "api_call": "network",
    "web_search": "network",
    "browser": "network",
}


def classify_tool(name: str | None) -> str:
    """Classify tool name into delay category (exact match only)."""
    if name is None:
        return "default"
    return _TOOL_CATEGORIES.get(name, "default")


def sample_delay(category: str, rng: random.Random) -> float:
    """Sample inter-turn delay (seconds) for a tool category.

    Distributions calibrated to Continuum (arXiv:2511.02230) Table 1:
    mean 925ms (light agents) to 1923ms (heavy agents).
    """
    if category == "fast":
        # ~100ms mean, Exponential
        return rng.expovariate(1.0 / 0.1)
    if category == "file_io":
        # median ~0.6s, LogNormal
        return min(rng.lognormvariate(-0.5, 0.6), 10.0)
    if category == "exec":
        # median ~2.0s, high variance, LogNormal
        return min(rng.lognormvariate(0.7, 0.8), 60.0)
    if category == "network":
        # median ~2.7s, LogNormal
        return min(rng.lognormvariate(1.0, 1.0), 30.0)
    # default: original bimodal for backward compat
    if rng.random() < 0.7:
        return rng.expovariate(1.0 / 0.3)
    return min(rng.lognormvariate(0.7, 0.8), 30.0)


def sample_tool_delay(tool_name: str | None, rng: random.Random) -> float:
    """Convenience: classify + sample in one call."""
    return sample_delay(classify_tool(tool_name), rng)
