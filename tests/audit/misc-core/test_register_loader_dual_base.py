# SPDX-License-Identifier: MIT
"""[audit:misc-core#H4] register_loader must surface dual-base classes.

A class that subclasses BOTH ``TraceLoader`` and ``TaskLoader`` should
not be silently registered only on the trace side. Either reject with a
clear error, or register in both registries. The previous implementation
fell through the ``elif`` branch and quietly dropped the task-side
registration with no log line above debug level.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from agentsurge import register_loader
from agentsurge.loaders.base import TaskLoader, TraceLoader


def test_dual_base_loader_is_rejected_or_double_registered() -> None:
    class DualLoader(TraceLoader, TaskLoader):  # type: ignore[misc]
        name = "dual-loader-h4"

        def load(self, limit: int | None = None) -> Iterator:
            return iter(())

    with pytest.raises((TypeError, ValueError)) as exc_info:
        register_loader(DualLoader)
    assert "both" in str(exc_info.value).lower() or "dual" in str(exc_info.value).lower(), (
        f"error message must call out the dual-base case, got: {exc_info.value!r}"
    )
