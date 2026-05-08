"""M11: Registry name collisions must be opt-in escalatable.

Default behaviour stays warn-and-overwrite (back-compat).  When
``AGENTSURGE_LOADER_STRICT=1`` is set, a duplicate ``name`` registration
must raise ``ValueError`` so a plugin-discovery collision becomes a
hard, reproducible failure.
"""

from __future__ import annotations

import pytest


def test_strict_mode_raises_on_trace_collision(monkeypatch):
    from agentsurge.loaders.base import TraceLoader
    from agentsurge.loaders.registry import LoaderRegistry

    monkeypatch.setenv("AGENTSURGE_LOADER_STRICT", "1")

    reg = LoaderRegistry()

    class A(TraceLoader):
        name = ""  # avoid auto-registration into global registry

        def load(self, limit=None):  # pragma: no cover
            return iter([])

    class B(TraceLoader):
        name = ""

        def load(self, limit=None):  # pragma: no cover
            return iter([])

    A.name = "dup-trace"
    B.name = "dup-trace"
    reg.register_trace(A)
    with pytest.raises(ValueError):
        reg.register_trace(B)


def test_strict_mode_raises_on_task_collision(monkeypatch):
    from agentsurge.loaders.base import TaskLoader
    from agentsurge.loaders.registry import LoaderRegistry

    monkeypatch.setenv("AGENTSURGE_LOADER_STRICT", "1")

    reg = LoaderRegistry()

    class A(TaskLoader):
        name = ""

        def load(self, limit=None):  # pragma: no cover
            return iter([])

    class B(TaskLoader):
        name = ""

        def load(self, limit=None):  # pragma: no cover
            return iter([])

    A.name = "dup-task"
    B.name = "dup-task"
    reg.register_task(A)
    with pytest.raises(ValueError):
        reg.register_task(B)


def test_default_mode_warns_does_not_raise(monkeypatch):
    """Without the env var, behaviour stays warn-and-overwrite."""
    from agentsurge.loaders.base import TraceLoader
    from agentsurge.loaders.registry import LoaderRegistry

    monkeypatch.delenv("AGENTSURGE_LOADER_STRICT", raising=False)

    reg = LoaderRegistry()

    class A(TraceLoader):
        name = ""

        def load(self, limit=None):  # pragma: no cover
            return iter([])

    class B(TraceLoader):
        name = ""

        def load(self, limit=None):  # pragma: no cover
            return iter([])

    A.name = "dup-soft"
    B.name = "dup-soft"
    reg.register_trace(A)
    reg.register_trace(B)  # must not raise
    assert reg.get("dup-soft") is B
