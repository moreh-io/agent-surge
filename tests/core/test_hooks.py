"""Tests for agentsurge._hooks - runtime registration of loaders and adapters."""

from __future__ import annotations

import pytest

from agentsurge._hooks import register_adapter, register_loader
from agentsurge.backends.base import BackendBase
from agentsurge.backends.registry import backend_registry
from agentsurge.loaders.base import TraceLoader
from agentsurge.loaders.registry import loader_registry


@pytest.fixture(autouse=True)
def _cleanup_registries():
    """Remove test entries from global registries after each test."""
    yield
    for name in list(loader_registry._trace):
        if name.startswith("_test-hooks"):
            del loader_registry._trace[name]
    for name in list(loader_registry._task):
        if name.startswith("_test-hooks"):
            del loader_registry._task[name]
    for name in list(backend_registry._backends):
        if name.startswith("_test-hooks"):
            del backend_registry._backends[name]


class TestRegisterLoader:
    def test_register_trace_loader(self):
        class _TestTrace(TraceLoader):
            name = "_test-hooks-trace"

            def load(self, limit=None):
                yield from ()

        result = register_loader(_TestTrace)
        assert result is _TestTrace
        assert loader_registry.get("_test-hooks-trace") is _TestTrace

    def test_register_returns_cls_for_decorator_use(self):
        @register_loader
        class _DecoratedTrace(TraceLoader):
            name = "_test-hooks-decorated"

            def load(self, limit=None):
                yield from ()

        assert _DecoratedTrace.name == "_test-hooks-decorated"
        assert loader_registry.get("_test-hooks-decorated") is _DecoratedTrace

    def test_register_loader_no_name_raises(self):
        class _NoName(TraceLoader):
            def load(self, limit=None):
                yield from ()

        _NoName.name = ""
        with pytest.raises(ValueError, match="non-empty 'name'"):
            register_loader(_NoName)

    def test_register_loader_wrong_type_raises(self):
        class _NotALoader:
            name = "bad"

        with pytest.raises(TypeError, match="not a subclass"):
            register_loader(_NotALoader)  # type: ignore[arg-type]

    def test_idempotent_reregistration(self):
        class _Idem(TraceLoader):
            name = "_test-hooks-idem"

            def load(self, limit=None):
                yield from ()

        register_loader(_Idem)
        register_loader(_Idem)
        assert loader_registry.get("_test-hooks-idem") is _Idem


class TestRegisterAdapter:
    def test_register_backend_adapter(self):
        class _TestBackend(BackendBase):
            name = "_test-hooks-backend"

            async def send_turn(self, sid, tidx, msgs, **kwargs):
                pass

        result = register_adapter(_TestBackend)
        assert result is _TestBackend
        assert backend_registry.get("_test-hooks-backend") is _TestBackend

    def test_register_adapter_no_name_raises(self):
        class _NoName(BackendBase):
            async def send_turn(self, sid, tidx, msgs, **kwargs):
                pass

        _NoName.name = ""
        with pytest.raises(ValueError, match="non-empty 'name'"):
            register_adapter(_NoName)

    def test_register_adapter_wrong_type_raises(self):
        class _NotABackend:
            name = "bad"

        with pytest.raises(TypeError, match="not a subclass"):
            register_adapter(_NotABackend)  # type: ignore[arg-type]

    def test_decorator_usage(self):
        @register_adapter
        class _DecBackend(BackendBase):
            name = "_test-hooks-dec-backend"

            async def send_turn(self, sid, tidx, msgs, **kwargs):
                pass

        assert backend_registry.get("_test-hooks-dec-backend") is _DecBackend
