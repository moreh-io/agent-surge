# SPDX-License-Identifier: MIT
"""Runtime hook functions for registering custom loaders and adapters.

These convenience functions provide a simple, explicit way to register
third-party extensions at runtime without relying on ``__init_subclass__``
auto-discovery (which only fires when the class is **defined** in a process
that has already imported the base class).

Usage
-----
Register a custom trace loader::

    from agentsurge import register_loader

    class MyLoader(TraceLoader):
        name = "my-loader"
        def load(self, limit=None):
            ...

    register_loader(MyLoader)

Register a custom backend adapter::

    from agentsurge import register_adapter

    class MyBackend(BackendBase):
        name = "my-backend"
        async def send_turn(self, session_id, turn_index, messages):
            ...

    register_adapter(MyBackend)

Both functions are idempotent - re-registering the same class is a no-op.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentsurge.backends.base import BackendBase
    from agentsurge.loaders.base import TaskLoader, TraceLoader

_log = logging.getLogger(__name__)

__all__ = ["register_loader", "register_adapter"]


def register_loader(
    cls: "type[TraceLoader] | type[TaskLoader]",
) -> "type[TraceLoader] | type[TaskLoader]":
    """Register a :class:`TraceLoader` or :class:`TaskLoader` with the global registry.

    The function inspects the class hierarchy to determine which namespace
    (trace or task) the loader belongs to.  The loader **must** have a non-empty
    ``name`` class attribute.

    Parameters
    ----------
    cls : type
        A concrete subclass of :class:`~agentsurge.loaders.base.TraceLoader` or
        :class:`~agentsurge.loaders.base.TaskLoader` with a ``name`` attribute.

    Returns
    -------
    type
        The same *cls*, unchanged.  This allows ``register_loader`` to be used
        as a decorator::

            @register_loader
            class MyLoader(TraceLoader):
                name = "my-loader"
                ...

    Raises
    ------
    TypeError
        If *cls* is not a subclass of ``TraceLoader`` or ``TaskLoader``.
    ValueError
        If *cls* has no ``name`` attribute or it is empty.

    Examples
    --------
    >>> from agentsurge import register_loader
    >>> from agentsurge.loaders.base import TraceLoader
    >>> class Demo(TraceLoader):
    ...     name = "demo"
    ...     def load(self, limit=None): yield from ()
    >>> register_loader(Demo)
    <class '...Demo'>
    """
    from agentsurge.loaders.base import TaskLoader, TraceLoader
    from agentsurge.loaders.registry import loader_registry

    name = getattr(cls, "name", "")
    if not name:
        raise ValueError(
            f"Loader class {cls.__qualname__!r} must have a non-empty 'name' "
            f"class attribute to be registered."
        )

    if issubclass(cls, TraceLoader) and issubclass(cls, TaskLoader):
        raise TypeError(
            f"{cls.__qualname__!r} subclasses both TraceLoader and TaskLoader; "
            f"pick one. Dual-base loaders cannot be routed unambiguously by the "
            f"registry (a single class would silently land in only one namespace)."
        )
    if issubclass(cls, TraceLoader):
        loader_registry.register_trace(cls)
        _log.debug("register_loader: registered trace loader %r", name)
    elif issubclass(cls, TaskLoader):
        loader_registry.register_task(cls)
        _log.debug("register_loader: registered task loader %r", name)
    else:
        raise TypeError(
            f"{cls.__qualname__!r} is not a subclass of TraceLoader or TaskLoader. Cannot register."
        )
    return cls


def register_adapter(
    cls: "type[BackendBase]",
) -> "type[BackendBase]":
    """Register a :class:`BackendBase` subclass with the global backend registry.

    The adapter **must** have a non-empty ``name`` class attribute.

    Parameters
    ----------
    cls : type
        A concrete subclass of :class:`~agentsurge.backends.base.BackendBase`
        with a ``name`` attribute.

    Returns
    -------
    type
        The same *cls*, unchanged.  This allows ``register_adapter`` to be
        used as a decorator::

            @register_adapter
            class MyBackend(BackendBase):
                name = "my-backend"
                ...

    Raises
    ------
    TypeError
        If *cls* is not a subclass of ``BackendBase``.
    ValueError
        If *cls* has no ``name`` attribute or it is empty.

    Examples
    --------
    >>> from agentsurge import register_adapter
    >>> from agentsurge.backends.base import BackendBase
    >>> class DummyBackend(BackendBase):
    ...     name = "dummy"
    ...     async def send_turn(self, sid, tidx, msgs):
    ...         pass
    >>> register_adapter(DummyBackend)
    <class '...DummyBackend'>
    """
    from agentsurge.backends.base import BackendBase
    from agentsurge.backends.registry import backend_registry

    name = getattr(cls, "name", "")
    if not name:
        raise ValueError(
            f"Backend class {cls.__qualname__!r} must have a non-empty 'name' "
            f"class attribute to be registered."
        )

    if not issubclass(cls, BackendBase):
        raise TypeError(f"{cls.__qualname__!r} is not a subclass of BackendBase. Cannot register.")

    backend_registry.register(cls)
    _log.debug("register_adapter: registered backend %r", name)
    return cls
