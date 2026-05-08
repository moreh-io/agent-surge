# SPDX-License-Identifier: MIT
"""Auto-discovery registry for trace and task loaders.

Loaders register themselves automatically when their class is defined, using
the ``name`` class attribute as the registry key.  This eliminates manual
if/elif chains and makes it trivial for third-party code to add loaders::

    from agentsurge.loaders import TraceLoader, loader_registry

    class MyCustomLoader(TraceLoader):
        name = "my-custom"
        def load(self, limit=None):
            ...

    # Now accessible via:
    loader_cls = loader_registry.get("my-custom")
"""

import importlib
import logging
import os
import pkgutil
from typing import TYPE_CHECKING

_STRICT_ENV = "AGENTSURGE_LOADER_STRICT"


def _strict_mode() -> bool:
    """Return True if the user opted into hard-fail-on-collision via env."""
    return os.environ.get(_STRICT_ENV, "").lower() in ("1", "true", "yes", "on")


if TYPE_CHECKING:
    from agentsurge.loaders.base import TaskLoader, TraceLoader

_log = logging.getLogger(__name__)

__all__ = ["LoaderRegistry", "loader_registry"]

_INTERNAL_MODULES: frozenset[str] = frozenset(["__init__", "base", "registry"])


class LoaderRegistry:
    """Central registry mapping ``name`` strings to loader classes.

    Loaders are separated into two namespaces - *trace* loaders and *task*
    loaders - mirroring the two abstract base classes.  A single registry
    instance manages both namespaces.

    Registration happens automatically via :meth:`register_trace` and
    :meth:`register_task`, which are called from
    :meth:`TraceLoader.__init_subclass__` and
    :meth:`TaskLoader.__init_subclass__`.
    """

    def __init__(self) -> None:
        self._trace: dict[str, type[TraceLoader]] = {}
        self._task: dict[str, type[TaskLoader]] = {}

    def register_trace(self, cls: "type[TraceLoader]") -> None:
        """Register a :class:`TraceLoader` subclass by its ``name`` attribute.

        When ``AGENTSURGE_LOADER_STRICT=1`` a duplicate ``name`` raises
        :class:`ValueError`; otherwise the existing entry is overwritten with
        a warning (back-compat default).
        """
        name = getattr(cls, "name", "")
        if not name:
            return
        if name in self._trace:
            existing = self._trace[name]
            if existing is not cls:
                if _strict_mode():
                    raise ValueError(
                        f"trace loader name collision on {name!r}: "
                        f"{existing.__qualname__} vs {cls.__qualname__} "
                        f"({_STRICT_ENV} is set)"
                    )
                _log.warning(
                    "Overwriting trace loader %r: %s -> %s",
                    name,
                    existing.__qualname__,
                    cls.__qualname__,
                )
        self._trace[name] = cls
        _log.debug("Registered trace loader %r -> %s", name, cls.__qualname__)

    def register_task(self, cls: "type[TaskLoader]") -> None:
        """Register a :class:`TaskLoader` subclass by its ``name`` attribute.

        When ``AGENTSURGE_LOADER_STRICT=1`` a duplicate ``name`` raises
        :class:`ValueError`; otherwise the existing entry is overwritten with
        a warning (back-compat default).
        """
        name = getattr(cls, "name", "")
        if not name:
            return
        if name in self._task:
            existing = self._task[name]
            if existing is not cls:
                if _strict_mode():
                    raise ValueError(
                        f"task loader name collision on {name!r}: "
                        f"{existing.__qualname__} vs {cls.__qualname__} "
                        f"({_STRICT_ENV} is set)"
                    )
                _log.warning(
                    "Overwriting task loader %r: %s -> %s",
                    name,
                    existing.__qualname__,
                    cls.__qualname__,
                )
        self._task[name] = cls
        _log.debug("Registered task loader %r -> %s", name, cls.__qualname__)

    def get(self, name: str) -> "type[TraceLoader] | type[TaskLoader]":
        """Look up *name* in both namespaces (trace first, then task).

        Raises :class:`KeyError` if not found in either namespace.
        """
        if name in self._trace:
            return self._trace[name]
        if name in self._task:
            return self._task[name]
        all_names = sorted(set(self._trace) | set(self._task))
        available = ", ".join(all_names) or "(none)"
        raise KeyError(f"Unknown loader {name!r}. Available: {available}") from None

    def list_trace(self) -> list[str]:
        """Return sorted list of registered trace loader names."""
        return sorted(self._trace)

    def list_task(self) -> list[str]:
        """Return sorted list of registered task loader names."""
        return sorted(self._task)

    def list_all(self) -> list[str]:
        """Return sorted list of all registered loader names."""
        return sorted(set(self._trace) | set(self._task))

    def __contains__(self, name: str) -> bool:
        return name in self._trace or name in self._task

    def discover(self, package: str = "agentsurge.loaders") -> list[str]:
        """Scan *package* for loader modules and import them eagerly.

        Each module found in *package* is imported via
        :func:`importlib.import_module`.  When a module is imported, any
        :class:`TraceLoader` or :class:`TaskLoader` subclass it defines
        triggers :meth:`object.__init_subclass__`, which in turn calls
        :meth:`register_trace` or :meth:`register_task` on this registry.

        Parameters
        ----------
        package:
            Dotted module name of the package to scan.  Defaults to
            ``"agentsurge.loaders"``.

        Returns
        -------
        list[str]
            Sorted list of **all** loader names in the registry after discovery.
        """
        try:
            pkg = importlib.import_module(package)
        except ImportError:
            _log.warning("discover: cannot import package %r - skipping", package)
            return self.list_all()

        pkg_path = getattr(pkg, "__path__", None)
        if pkg_path is None:
            _log.warning("discover: %r has no __path__; it is not a package", package)
            return self.list_all()

        discovered: list[str] = []
        for _finder, modname, _ispkg in pkgutil.iter_modules(pkg_path):
            if modname in _INTERNAL_MODULES:
                _log.debug("discover: skipping internal module %r", modname)
                continue
            full_name = f"{package}.{modname}"
            try:
                importlib.import_module(full_name)
                _log.debug("discover: imported %r", full_name)
                discovered.append(full_name)
            except Exception:
                _log.warning("discover: failed to import %r", full_name, exc_info=True)

        all_names = self.list_all()
        _log.debug(
            "discover(%r): imported %d module(s), registry now has %d loader(s): %s",
            package,
            len(discovered),
            len(all_names),
            all_names,
        )
        return all_names

    def __repr__(self) -> str:
        return f"LoaderRegistry(trace={self.list_trace()}, task={self.list_task()})"


loader_registry = LoaderRegistry()
