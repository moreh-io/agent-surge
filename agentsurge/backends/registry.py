# SPDX-License-Identifier: MIT
"""Auto-discovery registry for inference backends.

Backends register themselves automatically when their class is defined, using
the ``name`` class attribute as the registry key.  This eliminates manual
if/elif chains and makes it trivial for third-party code to add backends::

    from agentsurge.backends import BackendBase, backend_registry

    class MyCustomBackend(BackendBase):
        name = "my-custom"
        async def send_turn(self, session_id, turn_index, messages):
            ...

    # Now accessible via:
    backend_cls = backend_registry.get("my-custom")

The registry is a singleton - :data:`backend_registry` - shared across the
process.  The default backend name is ``"vllm"``.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentsurge.backends.base import BackendBase

_log = logging.getLogger(__name__)

__all__ = ["BackendRegistry", "backend_registry"]

DEFAULT_BACKEND = "vllm"


class BackendRegistry:
    """Central registry mapping ``name`` strings to backend classes.

    Registration happens automatically via :meth:`register`, which is called
    from :meth:`BackendBase.__init_subclass__`.

    Parameters
    ----------
    default : str
        The backend name returned by :meth:`get` when called without
        arguments or with ``name=None``.
    """

    def __init__(self, *, default: str = DEFAULT_BACKEND) -> None:
        self._backends: dict[str, type[BackendBase]] = {}
        self._default = default

    def register(self, cls: "type[BackendBase]") -> None:
        """Register a :class:`BackendBase` subclass by its ``name`` attribute.

        Called automatically from ``BackendBase.__init_subclass__``.  Backends
        whose ``name`` is empty (e.g. abstract intermediate classes) are
        silently skipped.
        """
        name = getattr(cls, "name", "")
        if not name:
            return  # abstract or unnamed - skip silently
        if name in self._backends:
            existing = self._backends[name]
            if existing is not cls:
                _log.warning(
                    "Overwriting backend %r: %s -> %s",
                    name,
                    existing.__qualname__,
                    cls.__qualname__,
                )
        self._backends[name] = cls
        _log.debug("Registered backend %r -> %s", name, cls.__qualname__)

    def get(self, name: str | None = None) -> "type[BackendBase]":
        """Return the :class:`BackendBase` subclass registered under *name*.

        If *name* is ``None``, the default backend (``"vllm"``) is returned.

        Raises
        ------
        KeyError
            If no backend with that name is registered.
        """
        if name is None:
            name = self._default
        try:
            return self._backends[name]
        except KeyError:
            available = ", ".join(sorted(self._backends)) or "(none)"
            raise KeyError(f"Unknown backend {name!r}. Available: {available}") from None

    @property
    def default(self) -> str:
        """The default backend name."""
        return self._default

    def list_backends(self) -> list[str]:
        """Return sorted list of registered backend names."""
        return sorted(self._backends)

    def __contains__(self, name: str) -> bool:
        return name in self._backends

    def __len__(self) -> int:
        return len(self._backends)

    def __repr__(self) -> str:
        return f"BackendRegistry(backends={self.list_backends()}, default={self._default!r})"


backend_registry = BackendRegistry()
