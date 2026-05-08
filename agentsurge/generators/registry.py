# SPDX-License-Identifier: MIT
"""Auto-discovery registry for workload generators.

Generators register themselves automatically when their class is defined, using
the ``name`` class attribute as the registry key.  This eliminates manual
if/elif chains and makes it trivial for third-party code to add generators::

    from agentsurge.generators import GeneratorBase, generator_registry

    class MyCustomGenerator(GeneratorBase):
        name = "my-custom"
        def generate(self, n_sessions, **kwargs):
            ...

    # Now accessible via:
    gen_cls = generator_registry.get("my-custom")

The registry is a singleton --- :data:`generator_registry` --- shared across the
process.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentsurge.generators.base import GeneratorBase

_log = logging.getLogger(__name__)

__all__ = ["GeneratorRegistry", "generator_registry"]


class GeneratorRegistry:
    """Central registry mapping ``name`` strings to generator classes.

    Registration happens automatically via :meth:`register`, which is called
    from :meth:`GeneratorBase.__init_subclass__`.

    Built-in generators are loaded lazily on first lookup/list to avoid
    circular imports between this package and its submodules.
    """

    def __init__(self) -> None:
        self._generators: dict[str, type[GeneratorBase]] = {}
        self._builtins_loaded = False

    def _ensure_builtins(self) -> None:
        """Import generator submodules to trigger auto-registration of built-in generators."""
        if self._builtins_loaded:
            return
        try:
            import agentsurge.generators.burst  # noqa: F401
            import agentsurge.generators.mixed  # noqa: F401
            import agentsurge.generators.steady  # noqa: F401
            import agentsurge.generators.synthetic  # noqa: F401
            import agentsurge.generators.task_converter  # noqa: F401
            import agentsurge.generators.trace_replay  # noqa: F401
        except ImportError as e:
            # Leave _builtins_loaded False so a subsequent call retries —
            # otherwise a transient failure would latch a partial registry.
            _log.warning("Could not load generator builtins: %s", e)
            return
        self._builtins_loaded = True

    def register(self, cls: "type[GeneratorBase]") -> None:
        """Register a :class:`GeneratorBase` subclass by its ``name`` attribute."""
        name = getattr(cls, "name", "")
        if not name:
            return  # abstract or unnamed --- skip silently
        if name in self._generators:
            existing = self._generators[name]
            if existing is not cls:
                _log.warning(
                    "Overwriting generator %r: %s -> %s",
                    name,
                    existing.__qualname__,
                    cls.__qualname__,
                )
        self._generators[name] = cls
        _log.debug("Registered generator %r -> %s", name, cls.__qualname__)

    def get(self, name: str) -> "type[GeneratorBase]":
        """Return the :class:`GeneratorBase` subclass registered under *name*.

        Raises :class:`KeyError` if no generator with that name exists.
        """
        self._ensure_builtins()
        try:
            return self._generators[name]
        except KeyError:
            available = ", ".join(sorted(self._generators)) or "(none)"
            raise KeyError(f"Unknown generator {name!r}. Available: {available}") from None

    def list_generators(self) -> list[str]:
        """Return sorted list of registered generator names."""
        self._ensure_builtins()
        return sorted(self._generators)

    def __contains__(self, name: str) -> bool:
        self._ensure_builtins()
        return name in self._generators

    def __repr__(self) -> str:
        self._ensure_builtins()
        return f"GeneratorRegistry(generators={self.list_generators()})"


generator_registry = GeneratorRegistry()
