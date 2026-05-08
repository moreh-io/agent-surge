# SPDX-License-Identifier: MIT
"""Base class for workload generators with auto-discovery registration."""

from agentsurge.generators.registry import generator_registry
from agentsurge.types import ReplaySession  # noqa: F401


class GeneratorBase:
    """Base class for workload generators.

    Subclasses should set a class-level ``name`` attribute to auto-register
    with the generator registry::

        class MyGenerator(GeneratorBase):
            name = "my-generator"

            def generate(self, n_sessions: int, **kwargs) -> list[ReplaySession]:
                ...

    The ``generate()`` method signature varies across generators (some take
    ``n_sessions``, some take domain-specific args).  The registry pattern
    provides *discovery*, not interface enforcement --- concrete generators
    are free to define their own public API.
    """

    name: str = ""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        _name = cls.__dict__.get("name", "")
        if _name:
            generator_registry.register(cls)
