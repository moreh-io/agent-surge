# SPDX-License-Identifier: MIT
"""agentsurge trace and task loaders."""

from agentsurge.loaders.abc_bench import ABCBenchLoader
from agentsurge.loaders.base import (
    BaseHFTaskLoader,
    Session,
    Task,
    TaskLoader,
    TraceLoader,
    Trajectory,
    Turn,
)
from agentsurge.loaders.featurebench import FeatureBenchLoader
from agentsurge.loaders.openhands import OpenHandsLoader
from agentsurge.loaders.registry import LoaderRegistry, loader_registry
from agentsurge.loaders.swe_bench import SWEBenchLoader
from agentsurge.loaders.swe_evo import SWEEvoLoader
from agentsurge.loaders.swe_smith import SWESmithLoader

_LOADER_ALIASES: dict[str, str] = {
    "swe-bench": "swe-bench-verified",
    "swe-smith-tool": "swe-smith",
    "swe-smith-xml": "swe-smith",
    "swe-smith-ticks": "swe-smith",
}

# Aliases that need to inject a constructor argument (split, etc.) so
# callers of ``get_loader(alias)()`` get a correctly-pinned loader.
_SWE_SMITH_SPLIT_BY_ALIAS: dict[str, str] = {
    "swe-smith-tool": "tool",
    "swe-smith-xml": "xml",
    "swe-smith-ticks": "ticks",
}


def get_loader(name: str) -> type[TraceLoader] | type[TaskLoader]:
    """Return the loader *class* (or factory callable) registered under *name*.

    Parameters
    ----------
    name:
        Loader name as registered (e.g. ``"openhands"``,
        ``"swe-bench-verified"``) **or** a supported alias
        (e.g. ``"swe-bench"``, ``"swe-smith-xml"``).

    Returns
    -------
    type[TraceLoader] | type[TaskLoader]
        The loader *class* - not an instance.  For aliases that pin a
        constructor argument (e.g. ``swe-smith-xml`` -> split='xml'),
        a factory callable is returned that, when called with no args,
        produces a loader instance with the correct argument set.

    Raises
    ------
    KeyError
        If *name* (after alias resolution) is not registered.
    """
    if name in _SWE_SMITH_SPLIT_BY_ALIAS:
        split = _SWE_SMITH_SPLIT_BY_ALIAS[name]
        cls = loader_registry.get("swe-smith")

        def _factory(*args, **kwargs):  # type: ignore[no-untyped-def]
            kwargs.setdefault("split", split)
            return cls(*args, **kwargs)  # type: ignore[operator]

        _factory.name = name  # type: ignore[attr-defined]
        return _factory  # type: ignore[return-value]

    resolved = _LOADER_ALIASES.get(name, name)
    return loader_registry.get(resolved)


def discover(package: str | None = None) -> list[str]:
    """Eagerly scan *package* for loader modules and populate the registry.

    Parameters
    ----------
    package:
        Dotted module name of the package to scan.  Defaults to
        ``"agentsurge.loaders"`` (this package).

    Returns
    -------
    list[str]
        Sorted list of all loader names known to the registry after the scan.
    """
    return loader_registry.discover(package or __name__)


__all__ = [
    "LoaderRegistry",
    "loader_registry",
    "get_loader",
    "discover",
    "Session",
    "Task",
    "Trajectory",
    "Turn",
    "TaskLoader",
    "TraceLoader",
    "BaseHFTaskLoader",
    "OpenHandsLoader",
    "SWEBenchLoader",
    "SWESmithLoader",
    "ABCBenchLoader",
    "SWEEvoLoader",
    "FeatureBenchLoader",
]
