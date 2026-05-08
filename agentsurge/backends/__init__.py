# SPDX-License-Identifier: MIT
"""Backend adapters for agentsurge.

This package defines the abstract interface that every inference backend must
implement (:class:`BackendBase`) and the associated configuration dataclass
(:class:`BackendConfig`).

Concrete backends (``vllm``, ``mock``, …) live in sub-modules and are
discovered via the backend registry.

The **vLLM** backend is the default and can be imported directly::

    from agentsurge.backends import VllmBackend, VllmConfig

Registry usage::

    from agentsurge.backends import backend_registry, get_backend

    # Get the default (vllm) backend class:
    cls = get_backend()

    # Get a specific backend by name:
    cls = get_backend("mock")

    # List all registered backends:
    backend_registry.list_backends()  # -> ["mock", "openai", "vllm"]
"""

from agentsurge.backends.base import (
    BackendBase,
    BackendConfig,
    DefaultRequestAdapter,
    MetricsAdapter,
    MetricsAdapterProtocol,
    PrometheusMetricsAdapter,
    RequestAdapter,
    RequestAdapterProtocol,
)
from agentsurge.backends.mock import MockBackend, MockConfig
from agentsurge.backends.openai import OpenAiBackend
from agentsurge.backends.registry import BackendRegistry, backend_registry
from agentsurge.backends.vllm import VllmBackend, VllmConfig


def get_backend(name: str | None = None) -> type[BackendBase]:
    """Return the backend class registered under *name*.

    If *name* is ``None``, the default backend (``"vllm"``) is returned.
    This is a thin wrapper around ``backend_registry.get(name)``.

    Parameters
    ----------
    name : str or None
        Backend name (e.g. ``"vllm"``, ``"mock"``).  ``None`` for default.

    Returns
    -------
    type[BackendBase]
        The backend class (not an instance).

    Raises
    ------
    KeyError
        If no backend with that name is registered.

    Examples
    --------
    >>> cls = get_backend("mock")
    >>> cls.name
    'mock'
    """
    return backend_registry.get(name)


__all__ = [
    "BackendRegistry",
    "backend_registry",
    "get_backend",
    "BackendBase",
    "BackendConfig",
    "DefaultRequestAdapter",
    "MetricsAdapter",
    "MetricsAdapterProtocol",
    "PrometheusMetricsAdapter",
    "RequestAdapter",
    "RequestAdapterProtocol",
    "MockBackend",
    "MockConfig",
    "OpenAiBackend",
    "VllmBackend",
    "VllmConfig",
]
