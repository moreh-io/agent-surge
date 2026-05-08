# SPDX-License-Identifier: MIT
"""agentsurge - Benchmark your LLM server with realistic coding agent workloads."""

__version__ = "0.1.0"

from agentsurge._api import arun, run
from agentsurge._hooks import register_adapter, register_loader
from agentsurge.backends import get_backend
from agentsurge.backends.base import BackendBase, BackendConfig
from agentsurge.callbacks import OnSessionCallback, OnTurnCallback
from agentsurge.exceptions import (
    AgentSurgeConnectionError,
    AgentSurgeError,
    BackendError,
    ConfigError,
    WorkloadValidationError,
)
from agentsurge.generators.base import GeneratorBase
from agentsurge.io import validate_workload
from agentsurge.loaders import get_loader
from agentsurge.loaders.base import TaskLoader, TraceLoader
from agentsurge.preset import PRESET_NAMES, resolve_preset
from agentsurge.types import (
    WORKLOAD_SCHEMA_VERSION,
    BenchmarkConfig,
    ConfigProfile,
    LmcacheV1Metrics,
    MetricsSnapshot,
    ReplaySession,
    RunResult,
    Session,
    SessionResult,
    SloComparisonResult,
    Task,
    Trajectory,
    Turn,
    TurnResult,
)

__all__ = [
    "PRESET_NAMES",
    "resolve_preset",
    "BackendError",
    "ConfigError",
    "AgentSurgeConnectionError",
    "AgentSurgeError",
    "WorkloadValidationError",
    "WORKLOAD_SCHEMA_VERSION",
    "BenchmarkConfig",
    "ConfigProfile",
    "LmcacheV1Metrics",
    "MetricsSnapshot",
    "ReplaySession",
    "RunResult",
    "Session",
    "SessionResult",
    "SloComparisonResult",
    "Task",
    "Trajectory",
    "Turn",
    "TurnResult",
    "validate_workload",
    "BackendBase",
    "BackendConfig",
    "get_backend",
    "TaskLoader",
    "TraceLoader",
    "get_loader",
    "GeneratorBase",
    "OnSessionCallback",
    "OnTurnCallback",
    "run",
    "arun",
    "register_adapter",
    "register_loader",
    "__version__",
]
