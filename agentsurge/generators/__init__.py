# SPDX-License-Identifier: MIT
"""agentsurge workload generators with auto-discovery registry.

Concrete generators live in submodules of this package and auto-register
when imported.  This package provides the registry and base class, plus
convenience re-exports::

    from agentsurge.generators import generator_registry
    gen_cls = generator_registry.get("synthetic")

    # Or import concrete generators directly:
    from agentsurge.generators import SyntheticGenerator
"""

from agentsurge.generators.base import GeneratorBase
from agentsurge.generators.burst import BurstPatternGenerator
from agentsurge.generators.mixed import MixedWorkloadGenerator
from agentsurge.generators.registry import GeneratorRegistry, generator_registry
from agentsurge.generators.steady import SteadyPatternGenerator
from agentsurge.generators.synthetic import SingleTurnGenerator, SyntheticGenerator
from agentsurge.generators.task_converter import TaskToTrajectoryConverter
from agentsurge.generators.trace_pool import TracePool
from agentsurge.generators.trace_replay import TraceReplayGenerator

__all__ = [
    "GeneratorRegistry",
    "generator_registry",
    "GeneratorBase",
    "TraceReplayGenerator",
    "SyntheticGenerator",
    "TaskToTrajectoryConverter",
    "SingleTurnGenerator",
    "MixedWorkloadGenerator",
    "TracePool",
    "BurstPatternGenerator",
    "SteadyPatternGenerator",
]
