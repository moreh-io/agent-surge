# SPDX-License-Identifier: MIT
"""Shared utilities for multi-agent workload generators.

Provides content generation, context accumulation, and burst timing
helpers used by both BurstPatternGenerator and SteadyPatternGenerator.
"""

from __future__ import annotations

import random
from typing import Any

from agentsurge.generators.synthetic import _make_synthetic_prompt, _pad_assistant_response


def make_role_system_prompt(role_name: str, tokens: int, seed: int) -> str:
    """Generate a deterministic, role-specific system prompt.

    Produces a system prompt of ~tokens length that includes the role
    description and realistic technical content. Identical for the same
    (role_name, tokens, seed) triple.
    """
    rng = random.Random(seed)
    role_desc = (
        f"You are a {role_name} agent. Your role is to {_ROLE_BEHAVIORS.get(role_name, 'assist with the task')}. "
        f"Follow instructions precisely and provide detailed, actionable output.\n\n"
    )
    body_tokens = max(10, tokens - len(role_desc) // 4)
    body = _make_synthetic_prompt(body_tokens, seed=rng.randint(0, 2**31))
    return role_desc + body


_ROLE_BEHAVIORS: dict[str, str] = {
    "planner": "decompose complex tasks into subtasks and coordinate execution",
    "coder": "write, debug, and refactor code based on requirements",
    "reviewer": "review code for correctness, style, and potential issues",
    "tester": "write and run tests to verify functionality",
    "researcher": "gather information and analyze relevant documentation",
    "debater": "argue a position with evidence and respond to counterarguments",
    "moderator": "summarize arguments, identify consensus, and guide discussion",
    "manager": "delegate tasks to subordinates and synthesize their results",
    "worker": "execute assigned subtasks and report results",
    "analyst": "analyze data and provide quantitative assessments",
}


def build_session_turns(
    system_prompt: str,
    n_turns: int,
    tokens_per_turn: int,
    seed: int,
    assistant_tokens: int = 200,
    trace_pool: Any = None,
) -> list[list[dict]]:
    """Build accumulated turn_messages for a multi-turn session.

    Returns a list of n_turns message snapshots, each ending with a
    user message. Each snapshot includes all prior messages.
    """
    rng = random.Random(seed)
    turn_messages: list[list[dict]] = []
    current_msgs: list[dict] = [{"role": "system", "content": system_prompt}]

    for t in range(n_turns):
        user_seed = seed * 1_000_000 + t * 1000
        if trace_pool is not None:
            user_content = trace_pool.sample(rng, target_tokens=tokens_per_turn)
        else:
            user_content = _make_synthetic_prompt(tokens_per_turn, seed=user_seed)

        if t == 0:
            current_msgs = current_msgs + [{"role": "user", "content": user_content}]
        else:
            asst_content = _pad_assistant_response(
                f"Processing turn {t}...", seed=user_seed + 500, min_tokens=assistant_tokens
            )
            current_msgs = current_msgs + [
                {"role": "assistant", "content": asst_content},
                {"role": "user", "content": user_content},
            ]

        turn_messages.append(list(current_msgs))

    return turn_messages


def schedule_burst_offsets(
    n_agents: int,
    burst_type: str = "simultaneous",
    seed: int = 42,
) -> list[float]:
    """Return per-agent timing offsets (ms) within a burst group.

    burst_type:
      "simultaneous": all agents fire at t=0
      "jittered": small random jitter (0-100ms)
      "cascading": staggered by 50ms per agent (tree-level simulation)
    """
    if burst_type == "simultaneous":
        return [0.0] * n_agents
    elif burst_type == "jittered":
        rng = random.Random(seed)
        return [rng.uniform(0, 100) for _ in range(n_agents)]
    elif burst_type == "cascading":
        return [i * 50.0 for i in range(n_agents)]
    return [0.0] * n_agents
