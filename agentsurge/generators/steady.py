# SPDX-License-Identifier: MIT
"""Steady-state multi-agent workload generators.

Implements patterns with no or low burst intensity:
  - pipeline: sequential stage-by-stage processing (1 agent active at a time)
  - decentralized: round-based peer messaging with topology-controlled visibility
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from agentsurge.generators._agent_utils import (
    make_role_system_prompt,
    schedule_burst_offsets,
)
from agentsurge.generators.base import GeneratorBase
from agentsurge.generators.synthetic import _make_synthetic_prompt, _pad_assistant_response
from agentsurge.types import ReplaySession

_DEFAULT_PIPELINE_ROLES = ["planner", "coder", "reviewer", "tester"]


@dataclass
class PipelineSession(ReplaySession):
    """A single stage within a pipeline execution."""

    _METADATA_FIELDS = (
        "pattern",
        "agent_role",
        "stage_index",
        "n_stages",
        "context_mode",
        "burst_group",
    )

    pattern: str = "pipeline"
    agent_role: str = ""
    stage_index: int = 0
    n_stages: int = 0
    context_mode: str = ""
    burst_group: int = 0  # unique per stage (no burst)


@dataclass
class DecentralizedSession(ReplaySession):
    """One agent's full conversation across all rounds in a decentralized group."""

    _METADATA_FIELDS = (
        "pattern",
        "agent_role",
        "agent_index",
        "n_agents",
        "n_rounds",
        "message_topology",
        "burst_group",
    )

    pattern: str = "decentralized"
    agent_role: str = ""
    agent_index: int = 0
    n_agents: int = 0
    n_rounds: int = 0
    message_topology: str = ""
    burst_group: int = 0


class SteadyPatternGenerator(GeneratorBase):
    """Generate steady-state multi-agent workloads (low/no burst).

    Methods:
      pipeline()       -- sequential stage handoff (carryover / artifact / reset)
      decentralized()  -- round-based peer messaging with topology control
    """

    name = "steady-pattern"

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def pipeline(
        self,
        n_pipelines: int,
        stages: int | list[str] = 4,
        tokens_per_stage: int = 2048,
        system_prompt_tokens_per_role: int = 1000,
        context_mode: str = "carryover",
        inter_stage_delay_ms: float = 500,
        n_concurrent_pipelines: int = 1,
    ) -> list[PipelineSession]:
        """Generate sequential pipeline workload sessions.

        Parameters
        ----------
        n_pipelines:
            Number of independent pipeline instances to generate.
        stages:
            If int, uses default roles ``["planner", "coder", "reviewer", "tester"][:stages]``.
            If list[str], each string is a role name.
        tokens_per_stage:
            Approximate token count for the user message at each stage.
        system_prompt_tokens_per_role:
            Token budget for each role's system prompt.
        context_mode:
            ``"carryover"`` -- full accumulated history carries to next stage.
            ``"artifact"``  -- only last assistant message passed forward.
            ``"reset"``     -- only final solution summary, context resets.
        inter_stage_delay_ms:
            Delay between stages (used for offset scheduling).
        n_concurrent_pipelines:
            When > 1, same-role stages across pipelines share system prompts.

        Returns
        -------
        list[PipelineSession]
            Ordered by (pipeline_index, stage_index).
        """
        if context_mode not in ("carryover", "artifact", "reset"):
            raise ValueError(
                f"Unknown context_mode {context_mode!r}. "
                f"Choose from: 'carryover', 'artifact', 'reset'"
            )

        role_names = _DEFAULT_PIPELINE_ROLES[:stages] if isinstance(stages, int) else list(stages)
        n_stages = len(role_names)

        rng = random.Random(self.seed)

        # Pre-generate role system prompts.  When n_concurrent_pipelines > 1
        # the SAME seed is used for each role so that identical-role stages
        # across pipelines share the same system prompt (prefix-sharing).
        role_prompts: dict[str, str] = {}
        for role in role_names:
            role_seed = rng.randint(0, 2**31)
            role_prompts[role] = make_role_system_prompt(
                role,
                system_prompt_tokens_per_role,
                seed=role_seed,
            )

        sessions: list[PipelineSession] = []

        for p_idx in range(n_pipelines):
            pipeline_seed = self.seed * 100_000 + p_idx * 1000

            lineage_id = f"pipeline_{p_idx}"
            prev_session_id = ""
            carry_messages: list[dict] = []

            for s_idx, role in enumerate(role_names):
                session_id = f"pipeline_{p_idx}_stage_{s_idx}_{role}"
                stage_seed = pipeline_seed + s_idx * 100

                sys_prompt = role_prompts[role]

                stage_messages = self._build_pipeline_stage_messages(
                    sys_prompt=sys_prompt,
                    carry_messages=carry_messages,
                    context_mode=context_mode,
                    tokens_per_stage=tokens_per_stage,
                    stage_seed=stage_seed,
                    stage_index=s_idx,
                    role=role,
                )

                turn_messages = [list(stage_messages)]

                asst_response = _pad_assistant_response(
                    f"Stage {s_idx} ({role}) output: task completed.",
                    seed=stage_seed + 50,
                    min_tokens=max(200, tokens_per_stage // 4),
                )

                carry_messages = self._update_carry(
                    carry_messages=carry_messages,
                    stage_messages=stage_messages,
                    asst_response=asst_response,
                    context_mode=context_mode,
                )

                parent_sid = prev_session_id if s_idx > 0 else ""

                sess = PipelineSession(
                    session_id=session_id,
                    turn_messages=turn_messages,
                    metadata={
                        "generator": "SteadyPatternGenerator",
                        "pipeline_index": p_idx,
                        "inter_stage_delay_ms": inter_stage_delay_ms,
                    },
                    fan_out=1,
                    lineage_id=lineage_id,
                    parent_session_id=parent_sid,
                    branch_id=0,
                    branch_depth=s_idx,
                    pattern="pipeline",
                    agent_role=role,
                    stage_index=s_idx,
                    n_stages=n_stages,
                    context_mode=context_mode,
                    burst_group=s_idx,
                )
                sessions.append(sess)
                prev_session_id = session_id

        return sessions

    def _build_pipeline_stage_messages(
        self,
        sys_prompt: str,
        carry_messages: list[dict],
        context_mode: str,
        tokens_per_stage: int,
        stage_seed: int,
        stage_index: int,
        role: str,
    ) -> list[dict]:
        """Assemble the message list for one pipeline stage."""
        messages: list[dict] = [{"role": "system", "content": sys_prompt}]

        if stage_index == 0:
            user_content = _make_synthetic_prompt(tokens_per_stage, seed=stage_seed)
            messages.append({"role": "user", "content": user_content})
        else:
            if context_mode == "carryover":
                messages.extend(carry_messages)
            elif (context_mode == "artifact" or context_mode == "reset") and carry_messages:
                messages.append(carry_messages[-1])

            stage_instruction = (
                f"[Stage {stage_index}: {role}] "
                f"Review the previous output and perform your role.\n\n"
            )
            user_body = _make_synthetic_prompt(tokens_per_stage, seed=stage_seed + 10)
            messages.append({"role": "user", "content": stage_instruction + user_body})

        return messages

    @staticmethod
    def _update_carry(
        carry_messages: list[dict],
        stage_messages: list[dict],
        asst_response: str,
        context_mode: str,
    ) -> list[dict]:
        """Update the carry-forward context after a stage completes."""
        if context_mode == "carryover":
            non_system = [m for m in stage_messages if m["role"] != "system"]
            new_carry = list(carry_messages)
            new_carry.extend(non_system)
            new_carry.append({"role": "assistant", "content": asst_response})
            return new_carry
        elif context_mode == "artifact":
            return [{"role": "assistant", "content": asst_response}]
        elif context_mode == "reset":
            summary = asst_response[:400] if len(asst_response) > 400 else asst_response
            return [{"role": "assistant", "content": f"[Summary] {summary}"}]
        return carry_messages

    def decentralized(
        self,
        n_groups: int,
        n_agents: int = 4,
        n_rounds: int = 5,
        tokens_per_message: int = 1000,
        shared_context_chars: int = 10000,
        message_topology: str = "ring",
        message_fraction: float = 0.5,
    ) -> list[DecentralizedSession]:
        """Generate decentralized peer-messaging workload sessions.

        Parameters
        ----------
        n_groups:
            Number of independent agent groups.
        n_agents:
            Agents per group.
        n_rounds:
            Communication rounds (becomes turns per session).
        tokens_per_message:
            Approximate token count per agent message.
        shared_context_chars:
            Characters of shared context all agents see (simulates a shared
            document / codebase).
        message_topology:
            ``"ring"``            -- agent *i* sees only agent *(i-1) mod N*.
            ``"fully_connected"`` -- all agents see all messages.
            ``"random_sparse"``   -- each agent sees ``message_fraction`` of others.
        message_fraction:
            Fraction of other agents' messages visible (only for ``"random_sparse"``).

        Returns
        -------
        list[DecentralizedSession]
            One session per agent (rounds = turns within the session).
        """
        if message_topology not in ("ring", "fully_connected", "random_sparse"):
            raise ValueError(
                f"Unknown message_topology {message_topology!r}. "
                f"Choose from: 'ring', 'fully_connected', 'random_sparse'"
            )

        all_sessions: list[DecentralizedSession] = []

        for g_idx in range(n_groups):
            group_seed = self.seed * 100_000 + g_idx * 10_000

            lineage_id = f"decentral_{g_idx}"

            shared_context = _make_synthetic_prompt(
                shared_context_chars // 4,  # chars -> approx tokens
                seed=group_seed,
            )

            agent_roles = [f"analyst_{a}" for a in range(n_agents)]

            agent_messages: list[list[str]] = []
            for r in range(n_rounds):
                round_msgs: list[str] = []
                for a in range(n_agents):
                    msg_seed = group_seed + r * 1000 + a * 10
                    content = _make_synthetic_prompt(tokens_per_message, seed=msg_seed)
                    round_msgs.append(content)
                agent_messages.append(round_msgs)

            for a_idx in range(n_agents):
                session_id = f"decentral_{g_idx}_agent_{a_idx}"
                role = agent_roles[a_idx]

                role_prefix = (
                    f"You are a {role} agent. Your role is to assist with the task. "
                    f"Follow instructions precisely and provide detailed, actionable output.\n\n"
                )
                sys_content = f"{role_prefix}# Shared project context:\n{shared_context}"

                turn_messages_list: list[list[dict]] = []
                accumulated: list[dict] = [{"role": "system", "content": sys_content}]

                for r in range(n_rounds):
                    if r == 0:
                        visible_msgs: list[str] = []
                    else:
                        visible_msgs = self._get_visible_messages(
                            agent_index=a_idx,
                            n_agents=n_agents,
                            round_messages=agent_messages[r - 1],
                            topology=message_topology,
                            message_fraction=message_fraction,
                            rng=random.Random(group_seed + r * 100 + a_idx),
                        )

                    round_user = self._build_decentralized_round_message(
                        agent_idx=a_idx,
                        round_idx=r,
                        own_prompt=agent_messages[r][a_idx],
                        visible_messages=visible_msgs,
                    )

                    if r > 0:
                        asst_seed = group_seed + (r - 1) * 1000 + a_idx * 10 + 5
                        asst_content = _pad_assistant_response(
                            f"Agent {a_idx} round {r - 1} analysis complete.",
                            seed=asst_seed,
                            min_tokens=200,
                        )
                        accumulated.append({"role": "assistant", "content": asst_content})

                    accumulated.append({"role": "user", "content": round_user})
                    turn_messages_list.append(list(accumulated))

                jitter_offsets = schedule_burst_offsets(
                    n_agents,
                    burst_type="jittered",
                    seed=group_seed,
                )

                sess = DecentralizedSession(
                    session_id=session_id,
                    turn_messages=turn_messages_list,
                    metadata={
                        "generator": "SteadyPatternGenerator",
                        "group_index": g_idx,
                        "jitter_offset_ms": jitter_offsets[a_idx],
                    },
                    fan_out=1,
                    lineage_id=lineage_id,
                    parent_session_id="",
                    branch_id=a_idx,
                    branch_depth=0,
                    pattern="decentralized",
                    agent_role=role,
                    agent_index=a_idx,
                    n_agents=n_agents,
                    n_rounds=n_rounds,
                    message_topology=message_topology,
                    burst_group=0,
                )
                all_sessions.append(sess)

        return all_sessions

    @staticmethod
    def _get_visible_messages(
        agent_index: int,
        n_agents: int,
        round_messages: list[str],
        topology: str,
        message_fraction: float,
        rng: random.Random,
    ) -> list[str]:
        """Return messages visible to ``agent_index`` based on topology."""
        if topology == "ring":
            sender = (agent_index - 1) % n_agents
            return [round_messages[sender]]

        elif topology == "fully_connected":
            return [round_messages[j] for j in range(n_agents) if j != agent_index]

        elif topology == "random_sparse":
            others = [j for j in range(n_agents) if j != agent_index]
            n_visible = max(1, int(len(others) * message_fraction))
            chosen = rng.sample(others, min(n_visible, len(others)))
            chosen.sort()
            return [round_messages[j] for j in chosen]

        return []

    @staticmethod
    def _build_decentralized_round_message(
        agent_idx: int,
        round_idx: int,
        own_prompt: str,
        visible_messages: list[str],
    ) -> str:
        """Assemble the user message for one agent in one round."""
        parts: list[str] = []
        parts.append(f"[Round {round_idx}, Agent {agent_idx}]")

        if visible_messages:
            parts.append("\n--- Messages from other agents ---")
            for i, msg in enumerate(visible_messages):
                preview = msg[:2000] if len(msg) > 2000 else msg
                parts.append(f"\n[Peer {i}]:\n{preview}")
            parts.append("\n--- End of peer messages ---\n")

        parts.append(own_prompt)
        return "\n".join(parts)
