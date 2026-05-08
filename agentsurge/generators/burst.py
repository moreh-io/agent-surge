# SPDX-License-Identifier: MIT
"""Burst-pattern multi-agent workload generators.

Implements Map-Reduce, Debate, and Hierarchical topologies -- the three
multi-agent patterns characterised by bursty, concurrent LLM calls that
stress KV cache prefix sharing and scheduling.

Each method returns a flat list of ReplaySession subclass instances with
lineage metadata that lets the runner reconstruct the DAG.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from agentsurge.generators._agent_utils import (
    build_session_turns,
    make_role_system_prompt,
    schedule_burst_offsets,
)
from agentsurge.generators._technical_doc import _make_technical_doc
from agentsurge.generators.base import GeneratorBase
from agentsurge.generators.synthetic import _make_synthetic_prompt, _pad_assistant_response
from agentsurge.types import ReplaySession


@dataclass
class MapReduceSession(ReplaySession):
    """A session within a map-reduce workload.

    Fields beyond ReplaySession:
      pattern      -- always "map_reduce"
      agent_role   -- "planner" | "worker" | "gather"
      burst_group  -- integer grouping simultaneous workers
      phase        -- "planner" | "scatter" | "gather"
    """

    _METADATA_FIELDS = ("pattern", "agent_role", "burst_group", "phase")

    pattern: str = "map_reduce"
    agent_role: str = ""
    burst_group: int = 0
    phase: str = ""


@dataclass
class DebateSession(ReplaySession):
    """A session within a multi-agent debate workload.

    Fields beyond ReplaySession:
      pattern       -- always "debate"
      agent_role    -- "debater" | "moderator"
      burst_group   -- round index (all debaters in same round share this)
      round_index   -- which debate round this session belongs to
      debater_index -- ordinal within the round (0-based); -1 for moderator
      context_mode  -- "cumulative" | "windowed"
    """

    _METADATA_FIELDS = (
        "pattern",
        "agent_role",
        "burst_group",
        "round_index",
        "debater_index",
        "context_mode",
    )

    pattern: str = "debate"
    agent_role: str = ""
    burst_group: int = 0
    round_index: int = 0
    debater_index: int = -1
    context_mode: str = "cumulative"


@dataclass
class HierarchicalSession(ReplaySession):
    """A session within a hierarchical tree workload.

    Fields beyond ReplaySession:
      pattern      -- always "hierarchical"
      agent_role   -- "root" | "manager" | "leaf"
      burst_group  -- tree level (0 = root, 1 = first children, ...)
      tree_level   -- same as burst_group, explicit alias
      node_path    -- dotted path in the tree, e.g. "0.1.2"
    """

    _METADATA_FIELDS = ("pattern", "agent_role", "burst_group", "tree_level", "node_path")

    pattern: str = "hierarchical"
    agent_role: str = ""
    burst_group: int = 0
    tree_level: int = 0
    node_path: str = ""


class BurstPatternGenerator(GeneratorBase):
    """Generate multi-agent workloads with bursty, concurrent LLM calls.

    Three topologies are available as methods:

      map_reduce()    -- planner -> N workers (burst) -> gather
      debate()        -- N debaters per round (burst), optional moderator
      hierarchical()  -- root -> B children -> B^2 grandchildren (cascading burst)

    All methods return a flat ``list[ReplaySession]`` (actually the
    pattern-specific subclass) with lineage fields populated so the
    caller can reconstruct the DAG.

    Usage::

        gen = BurstPatternGenerator(seed=42)
        sessions = gen.map_reduce(n_tasks=3, n_workers=4)
    """

    name = "burst-pattern"

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def map_reduce(
        self,
        n_tasks: int,
        n_workers: int = 4,
        planner_turns: int = 2,
        worker_turns: int = 3,
        gather_turns: int = 1,
        tokens_per_turn: int = 2048,
        shared_prefix_chars: int = 20_000,
        worker_divergence_tokens: int = 500,
    ) -> list[MapReduceSession]:
        """Generate map-reduce workload sessions.

        Topology per task::

            Planner (builds context over planner_turns)
              -> N Workers (simultaneous; share planner's accumulated context
                 + unique divergence suffix)
              -> Gather (planner context + worker summaries merged)

        Parameters
        ----------
        n_tasks : int
            Number of independent map-reduce tasks to generate.
        n_workers : int
            Fan-out: how many workers run in the scatter phase per task.
        planner_turns : int
            Turns the planner session accumulates before scattering.
        worker_turns : int
            Turns each worker session runs independently.
        gather_turns : int
            Turns in the gather (reduce) session.
        tokens_per_turn : int
            Approximate token count per user message.
        shared_prefix_chars : int
            Character count for the shared technical-doc system prompt
            (~shared_prefix_chars/4 tokens).
        worker_divergence_tokens : int
            Extra unique tokens appended to each worker's first user
            message so siblings diverge after the shared prefix.

        Returns
        -------
        list[MapReduceSession]
            Flat list: [planner, worker_0, ..., worker_N-1, gather] per task.
        """
        sessions: list[MapReduceSession] = []

        shared_doc = _make_technical_doc(shared_prefix_chars, seed=self.seed ^ 0xCAFE)
        system_prompt = (
            "You are a planning agent responsible for decomposing complex tasks, "
            "delegating work to specialist workers, and synthesising their results.\n\n"
            "# Reference Material\n" + shared_doc
        )

        for task_idx in range(n_tasks):
            task_seed = self.seed * 10_000 + task_idx * 100
            lineage_id = f"mr_task_{task_idx}_{hashlib.md5(f'{self.seed}_mr_{task_idx}'.encode()).hexdigest()[:8]}"

            planner_id = f"mr_t{task_idx}_planner"
            planner_turns_msgs = build_session_turns(
                system_prompt=system_prompt,
                n_turns=max(1, planner_turns),
                tokens_per_turn=tokens_per_turn,
                seed=task_seed,
            )

            planner = MapReduceSession(
                session_id=planner_id,
                turn_messages=planner_turns_msgs,
                metadata={"generator": "BurstPatternGenerator", "task_idx": task_idx},
                lineage_id=lineage_id,
                parent_session_id="",
                branch_id=0,
                branch_depth=0,
                pattern="map_reduce",
                agent_role="planner",
                burst_group=0,
                phase="planner",
            )
            sessions.append(planner)

            planner_context = list(planner_turns_msgs[-1])

            burst_offsets = schedule_burst_offsets(n_workers, "jittered", seed=task_seed + 1)

            for w in range(n_workers):
                worker_id = f"mr_t{task_idx}_worker_{w}"
                worker_seed = task_seed + 1000 + w * 100

                divergence = _make_synthetic_prompt(
                    worker_divergence_tokens,
                    seed=worker_seed,
                )
                first_user = f"[Worker {w}] Execute your assigned sub-task.\n\n" + divergence

                worker_msgs_list: list[list[dict]] = []
                current_msgs = list(planner_context)

                planner_asst = _pad_assistant_response(
                    f"Dispatching sub-task to worker {w}.",
                    seed=worker_seed + 50,
                    min_tokens=200,
                )
                current_msgs.append({"role": "assistant", "content": planner_asst})

                current_msgs.append({"role": "user", "content": first_user})
                worker_msgs_list.append(list(current_msgs))

                for t in range(1, worker_turns):
                    asst = _pad_assistant_response(
                        f"Worker {w} processing step {t}.",
                        seed=worker_seed + t * 10,
                        min_tokens=200,
                    )
                    current_msgs.append({"role": "assistant", "content": asst})
                    user = _make_synthetic_prompt(
                        tokens_per_turn,
                        seed=worker_seed + t * 1000,
                    )
                    current_msgs.append({"role": "user", "content": user})
                    worker_msgs_list.append(list(current_msgs))

                worker = MapReduceSession(
                    session_id=worker_id,
                    turn_messages=worker_msgs_list,
                    metadata={
                        "generator": "BurstPatternGenerator",
                        "task_idx": task_idx,
                        "burst_offset_ms": burst_offsets[w],
                    },
                    lineage_id=lineage_id,
                    parent_session_id=planner_id,
                    branch_id=w,
                    branch_depth=1,
                    pattern="map_reduce",
                    agent_role="worker",
                    burst_group=1,
                    phase="scatter",
                )
                sessions.append(worker)

            gather_id = f"mr_t{task_idx}_gather"
            gather_seed = task_seed + 9000

            gather_msgs = list(planner_context)

            planner_to_gather_asst = _pad_assistant_response(
                "All workers complete. Gathering results.",
                seed=gather_seed + 1,
                min_tokens=200,
            )
            gather_msgs.append({"role": "assistant", "content": planner_to_gather_asst})

            summary_parts = ["[Gather] Merge the following worker results:\n"]
            for w in range(n_workers):
                summary_content = _pad_assistant_response(
                    f"Worker {w} result summary.",
                    seed=task_seed + 1000 + w * 100 + 99,
                    min_tokens=100,
                )
                summary_parts.append(f"--- Worker {w} ---\n{summary_content[:600]}\n")
            gather_user = "\n".join(summary_parts)
            gather_msgs.append({"role": "user", "content": gather_user})

            gather_turns_msgs: list[list[dict]] = [list(gather_msgs)]

            for t in range(1, gather_turns):
                asst = _pad_assistant_response(
                    f"Synthesising gathered results, step {t}.",
                    seed=gather_seed + t * 10,
                    min_tokens=200,
                )
                gather_msgs.append({"role": "assistant", "content": asst})
                user = _make_synthetic_prompt(
                    tokens_per_turn,
                    seed=gather_seed + t * 1000,
                )
                gather_msgs.append({"role": "user", "content": user})
                gather_turns_msgs.append(list(gather_msgs))

            gather = MapReduceSession(
                session_id=gather_id,
                turn_messages=gather_turns_msgs,
                metadata={"generator": "BurstPatternGenerator", "task_idx": task_idx},
                lineage_id=lineage_id,
                parent_session_id=planner_id,
                branch_id=0,
                branch_depth=1,
                pattern="map_reduce",
                agent_role="gather",
                burst_group=2,
                phase="gather",
            )
            sessions.append(gather)

        return sessions

    def debate(
        self,
        n_debates: int,
        n_debaters: int = 3,
        n_rounds: int = 4,
        tokens_per_argument: int = 1500,
        shared_premise_chars: int = 15_000,
        has_moderator: bool = True,
        context_mode: str = "cumulative",
    ) -> list[DebateSession]:
        """Generate multi-agent debate workload sessions.

        Topology per debate::

            Round 0: N debaters (simultaneous) + optional moderator
            Round 1: N debaters (shared history from R0) + moderator
            ...
            Round R: N debaters (shared history from R0..R-1) + moderator

        Parameters
        ----------
        n_debates : int
            Number of independent debates to generate.
        n_debaters : int
            Agents arguing per round.
        n_rounds : int
            Number of debate rounds.
        tokens_per_argument : int
            Approximate token count for each debater's argument.
        shared_premise_chars : int
            Character count for the shared debate premise document
            (~shared_premise_chars/4 tokens).
        has_moderator : bool
            If True, a moderator session summarises after each round.
        context_mode : str
            "cumulative" -- full history grows across rounds.
            "windowed"   -- only the last 2 rounds of history are retained.

        Returns
        -------
        list[DebateSession]
            Flat list ordered by (debate, round, debater_index/moderator).
        """
        if context_mode not in ("cumulative", "windowed"):
            raise ValueError(
                f"context_mode must be 'cumulative' or 'windowed', got {context_mode!r}"
            )

        sessions: list[DebateSession] = []

        premise_doc = _make_technical_doc(shared_premise_chars, seed=self.seed ^ 0xDEBA)
        premise_system = (
            "You are participating in a structured multi-agent debate. "
            "Argue your position with evidence and respond to counterarguments.\n\n"
            "# Debate Premise\n" + premise_doc
        )

        for debate_idx in range(n_debates):
            debate_seed = self.seed * 10_000 + debate_idx * 1000
            lineage_id = f"debate_{debate_idx}_{hashlib.md5(f'{self.seed}_debate_{debate_idx}'.encode()).hexdigest()[:8]}"

            debate_history: list[tuple[str, str]] = []

            for round_idx in range(n_rounds):
                round_seed = debate_seed + round_idx * 100
                burst_offsets = schedule_burst_offsets(
                    n_debaters,
                    "jittered",
                    seed=round_seed,
                )

                history_text = self._build_debate_history(
                    debate_history,
                    context_mode,
                    n_rounds_window=2,
                    current_round=round_idx,
                    n_debaters=n_debaters,
                    has_moderator=has_moderator,
                )

                round_arguments: list[str] = []

                for d in range(n_debaters):
                    debater_id = f"debate_{debate_idx}_r{round_idx}_d{d}"
                    debater_seed = round_seed + d * 10 + 1

                    sys_content = premise_system
                    msgs: list[dict] = [{"role": "system", "content": sys_content}]

                    if history_text:
                        msgs.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Here is the debate history so far:\n\n{history_text}\n\n"
                                    "Now present your argument for this round."
                                ),
                            }
                        )
                        msgs.append(
                            {
                                "role": "assistant",
                                "content": "I've reviewed the debate history. Let me formulate my argument.",
                            }
                        )

                    argument_prompt = (
                        f"[Round {round_idx}, Debater {d}] "
                        "Present your argument with evidence. "
                        "Address counterpoints from previous rounds if any.\n\n"
                        + _make_synthetic_prompt(tokens_per_argument, seed=debater_seed)
                    )
                    msgs.append({"role": "user", "content": argument_prompt})

                    argument_content = _pad_assistant_response(
                        f"Debater {d} round {round_idx} argument.",
                        seed=debater_seed + 5,
                        min_tokens=tokens_per_argument // 2,
                    )
                    round_arguments.append(argument_content)

                    debater = DebateSession(
                        session_id=debater_id,
                        turn_messages=[list(msgs)],
                        metadata={
                            "generator": "BurstPatternGenerator",
                            "debate_idx": debate_idx,
                            "burst_offset_ms": burst_offsets[d],
                        },
                        lineage_id=lineage_id,
                        parent_session_id=(
                            f"debate_{debate_idx}_moderator_r{round_idx - 1}"
                            if round_idx > 0 and has_moderator
                            else ""
                        ),
                        branch_id=d,
                        branch_depth=round_idx,
                        pattern="debate",
                        agent_role="debater",
                        burst_group=round_idx,
                        round_index=round_idx,
                        debater_index=d,
                        context_mode=context_mode,
                    )
                    sessions.append(debater)

                for d, arg in enumerate(round_arguments):
                    debate_history.append((f"debater_{d}", arg))

                if has_moderator:
                    mod_id = f"debate_{debate_idx}_moderator_r{round_idx}"
                    mod_seed = round_seed + 90

                    mod_msgs: list[dict] = [{"role": "system", "content": premise_system}]

                    full_history = self._build_debate_history(
                        debate_history,
                        context_mode,
                        n_rounds_window=2,
                        current_round=round_idx + 1,
                        n_debaters=n_debaters,
                        has_moderator=has_moderator,
                    )
                    mod_msgs.append(
                        {
                            "role": "user",
                            "content": (
                                f"[Moderator, Round {round_idx}] "
                                "Summarise the arguments, identify areas of consensus "
                                "and disagreement, and guide the next round.\n\n"
                                f"Debate so far:\n{full_history}"
                            ),
                        }
                    )

                    moderator_summary = _pad_assistant_response(
                        f"Moderator summary for round {round_idx}.",
                        seed=mod_seed + 5,
                        min_tokens=300,
                    )
                    debate_history.append(("moderator", moderator_summary))

                    moderator = DebateSession(
                        session_id=mod_id,
                        turn_messages=[list(mod_msgs)],
                        metadata={
                            "generator": "BurstPatternGenerator",
                            "debate_idx": debate_idx,
                        },
                        lineage_id=lineage_id,
                        parent_session_id="",
                        branch_id=0,
                        branch_depth=round_idx,
                        pattern="debate",
                        agent_role="moderator",
                        burst_group=round_idx,
                        round_index=round_idx,
                        debater_index=-1,
                        context_mode=context_mode,
                    )
                    sessions.append(moderator)

        return sessions

    @staticmethod
    def _build_debate_history(
        history: list[tuple[str, str]],
        context_mode: str,
        n_rounds_window: int,
        current_round: int,
        n_debaters: int,
        has_moderator: bool,
    ) -> str:
        """Format accumulated debate history respecting context_mode.

        For "cumulative", returns all history entries.
        For "windowed", returns only entries from the last n_rounds_window
        rounds (each round = n_debaters entries + optional moderator).
        """
        if not history:
            return ""

        if context_mode == "cumulative":
            entries = history
        else:
            # "windowed": keep only last n_rounds_window rounds
            entries_per_round = n_debaters + (1 if has_moderator else 0)
            window_start_round = max(0, current_round - n_rounds_window)
            start_entry = window_start_round * entries_per_round
            entries = history[start_entry:]

        parts: list[str] = []
        for role_label, content in entries:
            # Truncate very long entries for context efficiency
            truncated = content[:2000] if len(content) > 2000 else content
            parts.append(f"[{role_label}]: {truncated}")
        return "\n\n".join(parts)

    def hierarchical(
        self,
        n_trees: int,
        branching_factor: int = 3,
        depth: int = 2,
        turns_per_agent: int = 2,
        tokens_per_turn: int = 2048,
        root_prefix_chars: int = 20_000,
        parent_prefix_fraction: float = 0.7,
    ) -> list[HierarchicalSession]:
        """Generate hierarchical tree workload sessions.

        Topology per tree::

            Root manager (large system prompt)
              -> B children (inherit fraction of parent context)
                -> B^2 grandchildren (inherit fraction of child context)
                  -> ... up to ``depth`` levels

        Parameters
        ----------
        n_trees : int
            Number of independent trees to generate.
        branching_factor : int
            Children per node (B). Total agents per tree = sum(B^l) for
            l in 0..depth.
        depth : int
            Maximum tree depth (0 = root only, 1 = root + children, ...).
        turns_per_agent : int
            Multi-turn conversation length per agent session.
        tokens_per_turn : int
            Approximate token count per user message.
        root_prefix_chars : int
            Character count for the root manager's system prompt
            (~root_prefix_chars/4 tokens).
        parent_prefix_fraction : float
            Fraction (0.0-1.0) of parent's accumulated context that
            children inherit. Siblings share this prefix, then diverge.

        Returns
        -------
        list[HierarchicalSession]
            Flat list with lineage fields encoding the tree structure.
            Ordered breadth-first (root, then level 1, then level 2, ...).
        """
        parent_prefix_fraction = max(0.0, min(1.0, parent_prefix_fraction))
        sessions: list[HierarchicalSession] = []

        for tree_idx in range(n_trees):
            tree_seed = self.seed * 10_000 + tree_idx * 500
            lineage_id = f"hier_tree_{tree_idx}_{hashlib.md5(f'{self.seed}_hier_{tree_idx}'.encode()).hexdigest()[:8]}"

            root_doc = _make_technical_doc(root_prefix_chars, seed=tree_seed ^ 0xBEEF)
            root_system = (
                "You are the root manager agent. Decompose the project into sub-tasks "
                "and delegate to your subordinate agents.\n\n"
                "# Project Specification\n" + root_doc
            )

            root_id = f"hier_t{tree_idx}_root"
            root_turn_msgs = build_session_turns(
                system_prompt=root_system,
                n_turns=max(1, turns_per_agent),
                tokens_per_turn=tokens_per_turn,
                seed=tree_seed,
            )

            root_session = HierarchicalSession(
                session_id=root_id,
                turn_messages=root_turn_msgs,
                metadata={"generator": "BurstPatternGenerator", "tree_idx": tree_idx},
                lineage_id=lineage_id,
                parent_session_id="",
                branch_id=0,
                branch_depth=0,
                pattern="hierarchical",
                agent_role="root",
                burst_group=0,
                tree_level=0,
                node_path="0",
            )
            sessions.append(root_session)

            self._spawn_children(
                sessions=sessions,
                parent=root_session,
                parent_context=root_turn_msgs[-1],
                lineage_id=lineage_id,
                branching_factor=branching_factor,
                max_depth=depth,
                current_depth=1,
                turns_per_agent=turns_per_agent,
                tokens_per_turn=tokens_per_turn,
                parent_prefix_fraction=parent_prefix_fraction,
                tree_seed=tree_seed,
                tree_idx=tree_idx,
                parent_path="0",
            )

        return sessions

    def _spawn_children(
        self,
        sessions: list[HierarchicalSession],
        parent: HierarchicalSession,
        parent_context: list[dict],
        lineage_id: str,
        branching_factor: int,
        max_depth: int,
        current_depth: int,
        turns_per_agent: int,
        tokens_per_turn: int,
        parent_prefix_fraction: float,
        tree_seed: int,
        tree_idx: int,
        parent_path: str,
    ) -> None:
        """Recursively create child sessions at the current tree level.

        Children inherit parent_prefix_fraction of the parent's accumulated
        context, then diverge with unique content. Appends to ``sessions``
        in place (breadth-first order within each level).
        """
        if current_depth > max_depth:
            return

        burst_offsets = schedule_burst_offsets(
            branching_factor,
            "cascading",
            seed=tree_seed + current_depth * 100,
        )

        # parent_prefix_fraction=0.0 means no inheritance (per docstring).
        # For positive fractions floor at 1 message so children share at
        # least the system prompt — preserves prior behaviour.
        if parent_prefix_fraction <= 0.0:
            n_inherit = 0
        else:
            n_inherit = max(1, int(len(parent_context) * parent_prefix_fraction))
        shared_prefix_msgs = list(parent_context[:n_inherit])

        children: list[HierarchicalSession] = []

        for child_idx in range(branching_factor):
            child_path = f"{parent_path}.{child_idx}"
            child_id = f"hier_t{tree_idx}_L{current_depth}_c{child_path.replace('.', '_')}"
            child_seed = tree_seed + current_depth * 1000 + child_idx * 100

            role = "leaf" if current_depth == max_depth else "manager"
            role_prompt = make_role_system_prompt(
                "worker" if role == "leaf" else "manager",
                tokens=200,
                seed=child_seed,
            )

            child_msgs = list(shared_prefix_msgs)

            delegation_asst = _pad_assistant_response(
                f"Delegating sub-task to child {child_idx} at level {current_depth}.",
                seed=child_seed + 50,
                min_tokens=150,
            )
            child_msgs.append({"role": "assistant", "content": delegation_asst})

            divergence = _make_synthetic_prompt(
                tokens_per_turn,
                seed=child_seed + 1,
            )
            child_user = (
                f"[Level {current_depth}, Node {child_path}] {role_prompt}\n\n"
                f"Execute the following sub-task:\n{divergence}"
            )
            child_msgs.append({"role": "user", "content": child_user})

            child_turn_msgs: list[list[dict]] = [list(child_msgs)]

            for t in range(1, turns_per_agent):
                asst = _pad_assistant_response(
                    f"Node {child_path} processing step {t}.",
                    seed=child_seed + t * 10,
                    min_tokens=200,
                )
                child_msgs.append({"role": "assistant", "content": asst})
                user = _make_synthetic_prompt(
                    tokens_per_turn,
                    seed=child_seed + t * 1000,
                )
                child_msgs.append({"role": "user", "content": user})
                child_turn_msgs.append(list(child_msgs))

            child_session = HierarchicalSession(
                session_id=child_id,
                turn_messages=child_turn_msgs,
                metadata={
                    "generator": "BurstPatternGenerator",
                    "tree_idx": tree_idx,
                    "burst_offset_ms": burst_offsets[child_idx],
                },
                lineage_id=lineage_id,
                parent_session_id=parent.session_id,
                branch_id=child_idx,
                branch_depth=current_depth,
                pattern="hierarchical",
                agent_role=role,
                burst_group=current_depth,
                tree_level=current_depth,
                node_path=child_path,
            )
            sessions.append(child_session)
            children.append(child_session)

        for child_session in children:
            self._spawn_children(
                sessions=sessions,
                parent=child_session,
                parent_context=child_session.turn_messages[-1],
                lineage_id=lineage_id,
                branching_factor=branching_factor,
                max_depth=max_depth,
                current_depth=current_depth + 1,
                turns_per_agent=turns_per_agent,
                tokens_per_turn=tokens_per_turn,
                parent_prefix_fraction=parent_prefix_fraction,
                tree_seed=tree_seed,
                tree_idx=tree_idx,
                parent_path=child_session.node_path,
            )
