# SPDX-License-Identifier: MIT
"""Loader and trace data types.

Defines the core data structures for conversation sessions, turns,
trajectories, and replay sessions used throughout agentsurge.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar

WORKLOAD_SCHEMA_VERSION: str = "1.0"
"""Semantic version embedded in every workload JSON produced by agentsurge.

Bump the *minor* component for backward-compatible additions (new optional
fields).  Bump the *major* component when old readers can no longer consume
the file without migration.
"""


@dataclass
class Turn:
    """A single conversation turn (one request-response pair)."""

    role: str
    content: str
    tool_calls: list[dict] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Session:
    """An ordered sequence of turns forming a conversation session."""

    session_id: str
    turns: list[Turn] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return len(self.turns)


@dataclass
class Trajectory:
    """A complete agent trajectory (may span multiple sessions)."""

    trajectory_id: str
    sessions: list[Session] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class Task:
    """A benchmark task (e.g. a SWE-bench instance)."""

    task_id: str
    prompt: str
    reference: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ReplaySession:
    """A session ready for replay: accumulated messages per turn.

    Subclasses that add pattern-specific fields should list them in
    ``_METADATA_FIELDS`` (a class variable).  :meth:`to_dict` will
    automatically merge those fields into the ``metadata`` dict so
    subclasses don't need to override ``to_dict()``.
    """

    _METADATA_FIELDS: ClassVar[tuple[str, ...]] = ()

    session_id: str
    turn_messages: list[list[dict]]  # turn_messages[i] = messages[0:turn_i_end]
    metadata: dict = field(default_factory=dict)
    fan_out: int = 1  # number of parallel LLM calls per turn (>1 simulates agent parallelism)
    lineage_id: str = ""  # root session's ID (all branches share this)
    parent_session_id: str = ""  # immediate parent (empty for root)
    branch_id: int = 0  # branch index within a fork (0 = main)
    branch_depth: int = 0  # nesting level (0 = root)
    pending_user_messages: list[str] = field(default_factory=list)
    # ^ user-role messages to feed on successive AI stops (tool-loop runner only)

    def __post_init__(self) -> None:
        if not isinstance(self.pending_user_messages, list):
            raise ValueError(
                "pending_user_messages must be a list of strings, "
                f"got {type(self.pending_user_messages).__name__}"
            )
        for i, msg in enumerate(self.pending_user_messages):
            if not isinstance(msg, str):
                raise ValueError(
                    "pending_user_messages must be a list of strings, "
                    f"item {i} is {type(msg).__name__}"
                )

    @property
    def n_turns(self) -> int:
        return len(self.turn_messages)

    @property
    def estimated_tokens(self) -> int:
        """Estimate total tokens in the final (longest) turn snapshot.

        Uses ~4 chars per token heuristic. This gives a rough KV cache
        footprint estimate without requiring a tokenizer.
        """
        if not self.turn_messages:
            return 0
        last_turn = self.turn_messages[-1]
        total_chars = sum(len(m.get("content", "")) for m in last_turn)
        return total_chars // 4

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "session_id": self.session_id,
            "turn_messages": self.turn_messages,
            "metadata": dict(self.metadata),
            "fan_out": self.fan_out,
        }
        if self.lineage_id:
            d["lineage_id"] = self.lineage_id
        if self.parent_session_id:
            d["parent_session_id"] = self.parent_session_id
        if self.branch_id != 0:
            d["branch_id"] = self.branch_id
        if self.branch_depth != 0:
            d["branch_depth"] = self.branch_depth
        if self.pending_user_messages:
            d["pending_user_messages"] = list(self.pending_user_messages)
        if self._METADATA_FIELDS:
            meta = d["metadata"]
            assert isinstance(meta, dict)
            meta.update({f: getattr(self, f) for f in self._METADATA_FIELDS})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ReplaySession":
        meta = dict(d.get("metadata", {}))
        extra_kwargs: dict[str, Any] = {}
        for field_name in getattr(cls, "_METADATA_FIELDS", ()):
            if field_name in meta:
                extra_kwargs[field_name] = meta.pop(field_name)
        pending = d.get("pending_user_messages", [])
        return cls(
            session_id=d["session_id"],
            turn_messages=d["turn_messages"],
            metadata=meta,
            fan_out=d.get("fan_out", 1),
            lineage_id=d.get("lineage_id", ""),
            parent_session_id=d.get("parent_session_id", ""),
            branch_id=d.get("branch_id", 0),
            branch_depth=d.get("branch_depth", 0),
            pending_user_messages=pending,
            **extra_kwargs,
        )

    def inject_response(self, turn_idx: int, response_text: str) -> None:
        """Replace assistant content in ALL subsequent turns with actual LLM response.

        After turn_idx completes, the assistant message corresponding to this
        turn must be updated in every subsequent turn snapshot (turn_idx+1,
        turn_idx+2, ...) because accumulated context is shared via shallow
        copy.  We find the target message position once (in turn_idx+1) and
        then patch every later snapshot at the same position.
        """
        if turn_idx + 1 >= len(self.turn_messages):
            return

        # Find the position of the assistant message to replace in turn_idx+1
        next_msgs = self.turn_messages[turn_idx + 1]
        target_pos = None
        for i in range(len(next_msgs) - 1, -1, -1):
            if next_msgs[i]["role"] == "assistant":
                target_pos = i
                break
        if target_pos is None:
            return

        # Patch this position in ALL subsequent turn snapshots.
        # Each turn gets its own dict copy to prevent mutation aliasing.
        base = dict(next_msgs[target_pos])
        base["content"] = response_text
        for t in range(turn_idx + 1, len(self.turn_messages)):
            msgs = self.turn_messages[t]
            if target_pos < len(msgs) and msgs[target_pos].get("role") == "assistant":
                msgs[target_pos] = dict(base)
