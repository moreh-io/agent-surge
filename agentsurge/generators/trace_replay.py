# SPDX-License-Identifier: MIT
"""Trace replay generator -- convert real agent trajectories to replay sessions.

Input tokens are deterministic (recorded context), so KV pressure is reproducible.
With use_model_reply_in_next_turn=True, the runner injects actual LLM output for accurate
intra-session prefix cache hits at the cost of cross-model reproducibility.
"""

import copy
from collections.abc import Iterator

from agentsurge.generators.base import GeneratorBase
from agentsurge.types import ReplaySession, Session, Trajectory, Turn

TOOL_OUTPUT_PREFIX = "[Tool output"


def flatten_tool_calls(turns: list[Turn]) -> list[dict]:
    """Convert turns with tool_calls into plain user/assistant messages.

    Normalize tool call format:
    - tool role -> user role with "[Tool output: name]" prefix
    - assistant tool_calls -> plain text "Call: name(args)"
    """
    flat: list[dict] = []
    for turn in turns:
        if turn.role == "tool":
            prefix = f"[Tool output: {turn.name}] " if turn.name else "[Tool output] "
            flat.append({"role": "user", "content": prefix + turn.content})
        elif turn.role == "assistant" and turn.tool_calls:
            parts = []
            if turn.content:
                parts.append(turn.content)
            for tc in turn.tool_calls:
                func = tc.get("function", tc)
                name = func.get("name", "unknown")
                args = func.get("arguments", "")
                parts.append(f"Call: {name}({args})")
            flat.append({"role": "assistant", "content": "\n".join(parts)})
        elif turn.role == "system":
            flat.append({"role": "system", "content": turn.content})
        else:
            flat.append({"role": turn.role, "content": turn.content})
    return flat


class TraceReplayGenerator(GeneratorBase):
    """Generate replay sessions from real agent trajectories."""

    name = "trace-replay"

    def __init__(
        self, system_prompt: str = "You are a helpful assistant.", flatten_tools: bool = True
    ) -> None:
        self.system_prompt = system_prompt
        self.flatten_tools = flatten_tools

    def from_trajectories(
        self, trajectories: Iterator[Trajectory], limit: int | None = None
    ) -> Iterator[ReplaySession]:
        """Yield ReplaySessions from trajectories lazily to avoid OOM on large datasets."""
        count = 0
        for traj in trajectories:
            for sess in traj.sessions:
                replay = self._build_replay(sess, trajectory=traj)
                if replay.n_turns > 0:
                    yield replay
                    count += 1
                    if limit is not None and count >= limit:
                        return

    def _preserve_tool_calls(self, turns: list[Turn]) -> list[dict]:
        """Preserve tool_calls structure in messages.

        Keeps ``tool_calls`` on assistant messages and ``tool_call_id`` /
        ``name`` on tool messages so the conversation is valid for the
        OpenAI chat-completions API.
        """
        result: list[dict] = []
        for turn in turns:
            msg: dict = {"role": turn.role, "content": turn.content}
            if turn.role == "assistant" and turn.tool_calls:
                msg["tool_calls"] = turn.tool_calls
            if turn.role == "tool":
                if turn.tool_call_id:
                    msg["tool_call_id"] = turn.tool_call_id
                if turn.name:
                    msg["name"] = turn.name
            result.append(msg)
        return result

    def _build_replay(
        self, session: Session, trajectory: "Trajectory | None" = None
    ) -> ReplaySession:
        """Build a ReplaySession with cumulative message prefixes.

        When ``flatten_tools`` is True the turn boundary is every ``user``
        message (original behaviour).

        When ``flatten_tools`` is False we treat each point where the model
        should generate a response as a turn boundary:
          - After the initial user message (turn 0).
          - Before every assistant message that follows one or more tool
            responses.  This captures the full agent loop where each LLM
            invocation sees the accumulated tool outputs.

        The resulting ``turn_messages[i]`` is the complete conversation
        prefix the model should receive for turn *i*.

        If the source loader advertises ``supports_multi_turn=True`` (via
        session or trajectory metadata), every user-role turn beyond the
        first is collected into ``pending_user_messages`` for the
        dynamic-tool-loop runner to feed after each AI stop.
        """
        if self.flatten_tools:
            flat = flatten_tool_calls(session.turns)
        else:
            flat = self._preserve_tool_calls(session.turns)
        if not flat:
            return ReplaySession(session_id=session.session_id, turn_messages=[])

        messages: list[dict] = []
        if flat[0].get("role") != "system":
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(flat)

        # First compute the turn-boundary end-indices into ``messages``, then
        # build each turn's prefix with fresh dict copies so entries in
        # ``turn_messages`` don't share references. Sharing would let
        # ``inject_response`` (which patches by position) corrupt earlier
        # turn snapshots.
        boundaries: list[int] = []
        if self.flatten_tools:
            for i, msg in enumerate(messages):
                if msg["role"] == "user":
                    boundaries.append(i + 1)
        else:
            prev_role: str | None = None
            for i, msg in enumerate(messages):
                role = msg["role"]
                if role == "user" and prev_role != "tool":
                    boundaries.append(i + 1)
                elif role == "assistant" and prev_role == "tool":
                    boundaries.append(i)
                prev_role = role
            if (
                messages
                and prev_role in ("tool", "user")
                and (not boundaries or boundaries[-1] != len(messages))
            ):
                boundaries.append(len(messages))

        turn_messages: list[list[dict]] = [
            [copy.deepcopy(m) for m in messages[:end]] for end in boundaries
        ]

        metadata = dict(session.metadata) if session.metadata else {}

        # Pre-compute boundary→first-assistant-content in one forward pass so
        # lookup per turn is O(1) rather than O(remaining messages), avoiding
        # O(N²) total work for N turns.
        #
        # For each boundary `end`, the answer is the content of the first
        # assistant message at index >= end.  Boundaries are sorted ascending;
        # keep a pointer into them and advance it as we scan messages.
        next_assistant: dict[int, str] = {}
        sorted_boundaries = sorted(set(boundaries))
        b_ptr = 0  # index into sorted_boundaries of next unfilled boundary
        for j, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content") or ""
            # This assistant message (at index j) answers all boundaries
            # end <= j that haven't been filled yet.
            while b_ptr < len(sorted_boundaries) and sorted_boundaries[b_ptr] <= j:
                next_assistant.setdefault(sorted_boundaries[b_ptr], content)
                b_ptr += 1

        turn_output_contents: list[str] = []
        for end in boundaries:
            turn_output_contents.append(next_assistant.get(end, ""))
        metadata["turn_output_contents"] = turn_output_contents

        supports_multi_turn = bool(
            session.metadata.get("supports_multi_turn")
            or (trajectory and trajectory.metadata.get("supports_multi_turn"))
        )
        pending_user_messages: list[str] = []
        if supports_multi_turn:
            seen_first_user = False
            for turn in session.turns:
                if turn.role != "user":
                    continue
                if not seen_first_user:
                    seen_first_user = True
                    continue
                content = (turn.content or "").strip()
                if content:
                    pending_user_messages.append(content)

        return ReplaySession(
            session_id=session.session_id,
            turn_messages=turn_messages,
            metadata=metadata,
            pending_user_messages=pending_user_messages,
        )
