# SPDX-License-Identifier: MIT
"""OpenHands trajectory loader from HuggingFace datasets."""

import json
import os
from collections.abc import Iterator

from agentsurge.loaders.base import TraceLoader, _ToolCallAligner, preflight_hf_dataset
from agentsurge.types import Session, Trajectory, Turn


class OpenHandsLoader(TraceLoader):
    """Load OpenHands agent trajectories.

    Tries HuggingFace datasets first, falls back to local JSON file.
    """

    name = "openhands"
    HF_DATASET = "nebius/SWE-rebench-openhands-trajectories"
    supports_multi_turn = True

    def __init__(self, local_path: str | None = None) -> None:
        self._local_path = local_path

    def load(self, limit: int | None = None) -> Iterator[Trajectory]:
        if self._local_path and os.path.exists(self._local_path):
            yield from self._load_local(limit)
        else:
            yield from self._load_hf(limit)

    def _load_hf(self, limit: int | None = None) -> Iterator[Trajectory]:
        """Load from HuggingFace streaming. Falls back with clear error on auth/network issues."""
        try:
            from datasets import load_dataset
        except ImportError as err:
            raise ImportError("Install datasets: pip install datasets") from err

        preflight_hf_dataset(self.HF_DATASET, split="train")
        try:
            ds = load_dataset(self.HF_DATASET, split="train", streaming=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to stream '{self.HF_DATASET}': {type(exc).__name__}: {exc}\n"
                f"Or download locally: agentsurge generate --source openhands --data-dir ./data"
            ) from exc
        for idx, sample in enumerate(ds):
            if limit is not None and idx >= limit:
                break
            yield self._parse_sample(sample, idx)

    def _load_local(self, limit: int | None = None) -> Iterator[Trajectory]:
        """Load from a local JSONL/JSON file."""
        import logging

        log = logging.getLogger(__name__)
        # ``utf-8-sig`` swallows a leading BOM if present.
        with open(self._local_path, encoding="utf-8-sig") as f:  # type: ignore[arg-type]
            idx = 0
            skipped = 0
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                if limit is not None and idx >= limit:
                    break
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as err:
                    skipped += 1
                    log.warning(
                        "%s: skipping malformed JSONL line %d: %s",
                        self._local_path,
                        line_no,
                        err,
                    )
                    continue
                yield self._parse_sample(sample, idx)
                idx += 1
            if skipped:
                log.warning(
                    "%s: skipped %d malformed JSONL line(s) total",
                    self._local_path,
                    skipped,
                )

    def _parse_sample(self, sample: dict, idx: int) -> Trajectory:
        """Parse a single sample into a Trajectory."""
        traj_id = sample.get("instance_id", sample.get("id", str(idx)))

        messages = None
        for key in ("trajectory", "messages", "history", "conversation"):
            val = sample.get(key)
            if val is not None:
                if isinstance(val, str):
                    try:
                        val = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        continue
                if isinstance(val, list):
                    messages = val
                    break

        if messages is None:
            return Trajectory(
                trajectory_id=str(traj_id),
                sessions=[Session(session_id=f"{traj_id}_0", turns=[])],
                metadata={"raw_keys": list(sample.keys())},
            )

        turns = []
        aligner = _ToolCallAligner()
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                if role not in ("user", "assistant", "system", "tool"):
                    continue
                content = msg.get("content", "")
                if isinstance(content, list):
                    # OpenAI/Anthropic-style content blocks
                    # ([{"type": "text", "text": "..."}, ...]).  Stringifying
                    # the list directly would emit a Python literal repr.
                    content = "\n".join(
                        item.get("text", str(item)) if isinstance(item, dict) else str(item)
                        for item in content
                    )
                content = str(content) if content else ""
                tool_calls = msg.get("tool_calls")
                name = msg.get("name")
                tool_call_id = msg.get("tool_call_id")
                if role == "assistant":
                    aligner.record_calls(tool_calls)
                if role == "tool":
                    tool_call_id, name = aligner.resolve(tool_call_id, name)
                turns.append(
                    Turn(
                        role=role,
                        content=content,
                        tool_calls=tool_calls,
                        name=name,
                        tool_call_id=tool_call_id,
                    )
                )

        session = Session(
            session_id=f"{traj_id}_0",
            turns=turns,
            metadata={
                "supports_multi_turn": type(self).supports_multi_turn,
                **{
                    k: v
                    for k, v in sample.items()
                    if k not in ("trajectory", "messages", "history", "conversation")
                },
            },
        )

        return Trajectory(
            trajectory_id=str(traj_id),
            sessions=[session],
            metadata={
                "source": "openhands",
                "supports_multi_turn": type(self).supports_multi_turn,
            },
        )
