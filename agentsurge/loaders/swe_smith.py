# SPDX-License-Identifier: MIT
"""SWE-smith trajectory loader from HuggingFace datasets."""

import json
import re
from collections.abc import Iterator

from agentsurge.loaders.base import TraceLoader, _ToolCallAligner, preflight_hf_dataset
from agentsurge.types import Session, Trajectory, Turn

_HEX_RE = re.compile(r"^[0-9a-f]{7,40}$")


def _derive_repo_metadata(instance_id: str) -> dict[str, str]:
    """Best-effort repo metadata recovery for SWE-smith instance ids.

    Observed ids include patterns such as:
    - ``owner__repo.commit_hash.variant__suffix``
    - ``owner__repo-issue_number``

    We recover:
    - ``repo`` as ``owner/repo``
    - ``base_commit`` when a commit-like token is present
    """
    from agentsurge.tool_call import _parse_owner_repo

    owner, repo_name = _parse_owner_repo(instance_id)
    if not owner:
        return {}

    meta: dict[str, str] = {"repo": f"{owner}/{repo_name}"}

    parts = instance_id.split("__")
    if len(parts) >= 2:
        dot_parts = parts[1].split(".")
        if len(dot_parts) >= 2 and _HEX_RE.fullmatch(dot_parts[1]):
            meta["base_commit"] = dot_parts[1]

    return meta


class SWESmithLoader(TraceLoader):
    """Load SWE-smith agent trajectories (SWE-bench/SWE-smith-trajectories)."""

    name = "swe-smith"
    HF_DATASET = "SWE-bench/SWE-smith-trajectories"
    VALID_SPLITS: frozenset[str] = frozenset({"tool", "xml", "ticks"})

    def __init__(self, split: str = "tool", local_path: str | None = None) -> None:
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"SWESmithLoader split must be one of {sorted(self.VALID_SPLITS)}, got {split!r}"
            )
        self.split = split  # "tool", "xml", "ticks"
        self._local_path = local_path

    @property
    def display_name(self) -> str:
        """Instance-level name including the split (e.g. ``swe-smith-tool``)."""
        return f"swe-smith-{self.split}"

    def download(self, dest_dir: str) -> str:
        """Download the dataset split to *dest_dir* and return the local JSONL path."""
        import json as _json
        import os

        from datasets import load_dataset

        os.makedirs(dest_dir, exist_ok=True)
        out_path = os.path.join(dest_dir, f"{self.name}-{self.split}.jsonl")
        if os.path.exists(out_path):
            return out_path

        # ``streaming=True`` keeps memory flat for the multi-GB SWE-smith
        # split dumps; we serialise rows to JSONL as they arrive.
        ds = load_dataset(self.HF_DATASET, split=self.split, streaming=True)
        with open(out_path, "w") as f:
            for row in ds:
                f.write(_json.dumps(row) + "\n")
        return out_path

    def load(self, limit: int | None = None) -> Iterator[Trajectory]:
        if self._local_path:
            import os

            if os.path.exists(self._local_path):
                yield from self._load_local(limit)
                return
        yield from self._load_hf(limit)

    def _load_hf(self, limit: int | None = None) -> Iterator[Trajectory]:
        try:
            from datasets import load_dataset
        except ImportError as err:
            raise ImportError("Install datasets: pip install datasets") from err

        preflight_hf_dataset(self.HF_DATASET, split=self.split)
        # Narrow wrap: only ``load_dataset`` (auth/network/repo) is mapped
        # to RuntimeError.  ``_parse_sample`` errors must propagate natively
        # so schema regressions are not masked as auth failures
        # (see test_swe_smith_load_hf_parse_error_not_masked_as_auth).
        try:
            ds = load_dataset(self.HF_DATASET, split=self.split, streaming=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to stream '{self.HF_DATASET}' (split={self.split}): "
                f"{type(exc).__name__}: {exc}\n"
                f"If gated/private, run `huggingface-cli login` or set HF_TOKEN."
            ) from exc
        for idx, sample in enumerate(ds):
            if limit is not None and idx >= limit:
                break
            yield self._parse_sample(sample, idx)

    def _load_local(self, limit: int | None = None) -> Iterator[Trajectory]:
        import logging

        log = logging.getLogger(__name__)
        if self._local_path is None:
            raise ValueError("local_path is required for local loading")
        # ``utf-8-sig`` swallows a leading BOM if present.
        with open(self._local_path, encoding="utf-8-sig") as f:
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
        traj_id = sample.get("instance_id", str(idx))

        messages = sample.get("messages")
        if messages is None:
            return Trajectory(
                trajectory_id=str(traj_id),
                sessions=[Session(session_id=f"{traj_id}_0", turns=[])],
                metadata={"raw_keys": list(sample.keys())},
            )

        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except (json.JSONDecodeError, TypeError):
                messages = []

        turns = []
        aligner = _ToolCallAligner()
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")
            if role not in ("user", "assistant", "system", "tool"):
                continue

            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    item.get("text", str(item)) if isinstance(item, dict) else str(item)
                    for item in content
                )
            content = str(content) if content else ""

            meta = {}
            for key in ("thought", "action", "agent", "message_type"):
                if key in msg:
                    meta[key] = msg[key]

            tool_calls = msg.get("tool_calls")
            tool_name = msg.get("name")
            tool_call_id = msg.get("tool_call_id")
            if role == "assistant":
                aligner.record_calls(tool_calls)
            if role == "tool":
                tool_call_id, tool_name = aligner.resolve(tool_call_id, tool_name)

            turns.append(
                Turn(
                    role=role,
                    content=content,
                    tool_calls=tool_calls,
                    name=tool_name,
                    tool_call_id=tool_call_id,
                    metadata=meta,
                )
            )

        session_meta = {k: v for k, v in sample.items() if k != "messages"}
        session_meta.setdefault("instance_id", str(traj_id))
        session_meta["supports_multi_turn"] = type(self).supports_multi_turn
        for k, v in _derive_repo_metadata(str(traj_id)).items():
            session_meta.setdefault(k, v)

        session = Session(
            session_id=f"{traj_id}_0",
            turns=turns,
            metadata=session_meta,
        )

        return Trajectory(
            trajectory_id=str(traj_id),
            sessions=[session],
            metadata={
                "source": f"swe-smith-{self.split}",
                "supports_multi_turn": type(self).supports_multi_turn,
            },
        )
