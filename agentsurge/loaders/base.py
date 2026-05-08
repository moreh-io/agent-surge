# SPDX-License-Identifier: MIT
"""Base types and ABCs for trace/task loaders."""

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator

from agentsurge.loaders.registry import loader_registry

# Canonical definitions live in agentsurge.types; re-exported here for backward
# compatibility so that ``from agentsurge.loaders.base import Turn`` still works.
from agentsurge.types import Session, Task, Trajectory, Turn  # noqa: F401

_log = logging.getLogger(__name__)


def preflight_hf_dataset(repo_id: str, split: str | None = None) -> None:
    """Fail fast with an actionable message if the HF dataset is unreachable.

    Wraps ``huggingface_hub.HfApi().dataset_info`` in a single call that
    terminates in under a second on auth, permission, or repo-id errors
    instead of stalling inside ``load_dataset(streaming=True)`` for up to
    the hub's full timeout. Re-raises as ``RuntimeError`` with a message
    naming the dataset and the concrete fix (export HF_TOKEN / run
    ``huggingface-cli login`` / check the repo id).

    Best-effort: if ``huggingface_hub`` is unavailable, silently returns so
    downstream ``load_dataset`` can still attempt the stream.
    """
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import (
            GatedRepoError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )
    except ImportError:
        return

    try:
        HfApi().dataset_info(repo_id)
    except (GatedRepoError, RepositoryNotFoundError) as err:
        raise RuntimeError(
            f"HuggingFace dataset {repo_id!r} is gated or not visible to this "
            f"token. Run `huggingface-cli login`, set HF_TOKEN=..., or request "
            f"access on https://huggingface.co/datasets/{repo_id}. "
            f"Original: {type(err).__name__}: {err}"
        ) from err
    except HfHubHTTPError as err:
        status = getattr(err.response, "status_code", None)
        if status == 401:
            raise RuntimeError(
                f"HuggingFace 401 on dataset {repo_id!r}: missing or expired "
                f"token. Run `huggingface-cli login` or export HF_TOKEN=... "
                f"before retrying."
            ) from err
        raise RuntimeError(
            f"HuggingFace hub error reaching dataset {repo_id!r} "
            f"(split={split!r}): HTTP {status}. {err}"
        ) from err
    except Exception as err:  # network, DNS, timeout
        _log.warning(
            "preflight for dataset %s failed (%s: %s); attempting stream anyway",
            repo_id,
            type(err).__name__,
            err,
        )


class _ToolCallAligner:
    """Match ``role:"tool"`` replies to prior assistant ``tool_calls``.

    Datasets often omit ``tool_call_id`` on the tool reply or ship it under
    ``name``.  Strict OpenAI-compatible chat-completions endpoints reject
    tool replies that don't reference a preceding tool_call.  This helper
    keeps a FIFO of pending calls plus an id index, mirroring the
    SWE-smith logic, so trace loaders can recover the linkage.
    """

    def __init__(self) -> None:
        self._pending: list[dict] = []
        self._by_id: dict[str, dict] = {}

    def record_calls(self, tool_calls: list | None) -> None:
        """Record an assistant message's ``tool_calls`` as pending replies."""
        if not isinstance(tool_calls, list):
            return
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            call_id = str(tc.get("id") or "")
            fn_raw = tc.get("function")
            fn = fn_raw if isinstance(fn_raw, dict) else {}
            call = {
                "id": call_id,
                "name": fn.get("name") or tc.get("name"),
            }
            self._pending.append(call)
            if call_id:
                self._by_id[call_id] = call

    def resolve(
        self, tool_call_id: str | None, tool_name: str | None
    ) -> tuple[str | None, str | None]:
        """Resolve a tool reply's id/name, filling gaps from pending calls.

        By-id match wins when ``tool_call_id`` is provided; falls back to
        FIFO popping the oldest pending call.
        """
        matched: dict | None = None
        if tool_call_id:
            matched = self._by_id.pop(str(tool_call_id), None)
            if matched is not None:
                self._pending = [c for c in self._pending if c is not matched]
        if matched is None and self._pending:
            matched = self._pending.pop(0)
            if matched.get("id"):
                self._by_id.pop(str(matched["id"]), None)
        if matched is not None:
            tool_call_id = tool_call_id or matched.get("id")
            tool_name = tool_name or matched.get("name")
        return tool_call_id, tool_name


class TraceLoader(ABC):
    """ABC for loading agent trajectories from datasets.

    Subclasses should set a class-level ``name`` attribute to auto-register
    with the loader registry::

        class MyLoader(TraceLoader):
            name = "my-loader"
            ...
    """

    # Subclasses MUST override this with a class-level str.
    name: str = ""

    #: HuggingFace dataset ID.  Set in subclasses that support ``download()``.
    HF_DATASET: str = ""

    #: Whether this source provides traces with multiple user-role turns
    #: (i.e. the user returns to the conversation after the assistant
    #: completes a step).  Sources that are only single-prompt tasks leave
    #: this False.  Consumers (e.g. TraceReplayGenerator) use this flag to
    #: decide whether to extract follow-up user messages into
    #: ReplaySession.pending_user_messages.
    supports_multi_turn: bool = False

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        _name = cls.__dict__.get("name", "")
        if _name:
            loader_registry.register_trace(cls)

    @abstractmethod
    def load(self, limit: int | None = None) -> Iterator[Trajectory]:
        """Yield Trajectory objects from the dataset."""
        ...

    def download(self, dest_dir: str) -> str:
        """Download the dataset to *dest_dir* and return the local JSONL path.

        Default implementation uses HuggingFace ``load_dataset`` with
        ``cache_dir`` and writes a JSONL file.  Subclasses may override.
        """
        if not self.HF_DATASET:
            raise NotImplementedError(f"{type(self).__name__} does not support download()")

        import json as _json

        from datasets import load_dataset

        os.makedirs(dest_dir, exist_ok=True)
        out_path = os.path.join(dest_dir, f"{self.name}.jsonl")
        if os.path.exists(out_path):
            return out_path

        # ``streaming=True`` keeps memory flat for multi-GB datasets; we
        # write each row to JSONL as it arrives.
        ds = load_dataset(self.HF_DATASET, split="train", streaming=True)
        with open(out_path, "w") as f:
            for row in ds:
                f.write(_json.dumps(row) + "\n")
        return out_path


class TaskLoader(ABC):
    """ABC for loading benchmark tasks.

    Subclasses should set a class-level ``name`` attribute to auto-register
    with the loader registry::

        class MyTaskLoader(TaskLoader):
            name = "my-task-loader"
            ...
    """

    # Subclasses MUST override this with a class-level str.
    name: str = ""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        _name = cls.__dict__.get("name", "")
        if _name:
            loader_registry.register_task(cls)

    @abstractmethod
    def load(self, limit: int | None = None) -> Iterator[Task]:
        """Yield Task objects from the dataset."""
        ...


class BaseHFTaskLoader(TaskLoader):
    """Base for HuggingFace-backed task loaders.

    Subclasses set ``name``, ``HF_DATASET``, ``HF_SPLIT`` and override
    ``_make_task()``.  The ``name`` class attribute is used for both
    registry auto-discovery.
    """

    HF_DATASET: str  # e.g. "princeton-nlp/SWE-bench_Verified"
    HF_SPLIT: str = "test"

    def __init__(self, local_path: str | None = None) -> None:
        self._local_path = local_path

    def load(self, limit: int | None = None) -> Iterator[Task]:
        if self._local_path and os.path.exists(self._local_path):
            yield from self._load_local(limit)
        else:
            yield from self._load_hf(limit)

    def _load_local(self, limit: int | None = None) -> Iterator[Task]:
        import json

        if self._local_path is None:
            raise ValueError("local_path is required for local loading")
        # ``utf-8-sig`` swallows a leading BOM if present and is otherwise
        # identical to UTF-8.
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
                    _log.warning(
                        "%s: skipping malformed JSONL line %d: %s",
                        self._local_path,
                        line_no,
                        err,
                    )
                    continue
                yield self._make_task(sample, idx)
                idx += 1
            if skipped:
                _log.warning(
                    "%s: skipped %d malformed JSONL line(s) total",
                    self._local_path,
                    skipped,
                )

    def _load_hf(self, limit: int | None = None) -> Iterator[Task]:
        try:
            from datasets import load_dataset
        except ImportError as err:
            raise ImportError("Install datasets: pip install datasets") from err

        preflight_hf_dataset(self.HF_DATASET, split=self.HF_SPLIT)
        # Narrow wrap: only ``load_dataset`` (auth/network/repo) is mapped
        # to RuntimeError.  ``_make_task`` errors below must propagate
        # natively so schema regressions are not masked as auth failures
        # (see test_base_hf_task_loader_parse_error_not_masked_as_auth).
        try:
            ds = load_dataset(self.HF_DATASET, split=self.HF_SPLIT, streaming=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to stream '{self.HF_DATASET}' (split={self.HF_SPLIT}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        for idx, sample in enumerate(ds):
            if limit is not None and idx >= limit:
                break
            yield self._make_task(sample, idx)

    def download(self, dest_dir: str) -> str:
        """Download the HF dataset to *dest_dir* and return local JSONL path."""
        import json as _json

        from datasets import load_dataset

        os.makedirs(dest_dir, exist_ok=True)
        out_path = os.path.join(dest_dir, f"{self.name}.jsonl")
        if os.path.exists(out_path):
            return out_path

        # ``streaming=True`` avoids materialising large datasets into RAM
        # before serialising to JSONL.
        ds = load_dataset(self.HF_DATASET, split=self.HF_SPLIT, streaming=True)
        with open(out_path, "w") as f:
            for row in ds:
                f.write(_json.dumps(row) + "\n")
        return out_path

    def _make_task(self, sample: dict, idx: int) -> Task:
        """Convert a HF dataset sample to a Task.

        Default implementation works for SWE-bench-style datasets with
        ``instance_id``, ``problem_statement``, and ``patch`` fields.
        Override for datasets with different schemas (e.g. ABC-bench).
        """
        instance_id = sample.get("instance_id", str(idx))
        metadata = {
            k: v
            for k, v in sample.items()
            if k not in ("problem_statement", "patch", "instance_id")
        }
        metadata.setdefault("instance_id", instance_id)
        reference = sample.get("patch")
        # Mirror onto metadata so downstream consumers (e.g. scoring) can
        # reach the gold answer; ``TaskToTrajectoryConverter`` only forwards
        # ``task.metadata``.
        if reference is not None:
            metadata.setdefault("reference", reference)
        return Task(
            task_id=instance_id,
            prompt=sample.get("problem_statement", ""),
            reference=reference,
            metadata=metadata,
        )
