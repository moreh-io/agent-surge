# SPDX-License-Identifier: MIT
"""ABC-Bench task loader from HuggingFace datasets."""

from agentsurge.loaders.base import BaseHFTaskLoader
from agentsurge.types import Task


class ABCBenchLoader(BaseHFTaskLoader):
    """Load tasks from OpenMOSS-Team/ABC-Bench."""

    name = "abc-bench"
    HF_DATASET = "OpenMOSS-Team/ABC-Bench"
    HF_SPLIT = "train"

    def _make_task(self, sample: dict, idx: int) -> Task:
        # HF schema uses ``task_id``; legacy/mock samples may use ``id``.
        task_id = sample.get("task_id") or sample.get("id") or str(idx)
        metadata = {
            k: v
            for k, v in sample.items()
            if k not in ("instruction", "prompt", "output", "answer", "id", "task_id")
        }
        # Preserve the real task_id for sandbox lookup and mark the source so
        # warmstart/Sandbox know to use the pre-extracted tarball path.
        metadata["task_id"] = task_id
        metadata["source"] = "abc-bench"
        # Preserve the alternate prompt key when both ``instruction`` and
        # ``prompt`` are present with different content so a schema rotation
        # cannot silently lose data.
        instruction = sample.get("instruction")
        alt = sample.get("prompt")
        if instruction is not None and alt is not None and instruction != alt:
            metadata["alt_prompt"] = alt
        reference = sample.get("output", sample.get("answer"))
        # Mirror onto metadata so downstream consumers (e.g. scoring) can
        # reach the gold answer; ``TaskToTrajectoryConverter`` only forwards
        # ``task.metadata``.
        if reference is not None:
            metadata.setdefault("reference", reference)
        return Task(
            task_id=task_id,
            prompt=sample.get("instruction", sample.get("prompt", "")),
            reference=reference,
            metadata=metadata,
        )
