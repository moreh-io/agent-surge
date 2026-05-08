# SPDX-License-Identifier: MIT
"""Task-to-trajectory converter - turn benchmark tasks into synthetic traces."""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Iterator
from typing import TYPE_CHECKING

from agentsurge.generators.base import GeneratorBase
from agentsurge.generators.synthetic import _pad_assistant_response
from agentsurge.types import Session, Trajectory, Turn

_FILE_PATTERN = re.compile(r"\S+\.(?:py|js|ts|go|rs|java|rb)\b")

if TYPE_CHECKING:
    from agentsurge.generators.trace_pool import TracePool


_INTENT_TOOL_TYPE: dict[str, str] = {
    "read_code": "str_replace_editor",
    "run_tests": "execute_bash",
    "validate_impl": "execute_bash",
    "add_integration": "str_replace_editor",
    "add_tests": "str_replace_editor",
    "run_checks": "execute_bash",
}

_INTENT_TEMPLATE: dict[str, str] = {
    "read_code": "[Tool output: read_file] {body}",
    "run_tests": "[Tool output: bash] {body}",
    "validate_impl": "[Tool output: bash] {body}",
    "add_integration": "[Tool output: read_file] {body}",
    "add_tests": "[Tool output: read_file] {body}",
    "run_checks": "[Tool output: bash] {body}",
}


def _extract_hints(task) -> dict[str, str]:
    """Extract context hints from a task for fallback content generation."""
    prompt = task.prompt or ""
    meta = task.metadata or {}

    files = _FILE_PATTERN.findall(prompt)
    file_hint = files[0] if files else "src/main.py"

    repo = meta.get("repo", "")
    if "/" in repo:
        repo = repo.split("/")[-1]
    if not repo:
        repo = task.task_id.split("_")[0] if "_" in task.task_id else "project"

    return {"file_hint": file_hint, "repo": repo}


def _fallback_body(task, turn_idx: int) -> str:
    """Generate fallback body when no fragment bank is available.

    Uses deterministic slices of the task prompt as content, ensuring
    each task produces unique output from the first character.
    """
    prompt = task.prompt or ""
    seed = int(hashlib.md5(f"{task.task_id}:{turn_idx}:body".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    # Target ~650 tokens (mean from real traces) ≈ 2600 chars.
    # Range [800, 4000] covers P25-P90 of OpenHands/SWE-smith tool output lengths
    # (trace_pool _aggregate LogNormal: mu=5.276, sigma=1.550).
    _FALLBACK_MIN_CHARS = 800
    _FALLBACK_MAX_CHARS = 4000
    target_chars = rng.randint(_FALLBACK_MIN_CHARS, _FALLBACK_MAX_CHARS)

    if len(prompt) >= target_chars:
        offset = seed % max(1, len(prompt) - target_chars)
        body = prompt[offset : offset + target_chars]
    else:
        body = prompt

    hints = _extract_hints(task)
    return f"{hints['repo']}/{hints['file_hint']}:\n{body}"


def _resolve_intent(intent: str, task, turn_idx: int, bank: TracePool | None) -> str:
    """Resolve an @intent marker to a realistic tool output message.

    When a fragment bank is available, samples real trace content at
    LogNormal-distributed lengths.  Otherwise falls back to task-prompt
    derived content.
    """
    seed = int(hashlib.md5(f"{task.task_id}:{turn_idx}:intent".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    template = _INTENT_TEMPLATE.get(intent, "[Tool output] {body}")
    tool_type = _INTENT_TOOL_TYPE.get(intent)

    if bank is not None:
        body = bank.sample(rng, tool_type=tool_type)
    else:
        body = _fallback_body(task, turn_idx)

    return template.format(body=body)


class TaskToTrajectoryConverter(GeneratorBase):
    """Convert Task objects to synthetic multi-turn Trajectories.

    Non-chunk user turns use ``@intent`` markers resolved to realistic
    tool-output messages.  When a :class:`TracePool` is provided, content
    is sampled from real agent traces at empirically correct lengths.
    Otherwise, task-prompt content is used as a fallback.

    Parameters
    ----------
    max_prompt_chars : int
        Max chars from task prompt to include.
    min_assistant_tokens : int
        Min tokens for padded assistant responses.
    trace_pool : TracePool or None
        Pre-extracted real trace fragments for realistic content.
    """

    name = "task-to-trajectory"

    # Turn counts based on agent trajectory analysis:
    #   - SWE-Effi (arXiv:2509.09853): per-agent LLM call counts
    #   - arXiv:2506.18824: median iterations per agent type
    #   - bug-fix: SWE-agent median 5 iters (arXiv:2506.18824), capped at 6 for 16K context
    #   - api-impl: ABC-Bench tasks are simpler, 4 turns typical
    #   - feature-impl: FeatureBench multi-step, 8 turns (arXiv:2602.10975)
    #   - migration: SWE-EVO long-horizon, 8 turns (arXiv:2512.18470)
    PATTERNS = {
        "bug-fix": [
            ("user", "Fix this bug:\n{chunk_0}"),
            (
                "assistant",
                "I'll analyze the issue and trace the root cause through the call stack.",
            ),
            ("user", "@read_code"),
            (
                "assistant",
                "Found the root cause. The issue is in the error handling path. Implementing fix now.",
            ),
            ("user", "@run_tests"),
            (
                "assistant",
                "Fix applied. Running the test suite to verify the change doesn't break anything.",
            ),
        ],
        "api-impl": [
            ("user", "Implement this:\n{chunk_0}"),
            (
                "assistant",
                "Setting up the route handler with request validation and response serialization.",
            ),
            ("user", "@validate_impl"),
            (
                "assistant",
                "Implementation complete with input validation, error handling, and response formatting.",
            ),
        ],
        "feature-impl": [
            ("user", "Feature spec:\n{chunk_0}"),
            (
                "assistant",
                "Breaking down into implementation steps and identifying affected modules.",
            ),
            ("user", "{chunk_1}"),
            (
                "assistant",
                "Implementing core logic with the necessary data transformations and state management.",
            ),
            ("user", "@add_integration"),
            (
                "assistant",
                "Integration points added. Connected to existing event system and data pipeline.",
            ),
            ("user", "@add_tests"),
            (
                "assistant",
                "Tests added covering happy path, edge cases, and error scenarios. All passing.",
            ),
        ],
        "migration": [
            ("user", "Migration needed:\n{chunk_0}"),
            (
                "assistant",
                "Identifying all affected files and mapping dependency chains for the migration.",
            ),
            ("user", "{chunk_1}"),
            (
                "assistant",
                "Starting migration. Updating imports, type signatures, and call sites systematically.",
            ),
            ("user", "{chunk_2}"),
            (
                "assistant",
                "Migration complete. All references updated and backward compatibility maintained.",
            ),
            ("user", "@run_checks"),
            (
                "assistant",
                "All compatibility checks pass. No breaking changes detected in the public API.",
            ),
        ],
    }

    # Retry patterns - test→fail→debug→fix→retest loops
    # Based on corpus analysis: 100% of real sessions have retry loops,
    # median 16 errors per session, 88.5% edit same files 2-5 times.
    RETRY_PATTERNS = {
        "bug-fix-retry": [
            ("user", "Fix this bug:\n{chunk_0}"),
            ("assistant", "I'll analyze the issue. Let me read the relevant code first."),
            ("user", "@read_code"),
            ("assistant", "I see the problem. Applying an initial fix to the error handling path."),
            ("user", "@run_tests"),
            (
                "assistant",
                "Tests failed. The fix exposed a secondary issue in the input validation.",
            ),
            ("user", "@read_code"),
            (
                "assistant",
                "Found the second issue. The validation logic needs to handle the edge case.",
            ),
            ("user", "@run_tests"),
            ("assistant", "Still one failure. The assertion expects a different return type."),
            ("user", "@read_code"),
            ("assistant", "Updating the return type to match the expected interface."),
            ("user", "@run_tests"),
            (
                "assistant",
                "All tests pass now. The fix handles both the original bug and the edge cases.",
            ),
        ],
        "api-impl-retry": [
            ("user", "Implement this:\n{chunk_0}"),
            ("assistant", "Setting up the route handler with request validation."),
            ("user", "@validate_impl"),
            ("assistant", "Validation errors found. Missing required fields in the schema."),
            ("user", "@read_code"),
            ("assistant", "Updated the schema. Adding proper error response formatting."),
            ("user", "@validate_impl"),
            ("assistant", "Type checking passes. Running integration tests."),
            ("user", "@run_tests"),
            (
                "assistant",
                "All tests pass. Implementation complete with validation and error handling.",
            ),
        ],
        "feature-impl-retry": [
            ("user", "Feature spec:\n{chunk_0}"),
            (
                "assistant",
                "Breaking down into implementation steps and identifying affected modules.",
            ),
            ("user", "{chunk_1}"),
            ("assistant", "Implementing core logic with data transformations."),
            ("user", "@run_tests"),
            (
                "assistant",
                "Tests failed. The transformation output format doesn't match the expected schema.",
            ),
            ("user", "@read_code"),
            ("assistant", "Fixed the output format. Also need to update the integration points."),
            ("user", "@add_integration"),
            ("assistant", "Integration added. Running the full test suite."),
            ("user", "@run_tests"),
            ("assistant", "Two tests still failing due to missing edge case handling."),
            ("user", "@read_code"),
            ("assistant", "Added edge case handling and updated the test fixtures."),
            ("user", "@add_tests"),
            ("assistant", "All tests pass including new edge case coverage."),
        ],
        "migration-retry": [
            ("user", "Migration needed:\n{chunk_0}"),
            ("assistant", "Identifying all affected files and mapping dependency chains."),
            ("user", "{chunk_1}"),
            ("assistant", "Starting migration. Updating imports and type signatures."),
            ("user", "@run_checks"),
            (
                "assistant",
                "Compatibility check failed. Three deprecated call sites still using old API.",
            ),
            ("user", "{chunk_2}"),
            ("assistant", "Updated the remaining call sites. Re-running compatibility checks."),
            ("user", "@run_checks"),
            ("assistant", "One more issue: a test fixture uses the old interface."),
            ("user", "@read_code"),
            ("assistant", "Fixed the test fixture. Running full regression suite."),
            ("user", "@run_checks"),
            ("assistant", "All compatibility checks pass. No breaking changes detected."),
        ],
    }

    # Probability of selecting the retry variant (based on corpus: ~60% of real
    # sessions have significant retry loops, the rest are quick fixes)
    RETRY_PROBABILITY = 0.6

    DATASET_PATTERN_MAP = {
        "swe-bench": "bug-fix",
        "swe-bench-verified": "bug-fix",
        "abc-bench": "api-impl",
        "featurebench": "feature-impl",
        "swe-evo": "migration",
    }

    def __init__(
        self,
        max_prompt_chars: int = 16384,
        min_assistant_tokens: int = 200,
        trace_pool: TracePool | None = None,
    ) -> None:
        self.max_prompt_chars = max_prompt_chars
        self.min_assistant_tokens = min_assistant_tokens
        self.trace_pool = trace_pool

    def from_tasks(
        self, tasks: Iterator, pattern: str, limit: int | None = None
    ) -> list[Trajectory]:
        results = []
        for task in tasks:
            traj = self._build_trajectory(task, pattern)
            results.append(traj)
            if limit is not None and len(results) >= limit:
                break
        return results

    def _chunk_prompt(self, prompt: str, n_chunks: int) -> list[str]:
        truncated = prompt[: self.max_prompt_chars]
        if n_chunks <= 1:
            return [truncated]
        chunk_size = max(1, len(truncated) // n_chunks)
        chunks = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = start + chunk_size if i < n_chunks - 1 else len(truncated)
            chunks.append(truncated[start:end])
        return chunks

    def _build_trajectory(self, task, pattern: str) -> Trajectory:
        seed = int(hashlib.md5(f"{task.task_id}:pattern".encode()).hexdigest()[:8], 16)
        use_retry = random.Random(seed).random() < self.RETRY_PROBABILITY
        retry_key = f"{pattern}-retry"
        if use_retry and retry_key in self.RETRY_PATTERNS:
            template = self.RETRY_PATTERNS[retry_key]
        else:
            template = self.PATTERNS[pattern]

        n_chunks = sum(1 for _, content in template if "{chunk_" in content)
        chunks = self._chunk_prompt(task.prompt, max(1, n_chunks))

        turns = []
        for turn_idx, (role, content) in enumerate(template):
            text = content
            if role == "user" and text.startswith("@"):
                text = _resolve_intent(text[1:], task, turn_idx, self.trace_pool)
            else:
                for i, chunk in enumerate(chunks):
                    text = text.replace(f"{{chunk_{i}}}", chunk)
            if role == "assistant":
                seed = int(hashlib.md5(f"{task.task_id}:{turn_idx}".encode()).hexdigest()[:8], 16)
                if self.trace_pool is not None:
                    sampled = self.trace_pool.sample_assistant(
                        random.Random(seed),
                        target_tokens=self.min_assistant_tokens,
                    )
                    if sampled:
                        text = sampled
                    else:
                        text = _pad_assistant_response(text, seed, self.min_assistant_tokens)
                else:
                    text = _pad_assistant_response(text, seed, self.min_assistant_tokens)
            turns.append(Turn(role=role, content=text))

        session = Session(
            session_id=f"{task.task_id}_0",
            turns=turns,
            metadata=task.metadata,
        )

        return Trajectory(
            trajectory_id=task.task_id,
            sessions=[session],
            metadata={"source": pattern, "task_id": task.task_id},
        )
