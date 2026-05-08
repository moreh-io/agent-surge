# SPDX-License-Identifier: MIT
"""Workload I/O - load, save, and validate workload sessions and corpus files."""

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from agentsurge.exceptions import WorkloadValidationError
from agentsurge.types import WORKLOAD_SCHEMA_VERSION, ReplaySession

_log = logging.getLogger(__name__)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write *payload* to *path* atomically via a sibling tmp file.

    A crash, OOM, or KeyboardInterrupt mid-write leaves the canonical
    path either at its previous good content or fully replaced — never
    at a half-written state shadowing the old file.
    """
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        # Best-effort cleanup: if write or replace raised, the tmp may
        # linger. Drop it so retries see a clean directory.
        if tmp.exists():
            with contextlib.suppress(OSError):
                tmp.unlink()


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Atomic counterpart of :meth:`Path.write_text` (tmp + ``os.replace``)."""
    _atomic_write_bytes(path, text.encode(encoding))


def _parse_workload_json(data: list | dict) -> list[dict]:
    """Extract session dicts from either legacy (bare list) or versioned envelope.

    Accepted formats:
      - Legacy: ``[{session_id, turn_messages, ...}, ...]``
      - Versioned: ``{"version": "1.0", "sessions": [...]}``
      - Corpus: ``{"version": "1.0", "manifest": {...}, "sessions": [...]}``

    Returns the list of raw session dicts.
    """
    if isinstance(data, list):
        result: list[dict] = data
        return result
    if isinstance(data, dict) and "sessions" in data:
        sessions: list[dict] = data["sessions"]
        return sessions
    raise ValueError(
        "Unrecognised workload JSON: expected a list of sessions or a dict with a 'sessions' key."
    )


def load_workload_sessions(path: str | Path) -> list[ReplaySession]:
    """Load workload sessions from JSON or HF URL.

    Handles both legacy bare-list workloads and the versioned envelope
    format introduced in schema version 1.0.  Accepts ``hf://`` URLs
    to download workloads from HuggingFace Hub.
    """
    path_str = str(path)
    if path_str.startswith("hf://"):
        from agentsurge.hf import download_workload_from_hf

        return download_workload_from_hf(path_str)

    raw = Path(path).read_bytes()
    try:
        import orjson

        data = orjson.loads(raw)
    except ImportError:
        data = json.loads(raw)

    session_dicts = _parse_workload_json(data)
    return [ReplaySession.from_dict(d) for d in session_dicts]


def load_corpus(path: str | Path) -> tuple[list[ReplaySession], dict]:
    """Load a pinned corpus from JSON. Returns (sessions, manifest)."""
    raw = Path(path).read_bytes()
    try:
        import orjson

        data = orjson.loads(raw)
    except ImportError:
        data = json.loads(raw)
    manifest = data["manifest"] if isinstance(data, dict) else {}
    sessions = [ReplaySession.from_dict(d) for d in _parse_workload_json(data)]
    return sessions, manifest


def save_corpus(sessions: list[ReplaySession], path: str | Path) -> dict:
    """Save a pinned corpus to JSON with manifest metadata."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    source_counts: dict[str, int] = {}
    total_tokens = 0
    for s in sessions:
        src = s.metadata.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
        total_tokens += s.estimated_tokens

    manifest = {
        "n_sessions": len(sessions),
        "source_counts": source_counts,
        "total_estimated_tokens": total_tokens,
        "avg_tokens_per_session": total_tokens // max(len(sessions), 1),
    }

    payload = {
        "version": WORKLOAD_SCHEMA_VERSION,
        "manifest": manifest,
        "sessions": [s.to_dict() for s in sessions],
    }
    try:
        import orjson

        _atomic_write_bytes(p, orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    except ImportError:
        _atomic_write_text(p, json.dumps(payload, ensure_ascii=False, indent=2))
    return manifest


def _major(version: str) -> int:
    """Extract the major version number from a ``"major.minor"`` string."""
    try:
        return int(version.split(".")[0])
    except (ValueError, IndexError, AttributeError) as err:
        raise WorkloadValidationError(
            f"Invalid version string: {version!r}",
            detail="Expected a 'major.minor' semver string (e.g. '1.0').",
        ) from err


def _check_version(version: str) -> None:
    current_major = _major(WORKLOAD_SCHEMA_VERSION)
    file_major = _major(version)
    if file_major != current_major:
        raise WorkloadValidationError(
            f"Unsupported workload schema version '{version}' "
            f"(expected major version {current_major}, got {file_major})",
        )


def _check_message(msg: Any, *, session_id: str, turn_idx: int, msg_idx: int) -> None:
    loc = f"session '{session_id}', turn {turn_idx}, message {msg_idx}"
    if not isinstance(msg, dict):
        raise WorkloadValidationError(f"Message at {loc} must be a dict, got {type(msg).__name__}")
    if "role" not in msg:
        raise WorkloadValidationError(f"Message at {loc} is missing required field 'role'")
    if not isinstance(msg["role"], str):
        raise WorkloadValidationError(
            f"Message at {loc} has non-string 'role': {type(msg['role']).__name__}"
        )
    if "content" not in msg:
        raise WorkloadValidationError(f"Message at {loc} is missing required field 'content'")


def _check_session(session: Any, idx: int) -> None:
    if not isinstance(session, dict):
        raise WorkloadValidationError(
            f"Session at index {idx} must be a dict, got {type(session).__name__}"
        )
    if "session_id" not in session:
        raise WorkloadValidationError(
            f"Session at index {idx} is missing required field 'session_id'"
        )
    sid = session["session_id"]
    if not isinstance(sid, str) or not sid:
        raise WorkloadValidationError(f"Session at index {idx} has invalid 'session_id': {sid!r}")
    if "turn_messages" not in session:
        raise WorkloadValidationError(f"Session '{sid}' is missing required field 'turn_messages'")
    tm = session["turn_messages"]
    if not isinstance(tm, list) or len(tm) == 0:
        raise WorkloadValidationError(f"Session '{sid}': 'turn_messages' must be a non-empty list")
    for turn_idx, turn_snapshot in enumerate(tm):
        if not isinstance(turn_snapshot, list) or len(turn_snapshot) == 0:
            raise WorkloadValidationError(
                f"Session '{sid}', turn {turn_idx}: expected non-empty list of messages"
            )
        for msg_idx, msg in enumerate(turn_snapshot):
            _check_message(msg, session_id=sid, turn_idx=turn_idx, msg_idx=msg_idx)
    if "fan_out" in session:
        fo = session["fan_out"]
        if not isinstance(fo, int) or fo < 1:
            raise WorkloadValidationError(
                f"Session '{sid}': 'fan_out' must be a positive integer, got {fo!r}"
            )
    if "metadata" in session and not isinstance(session["metadata"], dict):
        raise WorkloadValidationError(f"Session '{sid}': 'metadata' must be a dict")
    if "pending_user_messages" in session:
        pending = session["pending_user_messages"]
        if not isinstance(pending, list):
            raise WorkloadValidationError(
                f"Session '{sid}': 'pending_user_messages' must be a list of strings, "
                f"got {type(pending).__name__}"
            )
        for msg_idx, msg in enumerate(pending):
            if not isinstance(msg, str):
                raise WorkloadValidationError(
                    f"Session '{sid}': pending_user_messages[{msg_idx}] must be a string, "
                    f"got {type(msg).__name__}"
                )


def validate_workload(data: Any) -> None:
    """Validate a parsed workload payload against the agentsurge schema.

    Raises :class:`~agentsurge.exceptions.WorkloadValidationError` on the
    first detected problem.
    """
    if data is None:
        raise WorkloadValidationError("Workload data is None")
    if isinstance(data, list):
        sessions = data
    elif isinstance(data, dict):
        if "version" in data:
            _check_version(str(data["version"]))
        if "sessions" not in data:
            raise WorkloadValidationError("Workload envelope is missing 'sessions' key")
        sessions = data["sessions"]
        if not isinstance(sessions, list):
            raise WorkloadValidationError(
                f"'sessions' must be a list, got {type(sessions).__name__}"
            )
        if "manifest" in data and not isinstance(data["manifest"], dict):
            raise WorkloadValidationError("'manifest' must be a dict")
    else:
        raise WorkloadValidationError(f"Workload must be a dict or list, got {type(data).__name__}")
    if len(sessions) == 0:
        raise WorkloadValidationError("Workload contains no sessions")
    for idx, session in enumerate(sessions):
        _check_session(session, idx)
