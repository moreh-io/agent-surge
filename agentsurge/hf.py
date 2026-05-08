# SPDX-License-Identifier: MIT
"""HuggingFace dataset conversion and loading utilities.

Converts RunResult / ReplaySession objects to HF-compatible parquet files,
and supports downloading workloads from HuggingFace via ``hf://`` URLs.

All heavy dependencies (pandas, pyarrow, huggingface_hub) are lazily imported
so that ``import agentsurge.hf`` does not pull them in at module load time.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from agentsurge.types.results import RunResult, SessionResult
    from agentsurge.types.trace import ReplaySession

_log = logging.getLogger(__name__)

HF_URL_PREFIX = "hf://"


# ── Helpers ─────────────────────────────────────────────────────


def is_hf_url(path: str) -> bool:
    """Check if *path* is a HuggingFace URL (``hf://owner/repo/...``)."""
    return path.startswith(HF_URL_PREFIX)


def parse_hf_url(url: str) -> tuple[str, str]:
    """Split ``hf://owner/repo/sub/path`` into ``(repo_id, sub_path)``.

    >>> parse_hf_url("hf://your-org/agentsurge-workloads/qwen3.5-27b/agent-heavy")
    ('your-org/agentsurge-workloads', 'qwen3.5-27b/agent-heavy')
    """
    stripped = url.removeprefix(HF_URL_PREFIX)
    parts = stripped.split("/", 2)
    if len(parts) < 2:
        raise ValueError(f"Invalid hf:// URL: {url!r}. Expected format: hf://owner/repo[/sub/path]")
    repo_id = f"{parts[0]}/{parts[1]}"
    sub_path = parts[2] if len(parts) > 2 else ""
    return repo_id, sub_path


def generate_run_id(result: RunResult) -> str:
    """Deterministic run ID from config + timing + session count."""
    sessions = cast("list[SessionResult]", result.sessions)
    payload = json.dumps(result.config, sort_keys=True, default=str)
    payload += f"|{result.total_elapsed_s}|{len(sessions)}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


# ── Result → Parquet ────────────────────────────────────────────


def _turns_rows(result: RunResult, run_id: str) -> list[dict]:
    """Flatten all TurnResults into dicts for turns.parquet."""
    rows: list[dict] = []
    for sess in cast("list[SessionResult]", result.sessions):
        for t in sess.turns:
            rows.append(
                {
                    "run_id": run_id,
                    "session_id": t.session_id,
                    "turn_index": t.turn_index,
                    "completed": t.completed,
                    "ttft_ms": t.ttft_ms,
                    "wall_ttft_ms": t.wall_ttft_ms,
                    "total_ms": t.total_ms,
                    "input_tokens": t.input_tokens,
                    "output_tokens": t.output_tokens,
                    "cached_tokens": t.cached_tokens,
                    "prompt_tokens_server": t.prompt_tokens_server,
                    "error": t.error,
                    "retry_count": t.retry_count,
                    "interrupted": t.interrupted,
                    "fan_out_count": t.fan_out_count,
                    "response_text": t.response_text,
                    "tool_calls": json.dumps(t.tool_calls) if t.tool_calls is not None else None,
                    "tool_calls_detail": json.dumps(t.tool_calls_detail)
                    if t.tool_calls_detail is not None
                    else None,
                    "tool_valid": t.tool_valid,
                }
            )
    return rows


def _sessions_rows(result: RunResult, run_id: str) -> list[dict]:
    """Flatten all SessionResults into dicts for sessions.parquet."""
    rows: list[dict] = []
    for s in cast("list[SessionResult]", result.sessions):
        rows.append(
            {
                "run_id": run_id,
                "session_id": s.session_id,
                "completed": s.completed,
                "n_turns": s.n_turns,
                "expected_turns": s.expected_turns,
                "total_ms": s.total_ms,
                "llm_ms": s.llm_ms,
                "metadata": json.dumps(s.metadata, default=str),
            }
        )
    return rows


def _run_meta(result: RunResult, run_id: str, lineage: dict | None = None) -> dict:
    """Build run_meta.json content."""
    sessions = cast("list[SessionResult]", result.sessions)
    backend_metrics = cast("dict[str, object]", result.backend_metrics)
    return {
        "run_id": run_id,
        "config": result.config,
        "total_elapsed_s": result.total_elapsed_s,
        "isl_total": result.isl_total,
        "osl_total": result.osl_total,
        "prefix_cache_hit_rate": result.prefix_cache_hit_rate,
        "isl_osl_ratio": result.isl_osl_ratio,
        "requests_per_s": result.requests_per_s,
        "n_sessions": len(sessions),
        "n_completed_sessions": sum(1 for s in sessions if s.completed),
        "backend_metrics": dict(backend_metrics),
        "lineage": lineage or {},
    }


def _source_counts(items: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        metadata = getattr(item, "metadata", {}) or {}
        source = metadata.get("source")
        if isinstance(source, str) and source:
            counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def _source_info(source: str) -> dict[str, str]:
    key = "swe-smith" if source.startswith("swe-smith") else source
    return _UPSTREAM_LICENSES.get(key, {"repo": source, "license": "Unknown"})


def _attribution_entries(sources: list[str]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for source in sources:
        info = _source_info(source)
        repo = info["repo"]
        entries.append(
            {
                "source": source,
                "hf_repo": repo,
                "license": info["license"],
                "url": f"https://huggingface.co/datasets/{repo}",
            }
        )
    return entries


def _license_text(
    *,
    license_id: str,
    attribution: list[dict[str, str]],
) -> str:
    lines = [
        "AgentSurge export license notes",
        "",
        "This export is local/private by default. This file does not grant",
        "redistribution rights and does not override any upstream dataset,",
        "repository, prompt, or model-output terms that may apply.",
        "",
        f"Dataset license metadata requested for hosting: {license_id}",
        "AgentSurge code license: MIT",
        "",
        "Upstream sources:",
    ]
    if attribution:
        for entry in attribution:
            lines.append(
                f"- {entry['source']}: {entry['hf_repo']} ({entry['license']}) {entry['url']}"
            )
    else:
        lines.append("- No upstream source metadata was available in the exported sessions.")
    lines.append("")
    return "\n".join(lines)


def _write_dataset_metadata(
    out: Path,
    *,
    repo_type: str,
    model_name: str,
    config_name: str,
    n_sessions: int,
    license_id: str,
    source_counts: dict[str, int],
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    sources = sorted(source_counts)
    attribution = _attribution_entries(sources)
    readme_path = out / "README.md"
    license_path = out / "LICENSE.txt"
    attribution_path = out / "ATTRIBUTION.json"

    readme_path.write_text(
        make_dataset_card(
            repo_type,
            model_name,
            config_name,
            license_id=license_id,
            upstream_sources=sources,
            n_sessions=n_sessions,
            extra_metadata={**(extra_metadata or {}), "source_counts": source_counts},
        ),
        encoding="utf-8",
    )
    license_path.write_text(
        _license_text(license_id=license_id, attribution=attribution),
        encoding="utf-8",
    )
    attribution_path.write_text(
        json.dumps(
            {
                "generated_by": "agentsurge",
                "repo_type": repo_type,
                "model_name": model_name,
                "config_name": config_name,
                "n_sessions": n_sessions,
                "source_counts": source_counts,
                "upstream_sources": attribution,
                "sharing_note": (
                    "Local/private export by default. Review upstream licenses and "
                    "content terms before publishing."
                ),
                "extra_metadata": extra_metadata or {},
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return {
        "readme": readme_path,
        "license": license_path,
        "attribution": attribution_path,
    }


def run_result_to_parquet(
    result: RunResult,
    out_dir: str | Path,
    *,
    run_id: str | None = None,
    lineage: dict | None = None,
    model_name: str | None = None,
    config_name: str | None = None,
    license_id: str = "other",
) -> dict[str, Path]:
    """Write turns.parquet, sessions.parquet, run_meta.json, and metadata files.

    Returns a dict mapping file type to its path.
    """
    import pandas as pd

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rid = run_id or generate_run_id(result)

    # turns.parquet
    turns_df = pd.DataFrame(_turns_rows(result, rid))
    # Use nullable int types for columns that can be None
    for col in ("cached_tokens", "prompt_tokens_server"):
        if col in turns_df.columns:
            turns_df[col] = turns_df[col].astype("Int32")
    turns_path = out / "turns.parquet"
    turns_df.to_parquet(turns_path, index=False, engine="pyarrow")

    # sessions.parquet
    sessions_df = pd.DataFrame(_sessions_rows(result, rid))
    sessions_path = out / "sessions.parquet"
    sessions_df.to_parquet(sessions_path, index=False, engine="pyarrow")

    # run_meta.json
    meta_path = out / "run_meta.json"
    meta_path.write_text(
        json.dumps(_run_meta(result, rid, lineage), indent=2, default=str),
        encoding="utf-8",
    )

    source_counts = _source_counts(cast("list[SessionResult]", result.sessions))
    paths = {"turns": turns_path, "sessions": sessions_path, "meta": meta_path}
    paths.update(
        _write_dataset_metadata(
            out,
            repo_type="results",
            model_name=model_name or str(result.config.get("model", "unknown-model")),
            config_name=config_name or "local-results-export",
            n_sessions=len(sessions_df),
            license_id=license_id,
            source_counts=source_counts,
            extra_metadata={"run_id": rid},
        )
    )
    _log.info(
        "Wrote HF result dataset to %s (%d turns, %d sessions)",
        out,
        len(turns_df),
        len(sessions_df),
    )
    return paths


# ── Workload → Parquet ──────────────────────────────────────────


def _workload_rows(sessions: list[ReplaySession]) -> list[dict]:
    """Convert ReplaySessions to dicts for workload.parquet."""
    rows: list[dict] = []
    for s in sessions:
        rows.append(
            {
                "session_id": s.session_id,
                "n_turns": s.n_turns,
                "estimated_tokens": s.estimated_tokens,
                "fan_out": s.fan_out,
                "lineage_id": s.lineage_id,
                "parent_session_id": s.parent_session_id,
                "branch_id": s.branch_id,
                "branch_depth": s.branch_depth,
                "metadata": json.dumps(s.metadata, default=str),
                "turn_messages": json.dumps(s.turn_messages),
                "pending_user_messages": json.dumps(s.pending_user_messages or []),
            }
        )
    return rows


def workload_to_parquet(
    sessions: list[ReplaySession],
    out_dir: str | Path,
    *,
    model_name: str = "unknown-model",
    config_name: str = "local-workload-export",
    license_id: str = "other",
) -> dict[str, Path]:
    """Write workload.parquet and metadata files from a list of ReplaySessions.

    Returns a dict mapping file type to its path.
    """
    import pandas as pd

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(_workload_rows(sessions))
    wl_path = out / "workload.parquet"
    df.to_parquet(wl_path, index=False, engine="pyarrow")

    paths = {"workload": wl_path}
    paths.update(
        _write_dataset_metadata(
            out,
            repo_type="workloads",
            model_name=model_name,
            config_name=config_name,
            n_sessions=len(df),
            license_id=license_id,
            source_counts=_source_counts(sessions),
        )
    )
    _log.info("Wrote HF workload dataset to %s (%d sessions)", out, len(df))
    return paths


# ── DatasetCard ─────────────────────────────────────────────────


_UPSTREAM_LICENSES: dict[str, dict[str, str]] = {
    "openhands": {
        "repo": "nebius/SWE-rebench-openhands-trajectories",
        "license": "CC-BY-4.0",
    },
    "swe-bench-verified": {
        "repo": "princeton-nlp/SWE-bench_Verified",
        "license": "MIT",
    },
    "swe-smith": {
        "repo": "SWE-bench/SWE-smith-trajectories",
        "license": "MIT",
    },
    "abc-bench": {
        "repo": "OpenMOSS-Team/ABC-Bench",
        "license": "ODC-BY",
    },
    "swe-evo": {
        "repo": "Fsoft-AIC/SWE-EVO",
        "license": "Apache-2.0",
    },
    "featurebench": {
        "repo": "LiberCoders/FeatureBench",
        "license": "MIT",
    },
}


# ── HF Download ─────────────────────────────────────────────────


def download_workload_from_hf(
    hf_url: str,
    cache_dir: str | None = None,
) -> list[ReplaySession]:
    """Download a workload from HuggingFace and return ReplaySessions.

    Tries parquet format first, falls back to JSON.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    repo_id, sub_path = parse_hf_url(hf_url)

    # Try parquet first
    try:
        parquet_file = sub_path + "/workload.parquet" if sub_path else "workload.parquet"
        local_path = hf_hub_download(
            repo_id,
            parquet_file,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        return _parquet_to_sessions(local_path)
    except (FileNotFoundError, EntryNotFoundError) as e:
        _log.debug("Parquet not found at %s, trying JSON fallback: %s", parquet_file, e)

    # Fallback: JSON workload
    json_file = sub_path + "/workload.json" if sub_path else "workload.json"
    local_path = hf_hub_download(
        repo_id,
        json_file,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    from agentsurge.io import load_workload_sessions

    return load_workload_sessions(local_path)


def _parquet_to_sessions(path: str | Path) -> list[ReplaySession]:
    """Convert a workload parquet file back to ReplaySessions."""
    import pandas as pd

    from agentsurge.types.trace import ReplaySession

    df = pd.read_parquet(path)
    has_pending = "pending_user_messages" in df.columns
    sessions: list[ReplaySession] = []
    for _, row in df.iterrows():
        pending = (
            json.loads(row["pending_user_messages"])
            if has_pending and row["pending_user_messages"]
            else []
        )
        sessions.append(
            ReplaySession(
                session_id=row["session_id"],
                turn_messages=json.loads(row["turn_messages"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                fan_out=int(row["fan_out"]),
                lineage_id=row.get("lineage_id", "") or "",
                parent_session_id=row.get("parent_session_id", "") or "",
                branch_id=int(row.get("branch_id", 0)),
                branch_depth=int(row.get("branch_depth", 0)),
                pending_user_messages=pending,
            )
        )
    return sessions


def make_dataset_card(
    repo_type: str,
    model_name: str,
    config_name: str,
    *,
    license_id: str = "other",
    upstream_sources: list[str] | None = None,
    n_sessions: int = 0,
    extra_metadata: dict[str, Any] | None = None,
) -> str:
    """Generate a HuggingFace DatasetCard (README.md) string.

    Parameters
    ----------
    repo_type : "results" or "workloads"
    model_name : e.g. "Qwen/Qwen3.5-27B"
    config_name : e.g. "qwen3.5-27b/agent-heavy"
    license_id : SPDX identifier (default "other")
    upstream_sources : list of source keys (e.g. ["openhands", "swe-bench-verified"])
    """
    pretty_type = "Results" if repo_type == "results" else "Workloads"

    # Attribution table
    attr_lines = ""
    if upstream_sources:
        attr_lines = "| Source | HF Repo | License |\n|--------|---------|--------|\n"
        for src in upstream_sources:
            info = _source_info(src)
            attr_lines += f"| {src} | [{info['repo']}](https://huggingface.co/datasets/{info['repo']}) | {info['license']} |\n"

    extra_lines = ""
    if extra_metadata:
        source_counts = extra_metadata.get("source_counts")
        if source_counts:
            extra_lines += "\n- **Source counts**: "
            extra_lines += ", ".join(f"{k}={v}" for k, v in sorted(source_counts.items()))
            extra_lines += "\n"

    card = f"""---
license: {license_id}
task_categories:
  - text-generation
tags:
  - agentsurge
  - kv-cache
  - llm-serving
  - agent-workloads
  - benchmark
language:
  - en
pretty_name: "agentsurge {pretty_type} - {model_name}"
---

# agentsurge {pretty_type}

Local/private AgentSurge {pretty_type.lower()} export for LLM serving benchmark
analysis and reproduction. It was generated by
[agentsurge](https://github.com/moreh-io/agent-surge).

## Dataset Description

- **Model**: {model_name}
- **Configuration**: {config_name}
- **Sessions**: {n_sessions}
{extra_lines}

## Sharing Status

This export is private/local by default. Review the upstream sources, generated
content, and applicable terms before publishing it as a public dataset. The
license metadata below does not override upstream data licenses or prove that
derived artifacts are redistributable.

## Usage

```bash
# Run benchmark with this workload
agentsurge run --workload-file hf://your-org/agentsurge-workloads/{config_name}
```

```python
# Load with HuggingFace datasets
from datasets import load_dataset
ds = load_dataset("parquet", data_files="turns.parquet")
```
"""

    if attr_lines:
        card += f"""
## Upstream Sources & Attribution

{attr_lines}
"""
    else:
        card += """
## Upstream Sources & Attribution

No upstream source metadata was available in the exported sessions.
"""

    card += f"""
## License Notes

License metadata requested for hosting: `{license_id}`.
AgentSurge code is MIT licensed; exported records may include upstream workload
content and model responses governed by separate terms.
"""
    return card
