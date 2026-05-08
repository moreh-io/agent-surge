# SPDX-License-Identifier: MIT
"""SLO comparison helpers -- config profile loading and result formatting.

Small utility module used by ``agentsurge slo --configs`` to keep the main
slo.py lean.  No CLI argument parsing lives here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from agentsurge.types import ConfigProfile, SloComparisonResult


def load_config_profile(path: str) -> ConfigProfile:
    """Load a :class:`ConfigProfile` from a YAML file.

    Expected YAML format::

        name: TP8-APC
        vllm_url: http://localhost:8000
        model: Qwen/Qwen3.5-27B
        description: "TP=8 with APC enabled"
        overrides:
          max_model_len: 32768
          extra_body:
            chat_template_kwargs: {enable_thinking: false}

    Parameters
    ----------
    path:
        Path to a config profile YAML file.

    Returns
    -------
    ConfigProfile

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required fields (``name``, ``vllm_url``, ``model``) are missing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config profile not found: {path}")

    with open(p) as f:
        data = yaml.safe_load(f) or {}

    missing = [k for k in ("name", "vllm_url", "model") if k not in data]
    if missing:
        raise ValueError(f"Config profile {path} missing required fields: {', '.join(missing)}")

    return ConfigProfile(
        name=data["name"],
        vllm_url=data["vllm_url"],
        model=data["model"],
        description=data.get("description", ""),
        overrides=data.get("overrides", {}),
    )


def format_interleaved_table(
    profile_results: list[dict],
    slo_ms: float,
) -> str:
    """Format interleaved comparison results as a human-readable table.

    Parameters
    ----------
    profile_results:
        List of dicts returned by ``_run_interleaved_slo_search``, one per
        workload profile.  Each dict has keys: ``profile_label``, ``slo_ms``,
        ``workload``, ``configs``, ``capacity_ratios``, ``pairwise_ratios``.
    slo_ms:
        The SLO target in ms (for the header).

    Returns
    -------
    str
        Multi-line table string suitable for stdout.
    """
    lines: list[str] = []
    lines.append(f"SLO Target: p95 TTFT < {slo_ms:.0f}ms  (interleaved comparison)")
    lines.append("")

    for pr in profile_results:
        label = pr["profile_label"]
        wl = pr["workload"]
        lines.append(
            f"Profile: {label}  "
            f"(turns={wl['n_turns']}, tok/turn={wl['synthetic_tokens_per_turn']}, "
            f"max_out={wl['max_tokens']})"
        )
        lines.append("")

        header = (
            f"  {'Config':<24} {'max_n':>6} {'TTFT@boundary':>14} "
            f"{'KV%@boundary':>13} {'vs baseline':>12}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        configs = pr["configs"]
        ratios = pr["capacity_ratios"]
        baseline_name = configs[0]["config_name"] if configs else ""

        for cfg in configs:
            name = cfg["config_name"]
            max_n = cfg["max_n"]
            boundary = cfg.get("boundary") or {}
            ttft = boundary.get("ttft_p95_ms", 0)
            kv = boundary.get("kv_pct", 0)

            ttft_str = f"{ttft:,.1f}ms" if ttft else "N/A"
            kv_str = f"{kv:.1f}%" if kv else "N/A"

            ratio = ratios.get(name, 1.0)
            if name == baseline_name:
                ratio_str = "(baseline)"
            elif ratio == float("inf"):
                ratio_str = "inf"
            else:
                pct_diff = (ratio - 1.0) * 100
                sign = "+" if pct_diff >= 0 else ""
                ratio_str = f"{sign}{pct_diff:.0f}%"

            lines.append(f"  {name:<24} {max_n:>6} {ttft_str:>14} {kv_str:>13} {ratio_str:>12}")

        pairwise = pr.get("pairwise_ratios", {})
        if len(configs) == 2 and pairwise:
            lines.append("")
            c0 = configs[0]["config_name"]
            c1 = configs[1]["config_name"]
            n0 = configs[0]["max_n"]
            n1 = configs[1]["max_n"]
            if n1 > 0 and n0 > 0:
                if n0 > n1:
                    pct = ((n0 / n1) - 1) * 100
                    lines.append(f"  => {c0} capacity is {pct:.0f}% higher than {c1}")
                elif n1 > n0:
                    pct = ((n1 / n0) - 1) * 100
                    lines.append(f"  => {c1} capacity is {pct:.0f}% higher than {c0}")
                else:
                    lines.append(f"  => Both configs have equal capacity (n={n0})")
            elif n0 == 0 and n1 == 0:
                lines.append("  => Both configs breached at minimum load")
            elif n0 == 0:
                lines.append(f"  => {c0} breached at minimum; {c1} sustains n={n1}")
            else:
                lines.append(f"  => {c1} breached at minimum; {c0} sustains n={n0}")

        elif len(configs) > 2 and pairwise:
            lines.append("")
            lines.append("  Pairwise capacity ratios:")
            for key, ratio_val in sorted(pairwise.items()):
                if ratio_val == float("inf"):
                    lines.append(f"    {key}: inf")
                else:
                    lines.append(f"    {key}: {ratio_val:.3f}x")

        lines.append("")

    return "\n".join(lines)


def save_comparison_json(
    data: SloComparisonResult | list[dict],
    output_dir: str,
    tag: str,
    timestamp: str,
) -> str:
    """Save comparison result as JSON and return the output path.

    Accepts either:
      - A ``SloComparisonResult`` (legacy sequential mode)
      - A ``list[dict]`` of interleaved profile results (new mode)
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"slo_comparison_{tag}_{timestamp}.json")

    payload: dict[str, object]
    if isinstance(data, list):
        payload = {
            "mode": "interleaved",
            "tag": tag,
            "timestamp": timestamp,
            "profiles": data,
        }
    else:
        payload = {
            "mode": "sequential",
            "target_slo_ms": data.target_slo_ms,
            "tag": tag,
            "timestamp": timestamp,
            "configs": data.entries,
        }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return out_path


def _json_default(obj: object) -> str:
    """Handle non-serializable types (e.g. float('inf'))."""
    if isinstance(obj, float) and (obj == float("inf") or obj == float("-inf")):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
