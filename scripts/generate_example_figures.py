#!/usr/bin/env python3
"""Generate 5 example figures from AgentSurge RunResult JSON files.

Usage:
    uv run python scripts/generate_example_figures.py results/qwen35_full/run_*.json -o results/examples/

Outputs:
    fig_ttft_distribution.png   -- TTFT p50/p95/p99 histogram
    fig_ttft_vs_context.png     -- TTFT vs input_tokens scatter
    fig_context_growth.png      -- Input tokens per turn across sessions
    fig_output_distribution.png -- Output token distribution histogram
    fig_session_overview.png    -- Session-level stacked bar summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Dark theme colours (GitHub-dark inspired, matching generate_demo_plots.py)
# ---------------------------------------------------------------------------
BG = "#0d1117"
FG = "#c9d1d9"
GRID = "#21262d"
AXES_BG = "#161b22"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
RED = "#f85149"
ORANGE = "#f0883e"
PURPLE = "#d2a8ff"

DPI = 150
FIGSIZE = (8, 5)


def _apply_theme() -> None:
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": AXES_BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": FG,
            "xtick.color": FG,
            "ytick.color": FG,
            "text.color": FG,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.linestyle": "--",
            "legend.facecolor": BG,
            "legend.edgecolor": GRID,
            "legend.labelcolor": FG,
        }
    )


def _apply_dark(ax: plt.Axes) -> None:
    ax.set_facecolor(AXES_BG)
    ax.tick_params(colors=FG, which="both")
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.5, linestyle="--")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_runs(paths: list[Path]) -> dict:
    """Load and merge one or more RunResult JSON files into a unified dict."""
    all_sessions: list[dict] = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        all_sessions.extend(data.get("sessions", []))
    return {"sessions": all_sessions}


def _collect_turns(data: dict) -> list[dict]:
    turns: list[dict] = []
    for s in data["sessions"]:
        for t in s.get("turns", []):
            t["_session_id"] = s.get("session_id", "")
            turns.append(t)
    return turns


# ---------------------------------------------------------------------------
# Figure 1 -- TTFT distribution with percentile lines
# ---------------------------------------------------------------------------


def fig_ttft_distribution(data: dict, out_dir: Path) -> Path:
    turns = _collect_turns(data)
    ttfts = [
        t["ttft_ms"]
        for t in turns
        if t.get("ok", True) and t.get("ttft_ms", 0) > 0 and not t.get("interrupted", False)
    ]

    if not ttfts:
        print("  SKIP fig_ttft_distribution.png (no valid TTFT values)")
        return out_dir / "fig_ttft_distribution.png"

    arr = np.array(ttfts)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor(BG)
    _apply_dark(ax)

    bins = np.linspace(0, min(arr.max() * 1.05, p99 * 2.5), 50)
    ax.hist(arr, bins=bins, color=ACCENT, alpha=0.8, edgecolor=BG, linewidth=0.5, zorder=3)

    ymax = ax.get_ylim()[1]
    for pval, label, color, y_frac in [
        (p50, "p50", FG, 0.92),
        (p95, "p95", ORANGE, 0.75),
        (p99, "p99", RED, 0.58),
    ]:
        ax.axvline(pval, color=color, linestyle="--", linewidth=1.4, zorder=4)
        ax.text(
            pval + arr.max() * 0.01,
            ymax * y_frac,
            f"{label}\n{pval:,.0f} ms",
            ha="left",
            va="top",
            fontsize=9,
            color=color,
            bbox=dict(fc=AXES_BG, ec=GRID, pad=2, lw=0.6),
        )

    ax.set_xlabel("TTFT (ms)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("TTFT Distribution", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    note = f"n={len(ttfts):,}  p50={p50:,.0f}  p95={p95:,.0f}  p99={p99:,.0f} ms"
    ax.annotate(
        note,
        xy=(0.98, 0.98),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
        color=FG,
        bbox=dict(boxstyle="round,pad=0.4", fc=AXES_BG, ec=GRID, lw=0.8),
    )

    fig.tight_layout()
    out = out_dir / "fig_ttft_distribution.png"
    fig.savefig(out, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2 -- TTFT vs context length scatter
# ---------------------------------------------------------------------------


def fig_ttft_vs_context(data: dict, out_dir: Path) -> Path:
    turns = _collect_turns(data)
    valid = [
        t
        for t in turns
        if t.get("ok", True)
        and t.get("ttft_ms", 0) > 0
        and t.get("input_tokens", 0) > 0
        and not t.get("interrupted", False)
    ]

    if not valid:
        print("  SKIP fig_ttft_vs_context.png (no valid turns)")
        return out_dir / "fig_ttft_vs_context.png"

    session_ids = sorted({t["_session_id"] for t in valid})
    sid_to_idx = {sid: i for i, sid in enumerate(session_ids)}

    xs = np.array([t["input_tokens"] for t in valid])
    ys = np.array([t["ttft_ms"] for t in valid])
    cs = np.array([sid_to_idx[t["_session_id"]] for t in valid])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor(BG)
    _apply_dark(ax)

    cmap = plt.cm.cool
    scatter = ax.scatter(xs, ys, c=cs, cmap=cmap, s=18, alpha=0.7, edgecolors="none", zorder=3)

    ax.set_xlabel("Input Tokens", fontsize=11)
    ax.set_ylabel("TTFT (ms)", fontsize=11)
    ax.set_title("TTFT vs Context Length", fontsize=14, fontweight="bold")

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Session Index", fontsize=9, color=FG)
    cbar.ax.yaxis.set_tick_params(color=FG)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=FG)

    note = f"{len(valid):,} turns across {len(session_ids)} sessions"
    ax.annotate(
        note,
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=9,
        color=FG,
        bbox=dict(boxstyle="round,pad=0.4", fc=AXES_BG, ec=GRID, lw=0.8),
    )

    fig.tight_layout()
    out = out_dir / "fig_ttft_vs_context.png"
    fig.savefig(out, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3 -- Context growth per turn
# ---------------------------------------------------------------------------


def fig_context_growth(data: dict, out_dir: Path) -> Path:
    sessions = data.get("sessions", [])
    if not sessions:
        print("  SKIP fig_context_growth.png (no sessions)")
        return out_dir / "fig_context_growth.png"

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor(BG)
    _apply_dark(ax)

    max_turn = 0
    all_by_turn: dict[int, list[int]] = {}

    for s in sessions:
        turn_indices = []
        token_counts = []
        for t in s.get("turns", []):
            ti = t.get("turn", 0)
            it = t.get("input_tokens", 0)
            turn_indices.append(ti)
            token_counts.append(it)
            all_by_turn.setdefault(ti, []).append(it)
            max_turn = max(max_turn, ti)

        if turn_indices:
            ax.plot(turn_indices, token_counts, color=ACCENT, alpha=0.15, linewidth=0.8, zorder=2)

    if all_by_turn:
        sorted_turns = sorted(all_by_turn.keys())
        medians = [float(np.median(all_by_turn[ti])) for ti in sorted_turns]
        ax.plot(sorted_turns, medians, color=ORANGE, linewidth=2.5, label="Median", zorder=4)

    ax.set_xlabel("Turn Index", fontsize=11)
    ax.set_ylabel("Input Tokens", fontsize=11)
    ax.set_title("Context Growth Across Sessions", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    note = f"{len(sessions)} sessions, up to {max_turn} turns"
    ax.annotate(
        note,
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=9,
        color=FG,
        bbox=dict(boxstyle="round,pad=0.4", fc=AXES_BG, ec=GRID, lw=0.8),
    )

    fig.tight_layout()
    out = out_dir / "fig_context_growth.png"
    fig.savefig(out, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 4 -- Output token distribution
# ---------------------------------------------------------------------------


def fig_output_distribution(data: dict, out_dir: Path) -> Path:
    turns = _collect_turns(data)
    outputs = [t.get("tokens", t.get("output_tokens", 0)) for t in turns if t.get("ok", True)]

    if not outputs:
        print("  SKIP fig_output_distribution.png (no output tokens)")
        return out_dir / "fig_output_distribution.png"

    arr = np.array(outputs, dtype=float)
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor(BG)
    _apply_dark(ax)

    bins = np.linspace(0, min(arr.max() * 1.05, np.percentile(arr, 99) * 2), 50)
    ax.hist(arr, bins=bins, color=PURPLE, alpha=0.8, edgecolor=BG, linewidth=0.5, zorder=3)

    ymax = ax.get_ylim()[1]
    ax.axvline(median_val, color=FG, linestyle="--", linewidth=1.4, zorder=4)
    ax.text(
        median_val + arr.max() * 0.01,
        ymax * 0.92,
        f"median\n{median_val:,.0f}",
        ha="left",
        va="top",
        fontsize=9,
        color=FG,
        bbox=dict(fc=AXES_BG, ec=GRID, pad=2, lw=0.6),
    )

    ax.axvline(mean_val, color=ORANGE, linestyle="--", linewidth=1.4, zorder=4)
    ax.text(
        mean_val + arr.max() * 0.01,
        ymax * 0.72,
        f"mean\n{mean_val:,.0f}",
        ha="left",
        va="top",
        fontsize=9,
        color=ORANGE,
        bbox=dict(fc=AXES_BG, ec=GRID, pad=2, lw=0.6),
    )

    ax.set_xlabel("Output Tokens", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Output Token Distribution", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    note = f"n={len(outputs):,}  mean={mean_val:,.0f}  median={median_val:,.0f}"
    ax.annotate(
        note,
        xy=(0.98, 0.98),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
        color=FG,
        bbox=dict(boxstyle="round,pad=0.4", fc=AXES_BG, ec=GRID, lw=0.8),
    )

    fig.tight_layout()
    out = out_dir / "fig_output_distribution.png"
    fig.savefig(out, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 5 -- Session overview (stacked bars: ok vs error turns)
# ---------------------------------------------------------------------------


def fig_session_overview(data: dict, out_dir: Path) -> Path:
    sessions = data.get("sessions", [])
    if not sessions:
        print("  SKIP fig_session_overview.png (no sessions)")
        return out_dir / "fig_session_overview.png"

    ok_counts = []
    err_counts = []
    for s in sessions:
        ok = sum(1 for t in s.get("turns", []) if t.get("ok", True))
        err = sum(1 for t in s.get("turns", []) if not t.get("ok", True))
        ok_counts.append(ok)
        err_counts.append(err)

    x = np.arange(len(sessions))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor(BG)
    _apply_dark(ax)

    ax.bar(x, ok_counts, color=GREEN, edgecolor=BG, linewidth=0.5, label="OK turns", zorder=3)
    ax.bar(
        x,
        err_counts,
        bottom=ok_counts,
        color=RED,
        edgecolor=BG,
        linewidth=0.5,
        label="Error turns",
        zorder=3,
    )

    ax.set_xlabel("Session Index", fontsize=11)
    ax.set_ylabel("Number of Turns", fontsize=11)
    ax.set_title("Session Overview", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    if len(sessions) > 40:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=20, integer=True))

    ax.legend(loc="upper right", fontsize=10)

    n_ok_sessions = sum(1 for ok, err in zip(ok_counts, err_counts) if err == 0)
    total_turns = sum(ok_counts) + sum(err_counts)
    note = (
        f"{len(sessions)} sessions, {total_turns} turns total\n"
        f"{n_ok_sessions}/{len(sessions)} fully healthy"
    )
    ax.annotate(
        note,
        xy=(0.98, 0.98),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
        color=FG,
        bbox=dict(boxstyle="round,pad=0.4", fc=AXES_BG, ec=GRID, lw=0.8),
    )

    fig.tight_layout()
    out = out_dir / "fig_session_overview.png"
    fig.savefig(out, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate example figures from AgentSurge RunResult JSON files."
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="One or more RunResult JSON files (supports glob)"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("results/examples"),
        help="Output directory for PNG figures (default: results/examples/)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    existing = [p for p in args.inputs if p.exists()]
    if not existing:
        print(f"ERROR: none of the input files exist: {args.inputs}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(existing)} file(s)...")
    data = load_runs(existing)
    n_sessions = len(data["sessions"])
    n_turns = sum(len(s.get("turns", [])) for s in data["sessions"])
    print(f"  {n_sessions} sessions, {n_turns} turns")

    _apply_theme()

    generators = [
        ("fig_ttft_distribution.png", fig_ttft_distribution),
        ("fig_ttft_vs_context.png", fig_ttft_vs_context),
        ("fig_context_growth.png", fig_context_growth),
        ("fig_output_distribution.png", fig_output_distribution),
        ("fig_session_overview.png", fig_session_overview),
    ]

    generated = []
    for name, func in generators:
        out_path = func(data, args.output_dir)
        if out_path.exists():
            size = out_path.stat().st_size
            print(f"  Saved {out_path}  ({size:,} bytes)")
            generated.append(name)

    print(f"\n{len(generated)}/5 figures generated in {args.output_dir}/")


if __name__ == "__main__":
    main()
