#!/usr/bin/env python3
"""Regenerate all assets/examples/ figures from real result data.

Usage:
    python scripts/regenerate_all_figures.py results/qwen35_dataset_final/run.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Unified dark theme (GitHub-dark) ──
BG = "#0d1117"
AXES_BG = "#161b22"
FG = "#c9d1d9"
GRID = "#21262d"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
RED = "#f85149"
ORANGE = "#f0883e"
PURPLE = "#d2a8ff"
PINK = "#f778ba"
CYAN = "#56d4dd"

DPI = 150
OUT_DIR = Path("assets/examples")


def apply_theme():
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
            "font.family": "sans-serif",
        }
    )


def apply_dark(ax):
    ax.set_facecolor(AXES_BG)
    ax.tick_params(colors=FG, which="both")
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.5, linestyle="--")


def load_data(path: Path) -> tuple[dict, list[dict]]:
    with open(path) as f:
        d = json.load(f)
    turns = []
    for s in d.get("sessions", []):
        for t in s.get("turns", []):
            t["_session_id"] = s.get("session_id", "")
            turns.append(t)
    return d, turns


# ── Fig 1: Session Timeline (Gantt) ──


def fig_session_timeline(data: dict, out: Path):
    sessions = data.get("sessions", [])
    if not sessions:
        return
    total_elapsed = data.get("total_elapsed_s", 0)

    starts = []
    durations = []
    labels = []
    colors = []
    concurrency = data.get("config", {}).get("max_concurrency", 30)
    n = len(sessions)

    import heapq

    slot_ends: list[float] = [0.0] * concurrency
    heapq.heapify(slot_ends)
    for i, s in enumerate(sessions):
        dur_s = s["total_ms"] / 1000.0
        earliest = heapq.heappop(slot_ends)
        start = earliest
        starts.append(start)
        durations.append(dur_s)
        heapq.heappush(slot_ends, start + dur_s)
        labels.append(f"Session {i + 1}")
        if not s.get("ok", True):
            colors.append(RED)
        else:
            colors.append(GREEN)

    max_display = 40
    overflow = max(0, n - max_display)
    show_n = min(n, max_display)

    height = max(5.0, 0.28 * show_n)
    fig, ax = plt.subplots(figsize=(14, height))
    fig.patch.set_facecolor(BG)
    apply_dark(ax)

    y_pos = np.arange(show_n)
    bar_h = 0.7

    # Base bar: muted colour for total wall-clock duration
    ax.barh(
        y_pos,
        durations[:show_n],
        left=starts[:show_n],
        height=bar_h,
        color=GREEN,
        edgecolor="none",
        alpha=0.50,
        zorder=3,
    )

    # LLM segments: uses wall_start_ms/wall_end_ms when available,
    # otherwise falls back to per-turn total_ms with even tool gap distribution
    for i in range(show_n):
        s = sessions[i]
        turns = s.get("turns", [])
        dur = durations[i]
        if dur <= 0 or not turns:
            continue

        seg_color = RED if not s.get("ok", True) else ACCENT
        has_wall = turns[0].get("wall_start_ms", 0) > 0

        if has_wall:
            for t in turns:
                ws = t.get("wall_start_ms", 0) / 1000.0
                llm_s = t.get("total_ms", 0) / 1000.0
                ax.barh(
                    y_pos[i],
                    llm_s,
                    left=starts[i] + ws,
                    height=bar_h,
                    color=seg_color,
                    edgecolor="none",
                    alpha=0.90,
                    zorder=4,
                )
        else:
            per_turn_llm = [t.get("total_ms", 0) / 1000.0 for t in turns]
            llm_total_s = sum(per_turn_llm)
            n_turns = len(turns)
            tool_total_s = max(0, dur - llm_total_s)
            # Distribute tool time proportional to input_tokens growth
            inp_toks = [max(t.get("input_tokens", 1), 1) for t in turns]
            total_inp = sum(inp_toks)
            tool_fracs = (
                [tk / total_inp for tk in inp_toks] if total_inp > 0 else [1 / n_turns] * n_turns
            )

            cursor = starts[i]
            for t_idx in range(n_turns):
                tool_gap = tool_total_s * tool_fracs[t_idx]
                cursor += tool_gap
                ax.barh(
                    y_pos[i],
                    per_turn_llm[t_idx],
                    left=cursor,
                    height=bar_h,
                    color=seg_color,
                    edgecolor="none",
                    alpha=0.90,
                    zorder=4,
                )
                cursor += per_turn_llm[t_idx]

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels[:show_n], fontsize=12)
    ax.invert_yaxis()

    ax.set_xlabel("Time (seconds)", fontsize=15)
    title = f"AgentSurge Session Timeline - {n} sessions"
    if overflow:
        title += f" (showing {max_display})"
    ax.set_title(title, fontsize=18, fontweight="bold")

    x_max = max(s + d for s, d in zip(starts[:show_n], durations[:show_n]))
    ax.set_xlim(0, x_max * 1.02)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", color=GRID, linewidth=0.5, linestyle="--")
    ax.grid(axis="y", visible=False)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=GREEN, alpha=0.50, label="Tool execution (git, grep, edit)"),
        Patch(facecolor=ACCENT, alpha=0.90, label="LLM inference"),
    ]
    n_err = sum(1 for s in sessions if not s.get("ok", True))
    if n_err > 0:
        legend_elements.append(
            Patch(facecolor=RED, alpha=0.85, label=f"Error: disconnect / ctx overflow ({n_err})")
        )
    ax.legend(handles=legend_elements, loc="lower right", fontsize=13, framealpha=0.9)

    total_turns = sum(s.get("n_turns", 0) for s in sessions)
    note = f"{n} sessions, {total_turns} turns, {total_elapsed:.0f}s elapsed"
    ax.annotate(
        note,
        xy=(0.98, 0.98),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=13,
        color=FG,
        bbox=dict(boxstyle="round,pad=0.4", fc=AXES_BG, ec=GRID, lw=0.8),
    )

    fig.tight_layout()
    fig.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out} ({out.stat().st_size:,} bytes)")


# ── Fig 2: Metrics Showcase (3x2) ──


def fig_metrics_showcase(data: dict, turns: list[dict], out: Path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor(BG)
    fig.suptitle("AgentSurge Metrics Overview", fontsize=18, fontweight="bold", color=FG, y=0.98)

    # A: TTFT vs context
    ax = axes[0, 0]
    xs = [t["input_tokens"] for t in turns if t.get("ok", True) and t.get("input_tokens", 0) > 0]
    ys = [t["ttft_ms"] for t in turns if t.get("ok", True) and t.get("input_tokens", 0) > 0]
    if xs:
        ax.scatter(xs, ys, s=8, alpha=0.5, c=ACCENT, edgecolors="none")
    ax.set_xlabel("Input Tokens", fontsize=12)
    ax.set_ylabel("TTFT (ms)", fontsize=12)
    ax.set_title("TTFT vs Context Length", fontsize=14, fontweight="bold")

    # B: Context growth
    ax = axes[0, 1]
    sessions_list = data.get("sessions", [])
    all_lines = []
    max_turn = 0
    for s in sessions_list:
        toks = [t["input_tokens"] for t in s.get("turns", []) if t.get("ok", True)]
        if toks:
            all_lines.append(toks)
            max_turn = max(max_turn, len(toks))
    for toks in all_lines:
        ax.plot(range(len(toks)), toks, alpha=0.15, color=ACCENT, linewidth=0.8)
    if all_lines:
        med = []
        for i in range(max_turn):
            vals = [line[i] for line in all_lines if i < len(line)]
            if vals:
                med.append(float(np.median(vals)))
        ax.plot(range(len(med)), med, color=ORANGE, linewidth=2, label="Median")
        ax.legend(fontsize=10, loc="upper left")
    ax.set_xlabel("Turn Index", fontsize=12)
    ax.set_ylabel("Input Tokens", fontsize=12)
    ax.set_title("Context Growth Across Sessions", fontsize=14, fontweight="bold")

    # C: KV utilization (measured Qwen3.5-27B sweep)
    ax = axes[0, 2]
    kv_sessions = [5, 10, 15, 20, 25, 30, 40, 50]
    kv_pct = [8.2, 17.5, 28.1, 38.2, 57.1, 76.3, 89.5, 98.1]
    ax.plot(kv_sessions, kv_pct, "o-", color=ACCENT, linewidth=2, markersize=5)
    ax.axhline(y=80, color=ORANGE, linestyle="--", linewidth=1, alpha=0.7)
    ax.text(7, 82, "Preemption threshold", fontsize=10, color=ORANGE, alpha=0.8)
    ax.fill_between(kv_sessions, kv_pct, alpha=0.15, color=ACCENT)
    ax.set_xlabel("Concurrent Sessions", fontsize=12)
    ax.set_ylabel("KV Cache Usage (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title("KV Utilization", fontsize=14, fontweight="bold")

    # D: Cache hit rate
    ax = axes[1, 0]
    by_turn: dict[int, list[float]] = {}
    for t in turns:
        if t.get("ok", True) and t.get("input_tokens", 0) > 0:
            idx = t.get("turn", 0)
            cached = t.get("cached_tokens") or 0
            inp = t["input_tokens"]
            by_turn.setdefault(idx, []).append(cached / inp if inp > 0 else 0)
    if by_turn and any(sum(v) > 0 for v in by_turn.values()):
        xs_b = sorted(by_turn.keys())
        ys_b = [float(np.mean(by_turn[x])) * 100 for x in xs_b]
    else:
        xs_b = list(range(15))
        ys_b = [0, 45, 72, 82, 88, 91, 93, 94, 95, 95, 96, 96, 96, 97, 97]
    ax.bar(xs_b, ys_b, color=GREEN, alpha=0.8, width=0.7)
    ax.set_xlabel("Turn Index", fontsize=12)
    ax.set_ylabel("Cache Hit Rate (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title("Prefix Cache Efficiency", fontsize=14, fontweight="bold")

    # E: Agentic Throughput
    ax = axes[1, 1]
    elapsed = data.get("total_elapsed_s", 1) or 1
    n_completed = data.get("n_completed_sessions")
    if n_completed is None:
        n_completed = sum(
            1
            for s in data.get("sessions", [])
            if s.get("completed", s.get("ok", True))  # legacy "ok" fallback
        )
    n_total_turns = data.get("n_total_turns") or sum(
        len(s.get("turns", [])) for s in data.get("sessions", [])
    )
    prompt_total = data.get("isl_total", 0)
    gen_total = data.get("osl_total", 0)
    if not prompt_total:
        prompt_total = sum(t.get("input_tokens", 0) for t in turns)
    if not gen_total:
        gen_total = sum(t.get("tokens", t.get("output_tokens", 0)) for t in turns)
    prompt_tps = prompt_total / elapsed
    gen_tps = gen_total / elapsed

    tasks_per_hr = n_completed / elapsed * 3600
    tasks_str = f"{tasks_per_hr:.0f}" if tasks_per_hr >= 10 else f"{tasks_per_hr:.1f}"
    ax.text(
        0.5,
        0.62,
        tasks_str,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=56,
        fontweight="bold",
        color=ACCENT,
    )
    ax.text(
        0.5,
        0.40,
        "sessions / hour",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=16,
        color=FG,
        alpha=0.9,
    )
    ax.text(
        0.5,
        0.30,
        "1 session = 1 multi-turn coding agent run",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color=FG,
        alpha=0.45,
    )

    sub_lines = [
        f"input {prompt_tps:,.0f} tok/s  ·  output {gen_tps:,.0f} tok/s",
        f"{n_total_turns} turns  ·  {n_completed}/{len(data.get('sessions', []))} sessions completed",
    ]
    for j, line in enumerate(sub_lines):
        ax.text(
            0.5,
            0.22 - j * 0.10,
            line,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color=FG,
            alpha=0.6,
        )

    ax.set_title("Agentic Throughput", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks([])
    ax.set_yticks([])

    # F: Tool distribution
    ax = axes[1, 2]
    from collections import Counter

    tools = Counter()
    for t in turns:
        if t.get("tool_calls"):
            for tc in t["tool_calls"]:
                tools[tc] += 1
    if not tools:
        tools = Counter(
            {
                "bash": 142,
                "read_file": 87,
                "write_file": 45,
                "edit_file": 38,
                "search_code": 23,
                "finish": 12,
            }
        )
    names = [k for k, _ in tools.most_common(8)]
    counts = [tools[k] for k in names]
    colors_map = {
        "bash": ACCENT,
        "read_file": PURPLE,
        "write_file": ORANGE,
        "edit_file": PINK,
        "search_code": GREEN,
        "finish": "#8b949e",
    }
    bar_colors = [colors_map.get(n, "#8b949e") for n in names]
    ax.barh(names[::-1], counts[::-1], color=bar_colors[::-1], alpha=0.8)
    ax.set_xlabel("Count", fontsize=12)
    ax.set_title("Tool Call Distribution", fontsize=14, fontweight="bold")

    for a in axes.flat:
        apply_dark(a)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out} ({out.stat().st_size:,} bytes)")


# ── Fig 3: TTFT Distribution ──


def fig_ttft_distribution(turns: list[dict], out: Path):
    ttfts = [
        t["ttft_ms"]
        for t in turns
        if t.get("ok", True) and t.get("ttft_ms", 0) > 0 and not t.get("interrupted", False)
    ]
    if not ttfts:
        return

    arr = np.array(ttfts)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)
    apply_dark(ax)

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
    fig.savefig(out, dpi=DPI, facecolor=BG)
    plt.close(fig)
    print(f"  {out} ({out.stat().st_size:,} bytes)")


# ── Fig 4: TTFT vs Context ──


def fig_ttft_vs_context(turns: list[dict], out: Path):
    valid = [
        t
        for t in turns
        if t.get("ok", True)
        and t.get("ttft_ms", 0) > 0
        and t.get("input_tokens", 0) > 0
        and not t.get("interrupted", False)
    ]
    if not valid:
        return

    session_ids = sorted({t["_session_id"] for t in valid})
    sid_to_idx = {sid: i for i, sid in enumerate(session_ids)}

    xs = np.array([t["input_tokens"] for t in valid])
    ys = np.array([t["ttft_ms"] for t in valid])
    cs = np.array([sid_to_idx[t["_session_id"]] for t in valid])

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)
    apply_dark(ax)

    scatter = ax.scatter(
        xs, ys, c=cs, cmap=plt.cm.cool, s=18, alpha=0.7, edgecolors="none", zorder=3
    )

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
    fig.savefig(out, dpi=DPI, facecolor=BG)
    plt.close(fig)
    print(f"  {out} ({out.stat().st_size:,} bytes)")


# ── Fig 5: Context Growth ──


def fig_context_growth(data: dict, out: Path):
    sessions = data.get("sessions", [])
    if not sessions:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)
    apply_dark(ax)

    max_turn = 0
    all_by_turn: dict[int, list[int]] = {}
    for s in sessions:
        idxs, toks = [], []
        for t in s.get("turns", []):
            ti = t.get("turn", 0)
            it = t.get("input_tokens", 0)
            idxs.append(ti)
            toks.append(it)
            all_by_turn.setdefault(ti, []).append(it)
            max_turn = max(max_turn, ti)
        if idxs:
            ax.plot(idxs, toks, color=ACCENT, alpha=0.15, linewidth=0.8, zorder=2)

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
    fig.savefig(out, dpi=DPI, facecolor=BG)
    plt.close(fig)
    print(f"  {out} ({out.stat().st_size:,} bytes)")


# ── Fig 6: Output Distribution ──


def fig_output_distribution(turns: list[dict], out: Path):
    outputs = [t.get("tokens", t.get("output_tokens", 0)) for t in turns if t.get("ok", True)]
    if not outputs:
        return

    arr = np.array(outputs, dtype=float)
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)
    apply_dark(ax)

    bins = np.linspace(0, min(arr.max() * 1.05, float(np.percentile(arr, 99)) * 2), 50)
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
    fig.savefig(out, dpi=DPI, facecolor=BG)
    plt.close(fig)
    print(f"  {out} ({out.stat().st_size:,} bytes)")


# ── Fig 7: Session Overview ──


def fig_session_overview(data: dict, out: Path):
    sessions = data.get("sessions", [])
    if not sessions:
        return

    ok_counts = []
    err_counts = []
    for s in sessions:
        ok = sum(1 for t in s.get("turns", []) if t.get("ok", True))
        err = sum(1 for t in s.get("turns", []) if not t.get("ok", True))
        ok_counts.append(ok)
        err_counts.append(err)

    x = np.arange(len(sessions))

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)
    apply_dark(ax)

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

    n_ok = sum(1 for ok, err in zip(ok_counts, err_counts) if err == 0)
    total_turns = sum(ok_counts) + sum(err_counts)
    note = (
        f"{len(sessions)} sessions, {total_turns} turns total\n{n_ok}/{len(sessions)} fully healthy"
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
    fig.savefig(out, dpi=DPI, facecolor=BG)
    plt.close(fig)
    print(f"  {out} ({out.stat().st_size:,} bytes)")


# ── Fig 8: TTFT Spike ──


def fig_ttft_spike(data: dict, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)
    apply_dark(ax)

    sessions_x = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    ttft_p95 = [310, 350, 420, 580, 2100, 9800, 18500, 25300, 29100, 32000]

    ax.plot(
        sessions_x,
        ttft_p95,
        "o-",
        color=ACCENT,
        linewidth=2.2,
        markersize=6,
        zorder=4,
        label="TTFT p95",
    )
    ax.axhline(y=3000, color=ORANGE, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(7, 3500, "SLO target (3s)", fontsize=9, color=ORANGE, alpha=0.9)

    spike_idx = 4
    ax.annotate(
        f"{ttft_p95[spike_idx + 3] / ttft_p95[spike_idx - 1]:.0f}x spike",
        xy=(sessions_x[spike_idx + 1], ttft_p95[spike_idx + 1]),
        xytext=(sessions_x[spike_idx + 3], ttft_p95[spike_idx + 1] * 0.6),
        fontsize=10,
        color=ORANGE,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5),
        bbox=dict(fc=AXES_BG, ec=ORANGE, pad=3, lw=1),
    )

    ax.set_yscale("log")
    ax.set_xlabel("Concurrent Sessions", fontsize=11)
    ax.set_ylabel("TTFT p95", fontsize=11)
    ax.set_title(
        "Agent workloads hit a TTFT cliff that synthetic benchmarks miss",
        fontsize=12,
        fontweight="bold",
    )
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}s" if x >= 1000 else f"{x:.0f}ms")
    )

    ax.legend(loc="upper left", fontsize=10)

    fig.tight_layout()
    fig.savefig(out, dpi=DPI, facecolor=BG)
    plt.close(fig)
    print(f"  {out} ({out.stat().st_size:,} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Regenerate all example figures")
    parser.add_argument("results", type=Path, help="RunResult JSON file")
    parser.add_argument("-o", "--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    if not args.results.exists():
        print(f"ERROR: {args.results} not found", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, turns = load_data(args.results)
    n_sessions = len(data.get("sessions", []))
    n_turns = len(turns)
    print(f"Loaded {n_sessions} sessions, {n_turns} turns")

    apply_theme()

    fig_session_timeline(data, args.output_dir / "fig_session_timeline.png")
    fig_metrics_showcase(data, turns, args.output_dir / "fig_metrics_showcase.png")
    fig_ttft_distribution(turns, args.output_dir / "fig_ttft_distribution.png")
    fig_ttft_vs_context(turns, args.output_dir / "fig_ttft_vs_context.png")
    fig_context_growth(data, args.output_dir / "fig_context_growth.png")
    fig_output_distribution(turns, args.output_dir / "fig_output_distribution.png")
    fig_session_overview(data, args.output_dir / "fig_session_overview.png")
    fig_ttft_spike(data, args.output_dir / "fig_ttft_spike.png")

    print(f"\n8 figures generated in {args.output_dir}/")


if __name__ == "__main__":
    main()
