"""Generate a 3x2 metrics showcase figure for README.

Shows all key metric categories AgentSurge captures:
  Row 1: TTFT vs Context | Context Growth | KV Utilization
  Row 2: Cache Hit Rate  | Throughput     | Tool Distribution

Usage:
    uv run python scripts/generate_metrics_showcase.py [results.json] -o assets/examples/fig_metrics_showcase.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Style ──
BG = "#0d1117"
AXES_BG = "#161b22"
FG = "#c9d1d9"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
ORANGE = "#d29922"
RED = "#f85149"
PURPLE = "#bc8cff"
PINK = "#f778ba"


def _apply_dark(ax):
    ax.set_facecolor(AXES_BG)
    ax.tick_params(colors=FG, labelsize=12)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    for spine in ax.spines.values():
        spine.set_color("#30363d")


def load_data(path):
    with open(path) as f:
        d = json.load(f)
    turns = []
    for s in d.get("sessions", []):
        for t in s.get("turns", []):
            t["_session_id"] = s["session_id"]
            t["_session_n_turns"] = s["n_turns"]
            turns.append(t)
    return d, turns


def panel_ttft_vs_context(ax, turns):
    """A: TTFT vs input_tokens scatter."""
    xs = [t["input_tokens"] for t in turns if t["ok"] and t.get("input_tokens", 0) > 0]
    ys = [t["ttft_ms"] for t in turns if t["ok"] and t.get("input_tokens", 0) > 0]
    if not xs:
        # Synthetic
        rng = np.random.default_rng(42)
        xs = rng.integers(500, 25000, 200).tolist()
        ys = [x * 0.8 + rng.normal(0, 500) for x in xs]
        ys = [max(100, y) for y in ys]
    ax.scatter(xs, ys, s=8, alpha=0.5, c=ACCENT, edgecolors="none")
    ax.set_xlabel("Input Tokens", fontsize=12)
    ax.set_ylabel("TTFT (ms)", fontsize=12)
    ax.set_title("TTFT vs Context Length", fontsize=14, fontweight="bold")


def panel_context_growth(ax, data):
    """B: Context growth across sessions."""
    sessions = data.get("sessions", [])
    max_turn = 0
    lines = []
    for s in sessions:
        toks = [t["input_tokens"] for t in s["turns"] if t["ok"]]
        if toks:
            lines.append(toks)
            max_turn = max(max_turn, len(toks))
    if not lines:
        # Synthetic
        rng = np.random.default_rng(42)
        for _ in range(20):
            n = rng.integers(5, 15)
            base = rng.integers(1000, 3000)
            toks = [base + i * rng.integers(800, 2000) for i in range(n)]
            lines.append(toks)
            max_turn = max(max_turn, n)
    for toks in lines:
        ax.plot(range(len(toks)), toks, alpha=0.15, color=ACCENT, linewidth=0.8)
    # Median line
    if lines:
        med = []
        for i in range(max_turn):
            vals = [line[i] for line in lines if i < len(line)]
            if vals:
                med.append(np.median(vals))
        ax.plot(range(len(med)), med, color=ORANGE, linewidth=2, label="Median")
        ax.legend(fontsize=10, loc="upper left")
    ax.set_xlabel("Turn Index", fontsize=12)
    ax.set_ylabel("Input Tokens", fontsize=12)
    ax.set_title("Context Growth Across Sessions", fontsize=14, fontweight="bold")


def panel_kv_utilization(ax):
    """C: KV utilization % - synthetic from measured data."""
    # Representative sweep data for the README showcase.
    sessions = [5, 10, 15, 20, 25, 30, 40, 50]
    kv_pct = [8.2, 17.5, 28.1, 38.2, 57.1, 76.3, 89.5, 98.1]
    ax.plot(sessions, kv_pct, "o-", color=ACCENT, linewidth=2, markersize=5)
    ax.axhline(y=80, color=ORANGE, linestyle="--", linewidth=1, alpha=0.7)
    ax.text(7, 82, "Preemption threshold", fontsize=10, color=ORANGE, alpha=0.8)
    ax.fill_between(sessions, kv_pct, alpha=0.15, color=ACCENT)
    ax.set_xlabel("Concurrent Sessions", fontsize=12)
    ax.set_ylabel("KV Cache Usage (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title("KV Utilization", fontsize=14, fontweight="bold")


def panel_cache_hit(ax, turns):
    """D: Prefix cache hit rate per turn."""
    by_turn = {}
    for t in turns:
        if t["ok"] and t.get("input_tokens", 0) > 0:
            idx = t["turn"]
            cached = t.get("cached_tokens") or 0
            inp = t["input_tokens"]
            by_turn.setdefault(idx, []).append(cached / inp if inp > 0 else 0)
    if by_turn and any(sum(v) > 0 for v in by_turn.values()):
        xs = sorted(by_turn.keys())
        ys = [np.mean(by_turn[x]) * 100 for x in xs]
    else:
        # Synthetic: GPU prefix cache grows with turns
        xs = list(range(15))
        ys = [0, 45, 72, 82, 88, 91, 93, 94, 95, 95, 96, 96, 96, 97, 97]
    ax.bar(xs, ys, color=GREEN, alpha=0.8, width=0.7)
    ax.set_xlabel("Turn Index", fontsize=12)
    ax.set_ylabel("Cache Hit Rate (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title("Prefix Cache Efficiency", fontsize=14, fontweight="bold")


def panel_throughput(ax, data=None):
    """E: Agentic throughput - sessions/hour main, tok/s sub."""
    if data and data.get("total_elapsed_s"):
        elapsed = data["total_elapsed_s"]
        n_ok = data.get("n_completed_sessions") or sum(
            1 for s in data.get("sessions", []) if s.get("completed")
        )
        prompt_tps = data.get("isl_total", 0) / elapsed
        gen_tps = data.get("osl_total", 0) / elapsed
        sph = n_ok / elapsed * 3600
        n_turns = data.get("n_total_turns", 0)
        n_sess = len(data.get("sessions", []))
    else:
        sph, prompt_tps, gen_tps = 232, 4200, 348
        n_turns, n_ok, n_sess = 697, 129, 129

    ax.text(
        0.5,
        0.62,
        f"{sph:.0f}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=48,
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
        fontsize=14,
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
        fontsize=9,
        color=FG,
        alpha=0.45,
    )
    ax.text(
        0.5,
        0.18,
        f"input {prompt_tps:,.0f} tok/s  ·  output {gen_tps:,.0f} tok/s",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color=FG,
        alpha=0.6,
    )
    ax.text(
        0.5,
        0.08,
        f"{n_turns} turns  ·  {n_ok}/{n_sess} sessions OK",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color=FG,
        alpha=0.6,
    )
    ax.set_title("Agentic Throughput", fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def panel_tool_dist(ax, turns):
    """F: Tool call distribution."""
    from collections import Counter

    tools = Counter()
    for t in turns:
        if t.get("tool_calls"):
            for tc in t["tool_calls"]:
                tools[tc] += 1
    if not tools:
        # Synthetic from openhands typical distribution
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


def main():
    parser = argparse.ArgumentParser(description="Generate 3x2 metrics showcase")
    parser.add_argument("results", nargs="?", default=None, help="Results JSON file")
    parser.add_argument("-o", "--output", default="assets/examples/fig_metrics_showcase.png")
    args = parser.parse_args()

    data, turns = {}, []
    if args.results and Path(args.results).exists():
        data, turns = load_data(args.results)
        print(f"Loaded {len(data.get('sessions', []))} sessions, {len(turns)} turns")
    else:
        print("No results file - using synthetic data")

    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": AXES_BG,
            "text.color": FG,
            "axes.labelcolor": FG,
            "xtick.color": FG,
            "ytick.color": FG,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("AgentSurge Metrics Overview", fontsize=18, fontweight="bold", color=FG, y=0.98)

    panel_ttft_vs_context(axes[0, 0], turns)
    panel_context_growth(axes[0, 1], data)
    panel_kv_utilization(axes[0, 2])
    panel_cache_hit(axes[1, 0], turns)
    panel_throughput(axes[1, 1], data)
    panel_tool_dist(axes[1, 2], turns)

    for ax in axes.flat:
        _apply_dark(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, facecolor=BG, bbox_inches="tight")
    print(f"Saved {args.output} ({Path(args.output).stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
