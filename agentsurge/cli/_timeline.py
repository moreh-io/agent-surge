# SPDX-License-Identifier: MIT
"""Gantt-style session timeline visualisation (PNG and ASCII)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentsurge.types.results import RunResult


def save_session_timeline(result: RunResult, path: str) -> str:
    """Save a Gantt-style session timeline figure showing concurrent sessions.

    Each session is drawn as a horizontal bar from its ``start_time`` to its
    ``end_time``.  Bars are colour-coded: green (ok), red (failed).

    Returns *path* on success, or an empty string if no timestamp data is
    available.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sessions = [
        s
        for s in result.sessions
        if isinstance(getattr(s, "end_time", None), (int, float)) and s.end_time > 0
    ]
    if not sessions:
        return ""

    max_display = 30
    overflow = max(0, len(sessions) - max_display)
    display = sessions[:max_display]

    labels: list[str] = []
    starts: list[float] = []
    durations: list[float] = []
    colors: list[str] = []

    BG = "#0d1117"
    AXES_BG = "#161b22"
    FG = "#c9d1d9"
    GRID = "#21262d"
    GREEN = "#3fb950"
    RED = "#f85149"
    for i, s in enumerate(display):
        labels.append(f"Session {i + 1}")
        starts.append(s.start_time)
        durations.append(s.end_time - s.start_time)
        if not s.completed:
            colors.append(RED)
        else:
            colors.append(GREEN)

    height = max(5.0, 0.28 * len(display))
    fig, ax = plt.subplots(figsize=(14, height))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(AXES_BG)
    y_pos = range(len(display))

    ax.barh(
        y_pos,
        durations,
        left=starts,
        height=0.7,
        color=colors,
        edgecolor="none",
        alpha=0.85,
        zorder=3,
    )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8, color=FG)
    ax.invert_yaxis()

    ax.set_xlabel("Time (seconds)", fontsize=11, color=FG)
    title = f"AgentSurge Session Timeline - {len(sessions)} sessions"
    if overflow:
        title += f" (showing {max_display})"
    ax.set_title(title, fontsize=14, fontweight="bold", color=FG)
    ax.set_xlim(0, result.total_elapsed_s or max(s.end_time for s in sessions))

    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(left=False, colors=FG)
    ax.grid(axis="x", color=GRID, linewidth=0.5, linestyle="--")

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)
    return path


def save_session_timeline_ascii(result: RunResult) -> str:
    """Return a compact ASCII timeline string for terminal display."""
    sessions = [
        s
        for s in result.sessions
        if isinstance(getattr(s, "end_time", None), (int, float)) and s.end_time > 0
    ]
    if not sessions:
        return ""

    total = result.total_elapsed_s or max(s.end_time for s in sessions)
    width = 60

    def _time_label(t: float) -> str:
        if t >= 60:
            return f"{t / 60:.1f}m"
        return f"{t:.0f}s"

    header_ticks = 6
    tick_positions = [int(i * width / (header_ticks - 1)) for i in range(header_ticks)]
    tick_times = [total * i / (header_ticks - 1) for i in range(header_ticks)]

    label_line = "time  "
    for i, (pos, t) in enumerate(zip(tick_positions, tick_times)):
        lbl = _time_label(t)
        pad = pos - len(label_line) + len("time  ") if i > 0 else 0
        if pad < 0:
            pad = 1
        label_line += " " * pad + lbl
    ruler_chars = list(" " * (width + len("      ")))
    for pos in tick_positions:
        idx = pos + len("      ")
        if idx < len(ruler_chars):
            ruler_chars[idx] = "|"
    ruler_line = "".join(ruler_chars)

    max_display = 30
    overflow = max(0, len(sessions) - max_display)
    display = sessions[:max_display]

    max_id_len = max(len(s.session_id) for s in display)
    id_width = min(max_id_len, 6)

    lines: list[str] = []
    lines.append(f"Session Timeline ({len(sessions)} sessions, {total:.1f}s)")
    lines.append(label_line)
    lines.append(ruler_line)

    for s in display:
        sid = s.session_id[-id_width:]
        col_start = int(s.start_time / total * width) if total > 0 else 0
        col_end = int(s.end_time / total * width) if total > 0 else 0
        col_start = max(0, min(col_start, width - 1))
        col_end = max(col_start + 1, min(col_end, width))

        llm_frac = s.llm_ms / (s.total_ms or 1.0) if s.total_ms > 0 else 1.0
        active_cols = max(1, int((col_end - col_start) * llm_frac))

        bar = list(" " * width)
        for c in range(col_start, col_end):
            if c < col_start + active_cols:
                bar[c] = "\u2588"
            else:
                bar[c] = "\u2591"

        id_pad = " " * (id_width - len(sid))
        lines.append(f"{id_pad}{sid}  {''.join(bar)}")

    if overflow:
        lines.append(f"{'':>{id_width}}  ... ({overflow} more sessions)")

    return "\n".join(lines)
