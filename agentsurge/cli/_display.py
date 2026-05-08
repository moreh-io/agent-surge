# SPDX-License-Identifier: MIT
"""Rich box summary for benchmark run results."""

from __future__ import annotations

import contextlib
import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentsurge.types.results import RunResult


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


def _fmt_int(n: int | float) -> str:
    return f"{int(n):,}"


def _fmt_ms(v: float) -> str:
    return f"{_fmt_int(v)} ms"


def _fmt_pct(v: float) -> str:
    return f"{v:.1%}"


def _fmt_ratio(v: float) -> str:
    return f"{v:.0f}:1" if v >= 1 else f"1:{1 / v:.0f}" if v > 0 else "--"


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_latency_row(label: str, p50: float, p95: float, p99: float) -> str:
    return f"p50 {_fmt_ms(p50)}  p95 {_fmt_ms(p95)}  p99 {_fmt_ms(p99)}"


def _build_latency_table(
    rows: list[tuple[str, float, float, float]],
    placeholders: dict[str, str] | None = None,
):
    """Build a right-aligned latency table (p50/p95/p99) using rich.

    rows: list of (label, p50_ms, p95_ms, p99_ms)
    placeholders: map of label -> full-width placeholder string (e.g. for TPOT note)
    """
    from rich.table import Table

    t = Table.grid(padding=(0, 2), expand=False)
    t.add_column(style="cyan", no_wrap=True)
    t.add_column(justify="right", style="dim")
    t.add_column(justify="right", style="white", no_wrap=True)
    t.add_column(justify="right", style="dim")
    t.add_column(justify="right", style="white", no_wrap=True)
    t.add_column(justify="right", style="dim")
    t.add_column(justify="right", style="white", no_wrap=True)

    placeholders = placeholders or {}
    for label, p50, p95, p99 in rows:
        if label in placeholders:
            t.add_row(f"  {label}", placeholders[label], "", "", "", "", "")
        else:
            t.add_row(
                f"  {label}",
                "p50",
                _fmt_ms(p50),
                "p95",
                _fmt_ms(p95),
                "p99",
                _fmt_ms(p99),
            )
    return t


def _classify_failure(error: str) -> str:
    """Classify a turn error string into a short human-readable reason."""
    e = error.lower()
    if "context length" in e or "max_model_len" in e or "maximum context" in e:
        return "context_overflow"
    if "timeout" in e or "timed out" in e or "deadline" in e:
        return "timeout"
    if "connection" in e or "refused" in e or "reset" in e:
        return "connection"
    if "rate limit" in e or "429" in e or "too many" in e:
        return "rate_limit"
    if "oom" in e or "out of memory" in e or "cuda" in e:
        return "oom"
    if "enginecore" in e or "internal" in e:
        return "server_error"
    return "error"


def _failure_breakdown(sessions: list) -> dict[str, int]:
    """Count failure reasons across failed sessions."""
    counts: dict[str, int] = {}
    for s in sessions:
        if s.completed:
            continue
        reason = "error"
        for t in s.turns:
            if not t.completed and t.error:
                reason = _classify_failure(t.error)
                break
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def print_run_summary(
    result: RunResult,
    *,
    model: str = "",
    saved_files: list[str] | None = None,
) -> None:
    """Print a rich box summary of a benchmark run.

    Falls back to plain text when ``rich`` is not installed.
    """
    sessions = result.sessions
    n_sessions = len(sessions)
    n_completed = sum(1 for s in sessions if s.completed)
    n_failed = n_sessions - n_completed
    total_turns = sum(int(getattr(s, "n_turns", 0)) for s in sessions)
    turns_avg = total_turns / n_sessions if n_sessions > 0 else 0.0
    elapsed = float(getattr(result, "total_elapsed_s", 0) or 0)
    sessions_per_hour = n_completed / elapsed * 3600 if elapsed > 0 else 0.0

    try:
        ttfts = list(result.ttft_values())
    except (TypeError, AttributeError):
        ttfts = []
    ttft_p50 = statistics.median(ttfts) if ttfts else 0.0
    ttft_p95 = _percentile(ttfts, 95)
    ttft_p99 = _percentile(ttfts, 99)

    tpot_p50 = float(getattr(result, "tpot_p50", 0) or 0)
    tpot_p95 = float(getattr(result, "tpot_p95", 0) or 0)
    tpot_p99 = float(getattr(result, "tpot_p99", 0) or 0)

    e2e_p50 = float(getattr(result, "e2e_p50", 0) or 0)
    e2e_p95 = float(getattr(result, "e2e_p95", 0) or 0)
    e2e_p99 = float(getattr(result, "e2e_p99", 0) or 0)

    no_metrics = False
    with contextlib.suppress(TypeError, AttributeError):
        no_metrics = result.config.get("no_metrics", False)
    try:
        kv_util_peak = float(result._bm_float("kv_util_peak"))
    except (TypeError, AttributeError):
        kv_util_peak = float(getattr(result, "kv_util_peak", 0) or 0)
    try:
        prefix_hit = float(result.prefix_cache_hit_rate)
    except (TypeError, AttributeError):
        prefix_hit = 0.0

    isl_osl_ratio = float(getattr(result, "isl_osl_ratio", 0) or 0)

    isl_tokens = int(getattr(result, "isl_total", 0) or 0)
    osl_tokens = int(getattr(result, "osl_total", 0) or 0)

    all_failed = n_completed == 0 and n_sessions > 0
    high_error_rate = n_failed / n_sessions > 0.2 if n_sessions > 0 else False

    try:
        _print_rich(
            result=result,
            model=model,
            n_sessions=n_sessions,
            n_completed=n_completed,
            n_failed=n_failed,
            total_turns=total_turns,
            turns_avg=turns_avg,
            sessions_per_hour=sessions_per_hour,
            ttft_p50=ttft_p50,
            ttft_p95=ttft_p95,
            ttft_p99=ttft_p99,
            tpot_p50=tpot_p50,
            tpot_p95=tpot_p95,
            tpot_p99=tpot_p99,
            e2e_p50=e2e_p50,
            e2e_p95=e2e_p95,
            e2e_p99=e2e_p99,
            no_metrics=no_metrics,
            kv_util_peak=kv_util_peak,
            prefix_hit=prefix_hit,
            isl_osl_ratio=isl_osl_ratio,
            isl_tokens=isl_tokens,
            osl_tokens=osl_tokens,
            all_failed=all_failed,
            high_error_rate=high_error_rate,
            saved_files=saved_files,
        )
    except ImportError:
        _print_plain(
            result=result,
            model=model,
            n_sessions=n_sessions,
            n_completed=n_completed,
            n_failed=n_failed,
            total_turns=total_turns,
            turns_avg=turns_avg,
            sessions_per_hour=sessions_per_hour,
            ttft_p50=ttft_p50,
            ttft_p95=ttft_p95,
            ttft_p99=ttft_p99,
            tpot_p50=tpot_p50,
            tpot_p95=tpot_p95,
            tpot_p99=tpot_p99,
            e2e_p50=e2e_p50,
            e2e_p95=e2e_p95,
            e2e_p99=e2e_p99,
            no_metrics=no_metrics,
            kv_util_peak=kv_util_peak,
            prefix_hit=prefix_hit,
            isl_osl_ratio=isl_osl_ratio,
            isl_tokens=isl_tokens,
            osl_tokens=osl_tokens,
            saved_files=saved_files,
        )


def _print_rich(
    *,
    result: RunResult,
    model: str,
    n_sessions: int,
    n_completed: int,
    n_failed: int,
    total_turns: int,
    turns_avg: float,
    sessions_per_hour: float,
    ttft_p50: float,
    ttft_p95: float,
    ttft_p99: float,
    tpot_p50: float,
    tpot_p95: float,
    tpot_p99: float,
    e2e_p50: float,
    e2e_p95: float,
    e2e_p99: float,
    no_metrics: bool,
    kv_util_peak: float,
    prefix_hit: float,
    isl_osl_ratio: float,
    isl_tokens: int,
    osl_tokens: int,
    all_failed: bool,
    high_error_rate: bool,
    saved_files: list[str] | None,
) -> None:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.text import Text

    console = Console()

    if all_failed:
        body = Text()
        body.append("All sessions failed.\n\n", style="bold red")
        body.append(f"  Sessions     {_fmt_int(n_sessions)}\n")
        body.append(f"  Elapsed      {result.total_elapsed_s:.1f}s\n")
        first_err = ""
        for s in result.sessions:
            for t in s.turns:
                if t.error:
                    first_err = t.error
                    break
            if first_err:
                break
        if first_err:
            truncated = first_err[:120] + ("..." if len(first_err) > 120 else "")
            body.append(f"\n  First error: {truncated}\n", style="red")
        panel = Panel(
            body,
            title="[bold red]AgentSurge Results  \u00b7  ERROR[/bold red]",
            border_style="red",
            padding=(1, 2),
        )
        console.print(panel)
        _print_saved_files(console, saved_files)
        return

    title_parts = ["[bold green]AgentSurge Results[/bold green]"]
    if model:
        title_parts.append(f"[green]{model}[/green]")
    title_parts.append(f"[green]{result.total_elapsed_s:.1f}s[/green]")
    title = "  \u00b7  ".join(title_parts)

    body = Text()
    body.append("\n")

    body.append("  \u2500\u2500 Workload statistics \u2500\u2500\n", style="dim")

    session_detail = f"{_fmt_int(n_completed)} completed"
    if n_failed:
        breakdown = _failure_breakdown(result.sessions)
        parts = [f"{v} {k}" for k, v in sorted(breakdown.items(), key=lambda x: -x[1])]
        session_detail += f", {', '.join(parts)}"
    _add_metric(body, "Sessions", f"{_fmt_int(n_sessions)} ({session_detail})", high_error_rate)
    _add_metric(body, "Turns", f"{_fmt_int(total_turns)} ({turns_avg:.1f} avg/session)")
    _add_metric(body, "Input:Output ratio", f"{_fmt_ratio(isl_osl_ratio)} (tokens)")
    tokens_str = f"{_fmt_tokens(isl_tokens)} input, {_fmt_tokens(osl_tokens)} output"
    _add_metric(body, "Tokens", tokens_str)
    body.append("\n")

    body.append("  \u2500\u2500 Server statistics \u2500\u2500\n", style="dim")

    elapsed = float(getattr(result, "total_elapsed_s", 0) or 0)
    req_per_s = total_turns / elapsed if elapsed > 0 else 0.0
    isl_tps = isl_tokens / elapsed if elapsed > 0 else 0.0
    osl_tps = osl_tokens / elapsed if elapsed > 0 else 0.0
    _add_metric(body, "Sessions/hr", _fmt_int(sessions_per_hour))
    _add_metric(body, "Requests/s", f"{req_per_s:.1f}")
    _add_metric(body, "Input tok/s", _fmt_tokens(int(isl_tps)))
    _add_metric(body, "Output tok/s", _fmt_tokens(int(osl_tps)))

    latency_rows = [("Time to first token", ttft_p50, ttft_p95, ttft_p99)]
    placeholders: dict[str, str] = {}
    if tpot_p50 == 0 and tpot_p95 == 0 and osl_tokens > 0:
        latency_rows.append(("Time per output token", 0.0, 0.0, 0.0))
        placeholders["Time per output token"] = "-- (non-streaming, TTFT ≈ E2E)"
    else:
        latency_rows.append(("Time per output token", tpot_p50, tpot_p95, tpot_p99))
    latency_rows.append(("End-to-end latency", e2e_p50, e2e_p95, e2e_p99))
    latency_table = _build_latency_table(latency_rows, placeholders)

    tail = Text()
    if no_metrics:
        _add_metric(tail, "Peak KV cache utilization", "--")
        _add_metric(tail, "Prefix cache hit rate", "--")
    else:
        _add_metric(tail, "Peak KV cache utilization", f"{_fmt_pct(kv_util_peak)} (of GPU memory)")
        _add_metric(tail, "Prefix cache hit rate", f"{_fmt_pct(prefix_hit)} (per-request)")
    tail.append("\n")

    panel = Panel(
        Group(body, latency_table, tail),
        title=title,
        border_style="green",
        padding=(0, 2),
    )
    console.print(panel)
    _print_saved_files(console, saved_files)


def _add_metric(text, label: str, value: str, warn: bool = False) -> None:
    text.append(f"  {label:<27}", style="cyan")
    text.append(f"{value}\n", style="yellow" if warn else "white")


def _print_saved_files(console, saved_files: list[str] | None) -> None:
    if not saved_files:
        return
    for path in saved_files:
        console.print(f"  Saved: {path}", style="dim")


def _print_plain(
    *,
    result: RunResult,
    model: str,
    n_sessions: int,
    n_completed: int,
    n_failed: int,
    total_turns: int,
    turns_avg: float,
    sessions_per_hour: float,
    ttft_p50: float,
    ttft_p95: float,
    ttft_p99: float,
    tpot_p50: float,
    tpot_p95: float,
    tpot_p99: float,
    e2e_p50: float,
    e2e_p95: float,
    e2e_p99: float,
    no_metrics: bool,
    kv_util_peak: float,
    prefix_hit: float,
    isl_osl_ratio: float,
    isl_tokens: int,
    osl_tokens: int,
    saved_files: list[str] | None,
) -> None:
    header = f"Done in {result.total_elapsed_s:.1f}s"
    if model:
        header += f" ({model})"

    if no_metrics:
        metrics_line = "KV util peak=--, prefix cache=-- (metrics disabled)"
    else:
        metrics_line = f"KV util peak={kv_util_peak:.1%}, prefix cache hit={prefix_hit:.1%}"

    session_info = f"{n_completed} completed"
    if n_failed:
        breakdown = _failure_breakdown(result.sessions)
        parts = [f"{v} {k}" for k, v in sorted(breakdown.items(), key=lambda x: -x[1])]
        session_info += f", {', '.join(parts)}"

    print(f"{header}: {metrics_line}")
    print("  -- Workload statistics --")
    print(
        f"  Sessions: {n_sessions} ({session_info}) | "
        f"Turns: {total_turns} ({turns_avg:.1f} avg/session)"
    )
    print(
        f"  Input:Output {_fmt_ratio(isl_osl_ratio)} | "
        f"Tokens: {_fmt_tokens(isl_tokens)} input, {_fmt_tokens(osl_tokens)} output"
    )
    print("  -- Server statistics --")
    elapsed = float(getattr(result, "total_elapsed_s", 0) or 0)
    isl_tps = isl_tokens / elapsed if elapsed > 0 else 0.0
    osl_tps = osl_tokens / elapsed if elapsed > 0 else 0.0
    print(f"  Throughput: {_fmt_int(sessions_per_hour)} sessions/hr")
    print(
        f"              input {_fmt_tokens(int(isl_tps))} tok/s  \u00b7  output {_fmt_tokens(int(osl_tps))} tok/s"
    )
    print(f"  Time to first token:  p50={ttft_p50:.0f}ms p95={ttft_p95:.0f}ms p99={ttft_p99:.0f}ms")
    if tpot_p50 == 0 and tpot_p95 == 0 and osl_tokens > 0:
        print("  Time per output token: -- (non-streaming, TTFT ≈ E2E)")
    else:
        print(
            f"  Time per output token: p50={tpot_p50:.0f}ms p95={tpot_p95:.0f}ms p99={tpot_p99:.0f}ms"
        )
    print(f"  End-to-end latency:   p50={e2e_p50:.0f}ms p95={e2e_p95:.0f}ms p99={e2e_p99:.0f}ms")
    if saved_files:
        for path in saved_files:
            print(f"  Saved: {path}")
