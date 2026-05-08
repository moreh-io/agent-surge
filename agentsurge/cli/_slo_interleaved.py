# SPDX-License-Identifier: MIT
"""Interleaved multi-config SLO search.

Probes ALL configs at each concurrency level during the exponential scan
phase for fair comparison, then bisects each config independently.

Delegates probe execution to :func:`agentsurge.cli._slo_core.slo_probe` and
uses :func:`agentsurge.cli._slo_core.check_slo` /
:func:`agentsurge.cli._slo_core.next_n` for shared SLO logic.
"""

import collections.abc
import sys
from typing import Any

from agentsurge.cli._helpers import _smart_cooldown
from agentsurge.cli._slo_core import check_slo, next_n, slo_probe


async def _run_interleaved_slo_search(
    *,
    config_profiles: list,
    slo_ms: float,
    cooldown: int,
    max_n_cap: int,
    n_turns: int,
    tokens_per_turn: int,
    max_tokens: int,
    profile_label: str,
    max_error_rate: float = 0.05,
    no_cooldown: bool = False,
    tokenizer_model: str | None = None,
    trust_remote_code: bool = False,
) -> dict:
    """Interleaved two-pointer SLO search across multiple config profiles.

    Instead of running each config's full sweep sequentially, this probes
    ALL configs at each concurrency level during the exponential scan phase,
    then bisects each config independently once its breach bounds are known.

    Benefits over sequential:
      - Fairer comparison: each config sees the same concurrency sequence.
      - Early termination: configs that breach early are excluded from
        higher-level probes.
      - Shared scan phase reduces total probe count when configs have
        similar spike locations.

    Parameters
    ----------
    config_profiles:
        List of :class:`ConfigProfile` objects (each has vllm_url, model, etc.).
    slo_ms:
        SLO target -- max acceptable p95 TTFT in ms.
    cooldown:
        Max seconds to wait for KV cooldown between probes.
    max_n_cap:
        Upper bound for session count search.
    n_turns, tokens_per_turn, max_tokens:
        Workload shape for this profile.
    profile_label:
        Human-readable label for the workload profile being tested.
    tokenizer_model:
        Optional HF tokenizer ID or local path forwarded to ``BenchmarkConfig``.
        When ``None``, the runner falls back to ``model``.
    trust_remote_code:
        Forwarded to ``BenchmarkConfig`` to enable ``trust_remote_code`` when
        loading the tokenizer.

    Returns
    -------
    dict
        Keys: ``profile_label``, ``slo_ms``, ``workload``, ``configs`` (list of
        per-config result dicts), ``capacity_ratios`` (relative to first config),
        ``pairwise_ratios`` (all pairs).
    """
    n_configs = len(config_profiles)

    per_cfg_lo = [0] * n_configs
    per_cfg_hi = [0] * n_configs
    per_cfg_scan: list[list[dict]] = [[] for _ in range(n_configs)]
    per_cfg_active = [True] * n_configs  # still probing in scan phase?
    per_cfg_breached = [False] * n_configs
    per_cfg_prev_n = [0] * n_configs
    per_cfg_prev_kv = [0.0] * n_configs
    per_cfg_prev_err = [False] * n_configs

    print(f"\n{'=' * 70}")
    print(
        f"Interleaved scan: {profile_label} "
        f"(turns={n_turns}, tok/turn={tokens_per_turn}, max_out={max_tokens})"
    )
    print(f"SLO target: p95 TTFT < {slo_ms}ms")
    print(f"Configs: {', '.join(cp.name for cp in config_profiles)}")
    print(f"{'=' * 70}")
    print("\n--- Phase 1: Interleaved exponential scan ---")

    n = 10
    while n <= max_n_cap:
        if not any(per_cfg_active):
            break

        print(f"\n  n={n}:")
        for i, cp in enumerate(config_profiles):
            if not per_cfg_active[i]:
                print(f"    [{cp.name:>20}]  -- skipped (breached at n={per_cfg_hi[i]})")
                continue

            await _smart_cooldown(
                cp.vllm_url,
                current_n=per_cfg_prev_n[i],
                next_n=n,
                current_kv=per_cfg_prev_kv[i],
                had_error=per_cfg_prev_err[i],
                no_cooldown=no_cooldown,
                timeout=cooldown,
            )
            extra_body = cp.overrides.get("extra_body")
            r = await slo_probe(
                vllm_url=cp.vllm_url,
                model=cp.model,
                n_sess=n,
                n_turns=n_turns,
                tokens_per_turn=tokens_per_turn,
                max_tokens=max_tokens,
                extra_body=extra_body,
                tokenizer_model=tokenizer_model,
                trust_remote_code=trust_remote_code,
            )
            per_cfg_prev_n[i] = n
            per_cfg_prev_kv[i] = r["kv_pct"] / 100.0
            per_cfg_prev_err[i] = r["error_rate"] > 0
            per_cfg_scan[i].append(r)
            ok = check_slo(r, slo_ms, max_error_rate)
            if ok:
                status = "OK"
            elif r["ttft_p95_ms"] >= slo_ms:
                status = "TIMEOUT"
            else:
                status = "ERROR"
            print(
                f"    [{cp.name:>20}]  kv={r['kv_pct']:>5.1f}% "
                f"p95={r['ttft_p95_ms']:>8.1f}ms "
                f"err={r['error_rate'] * 100:.1f}% [{status}]"
            )
            sys.stdout.flush()

            if ok:
                per_cfg_lo[i] = n
            else:
                per_cfg_hi[i] = n
                per_cfg_active[i] = False
                per_cfg_breached[i] = True

        active_kv_pcts = []
        for i in range(n_configs):
            if per_cfg_active[i] and per_cfg_scan[i]:
                active_kv_pcts.append(per_cfg_scan[i][-1]["kv_pct"])

        if not active_kv_pcts:
            break

        max_kv = max(active_kv_pcts)
        n = next_n(n, max_kv)

    for i in range(n_configs):
        if not per_cfg_breached[i]:
            per_cfg_hi[i] = max_n_cap

    print("\n--- Phase 2: Independent bisect ---")

    per_cfg_results: list[dict] = []
    for i, cp in enumerate(config_profiles):
        lo, hi = per_cfg_lo[i], per_cfg_hi[i]
        scan = per_cfg_scan[i]

        if lo == 0 and hi <= 10:
            print(f"\n  [{cp.name}] SLO breached at n=10. Max N = 0")
            per_cfg_results.append(
                {
                    "config_name": cp.name,
                    "max_n": 0,
                    "boundary": scan[0] if scan else None,
                    "scan": scan,
                    "lo": lo,
                    "hi": hi,
                }
            )
            continue

        if hi - lo <= 5:
            best_n, best_r = 0, None
            for sr in scan:
                if check_slo(sr, slo_ms, max_error_rate) and sr["n"] >= best_n:
                    best_n, best_r = sr["n"], sr
            if best_n == 0 and lo > 0:
                best_n = lo
                for sr in scan:
                    if sr["n"] == lo and check_slo(sr, slo_ms, max_error_rate):
                        best_r = sr
                        break
            print(f"\n  [{cp.name}] Range [{lo}, {hi}] already tight. Max N = {best_n}")
            per_cfg_results.append(
                {
                    "config_name": cp.name,
                    "max_n": best_n,
                    "boundary": best_r,
                    "scan": sorted(scan, key=lambda x: x["n"]),
                    "lo": lo,
                    "hi": hi,
                }
            )
            continue

        def _make_bisect_probe(
            cp_inner: Any, i_inner: int
        ) -> collections.abc.Callable[[int], collections.abc.Awaitable[dict]]:
            """Create a closure-safe probe function for bisect."""
            _bp_prev_n = per_cfg_prev_n[i_inner]
            _bp_prev_kv = per_cfg_prev_kv[i_inner]
            _bp_prev_err = per_cfg_prev_err[i_inner]
            extra_body = cp_inner.overrides.get("extra_body")

            async def _bisect_probe(n_sess: int) -> dict:
                nonlocal _bp_prev_n, _bp_prev_kv, _bp_prev_err
                await _smart_cooldown(
                    cp_inner.vllm_url,
                    current_n=_bp_prev_n,
                    next_n=n_sess,
                    current_kv=_bp_prev_kv,
                    had_error=_bp_prev_err,
                    no_cooldown=no_cooldown,
                    timeout=cooldown,
                )
                r = await slo_probe(
                    vllm_url=cp_inner.vllm_url,
                    model=cp_inner.model,
                    n_sess=n_sess,
                    n_turns=n_turns,
                    tokens_per_turn=tokens_per_turn,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                    tokenizer_model=tokenizer_model,
                    trust_remote_code=trust_remote_code,
                )
                _bp_prev_n = n_sess
                _bp_prev_kv = r["kv_pct"] / 100.0
                _bp_prev_err = r["error_rate"] > 0
                return r

            return _bisect_probe

        bisect_probe_fn = _make_bisect_probe(cp, i)

        print(f"\n  [{cp.name}] Bisecting [{lo}, {hi}]")

        best_n, best_r = 0, None
        for sr in scan:
            if check_slo(sr, slo_ms, max_error_rate) and sr["n"] >= best_n:
                best_n, best_r = sr["n"], sr

        while hi - lo > 5:
            mid = ((lo + hi) // 2 // 5) * 5
            mid = max(lo + 5, mid)
            if mid >= hi:
                break

            r = await bisect_probe_fn(mid)
            scan.append(r)
            ok = check_slo(r, slo_ms, max_error_rate)
            if ok:
                status = "OK"
            elif r["ttft_p95_ms"] >= slo_ms:
                status = "TIMEOUT"
            else:
                status = "ERROR"
            print(
                f"    n={mid:>4}: kv={r['kv_pct']:>5.1f}% "
                f"p95={r['ttft_p95_ms']:>8.1f}ms "
                f"err={r['error_rate'] * 100:.1f}% [{status}]"
            )
            sys.stdout.flush()

            if ok:
                lo = mid
                if mid > best_n:
                    best_n, best_r = mid, r
            else:
                hi = mid

        if best_n == 0 and lo > 0:
            best_n = lo
            for sr in scan:
                if sr["n"] == lo and check_slo(sr, slo_ms, max_error_rate):
                    best_r = sr
                    break

        print(f"    >> Max N meeting SLO: {best_n}")
        if best_r:
            print(f"       kv={best_r['kv_pct']:.1f}% p95={best_r['ttft_p95_ms']:.1f}ms")

        per_cfg_results.append(
            {
                "config_name": cp.name,
                "max_n": best_n,
                "boundary": best_r,
                "scan": sorted(scan, key=lambda x: x["n"]),
                "lo": lo,
                "hi": hi,
            }
        )

    max_ns = [r["max_n"] for r in per_cfg_results]
    baseline_n = max_ns[0] if max_ns else 1

    capacity_ratios: dict[str, float] = {}
    for r in per_cfg_results:
        name = r["config_name"]
        my_n = r["max_n"]
        if baseline_n > 0:
            capacity_ratios[name] = round(my_n / baseline_n, 3)
        else:
            capacity_ratios[name] = float("inf") if my_n > 0 else 0.0

    pairwise: dict[str, float] = {}
    for ri in per_cfg_results:
        for rj in per_cfg_results:
            if ri["config_name"] == rj["config_name"]:
                continue
            key = f"{ri['config_name']}_vs_{rj['config_name']}"
            if rj["max_n"] > 0:
                pairwise[key] = round(ri["max_n"] / rj["max_n"], 3)
            else:
                pairwise[key] = float("inf") if ri["max_n"] > 0 else 1.0

    return {
        "profile_label": profile_label,
        "slo_ms": slo_ms,
        "workload": {
            "n_turns": n_turns,
            "synthetic_tokens_per_turn": tokens_per_turn,
            "max_tokens": max_tokens,
        },
        "configs": per_cfg_results,
        "capacity_ratios": capacity_ratios,
        "pairwise_ratios": pairwise,
    }
