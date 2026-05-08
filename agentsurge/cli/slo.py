# SPDX-License-Identifier: MIT
"""cmd_slo - SLO boundary binary search subcommand."""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime

from agentsurge.cli._helpers import _collision_suffix, _load_config, _tokenizer_kwargs
from agentsurge.cli._slo_compare import (
    format_interleaved_table,
    load_config_profile,
    save_comparison_json,
)
from agentsurge.cli._slo_interleaved import _run_interleaved_slo_search
from agentsurge.cli._slo_search import _run_slo_search

_DEFAULT_PROFILES = [
    (7, 2200, 512, "7Tx2200"),
    (3, 500, 256, "3Tx500"),
    (15, 3000, 512, "15Tx3000"),
    (5, 1500, 256, "5Tx1500"),
    (10, 4000, 512, "10Tx4000"),
]


def _parse_profiles(args) -> list[tuple[int, int, int, str]]:
    """Parse --profiles CLI argument into a list of (turns, tokens, max_tokens, label) tuples.

    cli.md M6: bad input now produces a clean SystemExit with a message
    naming the offending spec (instead of an uncaught ValueError traceback).
    """
    if not getattr(args, "profiles", None):
        return list(_DEFAULT_PROFILES)
    profiles = []
    for spec in args.profiles:
        parts = spec.split(",") if spec else []
        if len(parts) not in (2, 3):
            print(
                f"error: invalid --profiles entry {spec!r}: expected "
                f"'turns,tokens[,max_tokens]' (e.g. '3,1500,256')",
                file=sys.stderr,
            )
            raise SystemExit(1)
        try:
            if len(parts) == 3:
                t, tok, mt = int(parts[0]), int(parts[1]), int(parts[2])
                profiles.append((t, tok, mt, f"{t}T{tok}x{mt}"))
            else:
                t, tok = int(parts[0]), int(parts[1])
                profiles.append((t, tok, 256, f"{t}T{tok}"))
        except ValueError:
            print(
                f"error: invalid --profiles entry {spec!r}: each field must be an integer "
                f"(got {parts!r})",
                file=sys.stderr,
            )
            raise SystemExit(1) from None
    return profiles


def cmd_slo(args: argparse.Namespace) -> None:
    """Find max concurrent sessions meeting SLO target via binary search.

    For each workload profile, performs exponential scan then binary search
    to find the maximum N where p95 TTFT < SLO threshold.

    When ``--configs`` is provided, runs interleaved comparison across all
    config profiles (probing every config at each concurrency level for
    fairness, then bisecting each independently).
    """
    configs_arg = getattr(args, "configs", None)

    if configs_arg:
        _cmd_slo_compare(args)
    else:
        _cmd_slo_single(args)


def _cmd_slo_single(args: argparse.Namespace) -> None:
    """Original single-config SLO search."""
    cfg = _load_config(args.config)
    vllm = cfg.get("vllm") or {}
    vllm_url = args.vllm_url or vllm.get("url")
    model_path = args.model or vllm.get("model")
    if not vllm_url or not model_path:
        from agentsurge.cli._helpers import CLIConfigError

        missing = "vllm.url" if not vllm_url else "vllm.model"
        flag = "--vllm-url" if not vllm_url else "--model"
        raise CLIConfigError(
            f"missing {missing}: pass {flag} or set '{missing}' in your config YAML"
        )
    slo_ms = args.slo_ms
    cooldown = args.cooldown
    no_cooldown = getattr(args, "no_cooldown", False)
    max_n_cap = args.max_n
    hint_val = getattr(args, "hint", None)

    profiles = _parse_profiles(args)
    tag = args.tag or "default"
    max_error_rate = getattr(args, "max_error_rate", 0.05)

    collected: list = []

    async def _run() -> None:
        all_results = await _run_slo_search(
            vllm_url=vllm_url,
            model_path=model_path,
            slo_ms=slo_ms,
            cooldown=cooldown,
            max_n_cap=max_n_cap,
            profiles=profiles,
            tag=tag,
            extra_body=cfg.get("extra_body"),
            no_cooldown=no_cooldown,
            hint=hint_val,
            max_error_rate=max_error_rate,
            tokenizer_model=_tokenizer_kwargs(getattr(args, "tokenizer", "auto"))[
                "tokenizer_model"
            ],
            trust_remote_code=getattr(args, "trust_remote_code", False),
            api_type=getattr(args, "api", "chat"),
            backend_name=getattr(args, "backend", None),
            request_timeout=cfg.get("runner", {}).get("request_timeout", 7200),
            stream_idle_timeout=getattr(args, "stream_idle_timeout", 0.0) or 0.0,
            enable_thinking=getattr(args, "enable_thinking", False),
        )
        collected.extend(all_results)

        print(f"\n\n{'=' * 80}")
        print(f"SUMMARY: SLO Boundary ({tag}), p95 < {slo_ms}ms")
        print(f"{'=' * 80}")
        print(f"{'Profile':<20} {'Max N':>6} {'KV%':>6} {'p95(ms)':>9} {'p50(ms)':>9}")
        print(f"{'-' * 80}")
        for r in all_results:
            br = r.get("boundary") or {}
            print(
                f"{r['profile']:<20} {r['max_n']:>6} "
                f"{br.get('kv_pct', 0):>5.1f}% {br.get('ttft_p95_ms', 0):>8.1f} "
                f"{br.get('ttft_p50_ms', 0):>8.1f}"
            )

        os.makedirs(args.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            args.output_dir, f"slo_boundary_{tag}_{ts}_{_collision_suffix()}.json"
        )
        with open(out_path, "w") as f:
            json.dump(
                {
                    "tag": tag,
                    "vllm_url": vllm_url,
                    "model": model_path,
                    "slo_ms": slo_ms,
                    "timestamp": ts,
                    "profiles": all_results,
                },
                f,
                indent=2,
            )
        print(f"\nSaved: {out_path}")

    asyncio.run(_run())

    # Exit 2 if any profile failed to find a passing N (SLO breached).
    # See cli.md H1 — distinct exit code lets CI gate on SLO violations.
    if any((r.get("max_n") or 0) <= 0 for r in collected):
        sys.exit(2)


def _cmd_slo_compare(args: argparse.Namespace) -> None:
    """Multi-config SLO comparison using interleaved two-pointer search."""
    slo_ms = args.slo_ms
    cooldown = args.cooldown
    no_cooldown = getattr(args, "no_cooldown", False)
    max_n_cap = args.max_n
    tag = args.tag or "compare"
    max_error_rate = getattr(args, "max_error_rate", 0.05)

    config_paths = [p.strip() for p in args.configs.split(",") if p.strip()]
    if not config_paths:
        print("Error: --configs requires at least one config file path.", file=sys.stderr)
        sys.exit(1)

    config_profiles = []
    for path in config_paths:
        try:
            profile = load_config_profile(path)
            config_profiles.append(profile)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error loading config profile '{path}': {exc}", file=sys.stderr)
            sys.exit(1)

    print(
        f"\nSLO Comparison (interleaved): {len(config_profiles)} configs, "
        f"target p95 TTFT < {slo_ms}ms"
    )
    for cp in config_profiles:
        desc = f" ({cp.description})" if cp.description else ""
        print(f"  - {cp.name}: {cp.vllm_url} / {cp.model}{desc}")

    profiles = _parse_profiles(args)

    all_profile_results = []

    async def _run_all() -> None:
        for n_turns_val, tokens_val, max_tok_val, label in profiles:
            result = await _run_interleaved_slo_search(
                config_profiles=config_profiles,
                slo_ms=slo_ms,
                cooldown=cooldown,
                no_cooldown=no_cooldown,
                max_n_cap=max_n_cap,
                n_turns=n_turns_val,
                tokens_per_turn=tokens_val,
                max_tokens=max_tok_val,
                profile_label=label,
                max_error_rate=max_error_rate,
                tokenizer_model=_tokenizer_kwargs(getattr(args, "tokenizer", "auto"))[
                    "tokenizer_model"
                ],
                trust_remote_code=getattr(args, "trust_remote_code", False),
            )
            all_profile_results.append(result)

    asyncio.run(_run_all())

    print(f"\n\n{'=' * 70}")
    print(format_interleaved_table(all_profile_results, slo_ms))
    print(f"{'=' * 70}")

    os.makedirs(args.output_dir, exist_ok=True)
    ts = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_collision_suffix()}"
    out_path = save_comparison_json(
        all_profile_results,
        args.output_dir,
        tag,
        ts,
    )
    print(f"\nSaved comparison: {out_path}")

    # Exit 2 if any config-profile pair failed to find a passing N.
    # See cli.md H1.
    def _profile_breached(profile_result: dict) -> bool:
        # Result schema: {"label": ..., "configs": {name: {"max_n": int, ...}}}
        for cfg_entry in (profile_result.get("configs") or {}).values():
            if (cfg_entry.get("max_n") or 0) <= 0:
                return True
        return False

    if any(_profile_breached(r) for r in all_profile_results):
        sys.exit(2)
