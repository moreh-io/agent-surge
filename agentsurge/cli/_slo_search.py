# SPDX-License-Identifier: MIT
"""Single-config SLO boundary binary search core.

Implements exponential scan + binary search to find the maximum
number of concurrent sessions where p95 TTFT < SLO threshold.

Delegates probe execution and binary-search logic to
:mod:`agentsurge.cli._slo_core`.
"""

from agentsurge.cli._helpers import _smart_cooldown
from agentsurge.cli._slo_core import slo_binary_search, slo_probe


async def _run_slo_search(
    *,
    vllm_url: str,
    model_path: str,
    slo_ms: float,
    cooldown: int,
    max_n_cap: int,
    profiles: list[tuple[int, int, int, str]],
    tag: str,
    extra_body: dict | None = None,
    hint: int | None = None,
    no_cooldown: bool = False,
    max_error_rate: float = 0.05,
    tokenizer_model: str | None = None,
    trust_remote_code: bool = False,
    api_type: str = "chat",
    backend_name: str | None = None,
    request_timeout: int = 7200,
    stream_idle_timeout: float = 0.0,
    enable_thinking: bool = False,
) -> list[dict]:
    """Run SLO binary search for a list of workload profiles.

    Parameters
    ----------
    vllm_url:
        vLLM server URL.
    model_path:
        Model name/path for requests.
    slo_ms:
        SLO target -- max acceptable p95 TTFT in ms.
    cooldown:
        Max seconds to wait for KV cooldown between probes.
    max_n_cap:
        Upper bound for session count search.
    profiles:
        List of (n_turns, tokens_per_turn, max_tokens, label) tuples.
    tag:
        Human-readable tag for this search run.
    extra_body:
        Optional extra_body dict to pass to BenchmarkConfig.
    hint:
        Warm-start hint -- approximate spike N from a prior run.
        When provided, skips exponential scan and starts binary search
        in the range [max(5, hint*0.5), hint*2.0].
    no_cooldown:
        If True, skip KV cooldown entirely between probes.
    tokenizer_model:
        Optional HF tokenizer ID or local path forwarded to ``BenchmarkConfig``.
        When ``None``, the runner falls back to ``model``.
    trust_remote_code:
        Forwarded to ``BenchmarkConfig`` to enable ``trust_remote_code`` when
        loading the tokenizer.

    Returns
    -------
    list[dict]
        One result dict per profile with keys: profile, n_turns,
        tokens_per_turn, max_tokens, slo_ms, max_n, boundary, scan.
    """

    async def _search_one_profile(n_turns, tokens_per_turn, max_tokens, label) -> dict:
        """Search for max N meeting SLO for one workload profile."""
        print(f"\n{'=' * 70}")
        print(
            f"Profile: {label} (turns={n_turns}, tok/turn={tokens_per_turn}, max_out={max_tokens})"
        )
        print(f"SLO target: p95 TTFT < {slo_ms}ms")
        print(f"{'=' * 70}")

        # Build a probe function that includes smart-cooldown state tracking
        _sc_prev_n = 0
        _sc_prev_kv = 0.0
        _sc_prev_err = False

        async def _sc_probe(n_sess: int) -> dict:
            nonlocal _sc_prev_n, _sc_prev_kv, _sc_prev_err
            await _smart_cooldown(
                vllm_url,
                current_n=_sc_prev_n,
                next_n=n_sess,
                current_kv=_sc_prev_kv,
                had_error=_sc_prev_err,
                no_cooldown=no_cooldown,
                timeout=cooldown,
            )
            r = await slo_probe(
                vllm_url=vllm_url,
                model=model_path,
                n_sess=n_sess,
                n_turns=n_turns,
                tokens_per_turn=tokens_per_turn,
                max_tokens=max_tokens,
                extra_body=extra_body,
                tokenizer_model=tokenizer_model,
                trust_remote_code=trust_remote_code,
                api_type=api_type,
                backend_name=backend_name,
                request_timeout=request_timeout,
                stream_idle_timeout=stream_idle_timeout,
                enable_thinking=enable_thinking,
            )
            _sc_prev_n = n_sess
            _sc_prev_kv = r["kv_pct"] / 100.0
            _sc_prev_err = r["error_rate"] > 0
            return r

        result = await slo_binary_search(
            _sc_probe,
            slo_ms=slo_ms,
            max_n_cap=max_n_cap,
            hint=hint,
            label=label,
            max_error_rate=max_error_rate,
        )

        return {
            "profile": label,
            "n_turns": n_turns,
            "synthetic_tokens_per_turn": tokens_per_turn,
            "max_tokens": max_tokens,
            "slo_ms": slo_ms,
            "max_n": result["max_n"],
            "boundary": result["boundary"],
            "scan": result["scan"],
        }

    all_results = []
    for n_turns, tok, mt, label in profiles:
        r = await _search_one_profile(n_turns, tok, mt, label)
        all_results.append(r)

    return all_results
