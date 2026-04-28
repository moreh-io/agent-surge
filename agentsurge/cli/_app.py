# SPDX-License-Identifier: MIT
"""CLI entry point for agentsurge.

Argument-parsing assembly and main() dispatcher only.
All shared utilities live in ``agentsurge.cli._helpers``.
"""

import argparse
import contextlib
import signal
import sys

from agentsurge.cli.generate import cmd_generate
from agentsurge.cli.run import cmd_run
from agentsurge.cli.sweep import _probe_levels_type, cmd_sweep
from agentsurge.preset import PRESET_NAMES


class _SigTerm(BaseException):
    """Raised when SIGTERM is received; translated to exit 143 in main()."""


def _install_signal_handlers() -> None:
    """Install a SIGTERM handler that raises ``_SigTerm`` on the main thread.

    cli.md H2: prior to this, SIGTERM killed the process hard with no chance
    to flush partial results. Translating SIGTERM into a Python-level
    exception lets the same try/except in main() (and the run-time
    finallies in cmd_run) flush the JSON before exiting.
    """

    def _handler(signum, frame):  # pragma: no cover - trivial
        raise _SigTerm()

    # Not on the main thread (rare in tests) → silently skip.
    with contextlib.suppress(ValueError, OSError):
        signal.signal(signal.SIGTERM, _handler)


def _think_time_type(s: str) -> tuple[float, float]:
    """Argparse type= for --think-time MIN,MAX."""
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--think-time must be 'MIN,MAX' (e.g. '1.0,3.0'); got {s!r}"
        )
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"--think-time values must be floats; got {s!r}") from err


def _run_args() -> argparse.ArgumentParser:
    """Return a parent parser with all run-specific arguments (add_help=False)."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--client-concurrency",
        type=int,
        default=None,
        dest="client_concurrency",
        help="Client-side cap on in-flight sessions (asyncio.Semaphore). "
        "Unrelated to the vLLM server's --max-num-seqs. Default: config or 100.",
    )
    p.add_argument(
        "--arrival",
        choices=["burst", "poisson", "constant", "gamma", "ramp"],
        default=None,
        help=(
            "Inter-session arrival pattern. burst: all sessions launch at t=0. "
            "poisson/gamma: Poisson(rate=--arrival-rate) or gamma-jittered. "
            "constant: uniform spacing at --arrival-rate req/s. "
            "ramp: linearly ramp rate 0 -> --arrival-rate over --ramp-duration "
            "seconds. Overrides preset."
        ),
    )
    p.add_argument("--workload-file", help="Pre-generated workload JSON")
    p.add_argument(
        "--preset",
        choices=PRESET_NAMES,
        default=None,
        help=(
            "Workload preset. "
            "stress-*: burst arrival, 256 max_tokens (KV saturation). "
            "agent-*: gamma arrival, dynamic context (calibrated agent sim). "
            "measure: uncapped 32K output (capacity planning). "
            "simulate-*: gamma arrival, static context (reproducible A/B). "
            "See agentsurge/configs/default.yaml for full details."
        ),
    )
    p.add_argument(
        "--think-time",
        type=_think_time_type,
        default=None,
        help="LogNormal params MU,SIGMA (e.g. 3.5,1.0). Overrides preset.",
    )
    p.add_argument(
        "--interruption-rate",
        type=float,
        default=None,
        help="Probability of abandoning a session after each turn (0.0-1.0)",
    )
    p.add_argument(
        "--single-turn-ratio",
        type=float,
        default=None,
        help="Ratio of single-turn traffic to mix in (0-1). Overrides preset.",
    )
    p.add_argument(
        "--simulate-tool-delays",
        dest="tool_delay",
        type=str,
        default=None,
        choices=["true", "false"],
        help=(
            "'true'/'false'. Simulate realistic inter-turn pauses whenever the "
            "previous turn produced a tool output, sampled from per-tool-type "
            "distributions in agentsurge.tool_timing. Overrides preset."
        ),
    )
    p.add_argument(
        "--tool-output-mode",
        type=str,
        default=None,
        choices=["recorded", "placeholder"],
        help="Tool output handling: 'recorded' (original trace) or 'placeholder' (token-count-matched filler)",
    )
    p.add_argument(
        "--synthetic-prompt-scale",
        type=int,
        default=None,
        dest="synthetic_prompt_scale",
        help="Token-length scale for synthetic single-turn prompts "
        "(only used when --single-turn-ratio > 0; ratio x this value = input tokens). "
        "Not a server max_model_len setting.",
    )
    p.add_argument(
        "--context-distribution",
        type=str,
        default=None,
        choices=["chat", "agent", "mixed", "autocomplete"],
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--lmcache",
        action="store_true",
        default=None,
        help=(
            "Force-enable LMCache metrics collection. Tri-state: when "
            "omitted, lmcache is auto-enabled if --lmcache-url is set, if "
            "the preset is in LMCACHE_AUTO_PRESETS (e.g. stress-lmcache), "
            "or if the YAML config sets lmcache.enabled=true. Pass "
            "--no-lmcache to override the auto-enable triggers."
        ),
    )
    p.add_argument(
        "--no-lmcache",
        dest="lmcache",
        action="store_false",
        help=(
            "Disable LMCache metrics collection even if --lmcache-url, "
            "preset, or config would auto-enable it. Overrides the "
            "tri-state auto-enable behaviour of --lmcache."
        ),
    )
    p.add_argument(
        "--lmcache-url",
        type=str,
        default=None,
        help="LMCache metrics endpoint URL (e.g. http://localhost:9003/metrics)",
    )
    p.add_argument(
        "--arrival-rate",
        type=float,
        default=None,
        help="Requests/sec for poisson and gamma arrival modes (overrides preset/config)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens per request (overrides preset/config)",
    )
    p.add_argument(
        "--skip-validation",
        action="store_true",
        default=False,
        help="Skip workload validation (validate_workload) when loading workload files",
    )
    p.add_argument(
        "--trace-pool",
        type=str,
        default=None,
        help="Path to trace pool JSON. "
        "Enables real-trace content when generating sessions on-the-fly.",
    )
    p.add_argument(
        "--warm-up",
        type=int,
        default=0,
        help="Number of warm-up requests before measurement (default: 0)",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default="auto",
        help="HF model ID, local path, or 'auto' (default; same as --model) "
        "or 'none' (skip loading a tokenizer). "
        "Use when --model is a served name that differs from the HF repo.",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Enable trust_remote_code when loading tokenizer (default: disabled for security)",
    )
    p.add_argument(
        "--ramp-duration",
        type=float,
        default=None,
        help="Seconds to linearly ramp arrival rate from 0 to --arrival-rate (use with --arrival ramp)",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Run for N seconds (recycling sessions) instead of fixed session count",
    )
    p.add_argument(
        "--tool-workspace-dir",
        "--tool-call-repo-dir",
        dest="workspace_dir",
        type=str,
        default=None,
        help="Pre-prepared execution workspace root holding per-session repo clones "
        "(skips git clone during run). Each session gets a subdirectory "
        "matching its session_id. Use local NVMe, not NFS. "
        "--tool-call-repo-dir is a legacy alias.",
    )
    p.add_argument(
        "--tool-env",
        choices=["safe", "inherit"],
        default="safe",
        help="Environment passed to real tool subprocesses. "
        "'safe' (default) uses a conservative allowlist with HOME set to the workspace; "
        "'inherit' passes the parent environment for trusted workloads.",
    )
    p.add_argument(
        "--save-responses",
        action="store_true",
        default=False,
        help="Include LLM response text in result JSON (up to 2000 chars per turn).",
    )
    p.add_argument(
        "--enable-inflight-dump",
        action="store_true",
        default=False,
        help="Append one JSONL line per completed turn to <results>/<run>_turns.jsonl "
        "as the benchmark progresses. Final consolidated JSON is unchanged. Buffered "
        "writes with periodic flush; off by default.",
    )
    p.add_argument(
        "--no-stream",
        action="store_true",
        default=False,
        help="Use non-streaming API for tool-calling turns. "
        "Disables TPOT measurement (TTFT will equal E2E). "
        "Useful for debugging or servers that don't support streaming with tool_choice.",
    )
    p.add_argument(
        "--ignore-replay-output-length",
        action="store_true",
        default=False,
        help="Ignore the replay's per-turn output length and use --max-tokens "
        "for every turn. Default: size each turn's max_tokens to the number of "
        "output tokens the replay produced on that turn.",
    )
    p.add_argument(
        "--thinking-budget",
        type=int,
        default=8192,
        help="Extra token budget added to dynamic max_tokens for thinking/reasoning models (default: 8192)",
    )
    p.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Enable thinking mode in chat_template_kwargs (default: disabled)",
    )
    p.add_argument(
        "--use-model-reply-in-next-turn",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        dest="use_model_reply_in_next_turn",
        help="'auto' (default): derive from --tool-mode (real/off → true, replay → false). "
        "'true': inject actual model output into the next turn. "
        "'false': replay the recorded assistant message.",
    )
    p.add_argument(
        "--frontend",
        choices=["direct", "echo", "codex", "claude", "opencode"],
        default="direct",
        dest="frontend",
        help="Frontend harness driving each session (default: direct = in-process API client).",
    )
    p.add_argument(
        "--frontend-command-template",
        type=str,
        default=None,
        dest="frontend_command_template",
        help="Override command template used to spawn the frontend process.",
    )
    p.add_argument(
        "--frontend-workspace-dir",
        type=str,
        default=None,
        dest="frontend_workspace_dir",
        help="Per-session workspace root for frontend processes (one subdir per session).",
    )
    p.add_argument(
        "--frontend-prompt-mode",
        choices=["auto", "file", "stdin", "arg"],
        default="auto",
        dest="frontend_prompt_mode",
        help="How the prompt is delivered to the frontend process (default: auto).",
    )
    p.add_argument(
        "--frontend-output-format",
        choices=["auto", "jsonl", "stream-json", "text"],
        default="auto",
        dest="frontend_output_format",
        help="Expected stdout format from the frontend process (default: auto-detect).",
    )
    p.add_argument(
        "--frontend-model",
        type=str,
        default=None,
        dest="frontend_model",
        help="Model name passed through to the frontend process (overrides --model for the harness).",
    )
    p.add_argument(
        "--frontend-session-timeout",
        type=float,
        default=7200.0,
        dest="frontend_session_timeout",
        help="Per-session wall-clock timeout (seconds) for the frontend process (default: 7200).",
    )
    p.add_argument(
        "--frontend-keep-artifacts",
        choices=["always", "failed", "never"],
        default="failed",
        dest="frontend_keep_artifacts",
        help="When to retain per-session frontend artifacts (default: failed).",
    )
    p.add_argument(
        "--frontend-server-url",
        type=str,
        default=None,
        dest="frontend_server_url",
        help="Server URL the frontend should target (overrides --vllm-url for the harness).",
    )
    p.add_argument(
        "--frontend-extra-env",
        action="append",
        default=[],
        dest="frontend_extra_env",
        metavar="KEY=VALUE",
        help="Extra environment variable for the frontend process; repeat for multiple.",
    )
    return p


def _gen_args() -> argparse.ArgumentParser:
    """Return a parent parser with all generate-specific arguments (add_help=False)."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--source",
        choices=[
            "openhands",
            "swe-smith-tool",
            "swe-smith-xml",
            "swe-smith-ticks",
            "swe-bench-verified",
            "abc-bench",
            "swe-evo",
            "featurebench",
            "mixed",
        ],
        default=None,
        help="Data source for replay mode (omit for synthetic)",
    )
    p.add_argument(
        "--local-path", default=None, help="Local JSONL file path (instead of HuggingFace)"
    )
    p.add_argument(
        "--swe-smith-split",
        default="tool",
        choices=["tool", "xml", "ticks"],
        help="SWE-smith dataset split",
    )
    p.add_argument(
        "--mix-weights",
        default=None,
        help="source:weight pairs (e.g. openhands:0.3,swe-smith:0.3,swe-bench-verified:0.4)",
    )
    p.add_argument(
        "--flatten-tools",
        action="store_true",
        default=True,
        help=(
            "Collapse tool-call outputs into the assistant message (shorter "
            "context; loses role separation). Default: on."
        ),
    )
    p.add_argument(
        "--no-flatten-tools",
        dest="flatten_tools",
        action="store_false",
        help="Disable --flatten-tools; preserve separate tool/role messages.",
    )
    p.add_argument(
        "--max-prompt-chars",
        type=int,
        default=16384,
        help=(
            "Cap each generated task prompt at N characters during workload "
            "synthesis (applies to task_converter templates; does not affect "
            "replayed traces). Default: 16384."
        ),
    )
    p.add_argument(
        "--n-sessions",
        type=int,
        default=50,
        help="Number of sessions to generate / issue. Default: 50.",
    )
    p.add_argument(
        "--n-turns",
        type=int,
        default=None,
        help=(
            "Synthetic workload: turns per session (ignored for trace replay). "
            "Default: from config.yaml workload.n_turns."
        ),
    )
    p.add_argument(
        "--synthetic-tokens-per-turn",
        type=int,
        default=None,
        help="Target output tokens per turn for synthetic workload generation (ignored for trace replay)",
    )
    p.add_argument(
        "--trace-pool",
        type=str,
        default=None,
        help="Path to pre-extracted trace pool JSON. Mutually exclusive with --corpus-paths.",
    )
    p.add_argument(
        "--corpus-paths",
        nargs="+",
        default=None,
        help="Corpus JSON files to build trace pool on-the-fly. Supports glob patterns.",
    )
    return p


def main() -> None:
    """CLI entry point: parse arguments and dispatch to the appropriate subcommand.

    Subcommands: run, sweep, generate.

    Absorbed commands (no longer top-level):
    - saturate, scaffold, full, verify  -> removed
    - slo                               -> sweep --slo-ms
    - compare                           -> analyze --compare
    - plan, tier-boundary, validate     -> removed
    - matrix                            -> removed (use shell loop with 'agentsurge run')
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", "-c", help="Path to YAML config file", default=None)
    common.add_argument("--vllm-url", help="vLLM URL (overrides config)", default=None)
    common.add_argument("--model", help="Model name/path (overrides config)", default=None)
    common.add_argument(
        "--output-dir", "-o", default="agentsurge/results", help="Output directory for results"
    )
    common.add_argument(
        "--api",
        choices=["chat", "responses"],
        default="chat",
        help="API endpoint type: chat (/v1/chat/completions) or responses (/v1/responses)",
    )
    common.add_argument(
        "--backend",
        choices=["vllm", "openai", "mock"],
        default=None,
        help=(
            "Inference backend (default: vllm). 'openai' means an OpenAI-compatible "
            "endpoint, not hosted OpenAI API auth; metrics default off unless configured."
        ),
    )
    common.add_argument(
        "--no-metrics",
        action="store_true",
        default=False,
        help="Disable Prometheus metrics polling (degraded mode: latency only, KV%% shown as '--')",
    )
    common.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for arrival schedule, per-session tool-delay and think-time "
        "sampling, interruption draws, and synthetic generation. "
        "Same seed across runs = identical per-session request sequence.",
    )

    run_parent = _run_args()
    gen_parent = _gen_args()

    parser = argparse.ArgumentParser(
        prog="agentsurge",
        description="agentsurge - Benchmark your LLM server with realistic coding agent workloads",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    p_run = sub.add_parser(
        "run",
        help="Run benchmark against a vLLM server with configurable workloads",
        description="Run a benchmark against a vLLM server with configurable workloads",
        parents=[common, run_parent],
    )
    p_run.add_argument(
        "--n-sessions",
        type=int,
        default=50,
        help="Number of sessions to issue. Default: 50.",
    )
    p_run.add_argument(
        "--n-turns",
        type=int,
        default=None,
        help=(
            "Synthetic workload: turns per session (ignored for trace replay). "
            "Default: from config.yaml workload.n_turns."
        ),
    )
    p_run.add_argument(
        "--synthetic-tokens-per-turn",
        type=int,
        default=None,
        help="Target output tokens per turn for synthetic workload generation (ignored for trace replay)",
    )
    p_run.add_argument("--data-dir", default="data", help="Local data directory")
    p_run.add_argument(
        "--prefix-overlap-fraction",
        type=float,
        default=0.0,
        help="Fraction of sessions sharing a common prefix for LMCache hits (0.0-1.0, default: 0.0)",
    )
    p_run.add_argument(
        "--source",
        choices=[
            "openhands",
            "swe-smith-tool",
            "swe-smith-xml",
            "swe-smith-ticks",
            "swe-bench-verified",
            "abc-bench",
            "swe-evo",
            "featurebench",
            "mixed",
        ],
        default=None,
        help="Data source for workload generation (omit for synthetic)",
    )
    p_run.add_argument(
        "--local-path", default=None, help="Local JSONL file path (instead of HuggingFace)"
    )
    p_run.add_argument(
        "--tool-mode",
        default="off",
        choices=["real", "replay", "off"],
        dest="tool_mode",
        help="Tool calling mode: 'off' = no tools (default), 'real' = local execution workspace, 'replay' = prefill from trace.",
    )
    p_run.add_argument(
        "--tool-max-turns",
        type=int,
        default=None,
        dest="tool_max_turns",
        help="Maximum tool-calling turns per session. Overrides the default tool-mode cap.",
    )
    p_run.add_argument(
        "--tool-call-parser-fallback",
        default="off",
        choices=["off", "deepseek_text", "hermes_xml"],
        dest="tool_call_parser_fallback",
        help=(
            "Optional text-to-tool fallback. "
            "'deepseek_text' converts <run>...</run> and fenced bash blocks into execute_bash."
        ),
    )
    p_run.add_argument(
        "--sanitize-truncated-tool-calls",
        action="store_true",
        default=False,
        dest="sanitize_truncated_tool_calls",
        help="Recover truncated tool-call JSON with empty arguments ({}) instead of "
        "reporting a parse error. Truncated tool calls are already failed generations; "
        "empty args let the tool return an error naturally, keeping the session alive.",
    )
    p_run.add_argument(
        "--continue-turn-on",
        default="never",
        choices=["never", "text-only"],
        dest="continue_turn_on",
        help=(
            "[EXPERIMENTAL] When to force the agent loop to continue past a response "
            "that would normally end the session. "
            "'never' (default): loop exits on no tool calls -- the tool-call-driven "
            "pattern used by Gemini CLI "
            "(https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/core/turn.ts), "
            "OpenAI Codex "
            "(https://github.com/openai/codex/blob/main/codex-cli/src/utils/agent/agent-loop.ts), "
            "Aider "
            "(https://github.com/Aider-AI/aider/blob/main/aider/coders/base_coder.py), "
            "and OpenHands "
            "(https://github.com/All-Hands-AI/OpenHands/blob/main/openhands/controller/stuck.py). "
            "'text-only': on a text-only response, append the text and nudge the model "
            "with --continue-prompt -- the react() pattern from Inspect AI "
            "(https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/agent/_react.py). "
            "May be removed if production harnesses handle this internally."
        ),
    )
    p_run.add_argument(
        "--continue-prompt",
        default=None,
        dest="continue_prompt",
        help=(
            "User message injected after a text-only assistant response when "
            "--continue-turn-on=text-only. Defaults to Inspect AI's continuation "
            "nudge (src/inspect_ai/agent/_react.py DEFAULT_CONTINUE_PROMPT)."
        ),
    )
    p_run.add_argument(
        "--multi-turn-only",
        action="store_true",
        default=False,
        dest="multi_turn_only",
        help=(
            "Filter the generated workload to sessions that have at least one "
            "pending follow-up user message (i.e. multi-user-turn traces from "
            "sources that advertise supports_multi_turn=True, currently "
            "openhands). Single-turn sessions are dropped. Use this to benchmark "
            "only the realistic multi-turn load case."
        ),
    )
    p_run.add_argument(
        "--max-session-time",
        type=float,
        default=0.0,
        dest="max_session_time",
        help=(
            "[EXPERIMENTAL] Per-session wall-clock cap in seconds (0 = disabled). "
            "Borrowed from Gemini CLI's maxTimeMinutes "
            "(https://github.com/google-gemini/gemini-cli). Catches runaway turns "
            "that max-turns alone would miss (e.g. a single turn running a long build)."
        ),
    )
    p_run.add_argument(
        "--auto-finish-on-exhaust",
        action="store_true",
        default=False,
        dest="auto_finish_on_exhaust",
        help=(
            "[EXPERIMENTAL] When the session hits max-turns / max-session-time / "
            "max-retry-turns / max-budget-usd / stuck, tag "
            "result.metadata['exhaust_reason'] so partial results are still "
            "scorable. Borrowed from SWE-agent's attempt_autosubmission_after_error "
            "(https://github.com/SWE-agent/SWE-agent/blob/main/sweagent/agent/agents.py)."
        ),
    )
    p_run.add_argument(
        "--max-retry-turns",
        type=int,
        default=0,
        dest="max_retry_turns",
        help=(
            "[EXPERIMENTAL] Cap on consecutive turns with invalid tool calls "
            "(0 = disabled). Counter resets on any valid turn. Borrowed from "
            "Aider's max_reflections "
            "(https://github.com/Aider-AI/aider/blob/main/aider/coders/base_coder.py), "
            "which caps remediation loops (lint/test retries) separately from the "
            "overall turn count."
        ),
    )
    p_run.add_argument(
        "--max-budget-usd",
        type=float,
        default=0.0,
        dest="max_budget_usd",
        help=(
            "[EXPERIMENTAL] Per-session USD cost cap (0 = disabled). Session "
            "terminates when accumulated input+output token cost exceeds the "
            "cap. Requires --input-price-per-mtok and --output-price-per-mtok. "
            "Borrowed from Claude Code's --max-budget-usd "
            "(https://code.claude.com/docs/en/cli-reference)."
        ),
    )
    p_run.add_argument(
        "--input-price-per-mtok",
        type=float,
        default=0.0,
        dest="input_price_per_mtok",
        help="[EXPERIMENTAL] Input token price in USD per 1M tokens. Required with --max-budget-usd.",
    )
    p_run.add_argument(
        "--output-price-per-mtok",
        type=float,
        default=0.0,
        dest="output_price_per_mtok",
        help="[EXPERIMENTAL] Output token price in USD per 1M tokens. Required with --max-budget-usd.",
    )
    p_run.add_argument(
        "--stream-idle-timeout",
        type=float,
        default=0.0,
        dest="stream_idle_timeout",
        help=(
            "[EXPERIMENTAL] Stream idle timeout in seconds (0 = disabled). "
            "Kills the request if no data is received for N consecutive "
            "seconds, distinct from --request-timeout (full request). "
            "Borrowed from Codex CLI's stream_idle_timeout default 300s "
            "(https://github.com/openai/codex, codex-rs/core/src/codex.rs)."
        ),
    )
    p_run.add_argument(
        "--retry-profile",
        default="default",
        choices=["default", "codex"],
        dest="retry_profile",
        help=(
            "[EXPERIMENTAL] Retry backoff tuning. 'default' (current): "
            "1s * 2^n + up to 50%% jitter. 'codex': 200ms * 2^n + 10%% jitter "
            "(Codex CLI default, DEFAULT_STREAM_MAX_RETRIES=5; "
            "https://github.com/openai/codex, codex-rs/core/src/util.rs). "
            "Only affects timing when --max-retries > 0."
        ),
    )
    p_run.add_argument(
        "--max-retries",
        type=int,
        default=None,
        dest="max_retries",
        help="Max retries per turn on transient errors (default: 0, preset overrides apply).",
    )
    p_run.add_argument(
        "--make-dataset",
        action="store_true",
        default=False,
        dest="make_dataset",
        help="Also produce HF-ready parquet files alongside JSON results",
    )
    p_run.set_defaults(func=cmd_run)

    p_sweep = sub.add_parser(
        "sweep",
        help="Sweep concurrency levels to find capacity boundaries",
        description="Sweep concurrency levels to find capacity boundaries",
        parents=[common, run_parent],
    )
    p_sweep.add_argument(
        "--probe-levels",
        type=_probe_levels_type,
        default="50,100,200,300,400,500",
        help="Comma-separated session counts to probe (default: 50,100,200,300,400,500)",
    )
    p_sweep.add_argument(
        "--source",
        choices=[
            "openhands",
            "swe-smith-tool",
            "swe-smith-xml",
            "swe-smith-ticks",
            "swe-bench-verified",
            "abc-bench",
            "swe-evo",
            "featurebench",
            "mixed",
        ],
        default=None,
        help="Workload source (omit for synthetic)",
    )
    p_sweep.add_argument(
        "--n-turns",
        type=int,
        default=None,
        help="Turns per session (default: config workload.n_turns)",
    )
    p_sweep.add_argument(
        "--synthetic-tokens-per-turn",
        type=int,
        default=None,
        help="Target output tokens per turn for synthetic workload generation (ignored for trace replay)",
    )
    p_sweep.add_argument(
        "--cooldown",
        type=int,
        default=30,
        help="Max seconds to wait for KV cooldown between levels (default: 30)",
    )
    p_sweep.add_argument(
        "--no-cooldown",
        action="store_true",
        default=False,
        help="Skip KV cooldown entirely between probes (fast exploratory sweeps)",
    )
    p_sweep.add_argument(
        "--max-error-rate",
        type=float,
        default=0.05,
        dest="max_error_rate",
        help="Maximum error rate for SLO pass (default: 0.05 = 5%%).",
    )
    p_sweep.add_argument(
        "--ttft-threshold",
        type=float,
        default=None,
        help="Stop sweep when TTFT p95 exceeds this (ms). Prints saturation point.",
    )
    p_sweep.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Local data directory with pre-downloaded JSONL files",
    )
    p_sweep.add_argument(
        "--prefix-overlap-fraction",
        type=float,
        default=0.0,
        help="Fraction of sessions sharing a common prefix for LMCache hits (0.0-1.0, default: 0.0)",
    )
    p_sweep.add_argument(
        "--max-n",
        type=int,
        default=500,
        help="Upper bound for auto mode / session count search (default: 500)",
    )
    p_sweep.add_argument(
        "--hint",
        type=int,
        default=None,
        help="Warm-start hint: approximate spike/SLO boundary N from a prior run. "
        "Skips exponential scan and bisects around [hint*0.5, hint*2.0]. "
        "Used with --slo-ms auto mode.",
    )
    p_sweep.add_argument(
        "--multi-turn-only",
        action="store_true",
        default=False,
        dest="multi_turn_only",
        help=(
            "Filter the generated workload to sessions that have at least one "
            "pending follow-up user message (i.e. multi-user-turn traces from "
            "sources that advertise supports_multi_turn=True, currently "
            "openhands). Single-turn sessions are dropped."
        ),
    )
    p_sweep.add_argument(
        "--tool-mode",
        default="off",
        choices=["real", "replay", "off"],
        dest="tool_mode",
        help="Tool calling mode: 'off' = no tools (default), 'real' = local execution workspace, 'replay' = prefill from trace.",
    )

    slo_group = p_sweep.add_argument_group("SLO binary search mode (--slo-ms)")
    slo_group.add_argument(
        "--slo-ms",
        type=float,
        default=None,
        help="SLO target: max acceptable p95 TTFT in ms. "
        "Enables binary search mode to find max N meeting SLO.",
    )
    slo_group.add_argument(
        "--profiles",
        nargs="*",
        default=None,
        help="Workload profiles as turns,tokens[,max_tokens] "
        "(e.g., 3,1500,256). Omit for default 5 profiles.",
    )
    slo_group.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag for output file (e.g., 'with-lmcache', 'no-lmcache')",
    )
    slo_group.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Comma-separated paths to config profile YAML files for "
        "multi-config comparison (e.g., tp8_apc.yaml,tp4_apc.yaml)",
    )

    p_sweep.set_defaults(func=cmd_sweep)

    p_gen = sub.add_parser(
        "generate",
        help="Generate workload files from trace data or synthetic patterns",
        description="Generate workload files from trace data or synthetic patterns",
        parents=[common, gen_parent],
    )
    p_gen.add_argument("--data-dir", default="data", help="Local data directory")
    p_gen.set_defaults(func=cmd_generate)

    # Install SIGTERM handler as early as possible so k8s/slurm preempt
    # signals don't kill us before partial results can flush. cli.md H2.
    _install_signal_handlers()

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(130)
    except _SigTerm:
        # SIGTERM (k8s eviction / slurm preempt). Subcommand finallies
        # have already flushed partial results.
        sys.exit(143)
    except Exception as exc:
        # cli.md H4: typed config errors should produce a friendly message,
        # not a traceback dump.
        from agentsurge.cli._helpers import CLIConfigError

        if isinstance(exc, CLIConfigError):
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        raise
