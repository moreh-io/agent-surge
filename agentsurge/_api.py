# SPDX-License-Identifier: MIT
"""Public programmatic API for running benchmarks.

Provides :func:`run` (sync) and :func:`arun` (async) as the primary
entry points for running agentsurge benchmarks from Python code.

Usage::

    from agentsurge import run

    result = run(vllm_url="http://localhost:8000", model="my-model")
    print(result.total_elapsed_s, result.kv_util_peak)
"""

from __future__ import annotations

import asyncio

from agentsurge.callbacks import OnSessionCallback, OnTurnCallback
from agentsurge.types import BenchmarkConfig, ReplaySession, RunResult


async def arun(
    vllm_url: str | None = None,
    model: str | None = None,
    sessions: list[ReplaySession] | None = None,
    *,
    config: BenchmarkConfig | None = None,
    preset: str | None = None,
    n_sessions: int = 10,
    n_turns: int = 5,
    synthetic_tokens_per_turn: int = 500,
    max_concurrency: int | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    arrival_pattern: str | None = None,
    backend: str | None = None,
    no_metrics: bool = False,
    seed: int | None = None,
    on_turn: OnTurnCallback | None = None,
    on_session: OnSessionCallback | None = None,
    **extra_config: object,
) -> RunResult:
    """High-level async entry point for running benchmarks.

    At minimum, provide ``vllm_url`` and ``model`` (either directly or
    via a pre-built ``config``).  If no ``sessions`` are given, synthetic
    sessions are generated automatically.

    Parameters
    ----------
    vllm_url:
        Base URL of the vLLM server (e.g. ``"http://localhost:8000"``).
    model:
        Model name or path served by vLLM.
    sessions:
        Explicit list of :class:`ReplaySession` to replay.  When ``None``,
        synthetic sessions are generated using *n_sessions*, *n_turns*,
        and *synthetic_tokens_per_turn*.
    config:
        Pre-built :class:`BenchmarkConfig`.  When provided, *vllm_url*,
        *model*, and other config-level kwargs are merged on top (the
        explicit kwargs win).
    preset:
        Named preset (e.g. ``"measure"``, ``"agent"``).  Merged into the
        config before explicit kwargs.
    n_sessions:
        Number of synthetic sessions to generate (ignored when *sessions*
        is provided).
    n_turns:
        Turns per synthetic session.
    synthetic_tokens_per_turn:
        Approximate token budget per turn for synthetic sessions.
    max_concurrency:
        Maximum number of sessions running concurrently.
    max_tokens:
        Maximum tokens per LLM response.
    temperature:
        Sampling temperature.
    arrival_pattern:
        Session arrival pattern (``"poisson"``, ``"burst"``, ``"constant"``,
        ``"gamma"``, ``"ramp"``).
    backend:
        Backend adapter name (e.g. ``"openai"``, ``"mock"``).  ``None``
        uses the built-in HTTP path.
    no_metrics:
        Disable Prometheus metrics polling.
    seed:
        Random seed for reproducibility.
    on_turn:
        Callback invoked after each turn completes.
    on_session:
        Callback invoked after each session completes.
    **extra_config:
        Additional keyword arguments forwarded to :class:`BenchmarkConfig`.

    Returns
    -------
    RunResult
        Collected benchmark results.
    """
    from agentsurge.runner import BenchmarkRunner

    cfg = _build_config(
        vllm_url=vllm_url,
        model=model,
        config=config,
        preset=preset,
        max_concurrency=max_concurrency,
        max_tokens=max_tokens,
        temperature=temperature,
        arrival_pattern=arrival_pattern,
        no_metrics=no_metrics,
        seed=seed,
        backend_name=backend,
        **extra_config,
    )

    if sessions is None:
        from agentsurge.generators.synthetic import SyntheticGenerator

        gen = SyntheticGenerator(
            tokens_per_turn=synthetic_tokens_per_turn,
            n_turns=n_turns,
        )
        sessions = gen.generate(n_sessions=n_sessions)

    backend_obj = None
    if backend is not None:
        from agentsurge.backends import get_backend
        from agentsurge.backends.base import BackendConfig

        backend_cls = get_backend(backend)
        if backend == "mock":
            from agentsurge.backends.mock import MockConfig

            backend_obj = backend_cls(MockConfig())
        else:
            backend_obj = backend_cls(
                BackendConfig(
                    base_url=cfg.vllm_url,
                    model=cfg.model,
                    max_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                    stream_idle_timeout=cfg.stream_idle_timeout,
                )
            )

    runner = BenchmarkRunner(
        cfg,
        on_turn=on_turn,
        on_session=on_session,
        backend=backend_obj,
    )
    return await runner.run(sessions)


def run(
    vllm_url: str | None = None,
    model: str | None = None,
    sessions: list[ReplaySession] | None = None,
    *,
    config: BenchmarkConfig | None = None,
    preset: str | None = None,
    n_sessions: int = 10,
    n_turns: int = 5,
    synthetic_tokens_per_turn: int = 500,
    max_concurrency: int | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    arrival_pattern: str | None = None,
    backend: str | None = None,
    no_metrics: bool = False,
    seed: int | None = None,
    on_turn: OnTurnCallback | None = None,
    on_session: OnSessionCallback | None = None,
    **extra_config: object,
) -> RunResult:
    """Synchronous wrapper around :func:`arun`.

    Accepts the same parameters as :func:`arun`.  Calls
    ``asyncio.run(arun(...))`` internally.

    Returns
    -------
    RunResult
        Collected benchmark results.
    """
    return asyncio.run(
        arun(
            vllm_url=vllm_url,
            model=model,
            sessions=sessions,
            config=config,
            preset=preset,
            n_sessions=n_sessions,
            n_turns=n_turns,
            synthetic_tokens_per_turn=synthetic_tokens_per_turn,
            max_concurrency=max_concurrency,
            max_tokens=max_tokens,
            temperature=temperature,
            arrival_pattern=arrival_pattern,
            backend=backend,
            no_metrics=no_metrics,
            seed=seed,
            on_turn=on_turn,
            on_session=on_session,
            **extra_config,
        )
    )


def _build_config(
    *,
    vllm_url: str | None,
    model: str | None,
    config: BenchmarkConfig | None,
    preset: str | None,
    max_concurrency: int | None,
    max_tokens: int | None,
    temperature: float | None,
    arrival_pattern: str | None,
    no_metrics: bool,
    seed: int | None,
    backend_name: str | None,
    **extra: object,
) -> BenchmarkConfig:
    """Build a BenchmarkConfig from the various input sources."""
    kwargs: dict = {}

    if config is not None:
        from dataclasses import fields

        for f in fields(config):
            kwargs[f.name] = getattr(config, f.name)

    if preset is not None:
        from agentsurge.preset import preset_to_kwargs, resolve_preset_config

        raw = resolve_preset_config(preset)
        preset_kwargs = preset_to_kwargs(raw)
        kwargs.update(preset_kwargs)
        kwargs["preset"] = preset

    if vllm_url is not None:
        kwargs["vllm_url"] = vllm_url
    if model is not None:
        kwargs["model"] = model
    if max_concurrency is not None:
        kwargs["max_concurrency"] = max_concurrency

    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    if arrival_pattern is not None:
        kwargs["arrival_pattern"] = arrival_pattern
    if seed is not None:
        kwargs["seed"] = seed

    if no_metrics:
        kwargs["no_metrics"] = True
    if backend_name == "mock":
        kwargs["no_metrics"] = True

    kwargs.update(extra)

    if "vllm_url" not in kwargs:
        raise ValueError("vllm_url is required: pass it directly or via a BenchmarkConfig")
    if "model" not in kwargs:
        raise ValueError("model is required: pass it directly or via a BenchmarkConfig")

    return BenchmarkConfig(**kwargs)
