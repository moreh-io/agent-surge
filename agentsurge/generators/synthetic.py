# SPDX-License-Identifier: MIT
"""Synthetic workload generators - code-like prompts and single-turn sessions."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from agentsurge.generators.base import GeneratorBase
from agentsurge.types import ReplaySession

if TYPE_CHECKING:
    from agentsurge.generators.trace_pool import TracePool


def _make_synthetic_prompt(n_tokens: int, seed: int, tokenizer=None) -> str:
    """Generate synthetic code prompt of ~n_tokens tokens.

    If tokenizer is provided, uses it to measure actual token count
    and adjusts output to match target. Otherwise uses 21 tok/iter heuristic.
    """

    def _gen_lines(n_iters: int, s: int) -> str:
        lines = []
        for i in range(n_iters):
            j = i + s * 10000
            lines.append(f"def f_{j}(x): return x*{j % 99}+{j % 50}  # L{j}")
            lines.append(f"def g_{j}(v): assert v>0, 'e{j}'")
        return "\n".join(lines) + f"\n# {s}: Say OK."

    # ~21 tokens per generated 2-line block with GPT-family BPE tokenizers.
    # Model-dependent; use tokenizer param for accuracy.
    _TOKENS_PER_SYNTH_ITER = 21
    iters = max(1, int(n_tokens) // _TOKENS_PER_SYNTH_ITER)

    if tokenizer is None:
        return _gen_lines(iters, seed)

    # Tokenizer-aware: initial estimate then adjust up to 3 attempts.
    _TOLERANCE = 0.1  # 10% acceptable deviation from target token count
    for _attempt in range(3):
        text = _gen_lines(iters, seed)
        actual = len(tokenizer.encode(text))
        if abs(actual - n_tokens) < n_tokens * _TOLERANCE:
            return text
        iters = max(1, int(iters * n_tokens / max(actual, 1)))
    # Final pass: if still overshooting target (the iters-floor lower-bound
    # case for n_tokens < ~21), trim by token boundary using the tokenizer.
    actual = len(tokenizer.encode(text))
    if actual > n_tokens * (1 + _TOLERANCE):
        try:
            ids = tokenizer.encode(text)
            keep = max(1, int(n_tokens))
            if hasattr(tokenizer, "decode"):
                text = tokenizer.decode(ids[:keep])
            else:
                # No decode available: approximate trim by char ratio.
                ratio = keep / max(len(ids), 1)
                text = text[: max(1, int(len(text) * ratio))]
        except Exception:
            return text
    return text


def _pad_assistant_response(stub: str, seed: int, min_tokens: int = 200) -> str:
    """Pad a short assistant stub with realistic code-like content.

    Real agent assistant turns are 200-2000 tokens. This generates
    deterministic filler that mimics code diffs and reasoning to produce
    realistic KV cache pressure.
    """
    rng = random.Random(seed)
    lines = [stub, ""]
    # BPE average for English code: ~3.5-4.5 chars/token (Karpathy 2023).
    # Tokenizer-aware path in _make_synthetic_prompt() is preferred for accuracy.
    _CHARS_PER_TOKEN_ESTIMATE = 4
    target_chars = min_tokens * _CHARS_PER_TOKEN_ESTIMATE
    fnames = ["utils.py", "handler.py", "models.py", "config.py", "tests/test_main.py"]
    actions = ["read_file", "edit_file", "run_tests", "search_code", "write_file"]
    while len("\n".join(lines)) < target_chars:
        fname = rng.choice(fnames)
        action = rng.choice(actions)
        var = f"v{rng.randint(0, 999)}"
        lines.append(f"I'll {action}(`{fname}`) to check the implementation.")
        lines.append("```python")
        for j in range(rng.randint(3, 8)):
            lines.append(
                f"    {var}_{j} = process(data[{rng.randint(0, 99)}], key='{fname[:4]}_{j}')"
            )
        lines.append("```")
        lines.append(f"This handles the {action.replace('_', ' ')} step correctly.")
        lines.append("")
    return "\n".join(lines)


class SyntheticGenerator(GeneratorBase):
    """Generate synthetic multi-turn sessions for load testing."""

    name = "synthetic"

    def __init__(
        self,
        system_prompt: str = "You are a coding assistant. Keep responses under 50 words.",
        tokens_per_turn: int = 2000,
        n_turns: int = 5,
        turn_distribution: str
        | None = None,  # None = fixed, "geometric" = draw from geometric dist
        fan_out: int = 1,  # parallel LLM calls per turn (>1 simulates map-reduce sub-queries)
        system_prompt_tokens: int = 0,  # if > 0, pad system prompt to this many tokens
        tokenizer: object | None = None,  # Any object with encode(str) -> list method
        prefix_overlap_fraction: float = 0.0,  # 0.0 = no sharing, 1.0 = all sessions share same prefix
        trace_pool: TracePool | None = None,  # real trace content for user messages
    ) -> None:
        self.tokens_per_turn = tokens_per_turn
        self.trace_pool = trace_pool
        self.n_turns = n_turns
        self.turn_distribution = turn_distribution
        self.fan_out = fan_out
        self.system_prompt_tokens = system_prompt_tokens
        self.tokenizer = tokenizer
        self.prefix_overlap_fraction = max(0.0, min(1.0, prefix_overlap_fraction))
        if system_prompt_tokens > 0:
            padding = _make_synthetic_prompt(system_prompt_tokens, 999999, tokenizer=tokenizer)
            self.system_prompt = f"{system_prompt}\n\n# Tool definitions and context:\n{padding}"
        else:
            self.system_prompt = system_prompt
        self._shared_prefix = (
            self._build_shared_prefix() if self.prefix_overlap_fraction > 0 else None
        )

    def _build_shared_prefix(self) -> str:
        """Build a realistic shared prefix (~1500-2500 tokens): system prompt + code context.

        Mimics a real agent setup where multiple sessions share the same
        system instructions and repository context, enabling LMCache prefix hits.
        """
        parts = [
            "You are an expert software engineer working on a large Python codebase.",
            "Follow these rules strictly:",
            "1. Always write type-annotated Python 3.10+ code.",
            "2. Use descriptive variable names and add docstrings to all public functions.",
            "3. Handle errors gracefully with specific exception types.",
            "4. Write unit tests for any new functionality.",
            "5. Do not modify files outside the specified module unless necessary.",
            "",
            "# Repository structure:",
            "```",
            "src/",
            "  core/",
            "    engine.py        # Main execution engine",
            "    scheduler.py     # Task scheduling and queue management",
            "    cache.py         # Multi-tier caching (memory, disk, remote)",
            "    config.py        # Configuration dataclasses",
            "  api/",
            "    routes.py        # FastAPI route handlers",
            "    middleware.py     # Auth, rate limiting, logging middleware",
            "    schemas.py       # Pydantic request/response models",
            "  workers/",
            "    pool.py          # Worker pool management",
            "    executor.py      # Task execution with retry logic",
            "  utils/",
            "    metrics.py       # Prometheus metrics collection",
            "    logging.py       # Structured logging setup",
            "tests/",
            "  test_engine.py",
            "  test_scheduler.py",
            "  test_cache.py",
            "  conftest.py        # Shared fixtures",
            "```",
            "",
        ]
        # 1500 tokens is representative of agent system prompts:
        # OpenHands ~3K tok, SWE-agent ~1.5K tok.
        _SHARED_PREFIX_TOKENS = 1500
        code_context = _make_synthetic_prompt(
            _SHARED_PREFIX_TOKENS, seed=0xBEEF, tokenizer=self.tokenizer
        )
        parts.append("# Current file context:")
        parts.append("```python")
        parts.append(code_context)
        parts.append("```")
        parts.append("")
        parts.append("Analyze the code above and respond to the user's request.")
        return "\n".join(parts)

    def _sample_n_turns(self, rng: random.Random) -> int:
        """Sample turn count per session.

        Default: fixed n_turns.
        "geometric": Geometric(p=0.35) + 1, giving P50≈1, P90≈5.
        Matches production traces (KVCache in the Wild, ATC'25).
        """
        if self.turn_distribution == "geometric":
            # Geometric(p=0.35): P50≈1, P90≈5.
            # Fit to "KVCache in the Wild" (ATC'25, Yin et al.) Fig.9 turn-count CDF:
            # 54% single-turn, P90≈5 turns. Validated against Azure production traces.
            _GEOMETRIC_P = 0.35
            _MAX_TURNS_CAP = 50  # safety cap; prevents infinite loop
            k = 1
            while rng.random() > _GEOMETRIC_P and k < _MAX_TURNS_CAP:
                k += 1
            return k
        return self.n_turns

    def generate(self, n_sessions: int, seed: int = 42) -> list[ReplaySession]:
        """Generate n_sessions synthetic replay sessions.

        When prefix_overlap_fraction > 0, that fraction of sessions get a shared
        prefix (system prompt + code context) prepended to turn 0, enabling
        LMCache prefix cache hits across sessions.
        """
        sessions: list[ReplaySession] = []
        rng = random.Random(seed)

        n_shared = int(n_sessions * self.prefix_overlap_fraction)

        for s in range(n_sessions):
            n_t = self._sample_n_turns(rng)
            use_shared = self._shared_prefix is not None and s < n_shared

            if use_shared:
                sys_content = f"{self.system_prompt}\n\n{self._shared_prefix}"
            else:
                sys_content = self.system_prompt

            messages: list[dict] = [{"role": "system", "content": sys_content}]
            turn_boundaries: list[int] = []

            for t in range(n_t):
                if self.trace_pool is not None:
                    turn_rng = random.Random(seed * 1000000 + s * 1000 + t)
                    prompt = self.trace_pool.sample(turn_rng, target_tokens=self.tokens_per_turn)
                else:
                    prompt = _make_synthetic_prompt(
                        self.tokens_per_turn,
                        seed * 1000000 + s * 1000 + t,
                        tokenizer=self.tokenizer,
                    )
                messages.append({"role": "user", "content": prompt})
                turn_boundaries.append(len(messages))

                if t < n_t - 1:
                    if self.trace_pool is not None:
                        asst_rng = random.Random(seed * 10000000 + s * 10000 + t + 5000000)
                        asst_content = self.trace_pool.sample_assistant(asst_rng, target_tokens=200)
                        if not asst_content:
                            asst_content = f"Code in module {s}_{t} looks fine."
                    else:
                        asst_content = f"Code in module {s}_{t} looks fine."
                    messages.append(
                        {
                            "role": "assistant",
                            "content": asst_content,
                        }
                    )

            turn_messages: list[list[dict]] = [messages[:end] for end in turn_boundaries]

            sessions.append(
                ReplaySession(
                    session_id=f"synthetic_{s}",
                    turn_messages=turn_messages,
                    metadata={
                        "synthetic": True,
                        "base_seed": seed,
                        "session_index": s,
                        "shared_prefix": use_shared,
                    },
                    fan_out=self.fan_out,
                )
            )

        return sessions


class SingleTurnGenerator(GeneratorBase):
    """Generate single-turn ReplaySessions for traffic mixing.

    Profile token counts scale relative to max_model_len.
    Context distributions are use-case specific, grounded in empirical data.
    """

    name = "single-turn"

    DISTRIBUTIONS = {
        "chat": {
            "short": {"ratio": 0.03, "weight": 0.65},
            "medium": {"ratio": 0.12, "weight": 0.27},
            "long": {"ratio": 0.37, "weight": 0.08},
        },
        "agent": {
            "short": {"ratio": 0.06, "weight": 0.10},
            "medium": {"ratio": 0.25, "weight": 0.40},
            "long": {"ratio": 0.50, "weight": 0.50},
        },
        "mixed": {
            "short": {"ratio": 0.03, "weight": 0.50},
            "medium": {"ratio": 0.12, "weight": 0.30},
            "long": {"ratio": 0.37, "weight": 0.20},
        },
        "autocomplete": {
            "short": {"ratio": 0.03, "weight": 0.85},
            "medium": {"ratio": 0.12, "weight": 0.12},
            "long": {"ratio": 0.37, "weight": 0.03},
        },
        "ssd-stress": {
            "short": {"ratio": 0.80, "weight": 0.30},
            "medium": {"ratio": 0.90, "weight": 0.40},
            "long": {"ratio": 1.00, "weight": 0.30},
        },
    }

    def __init__(
        self,
        max_model_len: int = 16384,
        system_prompt: str = "You are a helpful assistant.",
        seed: int = 42,
        distribution: str = "mixed",
    ) -> None:
        self.max_model_len = max_model_len
        self.system_prompt = system_prompt
        self.seed = seed
        if distribution not in self.DISTRIBUTIONS:
            raise ValueError(
                f"Unknown distribution {distribution!r}. "
                f"Choose from: {list(self.DISTRIBUTIONS.keys())}"
            )
        self.distribution = distribution
        self._profiles = self.DISTRIBUTIONS[distribution]

    def generate(self, n_sessions: int) -> list[ReplaySession]:
        rng = random.Random(self.seed)
        sessions = []
        profile_names = list(self._profiles.keys())
        profile_weights = [p["weight"] for p in self._profiles.values()]
        for i in range(n_sessions):
            profile = rng.choices(profile_names, weights=profile_weights)[0]
            tokens = int(self.max_model_len * self._profiles[profile]["ratio"])
            prompt = _make_synthetic_prompt(tokens, self.seed * 1000000 + i)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            sessions.append(
                ReplaySession(
                    session_id=f"single_{i}",
                    turn_messages=[messages],
                    metadata={"single_turn": True, "profile": profile},
                )
            )
        return sessions
