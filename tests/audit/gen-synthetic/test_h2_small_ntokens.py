"""Audit H2: _make_synthetic_prompt must approach the requested token count
even for small n_tokens, when a tokenizer is supplied.

Existing code floors `iters` at 1 and emits ~21 tokens per iter; for small
targets (e.g. 5 tokens) the function silently returns 21+ tokens with no
mechanism to shrink. The contract is "≈n_tokens tokens" — within tolerance
the result must satisfy `actual <= n_tokens * (1 + tolerance)` even at the
lower bound.
"""

from __future__ import annotations

from agentsurge.generators.synthetic import _make_synthetic_prompt


class _CharTokenizer:
    """Trivial deterministic tokenizer: 1 token per ~3 chars.

    Encoded length is a stable integer derived from the text, so the
    convergence loop has a real target to converge to.
    """

    def encode(self, text: str) -> list[int]:
        # ~3 chars per token approximates GPT-family BPE on code.
        return [0] * max(1, len(text) // 3)


def test_make_synthetic_prompt_small_ntokens_within_tolerance():
    tok = _CharTokenizer()
    n_tokens = 5
    text = _make_synthetic_prompt(n_tokens, seed=1, tokenizer=tok)
    actual = len(tok.encode(text))
    # Allow generous 2x slack; the bug returns ~50+ tokens.
    assert actual <= n_tokens * 2, (
        f"requested {n_tokens} tokens, got {actual} — function cannot shrink "
        "below ~21 tokens-per-iter floor"
    )
