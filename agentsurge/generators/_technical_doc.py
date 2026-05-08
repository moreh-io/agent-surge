# SPDX-License-Identifier: MIT
"""Deterministic technical documentation content for synthetic prefix workloads."""

import json
import random
from pathlib import Path

_TECHNICAL_TOPICS_PATH = Path(__file__).parent.parent / "data" / "technical_topics.json"
_TOPICS: list[tuple[str, list[str]]] | None = None


def _load_topics() -> list[tuple[str, list[str]]]:
    global _TOPICS
    if _TOPICS is None:
        _TOPICS = [(k, v) for k, v in json.loads(_TECHNICAL_TOPICS_PATH.read_text()).items()]
    return _TOPICS


def _make_technical_doc(n_chars: int, seed: int) -> str:
    """Generate deterministic technical documentation content (~n_chars characters).

    Produces realistic distributed-systems / ML documentation paragraphs that
    are suitable as a shared system-prompt prefix for prefix cache testing.
    The output is IDENTICAL for the same (n_chars, seed) pair across calls.
    """
    rng = random.Random(seed)
    topics = _load_topics()

    paragraphs: list[str] = []
    total = 0
    topic_idx = 0
    while total < n_chars:
        title, facts = topics[topic_idx % len(topics)]
        topic_idx += 1
        shuffled = list(facts)
        rng.shuffle(shuffled)
        header = f"\n## {title}\n\n"
        body = ""
        for fact in shuffled:
            detail_idx = rng.randint(0, 999)
            body += (
                f"{fact} "
                f"This property is critical for system reliability at scale "
                f"(ref: design-doc-{detail_idx:04d}).\n"
            )
        block = header + body
        paragraphs.append(block)
        total += len(block)
        if total >= n_chars and topic_idx >= len(topics):
            break  # at least one full pass

    result = "".join(paragraphs)
    return result[:n_chars]
