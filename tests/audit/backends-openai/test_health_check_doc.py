"""M5: OpenAiBackend.health_check docstring must match its strict semantics.

Existing tests (and downstream callers) assert that health_check returns False
on transport error and on non-200 status. The original docstring claimed
"Falls back to always-true if the endpoint is unreachable", which is opposite
to the implementation. Make the docstring tell the truth.
"""

from __future__ import annotations

import re

from agentsurge.backends.openai import OpenAiBackend


def test_health_check_docstring_does_not_claim_always_true_fallback():
    doc = OpenAiBackend.health_check.__doc__ or ""
    assert "always-true" not in doc, (
        "docstring still claims always-true fallback while code returns False"
    )


def test_health_check_docstring_documents_strict_semantics():
    doc = OpenAiBackend.health_check.__doc__ or ""
    assert re.search(r"\bFalse\b", doc), (
        "docstring should explicitly mention returning False on failure"
    )
