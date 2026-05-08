# SPDX-License-Identifier: MIT
"""[audit:misc-core#M9] tokenizer cache must not cache None permanently.

A transient HF Hub outage that causes ``_load_tokenizer`` to soft-fail
once should not disable the tokenizer for the rest of the process. The
fix: only cache successful loads, retry on subsequent calls if the
previous load returned None.
"""

from __future__ import annotations

from agentsurge import runner as runner_mod


def test_failed_tokenizer_load_is_not_cached(monkeypatch) -> None:
    # Restore the real _load_tokenizer; the autouse conftest fixture
    # replaces it with a stub that always succeeds, hiding the bug.
    import importlib

    real_runner = importlib.reload(runner_mod)
    monkeypatch.setattr("agentsurge.runner._load_tokenizer", real_runner._load_tokenizer)

    # Use a unique key to avoid coupling to other tests.
    test_repo = "audit-misc-core-h9/transient-fake"
    real_runner._tokenizer_cache.pop((test_repo, False), None)

    call_count = 0

    class _FakeTokenizer:
        @staticmethod
        def from_pretrained(model: str, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("simulated transient HF Hub outage")
            return object()  # second call succeeds

    class _FakeTransformers:
        AutoTokenizer = _FakeTokenizer

    import sys as _sys

    monkeypatch.setitem(_sys.modules, "transformers", _FakeTransformers)

    # First call: simulated transient failure. Soft-fall to None.
    first = real_runner._load_tokenizer(test_repo)
    assert first is None
    # The bug: the None gets cached, so the second call short-circuits
    # without retrying. After the fix, the second call retries the load.
    second = real_runner._load_tokenizer(test_repo)
    assert second is not None, (
        "tokenizer cache stuck on None after a transient failure; "
        "subsequent loads must retry to recover when HF Hub is back"
    )
