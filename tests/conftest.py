# Shared pytest fixtures for agentsurge tests.

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: unit tests (fast, no external deps)")
    config.addinivalue_line("markers", "e2e: integration tests (multi-component, may need deps)")


def pytest_collection_modifyitems(items):
    for item in items:
        if not any(m.name == "e2e" for m in item.iter_markers()):
            item.add_marker(pytest.mark.unit)


@pytest.fixture(autouse=True)
def _mock_tokenizer(monkeypatch):
    """Provide a mock tokenizer so BenchmarkRunner doesn't raise ValueError.

    Production requires a real tokenizer when ignore_replay_output_length=False (per-turn sizing).
    Tests use dummy model names that can't load real tokenizers.
    """

    class _FakeTokenizer:
        def encode(self, text):
            return list(range(max(1, len(text) // 4)))

    monkeypatch.setattr(
        "agentsurge.runner._load_tokenizer",
        lambda model, *, trust_remote_code=False: _FakeTokenizer(),
    )


# Shared mock helpers for test reuse


class MockAsyncLineIterator:
    """Async iterator over byte lines, for mocking SSE streams."""

    def __init__(self, lines: list[bytes]):
        self._lines = lines
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if self._index >= len(self._lines):
            raise StopAsyncIteration
        line = self._lines[self._index]
        self._index += 1
        return line


def make_mock_aiohttp_response(*, status=200, body_text="", sse_lines=None):
    """Create a mock aiohttp response with async context manager support."""
    from unittest.mock import AsyncMock, MagicMock

    resp = MagicMock()
    resp.status = status
    resp.text = AsyncMock(return_value=body_text)
    if sse_lines is not None:
        resp.content = MockAsyncLineIterator(sse_lines)
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def make_mock_aiohttp_session(response=None):
    """Create a mock aiohttp ClientSession."""
    from unittest.mock import AsyncMock, MagicMock

    session = MagicMock()
    if response is not None:
        session.post = MagicMock(return_value=response)
        session.get = MagicMock(return_value=response)
    session.close = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session
