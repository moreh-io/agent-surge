"""Tests for trace pool extraction, serialization, and sampling."""

import json
import random

import pytest

from agentsurge.generators.trace_pool import Snippet, TracePool, _fit_lognormal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_corpus(tmp_path, n_sessions=5, n_tool_msgs=10):
    """Create a small synthetic corpus JSON for testing extraction."""
    sessions = []
    for s in range(n_sessions):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": f"Fix bug #{s} in handler.py"},
        ]
        for t in range(n_tool_msgs):
            msgs.append(
                {
                    "role": "assistant",
                    "content": f"I'll read file_{s}_{t}.py",
                }
            )
            body = f"def func_{s}_{t}(x):\n" + "\n".join(
                f"    line_{i} = process(data[{i}], key='k{i}')" for i in range(30 + t * 5)
            )
            msgs.append(
                {
                    "role": "user",
                    "content": f"[Tool output: read_file] {body}",
                }
            )
        sessions.append(
            {
                "session_id": f"s{s}",
                "turn_messages": [msgs],
                "metadata": {},
            }
        )
    corpus = {"version": "1.0", "sessions": sessions}
    path = tmp_path / "test_corpus.json"
    path.write_text(json.dumps(corpus))
    return path


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


class TestExtraction:
    def test_extract_produces_snippets(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        out = tmp_path / "pool.json"
        meta = TracePool.extract([corpus], out, min_body_chars=50)
        assert meta["n_snippets"] > 0
        assert out.exists()

    def test_extract_deduplicates(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        out = tmp_path / "pool.json"
        meta = TracePool.extract([corpus], out, min_body_chars=50)
        out2 = tmp_path / "pool2.json"
        meta2 = TracePool.extract([corpus, corpus], out2, min_body_chars=50)
        assert meta2["n_snippets"] == meta["n_snippets"]

    def test_extract_min_body_chars_filters(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        out_low = tmp_path / "low.json"
        out_high = tmp_path / "high.json"
        meta_low = TracePool.extract([corpus], out_low, min_body_chars=10)
        meta_high = TracePool.extract([corpus], out_high, min_body_chars=5000)
        assert meta_low["n_snippets"] >= meta_high["n_snippets"]


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_round_trip(self, tmp_path):
        snips = [
            Snippet(tool_type="bash", tok_estimate=100, body="$ pytest\nPASSED"),
            Snippet(tool_type="read_file", tok_estimate=200, body="def foo():\n    return 42"),
        ]
        pool = TracePool(
            snippets=snips,
            distributions={"bash": {"mu": 5.0, "sigma": 1.5, "n": 100}},
            meta={"test": True},
        )
        path = tmp_path / "pool.json"
        pool.to_json(path)
        loaded = TracePool.from_json(path)
        assert len(loaded.snippets) == 2
        assert loaded.snippets[0].tool_type == "bash"
        assert loaded.distributions["bash"]["mu"] == 5.0

    def test_loads_legacy_fragments_key(self, tmp_path):
        """from_json should accept old 'fragments' key for backward compat."""
        data = {
            "version": "1.0",
            "meta": {},
            "distributions": {},
            "fragments": [
                {"tool_type": "bash", "tok_estimate": 50, "body": "hello"},
            ],
        }
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps(data))
        pool = TracePool.from_json(path)
        assert len(pool.snippets) == 1


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class TestSampling:
    @pytest.fixture
    def pool(self):
        snips = []
        for i in range(100):
            tok = 50 + i * 20
            body = f"content_{i}\n" * (tok // 3)
            snips.append(Snippet(tool_type="bash", tok_estimate=tok, body=body))
        for i in range(80):
            tok = 30 + i * 25
            body = f"file_content_{i}\n" * (tok // 4)
            snips.append(Snippet(tool_type="read_file", tok_estimate=tok, body=body))
        return TracePool(
            snippets=snips,
            distributions={
                "bash": {"mu": 5.3, "sigma": 1.3, "n": 100},
                "read_file": {"mu": 5.5, "sigma": 1.4, "n": 80},
                "_aggregate": {"mu": 5.4, "sigma": 1.35, "n": 180},
            },
        )

    def test_sample_returns_string(self, pool):
        body = pool.sample(random.Random(42))
        assert isinstance(body, str)
        assert len(body) > 0
        assert body.strip(), "sample should contain non-whitespace content"

    def test_sample_deterministic(self, pool):
        a = pool.sample(random.Random(42))
        b = pool.sample(random.Random(42))
        assert a == b

    def test_sample_with_target(self, pool):
        body = pool.sample(random.Random(42), target_tokens=500)
        actual_tok = len(body) // 4
        assert 100 < actual_tok < 2000

    def test_sample_with_tool_type(self, pool):
        body = pool.sample(random.Random(42), tool_type="bash")
        assert isinstance(body, str)
        assert len(body) > 10, "tool-typed sample should have non-trivial length"

    def test_sample_lognormal(self, pool):
        rng = random.Random(42)
        lengths = [len(pool.sample_lognormal(rng)) // 4 for _ in range(50)]
        assert len(set(lengths)) > 5

    def test_tool_types(self, pool):
        assert set(pool.tool_types) == {"bash", "read_file"}


# ---------------------------------------------------------------------------
# LogNormal fit
# ---------------------------------------------------------------------------


class TestLogNormalFit:
    def test_fit_basic(self):
        values = [100, 200, 300, 500, 1000, 2000, 5000]
        mu, sigma = _fit_lognormal(values)
        assert 4.0 < mu < 8.0
        assert 0.5 < sigma < 3.0

    def test_fit_uniform_values(self):
        values = [100] * 20
        mu, sigma = _fit_lognormal(values)
        assert sigma == 0.1


# ---------------------------------------------------------------------------
# Integration: TaskToTrajectoryConverter with trace pool
# ---------------------------------------------------------------------------


class TestConverterWithPool:
    @pytest.fixture
    def pool(self):
        snips = []
        for i in range(50):
            tok = 100 + i * 30
            body = f"real_code_from_trace_{i}\n" * (tok // 6)
            snips.append(Snippet(tool_type="execute_bash", tok_estimate=tok, body=body))
            snips.append(Snippet(tool_type="str_replace_editor", tok_estimate=tok, body=body))
        return TracePool(
            snippets=snips,
            distributions={
                "execute_bash": {"mu": 5.3, "sigma": 1.3, "n": 50},
                "str_replace_editor": {"mu": 5.5, "sigma": 1.4, "n": 50},
                "_aggregate": {"mu": 5.4, "sigma": 1.35, "n": 100},
            },
        )

    def test_converter_uses_pool_content(self, pool):
        from agentsurge.generators.task_converter import TaskToTrajectoryConverter
        from agentsurge.types import Task

        conv = TaskToTrajectoryConverter(trace_pool=pool)
        task = Task(task_id="t1", prompt="Fix null pointer in views.py", metadata={"repo": "x/y"})
        trajs = conv.from_tasks(iter([task]), "bug-fix")
        turns = trajs[0].sessions[0].turns
        assert "real_code_from_trace_" in turns[2].content

    def test_pool_uses_read_file_alias_for_editor_request(self):
        pool = TracePool(
            snippets=[Snippet(tool_type="read_file", tok_estimate=100, body="read_file_body")],
            distributions={"read_file": {"mu": 5.0, "sigma": 1.0, "n": 1}},
        )

        body = pool.sample(random.Random(1), tool_type="str_replace_editor", target_tokens=100)

        assert body == "read_file_body"

    def test_converter_without_pool_still_works(self):
        from agentsurge.generators.task_converter import TaskToTrajectoryConverter
        from agentsurge.types import Task

        conv = TaskToTrajectoryConverter()
        task = Task(task_id="t1", prompt="Fix null pointer in views.py", metadata={"repo": "x/y"})
        trajs = conv.from_tasks(iter([task]), "bug-fix")
        turns = trajs[0].sessions[0].turns
        assert len(turns) in (6, 14)  # linear or retry variant
        assert "[Tool output" in turns[2].content
