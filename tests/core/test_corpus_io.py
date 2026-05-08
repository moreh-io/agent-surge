"""Tests for corpus I/O - save_corpus / load_corpus round-trip."""

import json
import os
import tempfile

from agentsurge.types import ReplaySession


class TestCorpusRoundTrip:
    def _make_sessions(self, n=3):
        sessions = []
        for i in range(n):
            sessions.append(
                ReplaySession(
                    session_id=f"sess_{i}",
                    turn_messages=[
                        [
                            {"role": "system", "content": "You are helpful."},
                            {"role": "user", "content": f"Question {i}"},
                        ],
                    ],
                    metadata={"source": "test", "idx": i},
                )
            )
        return sessions

    def test_round_trip(self):
        from agentsurge.io import load_corpus, save_corpus

        sessions = self._make_sessions(5)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corpus.json")
            manifest = save_corpus(sessions, path)
            assert manifest["n_sessions"] == 5
            assert os.path.exists(path)

            loaded, loaded_manifest = load_corpus(path)
            assert len(loaded) == 5
            assert loaded_manifest["n_sessions"] == 5
            for orig, reloaded in zip(sessions, loaded, strict=False):
                assert orig.session_id == reloaded.session_id
                assert orig.turn_messages == reloaded.turn_messages
                assert orig.metadata == reloaded.metadata

    def test_manifest_source_counts(self):
        from agentsurge.io import save_corpus

        sessions = self._make_sessions(3)
        # Override source metadata
        sessions[0].metadata["source"] = "alpha"
        sessions[1].metadata["source"] = "alpha"
        sessions[2].metadata["source"] = "beta"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corpus.json")
            manifest = save_corpus(sessions, path)
            assert manifest["source_counts"]["alpha"] == 2
            assert manifest["source_counts"]["beta"] == 1

    def test_creates_parent_dirs(self):
        from agentsurge.io import save_corpus

        sessions = self._make_sessions(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "deep", "nested", "corpus.json")
            save_corpus(sessions, path)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert len(data["sessions"]) == 1

    def test_fan_out_preserved(self):
        from agentsurge.io import load_corpus, save_corpus

        session = ReplaySession(
            session_id="fan",
            turn_messages=[[{"role": "user", "content": "hi"}]],
            fan_out=3,
            lineage_id="root",
            parent_session_id="parent",
            branch_id=2,
            branch_depth=1,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corpus.json")
            save_corpus([session], path)
            loaded, _ = load_corpus(path)
            s = loaded[0]
            assert s.fan_out == 3
            assert s.lineage_id == "root"
            assert s.parent_session_id == "parent"
            assert s.branch_id == 2
            assert s.branch_depth == 1

    def test_versioned_envelope(self):
        from agentsurge.io import save_corpus

        sessions = self._make_sessions(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corpus.json")
            save_corpus(sessions, path)
            with open(path) as f:
                data = json.load(f)
            assert "version" in data
            assert data["version"] == "1.0"
            assert "manifest" in data
            assert "sessions" in data
            assert isinstance(data["sessions"], list)
