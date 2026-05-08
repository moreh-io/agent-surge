"""M5: JSONL loaders must tolerate UTF-8 BOM and skip malformed lines.

Currently ``open(path)`` defaults to platform encoding and a single bad
line aborts the whole iteration.  Real-world JSONL (esp. Windows-produced)
sometimes ships a BOM; large dumps may contain one corrupt line.
"""

from __future__ import annotations

import json


def test_swe_smith_loader_handles_bom(tmp_path):
    from agentsurge.loaders.swe_smith import SWESmithLoader

    sample = {"instance_id": "smith-bom", "messages": []}
    path = tmp_path / "bom.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + (json.dumps(sample) + "\n").encode("utf-8"))

    trajs = list(SWESmithLoader(local_path=str(path)).load())
    assert len(trajs) == 1
    assert trajs[0].trajectory_id == "smith-bom"


def test_openhands_loader_handles_bom(tmp_path):
    from agentsurge.loaders.openhands import OpenHandsLoader

    sample = {"instance_id": "oh-bom", "messages": []}
    path = tmp_path / "bom.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + (json.dumps(sample) + "\n").encode("utf-8"))

    trajs = list(OpenHandsLoader(local_path=str(path)).load())
    assert len(trajs) == 1
    assert trajs[0].trajectory_id == "oh-bom"


def test_base_hf_task_loader_handles_bom(tmp_path):
    from agentsurge.loaders.swe_bench import SWEBenchLoader

    sample = {"instance_id": "id-1", "problem_statement": "p", "patch": "diff"}
    path = tmp_path / "bom.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + (json.dumps(sample) + "\n").encode("utf-8"))

    tasks = list(SWEBenchLoader(local_path=str(path)).load())
    assert len(tasks) == 1
    assert tasks[0].task_id == "id-1"


def test_swe_smith_loader_skips_malformed_line(tmp_path, caplog):
    from agentsurge.loaders.swe_smith import SWESmithLoader

    good = {"instance_id": "good", "messages": []}
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        json.dumps(good) + "\n"
        "{not valid json}\n" + json.dumps({**good, "instance_id": "good2"}) + "\n"
    )

    trajs = list(SWESmithLoader(local_path=str(path)).load())
    ids = [t.trajectory_id for t in trajs]
    assert ids == ["good", "good2"]


def test_base_hf_task_loader_skips_malformed_line(tmp_path):
    from agentsurge.loaders.swe_bench import SWEBenchLoader

    good = {"instance_id": "id-1", "problem_statement": "p", "patch": "diff"}
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        json.dumps(good) + "\n{garbage}\n" + json.dumps({**good, "instance_id": "id-2"}) + "\n"
    )

    tasks = list(SWEBenchLoader(local_path=str(path)).load())
    ids = [t.task_id for t in tasks]
    assert ids == ["id-1", "id-2"]
