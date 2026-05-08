"""M7: ``download()`` must use streaming to avoid materialising full datasets.

The non-streaming path materialises multi-GB datasets in RAM before
writing the JSONL.  ``load_dataset(..., streaming=True)`` keeps memory
flat at the cost of a sequential scan, which is what we want here since
``download`` immediately writes each row to disk.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_base_hf_task_loader_download_uses_streaming(tmp_path):
    from agentsurge.loaders.swe_bench import SWEBenchLoader

    rows = [
        {"instance_id": "i-1", "problem_statement": "p", "patch": "diff"},
        {"instance_id": "i-2", "problem_statement": "q", "patch": "diff"},
    ]
    mock_datasets = MagicMock()
    mock_datasets.load_dataset.return_value = iter(rows)

    loader = SWEBenchLoader()
    with patch.dict("sys.modules", {"datasets": mock_datasets}):
        out = loader.download(str(tmp_path))

    assert out
    # Streaming kwarg must be set on every download call.
    _args, kwargs = mock_datasets.load_dataset.call_args
    assert kwargs.get("streaming") is True, (
        f"BaseHFTaskLoader.download must pass streaming=True; got kwargs={kwargs}"
    )


def test_swe_smith_download_uses_streaming(tmp_path):
    from agentsurge.loaders.swe_smith import SWESmithLoader

    rows = [{"instance_id": "s-1", "messages": []}]
    mock_datasets = MagicMock()
    mock_datasets.load_dataset.return_value = iter(rows)

    loader = SWESmithLoader()
    with patch.dict("sys.modules", {"datasets": mock_datasets}):
        out = loader.download(str(tmp_path))

    assert out
    _args, kwargs = mock_datasets.load_dataset.call_args
    assert kwargs.get("streaming") is True, (
        f"SWESmithLoader.download must pass streaming=True; got kwargs={kwargs}"
    )
