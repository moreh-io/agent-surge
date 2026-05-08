#!/usr/bin/env python3
"""Public run() API - the primary programmatic entry point.

Shows how to use :func:`agentsurge.run` and :func:`agentsurge.arun` to
execute benchmarks from Python code.

Usage::

    # Live (requires vLLM server at the given URL)
    python examples/05_run_api.py http://localhost:8000 my-model

    # CI / offline (uses mock backend, no server needed)
    AGENTSURGE_EXAMPLE_MOCK=1 python examples/05_run_api.py
"""

from __future__ import annotations

import os
import sys

from agentsurge import RunResult, run


def main() -> None:
    mock = os.environ.get("AGENTSURGE_EXAMPLE_MOCK", "")
    if mock:
        vllm_url, model = "http://mock:8000", "mock-model"
        backend, no_metrics = "mock", True
    elif len(sys.argv) >= 3:
        vllm_url, model = sys.argv[1], sys.argv[2]
        backend, no_metrics = None, False
    else:
        print("Usage: python examples/05_run_api.py <vllm_url> <model>")
        print("       AGENTSURGE_EXAMPLE_MOCK=1 python examples/05_run_api.py")
        sys.exit(1)

    result: RunResult = run(
        vllm_url=vllm_url,
        model=model,
        n_sessions=5,
        n_turns=3,
        max_concurrency=4,
        backend=backend,
        no_metrics=no_metrics,
        ignore_replay_output_length=True,
    )

    print(f"Sessions : {len(result.sessions)}")
    print(f"Elapsed  : {result.total_elapsed_s:.2f}s")
    print(f"OK       : {sum(s.ok for s in result.sessions)}/{len(result.sessions)}")


if __name__ == "__main__":
    main()
