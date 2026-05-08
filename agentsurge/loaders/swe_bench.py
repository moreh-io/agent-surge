# SPDX-License-Identifier: MIT
"""SWE-bench Verified task loader from HuggingFace."""

from agentsurge.loaders.base import BaseHFTaskLoader


class SWEBenchLoader(BaseHFTaskLoader):
    """Load tasks from princeton-nlp/SWE-bench_Verified."""

    name = "swe-bench-verified"
    HF_DATASET = "princeton-nlp/SWE-bench_Verified"
    HF_SPLIT = "test"
