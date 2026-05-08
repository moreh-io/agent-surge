# SPDX-License-Identifier: MIT
"""FeatureBench task loader from HuggingFace datasets."""

from agentsurge.loaders.base import BaseHFTaskLoader


class FeatureBenchLoader(BaseHFTaskLoader):
    """Load tasks from LiberCoders/FeatureBench."""

    name = "featurebench"
    HF_DATASET = "LiberCoders/FeatureBench"
    HF_SPLIT = "lite"
