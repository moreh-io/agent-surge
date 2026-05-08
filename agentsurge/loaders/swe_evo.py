# SPDX-License-Identifier: MIT
"""SWE-EVO task loader from HuggingFace datasets."""

from agentsurge.loaders.base import BaseHFTaskLoader


class SWEEvoLoader(BaseHFTaskLoader):
    """Load tasks from Fsoft-AIC/SWE-EVO."""

    name = "swe-evo"
    HF_DATASET = "Fsoft-AIC/SWE-EVO"
    HF_SPLIT = "test"
