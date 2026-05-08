# SPDX-License-Identifier: MIT
"""H2: classes that define __eq__ without __hash__ should be intentionally unhashable.

Python silently sets __hash__ = None whenever __eq__ is defined on a regular
class without an accompanying __hash__. To prevent this from being a surprise,
the audit asks for the class to either explicitly declare __hash__ = None or
implement __hash__ to match __eq__. We assert the former: instances are
intentionally unhashable AND the class declares this fact.
"""

import inspect

import pytest

from agentsurge.types.metrics import MetricsSnapshot
from agentsurge.types.results import RunResult


def test_metrics_snapshot_is_intentionally_unhashable():
    s = MetricsSnapshot()
    with pytest.raises(TypeError):
        hash(s)
    assert MetricsSnapshot.__hash__ is None
    # The class must declare __hash__ = None explicitly in its body so that
    # the unhashability is a documented contract rather than a quiet
    # consequence of defining __eq__.
    src = inspect.getsource(MetricsSnapshot)
    assert "__hash__ = None" in src, (
        "MetricsSnapshot must explicitly declare __hash__ = None in the class "
        "body to document its intentional unhashability."
    )


def test_run_result_is_intentionally_unhashable():
    r = RunResult()
    with pytest.raises(TypeError):
        hash(r)
    assert RunResult.__hash__ is None
    src = inspect.getsource(RunResult)
    assert "__hash__ = None" in src, (
        "RunResult must explicitly declare __hash__ = None in the class "
        "body to document its intentional unhashability."
    )
