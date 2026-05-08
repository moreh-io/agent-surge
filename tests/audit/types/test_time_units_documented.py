# SPDX-License-Identifier: MIT
"""M1: timestamp / start_time / end_time / wall_*_ms must document units.

MetricsSnapshot.timestamp is monotonic-raw seconds (time.monotonic()).
SessionResult.start_time/end_time are run-relative seconds.
The same float-second type with similar names hides the offset divergence
(self._run_start). The fix is to document the unit/origin so code mixing
the two cannot silently produce wrong durations.
"""

import inspect

from agentsurge.types.metrics import MetricsSnapshot
from agentsurge.types.results import SessionResult


def test_metrics_snapshot_timestamp_documents_unit():
    src = inspect.getsource(MetricsSnapshot)
    haystack = src.lower()
    assert "monotonic" in haystack, (
        "MetricsSnapshot.timestamp must document its unit/origin "
        "(monotonic seconds, not run-relative)."
    )


def test_session_result_start_time_documents_unit():
    src = inspect.getsource(SessionResult)
    haystack = src.lower()
    # Look for unit clarification: must mention "run-relative" / "since run start"
    # / "monotonic" so consumers know which clock origin applies.
    assert (
        "run-relative" in haystack or "since run start" in haystack or "seconds since" in haystack
    ), (
        "SessionResult.start_time/end_time must document that they are "
        "run-relative seconds (audit types#M1)."
    )
