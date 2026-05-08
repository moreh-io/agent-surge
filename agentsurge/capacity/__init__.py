# SPDX-License-Identifier: MIT
"""agentsurge.capacity - KV-cache capacity instrumentation.

wasted_store - KV object lifecycle tracking (stored vs retrieved vs evicted).
"""

from agentsurge.capacity import wasted_store

__all__ = ["wasted_store"]
