# SPDX-License-Identifier: MIT
"""Mixed workload generator - combine sessions from multiple sources by weight."""

import dataclasses
import random

from agentsurge.generators.base import GeneratorBase
from agentsurge.types import ReplaySession


class MixedWorkloadGenerator(GeneratorBase):
    """Mix ReplaySessions from multiple sources by weight."""

    name = "mixed"

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def mix(
        self,
        sources: dict[str, list[ReplaySession]],
        weights: dict[str, float] | None = None,
        total: int | None = None,
    ) -> list[ReplaySession]:
        if not sources:
            return []

        rng = random.Random(self.seed)

        if weights is None:
            all_count = sum(len(v) for v in sources.values())
            if all_count == 0:
                return []
            weights = {k: len(v) / all_count for k, v in sources.items()}

        if total is None:
            total = sum(len(v) for v in sources.values())

        w_sum = sum(weights.get(k, 0) for k in sources)
        if w_sum == 0:
            return []

        result: list[ReplaySession] = []
        src_names = [s for s in sources if sources[s] and weights.get(s, 0) > 0]
        raw_counts = {s: total * (weights[s] / w_sum) for s in src_names}
        floor_counts = {s: int(raw_counts[s]) for s in src_names}
        remainder = total - sum(floor_counts.values())
        by_frac = sorted(src_names, key=lambda s: raw_counts[s] - floor_counts[s], reverse=True)
        picks = dict(floor_counts)
        for s in by_frac[:remainder]:
            picks[s] += 1

        for src_name, sessions in sources.items():
            if not sessions:
                continue
            n_pick = picks.get(src_name, 0)
            if n_pick > len(sessions):
                raise ValueError(
                    f"Generator '{src_name}': weighted demand of {n_pick} sessions "
                    f"exceeds pool size of {len(sessions)}. "
                    "Provide a larger pool or reduce the total/weight."
                )
            picked = rng.sample(sessions, k=n_pick)
            for sess in picked:
                patched = dataclasses.replace(sess, metadata={**sess.metadata, "source": src_name})
                result.append(patched)

        rng.shuffle(result)
        return result[:total]
