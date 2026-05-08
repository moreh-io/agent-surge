# SPDX-License-Identifier: MIT
"""Trace pool - real trace content for realistic workload generation.

Extracts tool output and assistant response snippets from agent trace corpora
(OpenHands, SWE-smith) into a compact JSON pool.  Generators sample from this
pool to produce workloads with real code, test output, and error messages at
empirically correct length distributions.

Usage::

    agentsurge generate --source swe-bench-verified --trace-pool trace_pool.json
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Tool output detection prefix (must match agentsurge.generators.trace_replay.TOOL_OUTPUT_PREFIX)
_TOOL_PREFIX = "[Tool output"

_TOOL_ALIASES: dict[str, tuple[str, ...]] = {
    "bash": ("execute_bash",),
    "execute_bash": ("bash",),
    "str_replace_editor": ("read_file", "edit_file", "write_file"),
    "read_file": ("str_replace_editor",),
    "edit_file": ("str_replace_editor",),
    "write_file": ("str_replace_editor",),
}


@dataclass(slots=True)
class Snippet:
    """A single content snippet extracted from a real trace."""

    tool_type: str
    tok_estimate: int
    body: str


@dataclass
class TracePool:
    """Pre-extracted real trace snippets with length-distribution sampling.

    The pool stores real tool output and assistant response content grouped
    by type, with measured LogNormal distribution parameters for each type.
    At generation time, :meth:`sample` draws a snippet whose length matches
    the empirical distribution of real agent traces.

    Parameters
    ----------
    snippets : list[Snippet]
        All snippets, will be indexed by tool_type.
    distributions : dict
        Per-type ``{"mu": float, "sigma": float, "n": int}``.
    meta : dict
        Provenance metadata (source corpus, extraction date, etc.).
    """

    snippets: list[Snippet] = field(default_factory=list)
    distributions: dict[str, dict[str, float]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # Built lazily by _ensure_index()
    _index: dict[str, list[Snippet]] = field(default_factory=dict, repr=False)
    _tok_keys: dict[str, list[int]] = field(default_factory=dict, repr=False)

    # Diagnostic: number of _pick_nearest calls that exited the concat loop
    # below the target. Helps callers detect systematic short-tail bias when
    # the pool lacks long snippets (audit:generators#H8).
    undershoot_count: int = field(default=0, repr=False)

    def _ensure_index(self) -> None:
        if self._index:
            return
        idx: dict[str, list[Snippet]] = {}
        for snip in self.snippets:
            idx.setdefault(snip.tool_type, []).append(snip)
        for snips in idx.values():
            snips.sort(key=lambda s: s.tok_estimate)
        self._index = idx
        self._tok_keys = {tt: [s.tok_estimate for s in snips] for tt, snips in idx.items()}

    @property
    def tool_types(self) -> list[str]:
        self._ensure_index()
        return list(self._index.keys())

    @property
    def aggregate_params(self) -> tuple[float, float]:
        """Return aggregate LogNormal (mu, sigma) across all types."""
        agg = self.distributions.get("_aggregate")
        if agg:
            return agg["mu"], agg["sigma"]
        # Default: empirical fit from OpenHands/SWE-smith corpus
        return 5.276, 1.550

    def sample(
        self,
        rng: random.Random,
        *,
        tool_type: str | None = None,
        target_tokens: int | None = None,
    ) -> str:
        """Draw a snippet, optionally matching a target token count.

        Parameters
        ----------
        rng : random.Random
            Seeded RNG for deterministic sampling.
        tool_type : str or None
            Specific tool type to draw from.  ``None`` picks proportionally.
        target_tokens : int or None
            Target token count.  ``None`` draws from the fitted LogNormal.

        Returns
        -------
        str
            Snippet body (without ``[Tool output: ...]`` prefix - caller adds it).
        """
        self._ensure_index()
        if not self._index:
            return "# no snippets available"

        if tool_type is not None and tool_type not in self._index:
            for alias in _TOOL_ALIASES.get(tool_type, ()):
                if alias in self._index:
                    tool_type = alias
                    break

        if tool_type is None or tool_type not in self._index:
            types = list(self._index.keys())
            weights = [len(self._index[t]) for t in types]
            tool_type = rng.choices(types, weights=weights, k=1)[0]

        if target_tokens is None:
            dist = self.distributions.get(tool_type)
            if dist:
                mu, sigma = dist["mu"], dist["sigma"]
            else:
                mu, sigma = self.aggregate_params
            target_tokens = max(10, int(rng.lognormvariate(mu, sigma)))
            target_tokens = min(target_tokens, 25000)

        return self._pick_nearest(tool_type, target_tokens, rng)

    def sample_lognormal(self, rng: random.Random, *, tool_type: str | None = None) -> str:
        """Convenience: sample with LogNormal-drawn target length."""
        return self.sample(rng, tool_type=tool_type, target_tokens=None)

    def sample_assistant(self, rng: random.Random, *, target_tokens: int | None = None) -> str:
        """Sample a real assistant response snippet.

        Returns empty string if no assistant snippets are available.
        """
        self._ensure_index()
        if "_assistant" not in self._index:
            return ""
        return self.sample(rng, tool_type="_assistant", target_tokens=target_tokens)

    def _pick_nearest(self, tool_type: str, target_tok: int, rng: random.Random) -> str:
        """Find the snippet closest to target_tok, with concat/trim if needed."""
        snips = self._index[tool_type]
        keys = self._tok_keys[tool_type]

        pos = bisect.bisect_left(keys, target_tok)
        candidates = []
        for i in (pos - 1, pos, pos + 1):
            if 0 <= i < len(snips):
                candidates.append(snips[i])
        if not candidates:
            return snips[rng.randint(0, len(snips) - 1)].body

        best = min(candidates, key=lambda s: abs(s.tok_estimate - target_tok))

        # If within 40% tolerance, use directly
        if best.tok_estimate >= target_tok * 0.6:
            if best.tok_estimate <= target_tok * 1.4:
                return best.body
            # Too long: trim to nearest line boundary
            target_chars = target_tok * 4
            body = best.body[:target_chars]
            nl = body.rfind("\n")
            return body[:nl] if nl > target_chars // 2 else body

        # Too short: concatenate snippets to reach target
        parts = [best.body]
        current_tok = best.tok_estimate
        safety = 10
        while current_tok < target_tok * 0.8 and safety > 0:
            safety -= 1
            extra = snips[rng.randint(0, len(snips) - 1)]
            parts.append(extra.body)
            current_tok += extra.tok_estimate
        if current_tok < target_tok * 0.8:
            # Exited the loop still under-target — record so callers can
            # detect systematic long-tail undershoot.
            self.undershoot_count += 1
        combined = "\n\n".join(parts)
        target_chars = target_tok * 4
        if len(combined) > target_chars:
            combined = combined[:target_chars]
            nl = combined.rfind("\n")
            if nl > target_chars // 2:
                combined = combined[:nl]
        return combined

    def to_json(self, path: str | Path) -> None:
        """Save the pool to a JSON file."""
        data = {
            "version": "1.0",
            "meta": self.meta,
            "distributions": self.distributions,
            "snippets": [
                {"tool_type": s.tool_type, "tok_estimate": s.tok_estimate, "body": s.body}
                for s in self.snippets
            ],
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> TracePool:
        """Load a pool from a previously extracted JSON file."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        # Support both "snippets" (new) and "fragments" (old) keys
        raw_snippets = raw.get("snippets") if "snippets" in raw else raw.get("fragments", [])
        snippets = [
            Snippet(tool_type=s["tool_type"], tok_estimate=s["tok_estimate"], body=s["body"])
            for s in raw_snippets
        ]
        return cls(
            snippets=snippets,
            distributions=raw.get("distributions", {}),
            meta=raw.get("meta", {}),
        )

    @classmethod
    def _process_corpus_message(
        cls,
        msg: dict,
        seen: set[str],
        by_type: dict[str, list[Snippet]],
        min_body_chars: int,
        max_body_chars: int,
    ) -> None:
        """Process a single corpus message, adding snippets to *by_type*."""
        if msg.get("role") == "assistant":
            abody = msg.get("content", "")
            if len(abody) < min_body_chars:
                return
            ah = hashlib.md5(abody[:500].encode()).hexdigest()
            if ah in seen:
                return
            seen.add(ah)
            if len(abody) > max_body_chars:
                abody = abody[:max_body_chars]
            by_type.setdefault("_assistant", []).append(
                Snippet(tool_type="_assistant", tok_estimate=len(abody) // 4, body=abody)
            )
            return

        body, tool_type = _extract_tool_body(msg)
        if body is None or len(body) < min_body_chars:
            return
        h = hashlib.md5(body[:500].encode()).hexdigest()
        if h in seen:
            return
        seen.add(h)

        if len(body) > max_body_chars:
            body = body[:max_body_chars]
            nl = body.rfind("\n")
            if nl > max_body_chars // 2:
                body = body[:nl]

        tok = len(body) // 4
        by_type.setdefault(tool_type, []).append(
            Snippet(tool_type=tool_type, tok_estimate=tok, body=body)
        )

    @classmethod
    def _fit_distributions(
        cls,
        by_type: dict[str, list[Snippet]],
    ) -> tuple[dict[str, dict[str, Any]], list[int]]:
        """Fit LogNormal distributions per type and compute aggregate."""
        distributions: dict[str, dict[str, Any]] = {}
        all_toks: list[int] = []
        for tt, snips in by_type.items():
            toks = [s.tok_estimate for s in snips if s.tok_estimate > 0]
            if len(toks) < 10:
                continue
            mu, sigma = _fit_lognormal(toks)
            distributions[tt] = {"mu": round(mu, 3), "sigma": round(sigma, 3), "n": len(toks)}
            all_toks.extend(toks)

        if all_toks:
            mu, sigma = _fit_lognormal(all_toks)
            distributions["_aggregate"] = {
                "mu": round(mu, 3),
                "sigma": round(sigma, 3),
                "n": len(all_toks),
            }
        return distributions, all_toks

    @classmethod
    def from_corpus(
        cls,
        corpus_paths: list[str | Path],
        *,
        min_body_chars: int = 100,
        max_body_chars: int = 30000,
        max_snippets_per_type: int = 3000,
    ) -> TracePool:
        """Build a trace pool in-memory from corpus files.

        Parameters
        ----------
        corpus_paths : list of paths
            Paths to agentsurge corpus JSON files.
        min_body_chars : int
            Skip snippets shorter than this.
        max_body_chars : int
            Truncate snippets longer than this.
        max_snippets_per_type : int
            Cap per tool type to control pool size.

        Returns
        -------
        TracePool
            Ready-to-use pool instance.
        """
        seen: set[str] = set()
        by_type: dict[str, list[Snippet]] = {}
        total_scanned = 0

        for cpath in corpus_paths:
            data = json.loads(Path(cpath).read_text(encoding="utf-8"))
            sessions = data.get("sessions", [])
            for session in sessions:
                turn_messages = session.get("turn_messages", [])
                if not turn_messages:
                    continue
                for msg in turn_messages[-1]:
                    total_scanned += 1
                    cls._process_corpus_message(
                        msg,
                        seen,
                        by_type,
                        min_body_chars,
                        max_body_chars,
                    )

        rng = random.Random(42)
        all_snips: list[Snippet] = []
        for _tt, snips in by_type.items():
            if len(snips) > max_snippets_per_type:
                rng.shuffle(snips)
                snips = snips[:max_snippets_per_type]
            all_snips.extend(snips)

        distributions, _ = cls._fit_distributions(by_type)

        return cls(
            snippets=all_snips,
            distributions=distributions,
            meta={
                "source_corpus": [str(p) for p in corpus_paths],
                "n_snippets": len(all_snips),
                "total_scanned": total_scanned,
                "min_body_chars": min_body_chars,
                "type_counts": {tt: len(ss) for tt, ss in by_type.items()},
            },
        )

    @staticmethod
    def extract(
        corpus_paths: list[str | Path],
        output_path: str | Path,
        *,
        min_body_chars: int = 100,
        max_body_chars: int = 30000,
        max_snippets_per_type: int = 3000,
    ) -> dict:
        """Extract snippets from corpus files into a trace pool JSON.

        Convenience wrapper: calls :meth:`from_corpus` then saves to disk.

        Returns
        -------
        dict
            Summary statistics of the extraction.
        """
        pool = TracePool.from_corpus(
            corpus_paths,
            min_body_chars=min_body_chars,
            max_body_chars=max_body_chars,
            max_snippets_per_type=max_snippets_per_type,
        )
        pool.to_json(output_path)
        return pool.meta

    @staticmethod
    def fit_workload_stats(corpus_paths: list[str | Path]) -> dict:
        """Extract statistical properties from real trace corpora.

        Reads workload corpus files and fits distributions for turn count,
        token length, tool usage, and prefix sharing.  The returned dict
        contains fitted parameters that can be fed directly into agentsurge
        config/presets for realistic workload generation.

        Parameters
        ----------
        corpus_paths : list of paths
            Paths to agentsurge corpus JSON files.

        Returns
        -------
        dict
            Fitted workload statistics with recommended config values.
        """
        turn_counts: list[int] = []
        tokens_per_turn: list[int] = []
        tool_turns = 0
        total_turns = 0
        system_prefixes: dict[str, int] = {}

        for cpath in corpus_paths:
            data = json.loads(Path(cpath).read_text(encoding="utf-8"))
            sessions = data.get("sessions", [])
            for session in sessions:
                tms = session.get("turn_messages", [])
                if not tms:
                    continue
                turn_counts.append(len(tms))

                first_turn = tms[0]
                sys_msgs = [m.get("content", "") for m in first_turn if m.get("role") == "system"]
                if sys_msgs:
                    prefix_key = hashlib.md5(sys_msgs[0][:200].encode()).hexdigest()
                    system_prefixes[prefix_key] = system_prefixes.get(prefix_key, 0) + 1

                for i, turn_msgs in enumerate(tms):
                    total_turns += 1
                    total_chars = sum(len(m.get("content", "")) for m in turn_msgs)
                    tokens_per_turn.append(total_chars // 4)

                    if i > 0:
                        prev_len = len(tms[i - 1])
                        new_msgs = turn_msgs[prev_len:]
                        for m in new_msgs:
                            if m.get("role") == "tool" or _TOOL_PREFIX in m.get("content", ""):
                                tool_turns += 1
                                break

        result: dict[str, Any] = {
            "n_sessions": len(turn_counts),
            "n_corpus_files": len(corpus_paths),
        }

        if turn_counts:
            tc_mu, tc_sigma = _fit_lognormal(turn_counts)
            import statistics

            result["turn_count"] = {
                "mean": round(statistics.mean(turn_counts), 1),
                "median": round(statistics.median(turn_counts)),
                "stdev": round(statistics.stdev(turn_counts), 1) if len(turn_counts) > 1 else 0.0,
                "min": min(turn_counts),
                "max": max(turn_counts),
                "lognormal_mu": round(tc_mu, 3),
                "lognormal_sigma": round(tc_sigma, 3),
            }

        if tokens_per_turn:
            tpt_mu, tpt_sigma = _fit_lognormal([t for t in tokens_per_turn if t > 0] or [1])
            import statistics

            result["tokens_per_turn"] = {
                "mean": round(statistics.mean(tokens_per_turn), 1),
                "median": round(statistics.median(tokens_per_turn)),
                "lognormal_mu": round(tpt_mu, 3),
                "lognormal_sigma": round(tpt_sigma, 3),
            }

        if total_turns > 0:
            result["tool_turn_fraction"] = round(tool_turns / total_turns, 3)

        if system_prefixes:
            counts = list(system_prefixes.values())
            result["prefix_sharing"] = {
                "unique_prefixes": len(system_prefixes),
                "mean_sessions_per_prefix": round(sum(counts) / len(counts), 1),
                "max_sessions_per_prefix": max(counts),
            }

        if turn_counts and tokens_per_turn:
            import statistics

            result["recommended_config"] = {
                "n_turns": round(statistics.median(turn_counts)),
                "tokens_per_turn": round(statistics.median(tokens_per_turn)),
                "turn_distribution": "geometric",
                "think_time_mu": result["tokens_per_turn"]["lognormal_mu"],
                "think_time_sigma": min(result["tokens_per_turn"]["lognormal_sigma"], 2.0),
            }

        return result


def _extract_tool_body(msg: dict) -> tuple[str | None, str]:
    """Extract tool output body and tool type from a message dict."""
    role = msg.get("role", "")
    content = msg.get("content", "")

    if role == "tool":
        tool_type = msg.get("name", "unknown")
        return content, tool_type

    if _TOOL_PREFIX in content:
        idx = content.index(_TOOL_PREFIX)
        bracket_end = content.find("]", idx)
        if bracket_end < 0:
            return None, "unknown"
        prefix = content[idx : bracket_end + 1]
        body = content[bracket_end + 1 :].lstrip()
        if ": " in prefix:
            tool_type = prefix.split(": ", 1)[1].rstrip("]").strip()
        else:
            tool_type = "unknown"
        return body, tool_type

    return None, "unknown"


def _fit_lognormal(values: list[int]) -> tuple[float, float]:
    """Fit LogNormal distribution to positive integer values via MLE."""
    logs = [math.log(max(v, 1)) for v in values]
    n = len(logs)
    mu = sum(logs) / n
    sigma = math.sqrt(sum((x - mu) ** 2 for x in logs) / n)
    return mu, max(sigma, 0.1)
