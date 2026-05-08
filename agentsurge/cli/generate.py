# SPDX-License-Identifier: MIT
"""cmd_generate - workload generation subcommand."""

import argparse
import glob as _glob
import json
import logging
import sys

from agentsurge.cli._helpers import (
    _get_task_loader,
    _get_trace_loader,
    _load_config,
    _output_path,
    _resolve_local,
)
from agentsurge.generators.mixed import MixedWorkloadGenerator
from agentsurge.generators.synthetic import SyntheticGenerator
from agentsurge.generators.task_converter import TaskToTrajectoryConverter
from agentsurge.generators.trace_pool import TracePool
from agentsurge.generators.trace_replay import TraceReplayGenerator
from agentsurge.types import WORKLOAD_SCHEMA_VERSION

_log = logging.getLogger(__name__)


def cmd_generate(args: argparse.Namespace) -> str | None:
    """Generate workload sessions (trace replay or synthetic)."""
    cfg = _load_config(args.config)

    source = getattr(args, "source", None)
    flatten_tools = getattr(args, "flatten_tools", True)
    max_prompt_chars = getattr(args, "max_prompt_chars", 16384)

    bank = None
    bank_path = getattr(args, "trace_pool", None)
    corpus_paths = getattr(args, "corpus_paths", None)
    if bank_path and corpus_paths:
        _log.error("--trace-pool and --corpus-paths are mutually exclusive")
        sys.exit(1)
    if bank_path:
        bank = TracePool.from_json(bank_path)
        n = bank.meta.get("n_snippets") or bank.meta.get("n_fragments", "?")
        print(f"Loaded trace pool: {n} snippets")
    elif corpus_paths:
        expanded = []
        for p in corpus_paths:
            expanded.extend(_glob.glob(p))
        if not expanded:
            _log.error("no files matched --corpus-paths: %s", corpus_paths)
            sys.exit(1)
        print(f"Building trace pool from {len(expanded)} corpus file(s)...")
        bank = TracePool.from_corpus(expanded)
        print(f"Trace pool ready: {bank.meta.get('n_snippets', 0)} snippets")

    if source == "mixed":
        gen = TraceReplayGenerator(flatten_tools=flatten_tools)
        converter = TaskToTrajectoryConverter(max_prompt_chars=max_prompt_chars, trace_pool=bank)
        mixer = MixedWorkloadGenerator()

        weights = None
        if getattr(args, "mix_weights", None):
            weights = {}
            for pair in args.mix_weights.split(","):
                if ":" not in pair:
                    sys.exit(f"Invalid --mix-weights format: '{pair}'. Expected 'name:weight,...'")
                name, w = pair.split(":", 1)
                try:
                    weights[name.strip()] = float(w.strip())
                except ValueError:
                    sys.exit(f"Invalid weight value in --mix-weights: '{pair}'")

        trace_sources = ["openhands", "swe-smith"]
        task_sources = ["swe-bench-verified", "abc-bench", "swe-evo", "featurebench"]

        all_sources: dict[str, list] = {}
        per_source = max(1, args.n_sessions // 6)

        for src in trace_sources:
            try:
                lp = _resolve_local(
                    src, getattr(args, "data_dir", None), getattr(args, "local_path", None)
                )
                if not lp and getattr(args, "data_dir", None):
                    print(f"  [info] {src} not found locally, downloading to {args.data_dir}...")
                    loader = _get_trace_loader(src)
                    lp = loader.download(args.data_dir)

                loader = _get_trace_loader(
                    src,
                    local_path=lp,
                    split=getattr(args, "swe_smith_split", "tool"),
                )
                trajs = loader.load(limit=per_source)
                sessions = list(gen.from_trajectories(trajs, limit=per_source))
                if sessions:
                    all_sources[src] = sessions
            except Exception as e:
                print(f"Warning: Could not load {src}: {e}")

        for src in task_sources:
            try:
                lp = _resolve_local(src, getattr(args, "data_dir", None))
                if not lp and getattr(args, "data_dir", None):
                    print(f"  [info] {src} not found locally, downloading to {args.data_dir}...")
                    task_loader = _get_task_loader(src)
                    lp = task_loader.download(args.data_dir)

                task_loader = _get_task_loader(src, local_path=lp)
                pattern = TaskToTrajectoryConverter.DATASET_PATTERN_MAP.get(
                    task_loader.name, "bug-fix"
                )
                tasks = task_loader.load(limit=per_source)
                trajs = converter.from_tasks(tasks, pattern, limit=per_source)
                sessions = list(gen.from_trajectories(iter(trajs), limit=per_source))
                if sessions:
                    all_sources[src] = sessions
            except Exception as e:
                print(f"Warning: Could not load {src}: {e}")

        sessions = mixer.mix(all_sources, weights=weights, total=args.n_sessions)
        print(f"Mixed {len(sessions)} sessions from {list(all_sources.keys())}")

    elif source is not None and (source == "openhands" or source.startswith("swe-smith")):
        lp = _resolve_local(
            source, getattr(args, "data_dir", None), getattr(args, "local_path", None)
        )
        if not lp and getattr(args, "data_dir", None):
            print(f"  [info] {source} not found locally, downloading to {args.data_dir}...")
            loader = _get_trace_loader(source)
            lp = loader.download(args.data_dir)

        loader = _get_trace_loader(
            source,
            lp,
            split=getattr(args, "swe_smith_split", None),
        )
        gen = TraceReplayGenerator(flatten_tools=flatten_tools)
        trajectories = loader.load(limit=args.n_sessions)
        sessions = list(gen.from_trajectories(trajectories, limit=args.n_sessions))
        print(f"Loaded {len(sessions)} trace replay sessions from '{source}'")

    elif source in ("swe-bench-verified", "abc-bench", "swe-evo", "featurebench"):
        lp = _resolve_local(
            source, getattr(args, "data_dir", None), getattr(args, "local_path", None)
        )
        if not lp and getattr(args, "data_dir", None):
            print(f"  [info] {source} not found locally, downloading to {args.data_dir}...")
            task_loader = _get_task_loader(source)
            lp = task_loader.download(args.data_dir)

        task_loader = _get_task_loader(source, lp)
        converter = TaskToTrajectoryConverter(max_prompt_chars=max_prompt_chars, trace_pool=bank)
        pattern = TaskToTrajectoryConverter.DATASET_PATTERN_MAP.get(task_loader.name, "bug-fix")
        tasks = task_loader.load(limit=args.n_sessions)
        trajs = converter.from_tasks(tasks, pattern, limit=args.n_sessions)
        gen = TraceReplayGenerator(flatten_tools=flatten_tools)
        sessions = list(gen.from_trajectories(iter(trajs), limit=args.n_sessions))
        print(f"Loaded {len(sessions)} task-to-trajectory sessions from '{source}'")

    elif source is None:
        synth_gen = SyntheticGenerator(
            tokens_per_turn=args.synthetic_tokens_per_turn
            or cfg["workload"].get("synthetic_tokens_per_turn", 2700),
            n_turns=args.n_turns or cfg["workload"].get("n_turns", 5),
            trace_pool=bank,
        )
        sessions = synth_gen.generate(n_sessions=args.n_sessions, seed=getattr(args, "seed", 42))
        if sessions:
            print(f"Generated {len(sessions)} synthetic sessions, {sessions[0].n_turns} turns each")
        else:
            print("Generated 0 sessions")
    else:
        raise ValueError(f"Unknown source: {source}")

    out = _output_path(args.output_dir, "workload")
    session_dicts = [
        {
            "session_id": s.session_id,
            "n_turns": s.n_turns,
            "turn_messages": s.turn_messages,
            "metadata": s.metadata,
        }
        for s in sessions
    ]
    data = {
        "version": WORKLOAD_SCHEMA_VERSION,
        "sessions": session_dicts,
    }
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {out}")
    return out
