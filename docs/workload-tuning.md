# agentsurge run - tuning

## How sessions execute

Each session is a sequence of turns: the model generates a response, then agentsurge injects a tool-execution delay before the next turn.

```
Real agent:            LLM -> tool_call -> execute(0.1-60s) -> tool_result -> LLM -> ...
agentsurge (replay):   LLM -> sleep(tool_category_delay)    -> LLM -> ...
                       ^ only when --simulate-tool-delays true (default: off);
                         sampled from the per-category distribution
                         matching the tool name (see Tool timing table)
```

## Delay layers

Two delay layers can be stacked. The runner picks the first applicable layer for each inter-turn gap:

| Layer | Flag | When | Distribution |
|-------|------|------|-------------|
| **Tool delay** | `--simulate-tool-delays true\|false` | Previous turn was a tool call | Per-category (see [Tool timing table](#tool-timing-table) below). Default: off (omit the flag). |
| **Think time** | `--think-time mu,sigma` | Human turn (no tool output) | LogNormal(mu, sigma). Default: off (no delay). |

```bash
# Agent-like pacing: tool delay on + think time (LogNormal median ~7s)
agentsurge run --simulate-tool-delays true --think-time 2.0,1.0 --n-sessions 50

# No delays at all (default): hammer the server back-to-back
agentsurge run --n-sessions 50
```

```
Wall-clock timeline for one session (3 turns, turn 1 = bash tool call):

 --simulate-tool-delays=false --think-time=off (default):
    [LLM 2.1s] [LLM 3.4s] [LLM 0.8s]            total ≈ 6.3s

 --simulate-tool-delays=true --think-time=2.0,1.0:
    [LLM 2.1s] sleep(1.9s) [LLM 3.4s] sleep(6.1s) [LLM 0.8s]   total ≈ 14.3s
              ^ exec tool_delay              ^ think time (no prior tool call)
```

## Arrival patterns

How sessions are scheduled to start:

| Pattern | Flag | Behavior |
|---------|------|----------|
| **poisson** (default) | `--arrival poisson` | Exponential inter-arrivals at `--arrival-rate` (sessions/s, default `auto` = `--client-concurrency × 0.3`) |
| **gamma** | `--arrival gamma` | Bursty inter-arrivals; burstiness set by the coefficient of variation (CV) via the `gamma_cv` preset field (CV=1 ≈ Poisson, CV>1 = increasingly bursty; default 2.0). No CLI flag; set in preset YAML under `presets.<name>.gamma_cv` (see `agentsurge/configs/default.yaml` for the full preset list). |
| **burst** | `--arrival burst` | All sessions start simultaneously |
| **constant** | `--arrival constant` | Evenly spaced at `1/rate` intervals |
| **ramp** | `--arrival ramp` | Rate linearly increases from 0 to `--arrival-rate` over `--ramp-duration` seconds |

The CLI default is `poisson`. The `agent-*`, `measure-*`, and `simulate-*` presets override arrival to `gamma`; `stress-default` and `stress-prefix-cache` use `burst`.

```bash
# Gamma (bursty) via preset - CV=2.0 from agent-light
agentsurge run --preset agent-light --n-sessions 100

# Ramp from 0 to 50 req/s over 60 seconds (find throughput knee)
agentsurge run --arrival ramp --arrival-rate 50 --ramp-duration 60 --n-sessions 200

# Duration mode: run for 5 minutes, recycling sessions
agentsurge run --duration 300 --arrival poisson --arrival-rate 20
```

```
Session start timestamps for 8 sessions (--arrival-rate 2.0):

 burst:    t=0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
 constant: t=0.00, 0.50, 1.00, 1.50, 2.00, 2.50, 3.00, 3.50
 poisson:  t=0.12, 0.38, 1.04, 1.42, 2.08, 2.31, 3.09, 4.02
 gamma:    t=0.01, 0.03, 0.05, 0.08, 3.90, 3.92, 3.95, 7.10
           ^ cluster of 4                ^ cluster of 3  ^ lone
```

## Tool timing table

Per-category delay distributions used when `--simulate-tool-delays true`:

| Tool category | Examples | Distribution |
|---------------|----------|--------------|
| fast | `think`, `finish` | Exponential, mean ~0.1s |
| file_io | `str_replace_editor`, `read_file` | LogNormal, median ~0.6s |
| exec | `execute_bash`, `run_tests` | LogNormal, median ~2.0s |
| network | `curl`, `web_search` | LogNormal, median ~2.7s |

## Assistant context: generated vs replayed

Each turn's prompt includes the prior assistant response. The source is derived from `--tool-mode` by default: `real`/`off` → inject the actual model output, `replay` → replay the recorded assistant message. Override per-run with `--use-model-reply-in-next-turn auto|true|false` (`auto` = follow `--tool-mode`).

| Mode | Preset field | Presets | Context prefix | When to use |
|------|--------------|---------|----------------|-------------|
| **Generated** | `true` | `agent-*`, `measure-*` | Real model output carries forward | Accurate context growth + prefix cache hit rate |
| **Replayed** | `false` | `simulate-*`, `stress-*` | Recorded trace replayed verbatim | Reproducible A/B comparisons |

In `replayed` mode, the recorded assistant text is usually not what the model would have generated for that exact context, so the second turn's prompt diverges from any state vLLM has cached - prefix cache hits on turns 2+ get undercounted.

```bash
# Real context growth
agentsurge run --preset agent-heavy --n-sessions 50

# Reproducible A/B comparison (identical inputs across runs)
agentsurge run --preset simulate-default --n-sessions 50
```

## Thinking budget

For reasoning models (e.g. with `--enable-thinking`), `--thinking-budget N` inflates `max_tokens` by N to leave headroom for chain-of-thought tokens on top of the recorded output length. The full N is always added whenever `--enable-thinking` is set - not adaptive to how many thinking tokens the model actually produces. Default: 8192.

```bash
# Reasoning model with 16K thinking budget
agentsurge run --enable-thinking --thinking-budget 16384
# per-turn max_tokens sent to server = recorded_output + 16384
```

```
Request payload for a turn whose recorded output was 200 tokens:

 --enable-thinking off:                                    {"max_tokens": 200, ...}
 --enable-thinking on, --thinking-budget 8192 (default):   {"max_tokens": 8392, ...}
 --enable-thinking on, --thinking-budget 16384:            {"max_tokens": 16584, ...}
```

Note: `--max-tokens` on its own doesn't change the per-turn cap - by default agentsurge matches each turn's `max_tokens` to the replay's recorded output length. Pass `--ignore-replay-output-length` to force `--max-tokens` for every turn.

## Tool output mode

Control how recorded tool outputs are injected during replay (requires `--tool-mode replay`; see the [README](../README.md#key-parameters) for `--tool-mode` basics):

| Mode | Flag | Behavior |
|------|------|----------|
| **recorded** (default) | `--tool-output-mode recorded` | Original tool output from the trace is injected verbatim |
| **placeholder** | `--tool-output-mode placeholder` | Char-count-matched filler text replaces original output (preserves context size without leaking trace content) |

```bash
# Placeholder mode - same context sizes, no trace content leakage
agentsurge run --tool-mode replay --tool-output-mode placeholder --n-sessions 50
```

```
Tool message content for one grep call:

 --tool-output-mode recorded:
   {"role": "tool", "content": "[Tool output: grep] src/auth.py:42:  def verify_token(t):
                                 src/auth.py:89:  return verify_token(request.headers[\"x-auth\"])"}

 --tool-output-mode placeholder:
   {"role": "tool", "content": "[Tool output: grep]     v = process(data, key=cfg)  # placeholder
                                                      v = process(data, key=cfg)  # placeholder"}
                                ^ same char count as original; body repeated/truncated
```

## Agent loop cap (--tool-max-turns)

For `--tool-mode real` and `--tool-mode replay`, each session runs a multi-turn agent loop. The loop exits when the model emits no tool calls, calls `finish`, errors out, or hits `--tool-max-turns N`.

Default: 15. Production agents run much higher - SWE-agent 50, OpenHands 100-500, mini-SWE-agent 250 - so 15 is a tight cap for non-trivial SWE-bench instances.

The cap has a large effect on both wall time and completion rate. On H100 + SWE-bench Verified (GLM-4.7-Flash, 100 sessions), a 100-turn cap completed in 632.9 s with a 23% `finish` rate. Lower caps finish sooner but stop more sessions before the agent can call `finish`; see [expected-performance.md](expected-performance.md#swe-bench-verified-real) for the reported H100 run.

```bash
# Higher cap for harder instances
agentsurge run --source swe-bench-verified --tool-mode real --tool-max-turns 100
```

```
Session lifecycle for one SWE-bench instance (GLM-4.7-Flash, --tool-mode real):

 --tool-max-turns=15 (default):
    turn 1  [grep "verify_token"]    -> tool output
    turn 2  [cat src/auth.py]        -> tool output
    ...
    turn 14 [str_replace_editor]     -> tool output
    turn 15 [execute_bash "ls"]      -> [cap hit - loop exits]
    session result: n_turns=15, finish called = no

 --tool-max-turns=100:
    turn 1-8  [exploration: grep, cat, ls]
    turn 9    [str_replace_editor(auth.py)]    -> first edit
    turn 10-23 [more edits + pytest runs]
    turn 24   [finish("all tests pass")]       -> clean exit
    session result: n_turns=24, finish called = yes
```

Session metadata can also set `max_turns` per-session, which overrides the CLI flag.

## Synthetic single-turn injection (opt-in)

By default, agentsurge replays the sessions loaded from the selected trace source (or the base `SyntheticGenerator`) verbatim. `--single-turn-ratio R` (default `0`, off) mixes *extra* synthetic single-turn sessions on top of the base workload. Use it to deliberately pollute prefix cache, stress the scheduler under mixed traffic, or model chat noise alongside a long-context agent trace. Keep `R=0` (or simply omit the flag) for faithful replay.

| Flag | Default | Role |
|------|---------|------|
| `--single-turn-ratio` | `0.0` | Fraction of *total* sessions that are synthetic single-turn. `0` disables injection. Must be `< 1.0`. |
| `--context-distribution` | `mixed` | Profile bundle picked when synthesizing prompts. One of `chat`, `agent`, `mixed`, `autocomplete`, `ssd-stress`. Has no effect when `--single-turn-ratio 0`. |
| `--synthetic-prompt-scale` | from config `models.<name>.max_model_len`, else `16384` | Token-length anchor for synthetic prompts. Each profile's `ratio x scale` = input tokens. |

When `R > 0`:

- `N_single = round(N_base × R / (1-R))` synthetic sessions are appended to the session list, then the combined list is shuffled.
- Each synthetic session is a single user turn whose prompt length is drawn from one of the profiles in the selected `--context-distribution`.
- Per-profile `ratio` and `weight` values live in `agentsurge/generators/synthetic.py` under `SingleTurnGenerator.DISTRIBUTIONS`.

```bash
# Pure replay (default): no synthetic injection
agentsurge run --source openhands --n-sessions 50

# 30% synthetic single-turn injected on top of OpenHands replay
agentsurge run --source openhands --n-sessions 50 \
  --single-turn-ratio 0.3 --synthetic-prompt-scale 32768

# Long-prompt stress: push prompts near the scale anchor
agentsurge run --source openhands --n-sessions 50 \
  --single-turn-ratio 0.5 --context-distribution ssd-stress \
  --synthetic-prompt-scale 32768
```

Presets that set `single_turn_ratio > 0` (e.g. `stress-lmcache`, which uses `0.3`) are opting into this mode on purpose. To opt back out, pass `--single-turn-ratio 0` on the CLI (overrides the preset).
