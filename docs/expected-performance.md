# Expected performance

Numbers from `agentsurge` runs on a single H100 SXM5 ×8 node under one unified server config. If your own run lands in the same ballpark, the pipeline and server are wired up correctly.

What this measures is the server side: throughput, TTFT, TPOT, KV and prefix-cache behavior, under the two workloads below.

Two workloads:

- OpenHands replay (`--tool-mode replay`). Recorded OpenHands traces from [`nebius/SWE-rebench-openhands-trajectories`](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) get replayed turn-by-turn. The model generates fresh responses; tool outputs come from the dataset. Long multi-turn prefill, batched decode. Per-turn `max_tokens` is sized from the trace's recorded output length.
- SWE-bench Verified real (`--tool-mode real --source swe-bench-verified`). The agent runs an actual bash/edit/search loop against a pre-cloned repository from [`princeton-nlp/SWE-bench_Verified`](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified), ending when it calls `finish`, errors, or hits the turn cap.

Metric:

- **Concurrency**: max in-flight sessions the client keeps open.
- **Sessions**: total conversations issued.
- **Sessions completed**: sessions that finished without transport or schema errors (matches the CLI's `N completed` line).
- **Sessions that called `finish`** (real mode only): sessions where the agent emitted the `finish` tool call before the turn cap. A semantic task-completion signal, not the same as *Sessions completed*.
- **Turn cap**: agentsurge's `max_turns`.
- **TTFT**: time-to-first-token per turn, in milliseconds.
- **TPOT**: time per output token, in milliseconds.
- **E2E latency**: per-turn end-to-end latency, in milliseconds.
- **Wall time**: total elapsed time for the run (agentsurge inference wall-clock, server-load excluded).
- **Sessions/hr**: `n_completed / wall_clock_s × 3600`.
- **ISL / OSL**: total input / output tokens across the run, summed over all turns.
- **Peak KV cache utilization**: high-water mark of KV-cache usage as a fraction of GPU memory.
- **Prefix cache hit rate**: per-request prefix-cache hit rate reported by the server.

## Environment

**Hardware**: NVIDIA H100 SXM5 ×8 (80 GB/GPU, 640 GB total).

| Component | Value |
|---|---|
| Server image | `vllm/vllm-openai:v0.20.0` |
| Weight precision | bfloat16 |
| `--max-model-len` | 131,072 |
| `--max-num-batched-tokens` | 32,768 |
| `--max-num-seqs` | 100 |
| `--gpu-memory-utilization` | 0.90 |
| Prefix caching | enabled |
| Auto tool choice | enabled |
| Client | `agentsurge` (this repo, master `d451261`) |
| Client `n_sessions` / concurrency | 50 / 50 |
| Client arrival | gamma |

Tensor-parallel size is fixed by the model's attention-head and KV-head divisibility; it is not a tuning knob:

| Model | TP | Tool parser | Reasoning parser |
|---|---:|---|---|
| Qwen3.5-27B | 4 | `qwen3_xml` | — |

## Results

### OpenHands replay

Per-turn `max_tokens` is dynamic from the trace; the client cap is 32,768.

| Model | Wall | Completed/N | Sess/hr | Req/s | In tok/s | Out tok/s | TTFT p50/p95/p99 (ms) | TPOT p50/p95/p99 (ms) | E2E p50/p95/p99 (ms) | KV peak | Prefix hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-27B (TP=4) | 249.2 s | 41/50 | 592 | 12.4 | 403 K | 129 | 637 / 1,632 / 4,869 | 1,832 / 3,355 / 3,857 | 3,017 / 5,282 / 11,524 | 25.1 % | 96.4 % |

The 9 errored sessions are aiohttp `ServerDisconnectedError` near the `--max-model-len` boundary. Comes from long-context prefill on this server build, not the model itself.

### SWE-bench Verified real

Client flags: `--tool-mode real --tool-max-turns 100 --max-tokens 16384`, temperature 0.3 (agentsurge default), 50 sessions / concurrency 50.

| Model | Wall | Completed/N | Turns | Sess/hr | Req/s | In tok/s | Out tok/s | TTFT p50/p95/p99 (ms) | TPOT p50/p95/p99 (ms) | E2E p50/p95/p99 (ms) | KV peak | Prefix hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-27B (TP=4) | 185.0 s | 42/50 | 1,334 | 817 | 7.2 | 58 K | 1.3 K | 122 / 378 / 2,846 | 21 / 41 / 68 | 2,345 / 11,881 / 19,545 | 16.0 % | 90.4 % |

## Reproduce

### 1. Start the server

Mount your local weight directory at `/models/...` and wait for `/health` to return 200 before launching the client.

**Qwen3.5-27B (TP=4):**

```bash
docker run -d --name qwen35-server \
  --gpus all --shm-size 64g --network host \
  -v /path/to/Qwen3.5-27B:/models/qwen3.5-27b:ro \
  vllm/vllm-openai:v0.20.0 \
  --model /models/qwen3.5-27b \
  --served-model-name qwen3.5-27b \
  --tensor-parallel-size 4 \
  --max-model-len 131072 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 100 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --host 0.0.0.0 --port 8002
```

### 2. Run the benchmark

OpenHands replay:

```bash
agentsurge run --source openhands \
  --model qwen3.5-27b \
  --tokenizer /path/to/Qwen3.5-27B --trust-remote-code \
  --vllm-url http://localhost:8002 \
  --n-sessions 50 --client-concurrency 50 \
  --max-tokens 32768 \
  --tool-mode replay --arrival gamma --save-responses
```

SWE-bench Verified real:

```bash
agentsurge run --source swe-bench-verified \
  --model qwen3.5-27b \
  --tokenizer /path/to/Qwen3.5-27B --trust-remote-code \
  --vllm-url http://localhost:8002 \
  --n-sessions 50 --client-concurrency 50 \
  --max-tokens 16384 --tool-max-turns 100 \
  --tool-mode real --arrival gamma --save-responses \
  --tool-workspace-dir /tmp/qwen35_repos
```

Point `--tool-workspace-dir` at a local-disk path. NFS or any other networked filesystem will inflate per-file-op latency enough to distort real-mode timings.

## Known issues

### Replay long-context disconnects

In the replay row, 9 of 50 sessions die with `aiohttp.ServerDisconnectedError` partway through a long-context turn near the `--max-model-len` boundary. Disconnects concentrate on the same handful of unusually long traces. The per-turn TTFT / TPOT / E2E numbers come from turns before the failing one, so the throughput columns stay representative.
