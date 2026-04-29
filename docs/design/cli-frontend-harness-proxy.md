# AgentSurge CLI Frontend Harness — Serving Trace Proxy

Date: 2026-04-28
Status: Design — pending approval before implementation (Phase N.4).

## Problem

CLI frontends (Codex, Claude Code, OpenCode) own their own model-call loop and
expose only the events the CLI chose to surface. Frontend-observed timing
includes CLI startup, config loading, prompt processing, and tool-loop work,
which is not equivalent to per-request server-side TTFT/TPOT. See
§Layer 2 Serving-Observed Metrics in
`agentsurge-cli-frontend-harness-design.md` for the layered measurement model.
To produce benchmark-grade serving TTFT/TPOT for CLI frontend runs, AgentSurge
needs a proxy that sits between the CLI and the upstream backend, records
per-request timing at the SSE boundary, and correlates each request to the
AgentSurge session that issued it.

## Goals

- Capture per-request TTFT, per-chunk timing, usage, and end-of-request markers
  for every model call a frontend CLI makes during a benchmark session.
- Correlate captured traces with the AgentSurge session that issued them.
- Persist traces in a streaming, low-overhead format colocated with frontend
  artifacts.
- Auto-manage the proxy in benchmark runs while still exposing it as a manual
  subcommand.

## Non-goals

- Authentication, rate limiting, or quota enforcement (single-user local
  proxy).
- Body validation or transformation. Proxy is a passthrough.
- Persistent storage beyond the JSONL files. No DB.
- Multi-target routing per request. The proxy is one-target-per-instance.
- Real upstream-key rotation. Configured once at proxy start.

## Architecture

```text
+-------------------+     synthetic API key     +---------------------+
| Frontend CLI      |  ---------------------->  | AgentSurge Proxy    |
| (codex/claude/    |   Authorization: Bearer   | aiohttp server      |
|  opencode)        |   as-<sid>-<rand8>        | session table       |
+-------------------+                           | TraceWriter per req |
                                                +----------+----------+
                                                           |
                                                upstream key substituted
                                                           |
                                                           v
                                                +---------------------+
                                                | Upstream backend    |
                                                | (vLLM / OpenAI /    |
                                                |  Anthropic)         |
                                                +---------------------+

renderer flow:
  BenchmarkRunner -> mints synthetic_key(session_id) -> extra_env injection
                  -> CLI process spawned with API-key env var set
                  -> proxy decodes session_id from inbound auth header
                  -> trace written under <workspace>/sessions/<sid>/proxy_traces/
```

The renderer mints one synthetic key per session and injects it via
`extra_env` into the CLI's API-key env var (`OPENAI_API_KEY` for codex and
opencode, `ANTHROPIC_API_KEY` for claude). The proxy maintains an implicit
session-id table by parsing the auth header; no out-of-band registration step
is required.

### Components

- `agentsurge/proxy/server.py` — aiohttp HTTP server. Hosts OpenAI-compatible
  chat/completions, OpenAI responses, and Anthropic messages endpoints
  (passthrough only — no schema awareness).
- `agentsurge/proxy/trace.py` — per-request `TraceWriter` that streams phase
  events to a JSONL file.
- `agentsurge/proxy/embed.py` — `start_embedded_proxy(target, trace_dir) ->
  ProxyHandle` async context manager used by `BenchmarkRunner`.
- `agentsurge/cli/proxy.py` — thin CLI driver for the sibling `agentsurge
  proxy` subcommand.

### Request lifecycle

1. CLI sends an HTTPS request to the proxy listen address.
2. Proxy reads the auth header (see §Auth header inspection).
3. Proxy decodes `session_id` from the synthetic-key prefix.
4. Proxy opens a `TraceWriter` rooted under
   `<trace_dir>/<session_id>/proxy_traces/req_<N>.jsonl`.
5. Proxy forwards the request to the upstream target with the real upstream
   key substituted into the auth header.
6. Proxy streams the upstream response back to the CLI verbatim, tee-ing each
   SSE chunk into the `TraceWriter` and recording phase events.
7. On completion, the `TraceWriter` flushes a `response.end` event and closes
   the file. On any failure or client disconnect, the writer still flushes a
   terminal event before closing.

### Synthetic key shape

`as-<session_id>-<rand8>` where `rand8` is 8 hex characters from
`secrets.token_hex(4)`. Total prefix length is `6 + len(session_id) + 1 + 8`.
For the typical UUID-like 24-char `session_id`, the synthetic key is around
39 characters — fits comfortably in any auth header.

The proxy uses the regex `^as-([A-Za-z0-9_-]+)-[a-f0-9]{8}$` to extract the
`session_id`. If the regex fails to match, the request is forwarded with the
auth header passed through verbatim only when `--allow-passthrough-auth` is
set; otherwise it is rejected with `401`. A `parser.error`-style trace entry
is recorded under a synthetic `unknown_session/` directory so misconfigured
runs surface in the trace tree.

### Auth header inspection

Proxy reads, in order:

1. `Authorization: Bearer <token>` — used by codex and opencode.
2. `x-api-key: <token>` — used by claude.
3. `Authorization: ApiKey <token>` — fallback for other CLIs.

If any matches the synthetic-key regex, that is the session-tagged request.
The proxy then reads its own configured upstream key
(`AGENTSURGE_PROXY_UPSTREAM_KEY` env var, or a `--upstream-key-file` flag) and
substitutes it in the outgoing request. If the upstream is keyless (some local
backends), substitution is skipped and the outgoing header is removed.

If multiple session-tagged headers are present, the first one in the order
above wins. The non-winning headers are stripped from the outgoing request.

### Trace file layout

```text
<workspace>/sessions/<session_id>/proxy_traces/
  req_001.jsonl
  req_002.jsonl
  ...
```

One file per request. Files numbered in arrival order per session, with
zero-padded width 3 for legible sort order; rollover to width 4+ is automatic
once a session exceeds 999 requests. The `<workspace>` path is the same one
the renderer uses for `prompt.md` and `session.json` artifacts, so proxy
traces live alongside the rest of a session's artifacts.

### Trace event schema

Each line in `req_<N>.jsonl` is one phase event. Example sequence for a
streaming chat-completions call:

```jsonl
{"phase":"request.start","ts_monotonic":...,"ts_wall":"2026-04-28T...","session_id":"...","upstream_method":"POST","upstream_path":"/v1/chat/completions","client_addr":"..."}
{"phase":"response.start","ts_monotonic":...,"upstream_status":200,"upstream_headers":{"content-type":"text/event-stream","x-request-id":"..."}}
{"phase":"response.chunk","ts_monotonic":...,"chunk_bytes":4096,"sse_event_count":3}
{"phase":"usage","ts_monotonic":...,"usage":{"input_tokens":...,"output_tokens":...}}
{"phase":"response.end","ts_monotonic":...,"total_chunks":47,"total_bytes":81920}
```

Failure cases append an `error` event:

```jsonl
{"phase":"error","ts_monotonic":...,"error_type":"upstream_timeout","detail":"..."}
```

Headers are sanitized: only the safe-list fields `content-type`,
`content-length`, `transfer-encoding`, and `x-request-id` are persisted.
`Authorization`, `x-api-key`, and any header whose name contains `key` or
`token` (case-insensitive) are NEVER written to traces.

The `response.start` event is the TTFT marker (first byte returned by the
upstream). The `usage` event is emitted only when the upstream stream contains
a usage chunk (vLLM/OpenAI-compatible streams emit one when
`stream_options.include_usage=true`; Anthropic emits `message_delta` with
`usage`).

### Aggregation contract

After a frontend run completes, `BenchmarkRunner` (or a post-processor) walks
`<workspace>/sessions/*/proxy_traces/*.jsonl` and computes:

Per session, populating
`SessionResult.serving_trace_metrics: SessionServingTraceMetrics` (new
dataclass added alongside `FrontendMetrics`):

- `request_count`
- `serving_ttft_ms` distribution: per-request
  `(response.start.ts_monotonic - request.start.ts_monotonic) * 1000`
- `serving_tpot_ms` distribution: per-request
  `(response.end.ts_monotonic - response.start.ts_monotonic)
  / max(output_tokens - 1, 1) * 1000` when a `usage` event is present;
  unset otherwise
- Total request wall time

Aggregate, populating `RunResult.serving_trace` (existing
`ServingTraceMetrics` dataclass extended):

- Total proxy requests across the run.
- Aggregate serving TTFT/TPOT distributions (p50/p95/p99 plus full sample
  vector).
- Per-session request count distribution.

`SessionResult.serving_trace_metrics` is `None` when `frontend.server_url` is
not `auto` and the user did not point the renderer at a proxy.

## Configuration

CLI flags on `agentsurge proxy`:

- `--listen URL` (required): listen address, e.g.
  `http://127.0.0.1:18080`.
- `--target URL` (required): upstream backend URL.
- `--trace-dir PATH` (required): root directory under which session subdirs
  are created.
- `--upstream-key-env VAR` (default `AGENTSURGE_PROXY_UPSTREAM_KEY`): env var
  holding the real upstream key. Proxy reads at startup; if missing, treats
  requests as keyless.
- `--upstream-key-file PATH` (optional): file containing the real upstream
  key. Mutually exclusive with `--upstream-key-env`.
- `--allow-passthrough-auth` (default false): if set, requests whose auth
  header does not match the synthetic-key shape are forwarded with the
  original header passed through. If false, those requests are rejected with
  `401`. Default false because clean benchmark runs should ALWAYS use
  synthetic keys.

CLI flag changes on `agentsurge run`:

- `--frontend-server-url auto` (already present after Task C): when set to the
  literal `auto`, `BenchmarkRunner` starts an embedded proxy and points the
  renderer at it. Other values: an explicit URL → renderer points at that URL
  (assume the user runs the proxy externally), or unset (default `direct`
  mode, no proxy).
- `--proxy-target URL`: only honored when `--frontend-server-url auto`. The
  upstream the embedded proxy forwards to. Defaults to `--vllm-url` if unset.

## Lifecycle

### Embedded mode

```text
BenchmarkRunner.run() {
  if frontend.server_url == "auto":
    proxy_handle = await start_embedded_proxy(
        target=cfg.proxy_target or cfg.vllm_url,
        trace_dir=cfg.frontend_workspace_dir / "sessions",
    )
    frontend.server_url = proxy_handle.listen_url
  try:
    ... existing dispatch (see agentsurge/runner.py BenchmarkRunner.run) ...
  finally:
    await proxy_handle.close()  # graceful shutdown, flush all writers
}
```

The proxy is started before session dispatch (after preflight, before the
metrics collector is started) and closed in a `finally` so trace files are
flushed even when the run is interrupted.

### Sibling subcommand

User invokes `agentsurge proxy --listen ... --target ... --trace-dir
<somewhere>` in a separate shell. Long-running until SIGINT. Subsequent
`agentsurge run --frontend-server-url http://127.0.0.1:18080 ...` calls hit
the user-managed proxy. Trace aggregation in this mode requires
`--frontend-workspace-dir` to point at the same directory the proxy is using
as `--trace-dir` (or a parent thereof).

## Failure modes and recovery

Key invariants:

- Trace files are flushed and closed on every code path, including SIGINT and
  SIGTERM.
- Client-disconnect mid-stream still writes a `response.end` event with a
  `client_disconnect: true` flag.
- Upstream errors propagate to the client AND record an `error` trace event.

Specific cases:

- **Upstream unreachable**: proxy returns `502 Bad Gateway` to the client and
  writes `error` with `error_type="upstream_unreachable"`.
- **Upstream returns non-2xx**: proxy passes status and body through verbatim;
  `response.start` records the upstream status; `response.end` is still
  written.
- **Client disconnects mid-stream**: cancellation propagates from the client
  reader; proxy cancels the upstream read, writes `response.end` with
  `client_disconnect: true`, then closes the writer.
- **Trace file write error**: logged once at WARN level; the request is NOT
  failed because trace writing is best-effort. A `trace_writer_error` flag is
  attached to the in-memory request record so the aggregator can mark the
  session as having incomplete traces.
- **Port already bound**: embedded proxy raises at startup so
  `BenchmarkRunner.run` aborts before any session is dispatched. Sibling
  subcommand exits with a clear message.
- **SIGTERM during active request**: server enters a 5-second graceful drain;
  in-flight requests complete and their writers flush; new requests are
  rejected with `503`.

## Test strategy

Tests will be created under `tests/proxy/` in N.4.c:

- `tests/proxy/test_server_happy_path.py` — in-process aiohttp proxy plus a
  tiny aiohttp mock upstream that emits a fixed SSE stream; assert TTFT,
  chunk count, usage parsing.
- `tests/proxy/test_server_upstream_error.py` — mock upstream returns `500`
  or hangs; assert `error` event recorded and client sees the propagated
  status.
- `tests/proxy/test_server_client_disconnect.py` — simulated client closes
  mid-stream; assert `response.end` carries `client_disconnect: true` and the
  trace file is flushed.
- `tests/proxy/test_concurrent_sessions.py` — two simulated CLI clients with
  distinct synthetic keys hitting the proxy concurrently; assert their traces
  land under the correct per-session directories and do not interleave inside
  any single trace file.
- `tests/proxy/test_aggregation.py` — given a fixture trace tree, assert the
  aggregator produces the expected `SessionServingTraceMetrics` and
  `RunResult.serving_trace` values.

No tests run against real upstreams (CI-incompat). Real-CLI smoke is left to
manual benchmark runs using the existing fixture-capture protocol from
§Phase 1.5 in the parent design doc.

## Open questions

These should be flagged by the implementer during N.4 if they are not
resolved in this document:

1. **Streaming primitive**: is `aiohttp.ClientSession.post(...).content
   .iter_chunked(N)` the right primitive for SSE-style upstream responses, or
   do we need a line-oriented reader to count `sse_event_count` correctly?
   Recommendation: use `iter_chunked` for byte-passthrough; parse SSE event
   boundaries (`\n\n`) only inside `TraceWriter` for the `sse_event_count`
   field.
2. **Retry policy**: does the proxy retry on transient upstream errors, or is
   that the CLI's job? Recommendation: do not retry. The proxy is a
   passthrough; retries would distort serving TTFT measurement.
3. **Trace timestamps**: monotonic only, or also wall-clock? Recommendation:
   include both. `ts_monotonic` for delta math, `ts_wall` ISO8601 for human
   reading. Cost is negligible.
4. **Anthropic streaming format**: the proxy is byte-passthrough so no schema
   awareness is required for forwarding, but `usage` extraction differs
   between OpenAI and Anthropic streams. Recommendation: parse both
   `data: {"usage": ...}` (OpenAI) and `event: message_delta` payloads
   (Anthropic) inside `TraceWriter`; treat absence as N/A rather than an
   error.
5. **Trace-tree quota**: should the proxy enforce a per-session trace size
   cap mirroring item 14 in the parent design doc's P2 list? Recommendation:
   defer to N.5; for N.4 just write everything.

## Out of scope (recap)

- Authentication, rate limiting, body transformation.
- Real upstream-key rotation.
- Persistence beyond JSONL files (no DB).
- Multi-target / load balancing.
- Anthropic-streaming SSE format auto-detection beyond the usage-extraction
  case above (proxy is byte-passthrough; it does not need to know).
