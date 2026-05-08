# Security Policy

## Reporting a Vulnerability

Please report security issues privately through GitHub Security Advisories:
https://github.com/moreh-io/agent-surge/security/advisories/new

Do not include sensitive details in public issues. We will acknowledge reports as soon as practical and coordinate fixes or disclosure there.

## Execution Safety

`--tool-mode real` runs model-requested tool calls in a local workload-reproduction execution workspace. It is not a security sandbox. By default, tool subprocesses run with `--tool-env safe`, which uses a small allowlist of environment variables, sets `HOME` to the workspace root, and does not inherit credentials or service-specific variables from the parent process.

Use `--tool-env inherit` only for trusted workloads that intentionally need the parent process environment. Use trusted workloads or provide your own isolation when running untrusted traces, repositories, or model outputs.
