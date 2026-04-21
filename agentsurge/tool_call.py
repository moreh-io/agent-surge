# SPDX-License-Identifier: MIT
"""Tool calling definitions, validation, and execution for agent workloads.

Provides OpenAI-compatible tool definitions for coding agent tools,
validates LLM tool_call responses, and supports real tool execution
and recorded output replay.
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function calling format)
# ---------------------------------------------------------------------------

CODING_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Start line number (1-indexed). Optional.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "End line number (1-indexed, inclusive). Optional.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating or overwriting it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace a specific string in a file with new content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact string to find and replace.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement string.",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for a pattern in files using grep.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search in. Defaults to current dir.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Signal that the task is complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of what was accomplished.",
                    }
                },
                "required": ["summary"],
            },
        },
    },
]

TOOL_NAMES: set[str] = {t["function"]["name"] for t in CODING_TOOLS}

_TOOL_CALL_PATTERNS = re.compile(
    r'"name"\s*:\s*"(bash|edit|view|write|finish)"'
    r"|<tool_call>|<function="
    r'|"function"\s*:\s*\{',
    re.IGNORECASE,
)
_RUN_TAG_RE = re.compile(r"<run>\s*(?P<cmd>.*?)\s*</run>", re.IGNORECASE | re.DOTALL)
_BASH_FENCE_RE = re.compile(
    r"```(?:bash|sh|shell)\s*\n(?P<cmd>.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_TOOL_CALL_XML_RE = re.compile(
    r"<tool_call>\s*(?P<json>\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def looks_like_unparsed_tool_call(text: str) -> bool:
    """Return True if text contains patterns that look like tool calls the
    server failed to parse (e.g. wrong --tool-call-parser)."""
    return bool(_TOOL_CALL_PATTERNS.search(text))


def _deterministic_fallback_id(*parts: str) -> str:
    """Build a deterministic fallback tool_call_id from string components.

    Same inputs → same id, so replay-mode benchmark runs produce stable
    tool_call_id strings (downstream inflight dumps + JSON outputs no longer
    diff on every run for identical seed/input). Keeps the ``fallback_``
    prefix and 24-hex-char width so existing log-grep tooling continues to
    match.
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="replace"))
        h.update(b"\x1f")  # field separator (unit separator)
    return f"fallback_{h.hexdigest()[:24]}"


def synthesize_tool_calls_from_text(
    text: str,
    allowed_tool_names: set[str] | None = None,
    mode: str = "off",
) -> list["ToolCallCheck"]:
    """Best-effort conversion of text/markup tool intent into tool calls.

    Modes:
      - ``"off"``: disabled
      - ``"deepseek_text"``: parse ``<run>`` tags and fenced bash blocks
      - ``"hermes_xml"``: parse ``<tool_call>{JSON}</tool_call>`` XML tags
        (Qwen3/Hermes format, client-side - bypasses vLLM server-side parser)
    """
    if mode == "off" or not text:
        return []

    allowed_tool_names = allowed_tool_names or TOOL_NAMES

    if mode == "hermes_xml":
        return _parse_hermes_xml_tool_calls(text, allowed_tool_names)

    # deepseek_text mode: <run> tags and fenced bash
    tool_name = None
    if "execute_bash" in allowed_tool_names:
        tool_name = "execute_bash"
    elif "bash" in allowed_tool_names:
        tool_name = "bash"
    if tool_name is None:
        return []

    command = None
    run_match = _RUN_TAG_RE.search(text)
    if run_match:
        command = run_match.group("cmd")
    else:
        fence_match = _BASH_FENCE_RE.search(text)
        if fence_match:
            command = fence_match.group("cmd")

    if not command:
        return []

    command = command.strip()
    if not command:
        return []

    arguments = {"command": command}
    if tool_name == "execute_bash":
        arguments["timeout"] = 30

    arguments_raw = json.dumps(arguments, ensure_ascii=False)
    return [
        ToolCallCheck(
            valid=True,
            tool_call_id=_deterministic_fallback_id("deepseek_text", tool_name, arguments_raw, "0"),
            name=tool_name,
            arguments_raw=arguments_raw,
            arguments_parsed=arguments,
        )
    ]


def _parse_hermes_xml_tool_calls(text: str, allowed_tool_names: set[str]) -> list["ToolCallCheck"]:
    """Parse ``<tool_call>{...}</tool_call>`` XML from model output."""
    results = []
    occurrence = 0
    for match in _TOOL_CALL_XML_RE.finditer(text):
        raw_json = match.group("json")
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError:
            _log.warning("hermes_xml: malformed JSON in <tool_call>: %.200s", raw_json)
            continue
        name = parsed.get("name", "")
        if name not in allowed_tool_names:
            _log.warning("hermes_xml: unknown tool %r (allowed: %s)", name, allowed_tool_names)
            continue
        arguments = parsed.get("arguments", {})
        if isinstance(arguments, str):
            with contextlib.suppress(json.JSONDecodeError):
                arguments = json.loads(arguments)
        arguments_raw = (
            json.dumps(arguments, ensure_ascii=False) if isinstance(arguments, dict) else arguments
        )
        results.append(
            ToolCallCheck(
                valid=True,
                # Include the occurrence index so two identical calls in the
                # same text get distinct ids; the hash itself is deterministic
                # given identical inputs across invocations (replaces uuid4).
                tool_call_id=_deterministic_fallback_id(
                    "hermes_xml", name, arguments_raw, str(occurrence)
                ),
                name=name,
                arguments_raw=arguments_raw,
                arguments_parsed=arguments if isinstance(arguments, dict) else None,
            )
        )
        occurrence += 1
    return results


# ---------------------------------------------------------------------------
# Coding tasks for sanity checks
# ---------------------------------------------------------------------------

SANITY_TASKS: list[str] = [
    "Create a Python function that checks if a string is a valid IPv4 address. Write it to `solution.py` and verify it works with a few test cases.",
    "Read the file `data.csv` and write a Python script `analyze.py` that counts the number of rows and prints column statistics.",
    "Find all Python files in the current directory that import `json`, then create a summary file `imports.txt` listing them.",
    "Write a bash script `setup.sh` that creates a virtual environment, installs requirements from `requirements.txt`, and runs tests.",
    "Create a Python class `LRUCache` with `get` and `put` methods in `cache.py`. Include basic unit tests.",
]

SYSTEM_PROMPT = """You are a coding agent. You have access to tools for reading files, writing files, editing files, running bash commands, and searching code. Use these tools to complete the given task. When you are done, call the `finish` tool with a summary.

Important rules:
- Always use tools to interact with the filesystem. Do not just describe what to do.
- Use `bash` for running commands and tests.
- Use `read_file` to inspect existing files before editing.
- Use `write_file` to create new files.
- Use `edit_file` to modify existing files.
- Use `search_code` to find relevant code.
- Call `finish` when the task is complete."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass
class ToolCallCheck:
    """Validation result for a single tool_call."""

    valid: bool = True
    tool_call_id: str = ""
    name: str = ""
    arguments_raw: str = ""
    arguments_parsed: dict | None = None
    errors: list[str] = field(default_factory=list)
    sanitized: bool = False
    sanitized_original: str = ""
    sanitize_error: str = ""


@dataclass
class TurnValidation:
    """Validation result for a full LLM turn response."""

    turn_index: int
    has_tool_calls: bool = False
    has_content: bool = False
    finish_reason: str = ""
    tool_calls: list[ToolCallCheck] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    raw_response: dict = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        if self.errors:
            return False
        if self.has_tool_calls:
            return all(tc.valid for tc in self.tool_calls)
        return True

    @property
    def is_finished(self) -> bool:
        """True if the LLM called `finish` or gave a text-only response."""
        if not self.has_tool_calls and self.has_content:
            return True
        return any(tc.name == "finish" for tc in self.tool_calls if tc.valid)


def validate_tool_calls(
    response: dict,
    turn_index: int = 0,
    allowed_tool_names: set[str] | None = None,
    sanitize_truncated: bool = False,
) -> TurnValidation:
    """Validate an OpenAI chat completions response for correct tool_call format.

    Parameters
    ----------
    response : dict
        The full JSON response from /v1/chat/completions (non-streaming).
    turn_index : int
        Turn index for reporting.

    Returns
    -------
    TurnValidation
        Detailed validation result with per-tool-call checks.
    """
    v = TurnValidation(turn_index=turn_index, raw_response=response)
    allowed_tool_names = allowed_tool_names or TOOL_NAMES

    choices = response.get("choices")
    if not choices:
        v.errors.append("response has no 'choices'")
        return v

    choice = choices[0]
    v.finish_reason = choice.get("finish_reason", "")
    message = choice.get("message", {})

    content = message.get("content")
    if content and content.strip():
        v.has_content = True

    tool_calls = message.get("tool_calls")
    if not tool_calls:
        # No tool calls - text-only response. Valid if content exists.
        if not v.has_content:
            v.errors.append("response has neither tool_calls nor content")
        return v

    v.has_tool_calls = True
    for i, tc in enumerate(tool_calls):
        check = ToolCallCheck()

        # Check structure
        tc_id = tc.get("id", "")
        check.tool_call_id = tc_id
        if not tc_id:
            check.errors.append(f"tool_calls[{i}]: missing 'id'")

        tc_type = tc.get("type", "")
        if tc_type != "function":
            check.errors.append(f"tool_calls[{i}]: type={tc_type!r}, expected 'function'")

        func = tc.get("function", {})
        if not isinstance(func, dict):
            check.errors.append(f"tool_calls[{i}]: 'function' is not a dict")
            check.valid = False
            v.tool_calls.append(check)
            continue

        name = func.get("name", "")
        check.name = name
        if not name:
            check.errors.append(f"tool_calls[{i}]: missing function name")
        elif name not in allowed_tool_names:
            if not sanitize_truncated:
                check.errors.append(f"tool_calls[{i}]: unknown tool {name!r}")

        args_raw = func.get("arguments", "")
        check.arguments_raw = args_raw if isinstance(args_raw, str) else json.dumps(args_raw)

        # Validate arguments JSON
        try:
            parsed = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            if not isinstance(parsed, dict):
                check.errors.append(
                    f"tool_calls[{i}]: arguments is {type(parsed).__name__}, expected dict"
                )
            else:
                check.arguments_parsed = parsed
        except (json.JSONDecodeError, TypeError) as e:
            if sanitize_truncated:
                check.sanitized_original = args_raw  # preserve truncated original
                check.sanitize_error = str(e)
                check.arguments_parsed = {}
                check.arguments_raw = "{}"
                check.sanitized = True
            else:
                check.errors.append(f"tool_calls[{i}]: arguments JSON parse error: {e}")

        # Check for garbage (heuristic: very long repetitive strings)
        if check.arguments_raw and _looks_like_garbage(check.arguments_raw):
            check.errors.append(f"tool_calls[{i}]: arguments look like garbage output")

        check.valid = len(check.errors) == 0
        v.tool_calls.append(check)

    return v


def _looks_like_garbage(text: str, threshold: float = 0.5) -> bool:
    """Heuristic check for garbage/repetitive output.

    Returns True if the text appears to be garbage (high repetition ratio,
    excessive special characters, or very long with low entropy).
    """
    if len(text) < 20:
        return False

    # Check for long repeated substrings
    if len(text) > 200:
        # Sample a 10-char window and check repetition
        chunk = text[:10]
        if text.count(chunk) > len(text) / (len(chunk) * 2):
            return True

    # Check for excessive non-printable characters (sample first 2000 chars to avoid O(n) full scan)
    sample = text[:2000]
    non_printable = sum(1 for c in sample if ord(c) < 32 and c not in "\n\r\t")
    if non_printable > len(sample) * 0.1:
        return True

    # Check for very long strings with low unique character ratio
    if len(text) > 500:
        unique_ratio = len(set(text)) / min(len(text), 1000)
        if unique_ratio < 0.02:
            return True

    return False


# ---------------------------------------------------------------------------
# Real tool execution (local workspace)
# ---------------------------------------------------------------------------

_MAX_OUTPUT_CHARS = 8000  # Truncate tool output to prevent context explosion
_INSTANCE_ID_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
_SAFE_TOOL_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_SAFE_TOOL_ENV_KEYS = ("LANG", "LC_ALL", "LC_CTYPE", "TZ", "TERM")

# Match `/workspace/` only at path-argument boundaries: start of string,
# whitespace, `=` (e.g. ``--cwd=/workspace/foo``), or `:` (PATH-like). Skip
# occurrences directly preceded by a quote (single/double/back) so that
# literal data like ``echo "/workspace/foo"`` or ``grep '/workspace/' .`` is
# not corrupted by the rewrite.
_WORKSPACE_PATH_RE = re.compile(r"(?:(?<=^)|(?<=[\s=:]))/workspace/")


def _parse_owner_repo(instance_id: str) -> tuple[str, str]:
    """Extract (owner, repo_name) from an instance_id.

    Supports:
    - Standard SWE-bench: ``owner__repo-issue_number``
    - SWE-smith:          ``owner__repo.commit.variant__suffix``
    - Hyphenated owners:  ``django-money__django-money.abc123...``

    Returns ``("", "")`` when the id cannot be parsed.
    """
    parts = instance_id.split("__")
    if len(parts) < 2 or not parts[0]:
        return "", ""
    owner = parts[0]
    repo_segment = parts[1]
    if not repo_segment:
        return "", ""
    repo_name = re.sub(r"-\d+$", "", repo_segment.split(".")[0])
    if not repo_name:
        return "", ""
    return owner, repo_name


class ExecutionWorkspace:
    """A local execution workspace for real tool execution.

    File operations resolve under a workspace root, and path traversal
    attempts are blocked.
    """

    def __init__(self, root: str | Path, *, tool_env: str = "safe") -> None:
        if tool_env not in ("safe", "inherit"):
            raise ValueError(f"Unknown tool_env: {tool_env!r}")
        self.root = Path(root).resolve()
        self.tool_env = tool_env
        self.root.mkdir(parents=True, exist_ok=True)

    def _subprocess_env(self) -> dict[str, str]:
        """Build the environment for local tool subprocesses."""
        if self.tool_env == "inherit":
            return {**os.environ, "HOME": str(self.root)}

        tmpdir = self.root / "tmp"
        tmpdir.mkdir(exist_ok=True)
        env = {
            "HOME": str(self.root),
            "PATH": _SAFE_TOOL_PATH,
            "TMPDIR": str(tmpdir),
        }
        for key in _SAFE_TOOL_ENV_KEYS:
            value = os.environ.get(key)
            if value:
                env[key] = value
        env.setdefault("LANG", "C.UTF-8")
        return env

    def _safe_path(self, path: str) -> Path:
        """Resolve path within the workspace root, blocking traversal.

        Absolute ``/workspace/`` paths are rewritten to the local
        workspace directory so that traces recorded with OpenHands-style
        absolute paths resolve correctly.
        """
        if path.startswith("/workspace/"):
            path = str(self.root / path.lstrip("/"))
        resolved = (self.root / path).resolve()
        root_str = str(self.root)
        if resolved != self.root and not str(resolved).startswith(root_str + "/"):
            raise ValueError(f"Path traversal blocked: {path}")
        return resolved

    def setup_files(self, files: dict[str, str]) -> None:
        """Write setup files into the local execution workspace."""
        for path, content in files.items():
            full = self._safe_path(path)
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content)

    def run_command(self, cmd: str, timeout: int = 30) -> str:
        """Run a shell command inside the local execution workspace.

        Note: commands run with shell=True and are NOT process-isolated.
        """
        workspace_dir = self.root / "workspace"
        if workspace_dir.exists():
            cmd = _WORKSPACE_PATH_RE.sub(f"{workspace_dir}/", cmd)
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=str(self.root),
                capture_output=True,
                timeout=timeout,
                env=self._subprocess_env(),
            )
            output = result.stdout.decode(errors="replace")
            if result.returncode != 0:
                output += result.stderr.decode(errors="replace")
            if not output.strip():
                if result.returncode == 0:
                    output = "[command completed successfully]\n"
                else:
                    output = (
                        f"[exit code {result.returncode}]\n{result.stderr.decode(errors='replace')}"
                    )
            return output[:_MAX_OUTPUT_CHARS]
        except subprocess.TimeoutExpired:
            return f"[TIMEOUT after {timeout}s]\n"
        except Exception as e:
            return f"[ERROR] {e}\n"

    async def run_command_async(self, cmd: str, timeout: int = 30) -> str:
        """Run a shell command inside the local execution workspace without blocking the event loop."""
        workspace_dir = self.root / "workspace"
        if workspace_dir.exists():
            cmd = _WORKSPACE_PATH_RE.sub(f"{workspace_dir}/", cmd)
        try:
            proc = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                cmd,
                cwd=str(self.root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._subprocess_env(),
                start_new_session=(os.name == "posix"),
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except TimeoutError:
                if os.name == "posix":
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGKILL)
                else:
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                return f"[TIMEOUT after {timeout}s]\n"

            stdout_text = stdout.decode(errors="replace")
            stderr_text = stderr.decode(errors="replace")
            output = stdout_text
            if proc.returncode != 0:
                output += stderr_text
            if not output.strip():
                if proc.returncode == 0:
                    output = "[command completed successfully]\n"
                else:
                    output = f"[exit code {proc.returncode}]\n{stderr_text}"
            return output[:_MAX_OUTPUT_CHARS]
        except Exception as e:
            return f"[ERROR] {e}\n"

    def list_files(self) -> list[str]:
        """List all files in the local execution workspace (relative paths)."""
        files = []
        for p in self.root.rglob("*"):
            if p.is_file():
                files.append(str(p.relative_to(self.root)))
        return sorted(files)

    def setup_from_instance_id(self, instance_id: str, metadata: dict | None = None) -> None:
        """Clone and checkout a repository from a SWE-bench instance_id.

        instance_id format: ``"owner__repo-issue_number"``
        (e.g. ``"tobymao__sqlglot-2295"``).

        *metadata* may contain ``repo``, ``base_commit``, ``version``.
        When ``metadata["source"] == "abc-bench"``, the ABC-Bench tarball
        is used instead of git clone (see :mod:`agentsurge.loaders.abc_bench_assets`).
        """
        if metadata and metadata.get("source") == "abc-bench":
            self._setup_from_abc_bench(metadata)
            return

        repo_url = metadata.get("repo") if metadata else None
        if not repo_url and metadata:
            repo_url = metadata.get("repository") or metadata.get("project")
        base_commit = metadata.get("base_commit") if metadata else None
        version = metadata.get("version") if metadata else None

        if repo_url and not repo_url.startswith("http"):
            repo_url = f"https://github.com/{repo_url}.git"

        if not repo_url:
            owner, repo_name = _parse_owner_repo(instance_id)
            if owner and repo_name:
                repo_url = f"https://github.com/{owner}/{repo_name}.git"
                parts = instance_id.split("__")
                if len(parts) >= 3:
                    dot_parts = parts[1].split(".")
                    if (
                        not base_commit
                        and len(dot_parts) >= 2
                        and _INSTANCE_ID_COMMIT_RE.fullmatch(dot_parts[1])
                    ):
                        base_commit = dot_parts[1]
                    if not version and len(dot_parts) >= 3:
                        version = dot_parts[2]

        if not repo_url:
            _log.warning(
                "Cannot derive repo URL from instance_id=%r - "
                "skipping workspace clone (session will run without a repo checkout)",
                instance_id,
            )
            return

        workspace = self.root / "workspace"
        workspace.mkdir(exist_ok=True)

        # Build clone directory name to match OpenHands trace paths:
        # traces reference /workspace/{owner}__{repo}__{version}
        parts = instance_id.split("__")
        if len(parts) == 2 and version:
            owner = parts[0]
            repo_name = re.sub(r"-\d+$", "", parts[1])
            workspace_dir_name = f"{owner}__{repo_name}__{version}"
        else:
            workspace_dir_name = instance_id
        clone_target = workspace / workspace_dir_name

        clone_args = ["git", "clone"]
        if not base_commit:
            clone_args.extend(["--depth", "1"])
        clone_args.extend([repo_url, str(clone_target)])

        _log.info("Cloning %s -> %s", repo_url, clone_target)
        proc = subprocess.run(clone_args, timeout=120, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"git clone failed for instance_id={instance_id!r}:\n"
                f"  URL: {repo_url}\n"
                f"  Error: {proc.stderr[:500]}\n"
                f"  Alternatives:\n"
                f"    - Verify the repo exists on GitHub\n"
                f"    - Use --tool-workspace-dir with pre-cloned repos\n"
                f"    - Use --tool-mode off to skip workspace setup"
            )

        if base_commit:
            _log.info("Checking out %s", base_commit)
            cp = subprocess.run(
                ["git", "checkout", base_commit],
                cwd=str(clone_target),
                timeout=30,
                capture_output=True,
                text=True,
            )
            if cp.returncode != 0:
                _log.warning(
                    "git checkout %s failed (rc=%d): %s",
                    base_commit,
                    cp.returncode,
                    cp.stderr.strip(),
                )

    def _setup_from_abc_bench(self, metadata: dict) -> None:
        """Populate workspace from a pre-extracted ABC-Bench task directory."""
        from agentsurge.loaders.abc_bench_assets import find_repo_dir, get_task_dir

        task_id = metadata.get("task_id")
        if not task_id:
            raise RuntimeError(
                "ABC-Bench workspace setup requires metadata['task_id'] "
                f"(got metadata keys: {sorted(metadata)!r})"
            )
        task_dir = get_task_dir(task_id)
        repo_src = find_repo_dir(task_dir)
        if repo_src is None:
            raise RuntimeError(
                f"ABC-Bench task {task_id}: no repo subdirectory found in {task_dir}"
            )

        workspace = self.root / "workspace"
        workspace.mkdir(exist_ok=True)
        clone_target = workspace / repo_src.name
        if clone_target.exists():
            return
        shutil.copytree(str(repo_src), str(clone_target), symlinks=True)
        _log.info("ABC-Bench: copied %s -> %s", repo_src, clone_target)

    @classmethod
    def from_directory(cls, path: str | Path, *, tool_env: str = "safe") -> "ExecutionWorkspace":
        """Create a workspace rooted at an existing directory."""
        sb = cls.__new__(cls)
        sb.root = Path(path).resolve()
        sb.tool_env = tool_env
        return sb


Sandbox = ExecutionWorkspace


def real_execute(name: str, arguments: dict | None, sandbox: ExecutionWorkspace) -> str:
    """Execute a tool call for real inside a local execution workspace.

    Parameters
    ----------
    name : str
        Tool function name.
    arguments : dict or None
        Parsed arguments from the tool call.
    sandbox : ExecutionWorkspace
        Workspace root for execution.

    Returns
    -------
    str
        Real tool output string.
    """
    args = arguments or {}

    if name == "bash":
        cmd = args.get("command", "echo 'no command'")
        _log.debug("bash: %s", cmd[:200])
        return sandbox.run_command(cmd)

    if name == "read_file":
        path = args.get("path", "")
        try:
            full = sandbox._safe_path(path)
            if not full.exists():
                return f"[ERROR] File not found: {path}\n"
            content = full.read_text(errors="replace")
            # Apply line range if specified
            start = args.get("start_line")
            end = args.get("end_line")
            if start is not None or end is not None:
                lines = content.splitlines(keepends=True)
                start_idx = max(0, (start or 1) - 1)
                end_idx = end or len(lines)
                content = "".join(lines[start_idx:end_idx])
            return content[:_MAX_OUTPUT_CHARS]
        except ValueError as e:
            return f"[ERROR] {e}\n"
        except Exception as e:
            return f"[ERROR] reading {path}: {e}\n"

    if name == "write_file":
        path = args.get("path", "")
        content = args.get("content", "")
        try:
            full = sandbox._safe_path(path)
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content)
            return f"File written: {path} ({len(content)} chars)\n"
        except ValueError as e:
            return f"[ERROR] {e}\n"
        except Exception as e:
            return f"[ERROR] writing {path}: {e}\n"

    if name == "edit_file":
        path = args.get("path", "")
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")
        try:
            full = sandbox._safe_path(path)
            if not full.exists():
                return f"[ERROR] File not found: {path}\n"
            text = full.read_text(errors="replace")
            if old_string not in text:
                return f"[ERROR] old_string not found in {path}\n"
            text = text.replace(old_string, new_string, 1)
            full.write_text(text)
            return f"Edited {path}: replaced {len(old_string)} chars with {len(new_string)} chars\n"
        except ValueError as e:
            return f"[ERROR] {e}\n"
        except Exception as e:
            return f"[ERROR] editing {path}: {e}\n"

    if name == "search_code":
        pattern = args.get("pattern", "")
        search_path = args.get("path", ".")
        cmd = f"grep -rn {shlex.quote(pattern)} {shlex.quote(search_path)} 2>&1 || true"
        return sandbox.run_command(cmd)

    if name == "finish":
        summary = args.get("summary", "")
        return f"Task finished: {summary}\n"

    if name == "execute_bash":
        cmd = args.get("command", "echo 'no command'")
        if str(args.get("is_input", "")).lower() == "true":
            return "[ERROR] interactive execute_bash input is not supported in AgentSurge local execution workspace\n"
        timeout = args.get("timeout")
        try:
            timeout = int(timeout) if timeout is not None else 30
        except (TypeError, ValueError):
            timeout = 30
        _log.debug("execute_bash: %s", cmd[:200])
        return sandbox.run_command(cmd, timeout=timeout)

    if name == "str_replace_editor":
        return _str_replace_editor(args, sandbox)

    if name == "think":
        thought = args.get("thought", "")
        return f"[thought] {thought}\n" if thought else "[thought recorded]\n"

    if name == "task_tracker":
        command = args.get("command", "view")
        task_list = args.get("task_list")
        if command in {"view", "plan"}:
            return json.dumps(task_list or [], indent=2)[:_MAX_OUTPUT_CHARS]
        return f"[ERROR] Unknown task_tracker command: {command}\n"

    return f"[ERROR] Unknown tool: {name}\n"


def _str_replace_editor(args: dict, sandbox: ExecutionWorkspace) -> str:
    """Handle str_replace_editor tool calls (OpenHands file editor)."""
    command = args.get("command", "view")
    path = args.get("path", "")
    if not path:
        return "[ERROR] path is required\n"
    try:
        full = sandbox._safe_path(path)
    except ValueError as e:
        return f"[ERROR] {e}\n"

    if command == "view":
        if full.is_dir():
            entries = sorted(p.name for p in full.iterdir())
            return (
                ("\n".join(entries) + "\n")[:_MAX_OUTPUT_CHARS]
                if entries
                else "[empty directory]\n"
            )
        if not full.exists():
            return f"[ERROR] File not found: {path}\n"
        content = full.read_text(errors="replace")
        view_range = args.get("view_range")
        if isinstance(view_range, list) and view_range:
            lines = content.splitlines(keepends=True)
            try:
                start = max(0, int(view_range[0]) - 1)
                end = (
                    len(lines)
                    if len(view_range) < 2 or int(view_range[1]) == -1
                    else int(view_range[1])
                )
            except (ValueError, TypeError, IndexError):
                start, end = 0, len(lines)
            content = "".join(lines[start:end])
        return content[:_MAX_OUTPUT_CHARS]

    if command == "create":
        file_text = args.get("file_text", "")
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(file_text)
        return f"File created: {path} ({len(file_text)} chars)\n"

    if command == "str_replace":
        if not full.exists():
            return f"[ERROR] File not found: {path}\n"
        old_str = args.get("old_str", "")
        new_str = args.get("new_str", "")
        text = full.read_text(errors="replace")
        if old_str not in text:
            return f"[ERROR] old_str not found in {path}\n"
        full.write_text(text.replace(old_str, new_str, 1))
        return f"Edited {path}: replaced {len(old_str)} chars with {len(new_str)} chars\n"

    if command == "insert":
        if not full.exists():
            return f"[ERROR] File not found: {path}\n"
        insert_line = args.get("insert_line")
        new_str = args.get("new_str", "")
        if insert_line is None:
            return "[ERROR] insert_line is required for insert\n"
        lines = full.read_text(errors="replace").splitlines(keepends=True)
        try:
            idx = max(0, min(len(lines), int(insert_line)))
        except (ValueError, TypeError):
            return f"[ERROR] insert_line must be an integer, got {insert_line!r}\n"
        payload = new_str if new_str.endswith("\n") else new_str + "\n"
        lines.insert(idx, payload)
        full.write_text("".join(lines))
        return f"Inserted text into {path} after line {insert_line}\n"

    if command == "undo_edit":
        return "[ERROR] undo_edit not supported in local execution workspace\n"

    return f"[ERROR] Unknown str_replace_editor command: {command}\n"


async def real_execute_async(name: str, arguments: dict | None, sandbox: ExecutionWorkspace) -> str:
    """Async variant - uses non-blocking subprocess for command tools.

    Subprocess-spawning tools (bash, execute_bash, search_code) get a
    native asyncio.create_subprocess implementation. Everything else (the
    filesystem-tool branches: read_file, write_file, edit_file,
    str_replace_editor, think, task_tracker, finish) is dispatched to a
    worker thread via :func:`asyncio.to_thread` so a slow
    ``Path.read_text``/``write_text`` on a multi-MB file does not pin the
    event loop and stall every other in-flight session sharing it. Mirrors
    the existing async-dispatch treatment of ``_setup_sandbox`` and
    ``_maybe_warmstart`` in the runner (see master commits 0958bde, c9268ad).
    """
    args = arguments or {}

    if name == "execute_bash":
        cmd = args.get("command", "echo 'no command'")
        if str(args.get("is_input", "")).lower() == "true":
            return "[ERROR] interactive execute_bash input is not supported in AgentSurge local execution workspace\n"
        timeout = args.get("timeout")
        try:
            timeout = int(timeout) if timeout is not None else 30
        except (TypeError, ValueError):
            timeout = 30
        _log.debug("execute_bash: %s", cmd[:200])
        return await sandbox.run_command_async(cmd, timeout=timeout)

    if name == "bash":
        cmd = args.get("command", "echo 'no command'")
        _log.debug("bash: %s", cmd[:200])
        return await sandbox.run_command_async(cmd)

    if name == "search_code":
        pattern = args.get("pattern", "")
        search_path = args.get("path", ".")
        cmd = f"grep -rn {shlex.quote(pattern)} {shlex.quote(search_path)} 2>&1 || true"
        return await sandbox.run_command_async(cmd)

    return await asyncio.to_thread(real_execute, name, arguments, sandbox)


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------


def build_tool_response_messages(
    assistant_message: dict,
    tool_calls: list[ToolCallCheck],
    *,
    sandbox: "ExecutionWorkspace | None" = None,
    **_kwargs,
) -> list[dict]:
    """Build tool response messages for the next turn via local workspace execution.

    Parameters
    ----------
    assistant_message : dict
        The assistant message dict from the LLM response (with tool_calls).
    tool_calls : list[ToolCallCheck]
        Validated tool call checks.
    sandbox : ExecutionWorkspace or None
        Local execution workspace for real tool execution.

    Returns
    -------
    list[dict]
        [assistant_message, tool_response_1, tool_response_2, ...] to append
        to the conversation.
    """
    messages: list[dict] = [assistant_message]

    for tc in tool_calls:
        if not tc.valid:
            output = f"[ERROR] Invalid tool call: {'; '.join(tc.errors)}"
        elif tc.sanitized:
            truncated_preview = tc.sanitized_original[:500] if tc.sanitized_original else "(empty)"
            output = (
                f"[ERROR] Your tool call to '{tc.name}' had truncated/malformed arguments "
                f"and could not be executed. Parse error: {tc.sanitize_error}\n"
                f"Your truncated arguments were:\n{truncated_preview}\n\n"
                f"Please try again. If the arguments are long (e.g. large code blocks "
                f"in old_str/new_str), try breaking the edit into smaller pieces or "
                f"use a bash command with sed instead."
            )
        elif sandbox is not None:
            output = real_execute(tc.name, tc.arguments_parsed, sandbox)
        else:
            output = "[ERROR] No local execution workspace provided for tool execution"

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.tool_call_id,
                "name": tc.name,
                "content": output,
            }
        )

    return messages


async def build_tool_response_messages_async(
    assistant_message: dict,
    tool_calls: list[ToolCallCheck],
    *,
    sandbox: "ExecutionWorkspace | None" = None,
    **_kwargs,
) -> list[dict]:
    """Async variant of tool response building.

    Uses native asyncio subprocess execution for command tools and preserves the
    existing synchronous implementation for pure filesystem operations.
    """
    messages: list[dict] = [assistant_message]

    for tc in tool_calls:
        if not tc.valid:
            output = f"[ERROR] Invalid tool call: {'; '.join(tc.errors)}"
        elif tc.sanitized:
            truncated_preview = tc.sanitized_original[:500] if tc.sanitized_original else "(empty)"
            output = (
                f"[ERROR] Your tool call to '{tc.name}' had truncated/malformed arguments "
                f"and could not be executed. Parse error: {tc.sanitize_error}\n"
                f"Your truncated arguments were:\n{truncated_preview}\n\n"
                f"Please try again. If the arguments are long (e.g. large code blocks "
                f"in old_str/new_str), try breaking the edit into smaller pieces or "
                f"use a bash command with sed instead."
            )
        elif sandbox is not None:
            output = await real_execute_async(tc.name, tc.arguments_parsed, sandbox)
        else:
            output = "[ERROR] No local execution workspace provided for tool execution"

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.tool_call_id,
                "name": tc.name,
                "content": output,
            }
        )

    return messages
