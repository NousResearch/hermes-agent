#!/usr/bin/env python3
"""Mac local-node tool surface and policy helpers.

The Mac local node is intentionally a compact workstation surface.  It is not a
SaaS connector bundle; it exposes six agent-facing tools with action enums and a
Claude Code-like local development policy.  The relay/client implementation can
be layered behind these handlers without changing the public schema.
"""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import fnmatch
import glob
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from tools.registry import registry

TOOLSET = "mac_local"

SYSTEM_ACTIONS = ["status"]
FS_ACTIONS = ["read", "search", "write", "patch"]
TERMINAL_ACTIONS = ["run", "start", "poll", "wait", "kill", "input", "exec_code"]
PROJECT_CONTEXT_ACTIONS = ["summarize"]
UI_ACTIONS = ["screenshot", "open", "clipboard", "osascript"]
AGENT_ACTIONS = ["spawn", "status", "logs", "kill"]
NOISY_SEARCH_DIRS = frozenset({".git", "node_modules", ".venv", "venv", "dist", "build", "__pycache__", ".cache"})
STRUCTURED_ERROR_CODES = [
    "MAC_OFFLINE",
    "ACTION_DENIED",
    "APPROVAL_REQUIRED",
    "PATH_DENIED",
    "SECRET_DENIED",
    "TIMEOUT",
    "PROCESS_NOT_FOUND",
]
MAX_TERMINAL_OUTPUT_CHARS = 50_000
MAX_TERMINAL_INPUT_CHARS = 16_384
_MANAGED_PROCESSES: dict[str, "ManagedProcess"] = {}

# Keep these as data so tests can assert the intentionally small surface.
REMOVED_STANDALONE_TOOL_NAMES = frozenset(
    {
        "mac_status",
        "mac_capabilities",
        "mac_read_file",
        "mac_search_files",
        "mac_write_file",
        "mac_patch",
        "mac_process_start",
        "mac_process_poll",
        "mac_process_wait",
        "mac_process_kill",
        "mac_process_input",
        "mac_execute_code",
        "mac_git_status",
        "mac_git_diff",
        "mac_git_commit",
        "mac_screenshot",
        "mac_browser",  # direct browser is deferred in V1
        "mac_clipboard",
        "mac_open",
        "mac_osascript",
        "mac_spawn_agent",
        "mac_agent_status",
        "mac_agent_logs",
        "mac_agent_kill",
    }
)


@dataclass(frozen=True)
class PolicyVerdict:
    """Decision returned by the Mac local-node policy classifier."""

    decision: str  # allow | ask | deny
    reason: str
    scope: str = "unknown"


@dataclass(frozen=True)
class TrustedRoot:
    """Trusted local root known to the Mac local-node policy."""

    path: str
    scope: str

    @property
    def canonical(self) -> str:
        expanded = os.path.expanduser(os.path.expandvars(self.path))
        return _normalize_path_for_policy(expanded)


@dataclass
class ManagedProcess:
    """Bounded-output background process handle for Mac terminal actions."""

    process: subprocess.Popen[str]
    stdout_chunks: list[str]
    stderr_chunks: list[str]
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    output_limit_exceeded: bool = False
    stdout_thread: threading.Thread | None = None
    stderr_thread: threading.Thread | None = None
    cwd: str = ""
    policy: Any = None
    stdout_poll_offset: int = 0
    stderr_poll_offset: int = 0


class MacLocalPolicy:
    """Claude Code-like policy for local Mac workstation actions.

    The policy is intentionally flexible inside trusted roots: local dev reads,
    writes, patches, tests, builds, background processes, local commits, and
    worker agents are allowed.  It asks only for actions with real risk:
    external/publicating side effects, broad destructive operations, global or
    privileged system changes, secrets, or paths outside trusted roots.
    """

    secret_name_patterns = (
        re.compile(r"(^|/)\.env($|[./*?\[\]{},'\")\s])"),
        re.compile(r"(^|/)\.npmrc($|[/*?\[\]{},'\")\s])"),
        re.compile(r"(^|/)\.pypirc($|[/*?\[\]{},'\")\s])"),
        re.compile(r"(^|/)\.netrc($|[/*?\[\]{},'\")\s])"),
        re.compile(r"(^|/)\.aws/credentials($|[/*?\[\]{},'\")\s])"),
        re.compile(r"(^|/)id_(rsa|dsa|ecdsa|ed25519)($|[/*?\[\]{},'\")\s])"),
        re.compile(r"(^|/)\.ssh(/|$)"),
        re.compile(r"(^|/)Library/Keychains(/|$)"),
        re.compile(r"(^|/)Keychains(/|$)"),
        re.compile(r"(^|/)(Cookies|Login Data|Local State)$"),
        re.compile(r"(^|/)(auth|token|credentials?)-?cache(/|$)", re.I),
        re.compile(r"(^|/)\.config/gh/hosts\.yml($|[/*?\[\]{},'\")\s])"),
        re.compile(r"(^|/)\.git-credentials($|[/*?\[\]{},'\")\s])"),
        re.compile(r"(^|/)\.docker/config\.json($|[/*?\[\]{},'\")\s])"),
        re.compile(r"(^|/)\.config/gcloud/application_default_credentials\.json($|[/*?\[\]{},'\")\s])"),
        re.compile(r"(^|/)\.kube/config($|[/*?\[\]{},'\")\s])"),
        re.compile(r"(^|/)\.codex(/|$)"),
        re.compile(r"(^|/)\.claude(/|$)"),
        re.compile(r"(^|/)\.mcp-auth[^/]*(/|$)"),
    )

    destructive_patterns = (
        re.compile(r"\brm\b[^\n;|&]*(?:--recursive\b|-[A-Za-z]*r[A-Za-z]*)", re.I),
        re.compile(r"\bfind\b[^\n;|&]*\s-delete\b"),
        re.compile(r"\bfind\b[^\n;|&]*\s-(?:exec|execdir)\s+rm\b"),
        re.compile(r"\bperl\b[^\n;|&]*\b(unlink|rmtree)\b"),
        re.compile(r"\bgit\b.*\breset\s+--hard\b"),
        re.compile(r"\bgit\b.*\bclean\s+-[^\n;|&]*[fdx][^\n;|&]*\b"),
        re.compile(r"\bgit\b.*\bpush\b.*\s--force(?:-with-lease)?\b"),
        re.compile(r"\bdocker\b.*\bcompose\b.*\bdown\b.*\s-v\b"),
        re.compile(r"\bdocker\b.*\bcompose\b.*\brm\b.*-[A-Za-z]*[svf][A-Za-z]*\b"),
        re.compile(r"\bdocker\b.*\b(system|volume)\s+(rm|prune)\b"),
        re.compile(r"\b(dd|mkfs|diskutil)\b"),
    )

    external_patterns = (
        re.compile(r"\bgit\b.*\bpush\b"),
        re.compile(r"\bgh\b.*\bpr\s+(create|comment|review|merge)\b"),
        re.compile(r"\bgh\b.*\bissue\s+(create|comment|edit|close|reopen)\b"),
        re.compile(r"\bgh\b.*\brelease\s+(create|upload|delete)\b"),
        re.compile(r"\brailway\s+(deploy|up|variables\s+set)\b"),
        re.compile(r"\b(vercel|netlify|flyctl)\s+(deploy|publish)\b"),
        re.compile(r"\bnpm\s+publish\b"),
        re.compile(r"\b(pnpm|yarn)\s+publish\b"),
        re.compile(r"\b(curl|wget)\b.*(--data|--data-binary|--post-file|-d|-F|-T|--upload-file|-X\s*POST|-X\s*PUT)", re.I),
        re.compile(r"\b(scp|rsync|nc|netcat|ssh|sftp|ftp)\b"),
        re.compile(r"\b(socket|requests|urllib|httpx|fetch|http\.client|asyncio\.open_connection|aiohttp|websockets|ftplib|smtplib)\b"),
        re.compile(r"\brequire\(\s*['\"](?:node:)?(?:net|tls|dgram|http|https)['\"]\s*\)"),
        re.compile(r"\bimport\(\s*['\"](?:node:)?(?:net|tls|dgram|http|https)['\"]\s*\)"),
        re.compile(r"\bfrom\s+['\"](?:node:)?(?:net|tls|dgram|http|https)['\"]"),
    )

    global_or_privileged_patterns = (
        re.compile(r"\bsudo\b"),
        re.compile(r"\b(chmod|chown)\s+-R\b"),
        re.compile(r"\bbrew\s+(install|upgrade|remove)\b"),
        re.compile(r"\bnpm\s+install\s+-g\b"),
        re.compile(r"\bpip\s+install\s+(--user|-U|--upgrade)\b"),
    )

    guarded_mac_surface_patterns = (
        re.compile(r"\bsecurity\b"),
        re.compile(r"\bosascript\b"),
        re.compile(r"\bpb(paste|copy)\b"),
    )

    broad_secret_discovery_patterns = (
        re.compile(r"\b(cat|less|more|head|tail)\s+\.\*"),
        re.compile(r"\b(grep|rg)\b.*\b(token|secret|password|credential|api[_-]?key)\b.*\s\.($|\s)"),
        re.compile(r"\bfind\s+\.(?:\s|$).*-name\s+['\"]?\.(env|npmrc|pypirc|netrc)\b"),
        re.compile(r"\bfind\s+\.(?:\s|$).*-type\s+f\b"),
        re.compile(r"\btar\b.*\s\.($|\s)"),
    )

    unmodeled_shell_expansion_patterns = (
        re.compile(r"(?<!\\)\$[A-Za-z_][A-Za-z0-9_]*"),
        re.compile(r"\{[^}\n]+\}"),
        re.compile(r"\$\("),
        re.compile(r"`"),
        re.compile(r"[<>]\("),
        re.compile(r"\b(alias|function)\b"),
    )

    exec_code_sensitive_patterns = (
        re.compile(r"\bimport\s+"),
        re.compile(r"\bfrom\s+\S+\s+import\b"),
        re.compile(r"\bimportlib\b"),
        re.compile(r"\b__import__\b"),
        re.compile(r"\brequire\s*\("),
        re.compile(r"\bimport\s*\("),
        re.compile(r"\beval\s*\("),
        re.compile(r"\bexec\s*\("),
        re.compile(r"\bopen\s*\("),
        re.compile(r"\bPath\s*\("),
        re.compile(r"\b(pathlib|os|subprocess|shutil)\b"),
    )

    interactive_shell_patterns = (
        re.compile(r"^(?:.*/)?(bash|sh|zsh|fish)(?:\s+.*)?$"),
        re.compile(r"^(?:(?:/usr/bin/)?env(?:\s+-\S+|\s+[A-Za-z_][A-Za-z0-9_]*=\S+)*|command)\s+(?:.*/)?(bash|sh|zsh|fish)(?:\s+.*)?$"),
        re.compile(r"^(python|python3|node)\s*(-i)?\s*$"),
    )

    relative_secret_command_patterns = (
        re.compile(r"(^|[\s'\"(])\.env($|[./*?\[\]{},'\")\s])"),
        re.compile(r"(^|[\s'\"(])\.aws/credentials($|[/*?\[\]{},'\")\s])"),
        re.compile(r"(^|[\s'\"(])\.npmrc($|[/*?\[\]{},'\")\s])"),
        re.compile(r"(^|[\s'\"(])\.pypirc($|[/*?\[\]{},'\")\s])"),
        re.compile(r"(^|[\s'\"(])\.netrc($|[/*?\[\]{},'\")\s])"),
    )

    def __init__(self, trusted_roots: Iterable[TrustedRoot]):
        self.trusted_roots = tuple(trusted_roots)

    @classmethod
    def default(cls) -> "MacLocalPolicy":
        return cls(
            [
                TrustedRoot("~/personal-projects", "personal"),
                TrustedRoot("~/projects", "personal"),
                TrustedRoot("~/Documents", "personal"),
                TrustedRoot("~/Downloads", "personal"),
                TrustedRoot("~/Desktop", "personal"),
                TrustedRoot("~/Obsidian", "personal"),
                # Rafael/Pazzi's work scope: paggo-project is mounted/exposed as /work.
                TrustedRoot("/work", "work"),
                TrustedRoot("/tmp", "scratch"),
            ]
        )

    def classify_path(self, path: str, action: str) -> PolicyVerdict:
        normalized = _normalize_path_for_policy(path)
        if self._is_secret_path(normalized):
            return PolicyVerdict("deny", "SECRET_DENIED", self._scope_for_path(normalized))
        scope = self._scope_for_path(normalized)
        if scope == "unknown":
            return PolicyVerdict("ask", "APPROVAL_REQUIRED", scope)
        if action in {"read", "search", "write", "patch", "screenshot", "open", "spawn"}:
            return PolicyVerdict("allow", "TRUSTED_ROOT", scope)
        return PolicyVerdict("ask", "APPROVAL_REQUIRED", scope)

    def classify_command(self, command: str, cwd: str | None = None) -> PolicyVerdict:
        cwd_scope = self._scope_for_path(_normalize_path_for_policy(cwd or os.getcwd()))
        if cwd_scope == "unknown":
            return PolicyVerdict("ask", "APPROVAL_REQUIRED", cwd_scope)

        compact = " ".join(command.strip().split())
        if not compact:
            return PolicyVerdict("deny", "EMPTY_COMMAND", cwd_scope)
        if self._matches_any(compact, self.broad_secret_discovery_patterns):
            return PolicyVerdict("ask", "APPROVAL_REQUIRED", cwd_scope)
        if self._has_shell_assignment_syntax(compact):
            return PolicyVerdict("ask", "APPROVAL_REQUIRED", cwd_scope)
        if self._command_mentions_secret(compact, cwd or os.getcwd()):
            return PolicyVerdict("deny", "SECRET_DENIED", cwd_scope)
        if self._matches_any(compact, self.unmodeled_shell_expansion_patterns):
            return PolicyVerdict("ask", "APPROVAL_REQUIRED", cwd_scope)
        if self._matches_any(compact, self.interactive_shell_patterns):
            return PolicyVerdict("ask", "APPROVAL_REQUIRED", cwd_scope)
        if self._inline_interpreter_requires_approval(compact):
            return PolicyVerdict("ask", "APPROVAL_REQUIRED", cwd_scope)
        if self._command_mentions_untrusted_path(compact, cwd or os.getcwd()):
            return PolicyVerdict("ask", "APPROVAL_REQUIRED", cwd_scope)
        if self._matches_any(compact, self.destructive_patterns):
            return PolicyVerdict("ask", "APPROVAL_REQUIRED", cwd_scope)
        if self._matches_any(compact, self.external_patterns):
            return PolicyVerdict("ask", "APPROVAL_REQUIRED", cwd_scope)
        if self._matches_any(compact, self.global_or_privileged_patterns):
            return PolicyVerdict("ask", "APPROVAL_REQUIRED", cwd_scope)
        if self._matches_any(compact, self.guarded_mac_surface_patterns):
            return PolicyVerdict("ask", "APPROVAL_REQUIRED", cwd_scope)
        return PolicyVerdict("allow", "LOCAL_DEV_ALLOWED", cwd_scope)

    def classify_exec_code(self, code: str, cwd: str | None = None) -> PolicyVerdict:
        """Classify short code snippets before mac_terminal.exec_code runs them.

        `exec_code` is intentionally for small local snippets.  Code with imports
        or dynamic loaders can hide network/file-exfiltration behavior from a
        regex command classifier, so route it through approval instead of trying
        to statically prove it safe.
        """

        cwd_scope = self._scope_for_path(_normalize_path_for_policy(cwd or os.getcwd()))
        if cwd_scope == "unknown":
            return PolicyVerdict("ask", "APPROVAL_REQUIRED", cwd_scope)
        compact = " ".join(code.strip().split())
        if not compact:
            return PolicyVerdict("allow", "LOCAL_DEV_ALLOWED", cwd_scope)
        if self._matches_any(compact, self.exec_code_sensitive_patterns):
            return PolicyVerdict("ask", "APPROVAL_REQUIRED", cwd_scope)
        if self._matches_any(compact, self.external_patterns):
            return PolicyVerdict("ask", "APPROVAL_REQUIRED", cwd_scope)
        if self._command_mentions_secret(compact, cwd or os.getcwd()):
            return PolicyVerdict("deny", "SECRET_DENIED", cwd_scope)
        if self._command_mentions_untrusted_path(compact, cwd or os.getcwd()):
            return PolicyVerdict("ask", "APPROVAL_REQUIRED", cwd_scope)
        return PolicyVerdict("allow", "LOCAL_DEV_ALLOWED", cwd_scope)

    def _is_secret_path(self, normalized: str) -> bool:
        # Explicit examples are safe to read/write in repos.
        if normalized.endswith("/.env.example") or normalized.endswith("/.env.sample"):
            return False
        return self._contains_secret_path_text(normalized)

    def _contains_secret_path_text(self, text: str) -> bool:
        normalized = text.replace("\\\\", "/")
        return any(pattern.search(normalized) for pattern in self.secret_name_patterns)

    def _command_mentions_secret(self, command: str, cwd: str | None = None) -> bool:
        if self._contains_secret_path_text(command):
            return True
        if any(self._is_secret_path(_normalize_path_for_policy(path)) for path in _absolute_path_mentions(command)):
            return True
        if self._matches_any(command, self.relative_secret_command_patterns):
            return True
        cwd_path = _normalize_path_for_policy(cwd or os.getcwd())
        try:
            parts = shlex.split(command)
        except ValueError:
            parts = command.split()
        for part in parts:
            cleaned = _clean_command_path_token(part)
            if not cleaned:
                continue
            expanded = _expand_command_path_token(cleaned, cwd_path)
            path_candidates = [expanded]
            if any(marker in expanded for marker in "*?[]"):
                glob_pattern = expanded if expanded.startswith("/") else str(Path(cwd_path, expanded))
                path_candidates.extend(str(Path(match).resolve(strict=False)) for match in glob.glob(glob_pattern, recursive=False))
            elif expanded.startswith("/") or "/" in expanded or expanded.startswith("."):
                base = Path(expanded) if expanded.startswith("/") else Path(cwd_path, expanded)
                path_candidates.append(str(base.resolve(strict=False)))
            for candidate in path_candidates:
                if self._is_secret_path(_normalize_path_for_policy(candidate)):
                    return True
        return False

    def _has_shell_assignment_syntax(self, command: str) -> bool:
        try:
            lexer = shlex.shlex(command, posix=True, punctuation_chars=";|&")
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError:
            return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", command.strip()))
        previous_is_separator = True
        for token in tokens:
            if token in {";", "|", "&", "&&", "||"}:
                previous_is_separator = True
                continue
            if previous_is_separator and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", token):
                return True
            previous_is_separator = False
        return False

    def _inline_interpreter_requires_approval(self, command: str) -> bool:
        if re.search(
            r"(?:^|[;&|]\s*)(?:(?:env|command)(?:\s+(?:-\S+|[A-Za-z_][A-Za-z0-9_]*=\S+))*\s+)*(?:python3?|node)\b[^\n;|&]*(?:<<|<\s*[^\s])",
            command,
        ):
            return True
        try:
            parts = shlex.split(command)
        except ValueError:
            return True
        if not parts:
            return False
        executable = Path(parts[0]).name
        if executable == "node" and "-e" in parts:
            return True
        if executable in {"python", "python3", Path(sys.executable).name}:
            for flag in ("-c", "-m"):
                if flag in parts:
                    if flag == "-m":
                        return True
                    index = parts.index(flag)
                    code = parts[index + 1] if index + 1 < len(parts) else ""
                    return self._matches_any(code, self.exec_code_sensitive_patterns) or self._matches_any(
                        code, self.external_patterns
                    )
        return False

    def _command_mentions_untrusted_path(self, command: str, cwd: str) -> bool:
        cwd_path = _normalize_path_for_policy(cwd)
        candidates = set(_absolute_path_mentions(command))
        try:
            parts = shlex.split(command)
        except ValueError:
            parts = command.split()
        for part in parts:
            cleaned = _clean_command_path_token(part)
            if not cleaned:
                continue
            expanded = _expand_command_path_token(cleaned, cwd_path)
            candidates.update(_absolute_path_mentions(expanded))
            if expanded.startswith("/"):
                candidates.add(expanded)
            elif expanded.startswith("../") or expanded == ".." or "/../" in expanded:
                candidates.add(str(Path(cwd_path, expanded)))
            elif any(marker in expanded for marker in "*?[]"):
                for match in glob.glob(str(Path(cwd_path, expanded)), recursive=False):
                    candidates.add(str(Path(match).resolve(strict=False)))
            elif "/" in expanded:
                candidates.add(str(Path(cwd_path, expanded).resolve(strict=False)))
            else:
                try:
                    exists = Path(cwd_path, expanded).exists()
                except OSError:
                    exists = False
                if exists:
                    candidates.add(str(Path(cwd_path, expanded).resolve(strict=False)))
        return any(self._scope_for_path(_normalize_path_for_policy(candidate)) == "unknown" for candidate in candidates)

    def _scope_for_path(self, normalized: str) -> str:
        for root in self.trusted_roots:
            canonical = root.canonical
            if normalized == canonical or normalized.startswith(canonical.rstrip("/") + "/"):
                return root.scope
        return "unknown"

    @staticmethod
    def _matches_any(command: str, patterns: Iterable[re.Pattern[str]]) -> bool:
        return any(pattern.search(command) for pattern in patterns)


def _normalize_path_for_policy(path: str) -> str:
    if not path:
        return ""
    expanded = os.path.expanduser(os.path.expandvars(path))
    try:
        return os.path.normpath(str(Path(expanded).resolve(strict=False)))
    except (OSError, RuntimeError, ValueError):
        return os.path.normpath(expanded)


def _absolute_path_mentions(text: str) -> list[str]:
    mentions = re.findall(r"(?<![\w@.-])(/[^\s'\"`<>|&;,)\]]+)", text)
    return [_clean_command_path_token(mention) for mention in mentions]


def _clean_command_path_token(token: str) -> str:
    cleaned = token.strip().strip("'\"`()[]{}<>,;|&")
    if cleaned.startswith("@/"):
        cleaned = cleaned[1:]
    return cleaned


def _expand_command_path_token(token: str, cwd: str) -> str:
    """Expand shell path variables that would otherwise hide path escapes."""

    expanded = token.replace("${PWD}", cwd).replace("$PWD", cwd)
    expanded = os.path.expanduser(os.path.expandvars(expanded))
    return expanded


def _action_schema(actions: list[str], *, extra_properties: dict[str, Any] | None = None) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "action": {
            "type": "string",
            "enum": actions,
            "description": "Action to perform within this compact Mac local-node tool.",
        },
        "response_format": {
            "type": "string",
            "enum": ["concise", "detailed"],
            "description": "Return compact output by default; use detailed only when follow-up IDs/metadata are needed.",
            "default": "concise",
        },
    }
    if extra_properties:
        properties.update(extra_properties)
    return {
        "type": "object",
        "properties": properties,
        "required": ["action"],
        "additionalProperties": False,
    }


def _tool_schema(name: str, description: str, actions: list[str], extra_properties: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": _action_schema(actions, extra_properties=extra_properties),
    }


def get_mac_local_tool_schemas() -> dict[str, dict[str, Any]]:
    """Return the intentionally compact six-tool Mac local-node schema map."""

    return {
        "mac_system": _tool_schema(
            "mac_system",
            "Use this when you need Mac online state, trusted roots, policy mode, and current local-node capabilities.",
            SYSTEM_ACTIONS,
        ),
        "mac_fs": _tool_schema(
            "mac_fs",
            "Use this when you need token-efficient Mac file read/search/write/patch with path and secret policy.",
            FS_ACTIONS,
            {
                "path": {"type": "string", "description": "File or directory path on the Mac."},
                "pattern": {"type": "string", "description": "Search pattern or patch/find text."},
                "content": {"type": "string", "description": "New file content or replacement text."},
                "offset": {"type": "integer", "minimum": 1, "description": "1-indexed line offset for reads."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 2000, "description": "Maximum lines/results to return."},
            },
        ),
        "mac_terminal": _tool_schema(
            "mac_terminal",
            "Use this when you need Mac shell, managed processes, stdin, or short local Python/JS code execution.",
            TERMINAL_ACTIONS,
            {
                "command": {"type": "string", "description": "Shell command for run/start actions."},
                "cwd": {"type": "string", "description": "Working directory on the Mac."},
                "process_id": {"type": "string", "description": "Managed process handle for poll/wait/kill/input."},
                "data": {"type": "string", "description": "Input text or code payload."},
                "language": {"type": "string", "enum": ["python", "javascript", "bash"], "description": "Language for exec_code."},
                "timeout": {"type": "integer", "minimum": 1, "maximum": 3600, "description": "Timeout in seconds."},
            },
        ),
        "mac_project_context": _tool_schema(
            "mac_project_context",
            "Use this when you need one compact preflight summary of a Mac project/repo before editing or testing.",
            PROJECT_CONTEXT_ACTIONS,
            {"path": {"type": "string", "description": "Project or repository path on the Mac."}},
        ),
        "mac_ui": _tool_schema(
            "mac_ui",
            "Use this when you need Mac screenshot, open, clipboard, or guarded AppleScript/JXA automation.",
            UI_ACTIONS,
            {
                "target": {"type": "string", "description": "URL, file, app, clipboard text, or AppleScript/JXA target."},
                "data": {"type": "string", "description": "Clipboard text or script body."},
            },
        ),
        "mac_agent": _tool_schema(
            "mac_agent",
            "Use this when you need to spawn, inspect, stream logs from, or stop a local Codex/Claude/OpenCode/Pi worker.",
            AGENT_ACTIONS,
            {
                "kind": {"type": "string", "enum": ["codex", "claude", "opencode", "pi"], "description": "Local worker to use when spawning."},
                "mode": {"type": "string", "enum": ["read_only", "review", "dev_autonomous"], "description": "Worker permission posture."},
                "workdir": {"type": "string", "description": "Trusted Mac workdir/worktree."},
                "prompt": {"type": "string", "description": "Task prompt for spawn."},
                "agent_id": {"type": "string", "description": "Worker handle for status/logs/kill."},
            },
        ),
    }


def get_action_enum(schema: dict[str, Any]) -> list[str]:
    """Return a schema's action enum; small helper used by tests/docs."""

    return list(schema["parameters"]["properties"]["action"]["enum"])


def check_mac_local_node_requirements() -> bool:
    """Return True so enabled mac_local tools stay discoverable offline.

    The mac_local toolset is opt-in, so availability should be controlled by
    toolset selection.  Once selected, handlers return structured MAC_OFFLINE
    errors when no node URL/relay is configured instead of disappearing from the
    model's schema list.
    """

    return True


def _capability_contract() -> dict[str, list[str]]:
    return {name: get_action_enum(schema) for name, schema in _SCHEMAS.items()}


def _trusted_root_contract() -> list[dict[str, str]]:
    return [
        {"path": root.path, "scope": root.scope, "canonical": root.canonical}
        for root in MacLocalPolicy.default().trusted_roots
    ]


def _offline_payload(tool: str, action: str | None, *, message: str | None = None) -> dict[str, Any]:
    return {
        "ok": False,
        "error_code": "MAC_OFFLINE",
        "message": message or "Mac local node relay is not connected.",
        "tool": tool,
        "action": action,
    }


def _offline_result(tool: str, action: str | None, *, message: str | None = None) -> str:
    return json.dumps(_offline_payload(tool, action, message=message))


def _invalid_action_result(tool: str, action: str | None, allowed_actions: list[str]) -> str:
    return json.dumps(
        {
            "ok": False,
            "error_code": "ACTION_DENIED",
            "message": "Unsupported action for Mac local-node tool.",
            "tool": tool,
            "action": action,
            "allowed_actions": allowed_actions,
        }
    )


def _mac_system_status_result(action: str | None) -> str:
    payload = _offline_payload("mac_system", action)
    payload.update(
        {
            "online": False,
            "heartbeat_age_seconds": None,
            "hostname": None,
            "os": "macos",
            "user": None,
            "session": None,
            "executor_version": None,
            "trusted_roots": _trusted_root_contract(),
            "denied_roots": [
                "~/.ssh",
                "~/Library/Keychains",
                "~/Library/Application Support/*/Cookies",
            ],
            "capabilities": _capability_contract(),
            "policy": {
                "mode": "claude_code_like_high_autonomy",
                "default_inside_trusted_roots": "allow_local_dev",
                "approval_required_for": [
                    "destructive",
                    "external_or_publishing",
                    "global_or_system",
                    "secret_sensitive",
                    "outside_trusted_roots",
                ],
            },
            "structured_error_codes": STRUCTURED_ERROR_CODES,
        }
    )
    return json.dumps(payload)


def _error_payload(tool: str, action: str | None, error_code: str, message: str, **extra: Any) -> str:
    payload = {
        "ok": False,
        "error_code": error_code,
        "message": message,
        "tool": tool,
        "action": action,
    }
    payload.update(extra)
    return json.dumps(payload)


def _terminal_policy_error(action: str, verdict: PolicyVerdict, *, cwd: str | None = None) -> str | None:
    if verdict.decision == "allow":
        return None
    code = "SECRET_DENIED" if verdict.reason == "SECRET_DENIED" else "APPROVAL_REQUIRED"
    message = "Secret/auth paths are denied by default." if code == "SECRET_DENIED" else "Command requires approval by Mac local-node policy."
    return _error_payload(
        "mac_terminal",
        action,
        code,
        message,
        cwd=cwd,
        scope=verdict.scope,
        policy_reason=verdict.reason,
    )


def _truncate_text(text: str, limit: int = MAX_TERMINAL_OUTPUT_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _safe_terminal_env() -> dict[str, str]:
    allowed = {"PATH", "LANG", "LC_ALL", "TERM", "TMPDIR", "SHELL"}
    return {key: value for key, value in os.environ.items() if key in allowed}


def _terminal_command_payload(
    action: str,
    *,
    cwd: str,
    verdict: PolicyVerdict,
    completed: subprocess.CompletedProcess[str],
) -> str:
    stdout, stdout_truncated = _truncate_text(completed.stdout or "")
    stderr, stderr_truncated = _truncate_text(completed.stderr or "")
    stdout_truncated = stdout_truncated or getattr(completed, "stdout_truncated", False)
    stderr_truncated = stderr_truncated or getattr(completed, "stderr_truncated", False)
    return json.dumps(
        {
            "ok": completed.returncode == 0 and not getattr(completed, "output_limit_exceeded", False),
            "action": action,
            "cwd": cwd,
            "exit_code": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "output_limit_exceeded": getattr(completed, "output_limit_exceeded", False),
            "policy_reason": verdict.reason,
            "scope": verdict.scope,
        }
    )


def _terminal_timeout_payload(action: str, cwd: str | None, timeout: int) -> str:
    return _error_payload("mac_terminal", action, "TIMEOUT", f"Command timed out after {timeout}s.", cwd=cwd)


def _extract_action(args: dict[str, Any] | None) -> str | None:
    if not isinstance(args, dict):
        return None
    action = args.get("action")
    return action if isinstance(action, str) else None


def _policy_error_for_path(action: str, path: str, verdict: PolicyVerdict) -> str | None:
    if verdict.decision == "allow":
        return None
    code = "SECRET_DENIED" if verdict.reason == "SECRET_DENIED" else "PATH_DENIED"
    message = "Secret/auth paths are denied by default." if code == "SECRET_DENIED" else "Path is outside trusted Mac roots."
    return _error_payload("mac_fs", action, code, message, path=path, scope=verdict.scope)


def _bounded_limit(value: Any, default: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(1, min(parsed, maximum))


def _canonical_for_payload(path: str) -> str:
    return _normalize_path_for_policy(path)


def _mac_fs_read(args: dict[str, Any], policy: MacLocalPolicy) -> str:
    action = "read"
    canonical = _canonical_for_payload(str(args.get("path") or ""))
    policy_error = _policy_error_for_path(action, canonical, policy.classify_path(canonical, action))
    if policy_error:
        return policy_error
    offset = _bounded_limit(args.get("offset"), 1, 1_000_000)
    limit = _bounded_limit(args.get("limit"), 200, 2_000)
    try:
        all_lines = Path(canonical).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return _error_payload("mac_fs", action, "PATH_DENIED", "File does not exist.", path=canonical)
    except UnicodeDecodeError:
        return _error_payload("mac_fs", action, "ACTION_DENIED", "File is not valid UTF-8 text.", path=canonical)
    except OSError as exc:
        return _error_payload("mac_fs", action, "ACTION_DENIED", f"File could not be read: {exc}", path=canonical)
    start = offset - 1
    selected = all_lines[start : start + limit]
    next_offset = offset + len(selected)
    truncated = next_offset <= len(all_lines)
    payload = {
        "ok": True,
        "action": action,
        "path": canonical,
        "offset": offset,
        "limit": limit,
        "lines": [{"number": start + index + 1, "text": text} for index, text in enumerate(selected)],
        "truncated": truncated,
    }
    if truncated:
        payload["next_offset"] = next_offset
    return json.dumps(payload)


def _mac_fs_search(args: dict[str, Any], policy: MacLocalPolicy) -> str:
    action = "search"
    canonical = _canonical_for_payload(str(args.get("path") or ""))
    policy_error = _policy_error_for_path(action, canonical, policy.classify_path(canonical, action))
    if policy_error:
        return policy_error
    pattern = str(args.get("pattern") or "")
    if not pattern:
        return _error_payload("mac_fs", action, "ACTION_DENIED", "Search pattern is required.", path=canonical)
    limit = _bounded_limit(args.get("limit"), 50, 500)
    is_glob = any(marker in pattern for marker in "*?[]")
    try:
        regex = None if is_glob else re.compile(pattern)
    except re.error as exc:
        return _error_payload("mac_fs", action, "ACTION_DENIED", f"Invalid search regex: {exc}", path=canonical)
    matches: list[dict[str, Any]] = []

    def search_file(file_path: str, filename: str) -> str | None:
        if policy.classify_path(file_path, action).decision != "allow":
            return None
        if is_glob and fnmatch.fnmatch(filename, pattern):
            matches.append({"path": file_path, "line": None, "text": None, "kind": "filename"})
            return "stop" if len(matches) >= limit else None
        try:
            lines = Path(file_path).read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            return None
        for line_number, text in enumerate(lines, start=1):
            if regex and regex.search(text):
                matches.append({"path": file_path, "line": line_number, "text": text})
                if len(matches) >= limit:
                    return "stop"
        return None

    root_path = Path(canonical)
    if not root_path.exists():
        return _error_payload("mac_fs", action, "PATH_DENIED", "Search root does not exist.", path=canonical)
    if root_path.is_file():
        truncated = search_file(canonical, root_path.name) == "stop"
        return json.dumps({"ok": True, "action": action, "path": canonical, "matches": matches, "truncated": truncated})

    try:
        walker = os.walk(canonical)
        for current_root, dirs, files in walker:
            dirs[:] = [dirname for dirname in dirs if dirname not in NOISY_SEARCH_DIRS]
            for filename in files:
                file_path = _canonical_for_payload(str(Path(current_root, filename)))
                if search_file(file_path, filename) == "stop":
                    return json.dumps({"ok": True, "action": action, "path": canonical, "matches": matches, "truncated": True})
    except OSError as exc:
        return _error_payload("mac_fs", action, "ACTION_DENIED", f"Search root could not be scanned: {exc}", path=canonical)
    return json.dumps({"ok": True, "action": action, "path": canonical, "matches": matches, "truncated": False})


def _mac_fs_write(args: dict[str, Any], policy: MacLocalPolicy) -> str:
    action = "write"
    canonical = _canonical_for_payload(str(args.get("path") or ""))
    policy_error = _policy_error_for_path(action, canonical, policy.classify_path(canonical, action))
    if policy_error:
        return policy_error
    target = Path(canonical)
    try:
        previous_exists = target.exists()
        previous_size = target.stat().st_size if previous_exists else None
    except OSError as exc:
        return _error_payload("mac_fs", action, "ACTION_DENIED", f"Existing file metadata could not be read: {exc}", path=canonical)
    content = str(args.get("content") or "")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return _error_payload("mac_fs", action, "ACTION_DENIED", f"File could not be written: {exc}", path=canonical)
    return json.dumps(
        {
            "ok": True,
            "action": action,
            "path": canonical,
            "bytes_written": len(content.encode("utf-8")),
            "previous_exists": previous_exists,
            "previous_size": previous_size,
        }
    )


def _mac_fs_patch(args: dict[str, Any], policy: MacLocalPolicy) -> str:
    action = "patch"
    canonical = _canonical_for_payload(str(args.get("path") or ""))
    policy_error = _policy_error_for_path(action, canonical, policy.classify_path(canonical, action))
    if policy_error:
        return policy_error
    pattern = str(args.get("pattern") or "")
    replacement = str(args.get("content") or "")
    if not pattern:
        return _error_payload("mac_fs", action, "ACTION_DENIED", "Patch pattern is required.", path=canonical)
    target = Path(canonical)
    try:
        original = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _error_payload("mac_fs", action, "PATH_DENIED", "File does not exist.", path=canonical)
    except UnicodeDecodeError:
        return _error_payload("mac_fs", action, "ACTION_DENIED", "File is not valid UTF-8 text.", path=canonical)
    except OSError as exc:
        return _error_payload("mac_fs", action, "ACTION_DENIED", f"File could not be read: {exc}", path=canonical)
    if pattern not in original:
        return _error_payload("mac_fs", action, "ACTION_DENIED", "Patch pattern was not found.", path=canonical)
    updated = original.replace(pattern, replacement, 1)
    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=f"a/{target.name}",
            tofile=f"b/{target.name}",
        )
    )
    try:
        target.write_text(updated, encoding="utf-8")
    except OSError as exc:
        return _error_payload("mac_fs", action, "ACTION_DENIED", f"File could not be patched: {exc}", path=canonical)
    return json.dumps({"ok": True, "action": action, "path": canonical, "diff": diff, "replacements": 1})


def handle_mac_fs_local(args: dict[str, Any] | None = None, *, policy: MacLocalPolicy | None = None) -> str:
    """Run the Mac-node side filesystem action layer with policy checks."""

    action = _extract_action(args)
    if action not in FS_ACTIONS:
        return _invalid_action_result("mac_fs", action, FS_ACTIONS)
    safe_args = args if isinstance(args, dict) else {}
    active_policy = policy or MacLocalPolicy.default()
    if action == "read":
        return _mac_fs_read(safe_args, active_policy)
    if action == "search":
        return _mac_fs_search(safe_args, active_policy)
    if action == "write":
        return _mac_fs_write(safe_args, active_policy)
    if action == "patch":
        return _mac_fs_patch(safe_args, active_policy)
    return _invalid_action_result("mac_fs", action, FS_ACTIONS)


def _terminal_cwd(args: dict[str, Any]) -> str:
    return _canonical_for_payload(str(args.get("cwd") or os.getcwd()))


def _terminate_process_group(process: subprocess.Popen[str], sig: int = signal.SIGTERM) -> None:
    try:
        os.killpg(process.pid, sig)
    except (OSError, ProcessLookupError):
        if process.poll() is None:
            process.terminate()


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        if process.poll() is None:
            process.kill()


def _run_shell_command(command: str, cwd: str, timeout: int) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        ["/bin/bash", "-c", command],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_safe_terminal_env(),
        start_new_session=True,
    )
    managed = ManagedProcess(process, [], [], cwd=cwd)
    _start_output_threads(managed)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _kill_process_group(process)
            process.wait(timeout=2)
        _collect_managed_output(managed)
        raise
    stdout, stderr, stdout_truncated, stderr_truncated = _collect_managed_output(managed)
    completed = subprocess.CompletedProcess(["/bin/bash", "-c", command], process.returncode, stdout, stderr)
    completed.output_limit_exceeded = managed.output_limit_exceeded
    completed.stdout_truncated = stdout_truncated
    completed.stderr_truncated = stderr_truncated
    return completed


def _mac_terminal_run(args: dict[str, Any], policy: MacLocalPolicy) -> str:
    action = "run"
    command = str(args.get("command") or "")
    cwd = _terminal_cwd(args)
    verdict = policy.classify_command(command, cwd=cwd)
    policy_error = _terminal_policy_error(action, verdict, cwd=cwd)
    if policy_error:
        return policy_error
    timeout = _bounded_limit(args.get("timeout"), 180, 3_600)
    try:
        completed = _run_shell_command(command, cwd, timeout)
    except subprocess.TimeoutExpired:
        return _terminal_timeout_payload(action, cwd, timeout)
    except OSError as exc:
        return _error_payload("mac_terminal", action, "ACTION_DENIED", f"Command could not be started: {exc}", cwd=cwd)
    return _terminal_command_payload(action, cwd=cwd, verdict=verdict, completed=completed)


def _mac_terminal_exec_code(args: dict[str, Any], policy: MacLocalPolicy) -> str:
    action = "exec_code"
    language = str(args.get("language") or "python")
    code = str(args.get("data") or "")
    cwd = _terminal_cwd(args)
    path_verdict = policy.classify_command("true", cwd=cwd)
    policy_error = _terminal_policy_error(action, path_verdict, cwd=cwd)
    if policy_error:
        return policy_error
    code_verdict = policy.classify_command(code, cwd=cwd) if language == "bash" else policy.classify_exec_code(code, cwd=cwd)
    policy_error = _terminal_policy_error(action, code_verdict, cwd=cwd)
    if policy_error:
        return policy_error
    timeout = _bounded_limit(args.get("timeout"), 30, 300)
    suffix_by_language = {"python": ".py", "javascript": ".mjs", "bash": ".sh"}
    command_by_language = {
        "python": lambda path: f"{shlex.quote(sys.executable)} {shlex.quote(path)}",
        "javascript": lambda path: f"node {shlex.quote(path)}",
        "bash": lambda path: f"/bin/bash {shlex.quote(path)}",
    }
    if language not in suffix_by_language:
        return _error_payload("mac_terminal", action, "ACTION_DENIED", "Unsupported exec_code language.", cwd=cwd)
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=suffix_by_language[language], dir=cwd, delete=False) as handle:
            handle.write(code)
            temp_path = handle.name
        completed = _run_shell_command(command_by_language[language](temp_path), cwd, timeout)
    except subprocess.TimeoutExpired:
        return _terminal_timeout_payload(action, cwd, timeout)
    except OSError as exc:
        return _error_payload("mac_terminal", action, "ACTION_DENIED", f"Code could not be executed: {exc}", cwd=cwd)
    finally:
        if "temp_path" in locals():
            try:
                Path(temp_path).unlink()
            except OSError:
                pass
    return _terminal_command_payload(action, cwd=cwd, verdict=code_verdict, completed=completed)


def _capture_process_stream(managed: ManagedProcess, stream_name: str) -> None:
    stream = getattr(managed.process, stream_name)
    chunks = managed.stdout_chunks if stream_name == "stdout" else managed.stderr_chunks
    if stream is None:
        return
    total = 0
    try:
        while True:
            chunk = stream.readline(MAX_TERMINAL_OUTPUT_CHARS + 1)
            if not chunk:
                break
            remaining = MAX_TERMINAL_OUTPUT_CHARS - total
            if remaining > 0:
                chunks.append(chunk[:remaining])
                total += min(len(chunk), remaining)
            if len(chunk) > remaining:
                if stream_name == "stdout":
                    managed.stdout_truncated = True
                else:
                    managed.stderr_truncated = True
                managed.output_limit_exceeded = True
                _terminate_process_group(managed.process)
    except OSError:
        return


def _start_output_threads(managed: ManagedProcess) -> None:
    managed.stdout_thread = threading.Thread(target=_capture_process_stream, args=(managed, "stdout"), daemon=True)
    managed.stderr_thread = threading.Thread(target=_capture_process_stream, args=(managed, "stderr"), daemon=True)
    managed.stdout_thread.start()
    managed.stderr_thread.start()


def _snapshot_managed_output(managed: ManagedProcess) -> tuple[str, str, bool, bool]:
    return (
        "".join(managed.stdout_chunks),
        "".join(managed.stderr_chunks),
        managed.stdout_truncated,
        managed.stderr_truncated,
    )


def _consume_managed_poll_output(managed: ManagedProcess) -> tuple[str, str, bool, bool]:
    stdout_all = "".join(managed.stdout_chunks)
    stderr_all = "".join(managed.stderr_chunks)
    stdout = stdout_all[managed.stdout_poll_offset :]
    stderr = stderr_all[managed.stderr_poll_offset :]
    managed.stdout_poll_offset = len(stdout_all)
    managed.stderr_poll_offset = len(stderr_all)
    return stdout, stderr, managed.stdout_truncated, managed.stderr_truncated


def _collect_managed_output(managed: ManagedProcess) -> tuple[str, str, bool, bool]:
    for thread in (managed.stdout_thread, managed.stderr_thread):
        if thread is not None:
            thread.join(timeout=2)
    return _snapshot_managed_output(managed)


def _mac_terminal_start(args: dict[str, Any], policy: MacLocalPolicy) -> str:
    action = "start"
    command = str(args.get("command") or "")
    cwd = _terminal_cwd(args)
    verdict = policy.classify_command(command, cwd=cwd)
    policy_error = _terminal_policy_error(action, verdict, cwd=cwd)
    if policy_error:
        return policy_error
    try:
        process = subprocess.Popen(
            ["/bin/bash", "-c", command],
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_safe_terminal_env(),
            start_new_session=True,
        )
    except OSError as exc:
        return _error_payload("mac_terminal", action, "ACTION_DENIED", f"Process could not be started: {exc}", cwd=cwd)
    process_id = f"macproc-{uuid.uuid4().hex[:12]}"
    managed = ManagedProcess(process, [], [], cwd=cwd, policy=policy)
    _MANAGED_PROCESSES[process_id] = managed
    _start_output_threads(managed)
    return json.dumps(
        {
            "ok": True,
            "action": action,
            "process_id": process_id,
            "pid": process.pid,
            "cwd": cwd,
            "started_at": time.time(),
            "policy_reason": verdict.reason,
            "scope": verdict.scope,
        }
    )


def _process_not_found(action: str, process_id: str | None) -> str:
    return _error_payload("mac_terminal", action, "PROCESS_NOT_FOUND", "Managed process was not found.", process_id=process_id)


def _mac_terminal_poll(args: dict[str, Any]) -> str:
    action = "poll"
    process_id = str(args.get("process_id") or "")
    managed = _MANAGED_PROCESSES.get(process_id)
    if managed is None:
        return _process_not_found(action, process_id)
    exit_code = managed.process.poll()
    if exit_code is None:
        stdout, stderr, stdout_truncated, stderr_truncated = _consume_managed_poll_output(managed)
        return json.dumps(
            {
                "ok": True,
                "action": action,
                "process_id": process_id,
                "running": True,
                "exit_code": None,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "output_limit_exceeded": managed.output_limit_exceeded,
            }
        )
    _MANAGED_PROCESSES.pop(process_id, None)
    _collect_managed_output(managed)
    stdout, stderr, stdout_truncated, stderr_truncated = _consume_managed_poll_output(managed)
    return json.dumps(
        {
            "ok": exit_code == 0 and not managed.output_limit_exceeded,
            "action": action,
            "process_id": process_id,
            "running": False,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "output_limit_exceeded": managed.output_limit_exceeded,
        }
    )


def _mac_terminal_wait(args: dict[str, Any]) -> str:
    action = "wait"
    process_id = str(args.get("process_id") or "")
    managed = _MANAGED_PROCESSES.get(process_id)
    if managed is None:
        return _process_not_found(action, process_id)
    timeout = _bounded_limit(args.get("timeout"), 180, 3_600)
    try:
        managed.process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        stdout, stderr, stdout_truncated, stderr_truncated = _snapshot_managed_output(managed)
        return _error_payload(
            "mac_terminal",
            action,
            "TIMEOUT",
            f"Process did not complete within {timeout}s.",
            process_id=process_id,
            running=True,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            output_limit_exceeded=managed.output_limit_exceeded,
        )
    _MANAGED_PROCESSES.pop(process_id, None)
    stdout, stderr, stdout_truncated, stderr_truncated = _collect_managed_output(managed)
    return json.dumps(
        {
            "ok": managed.process.returncode == 0 and not managed.output_limit_exceeded,
            "action": action,
            "process_id": process_id,
            "running": False,
            "exit_code": managed.process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "output_limit_exceeded": managed.output_limit_exceeded,
        }
    )


def _mac_terminal_kill(args: dict[str, Any]) -> str:
    action = "kill"
    process_id = str(args.get("process_id") or "")
    managed = _MANAGED_PROCESSES.pop(process_id, None)
    if managed is None:
        return _process_not_found(action, process_id)
    _terminate_process_group(managed.process)
    try:
        managed.process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _kill_process_group(managed.process)
        managed.process.wait(timeout=5)
    _collect_managed_output(managed)
    return json.dumps({"ok": True, "action": action, "process_id": process_id, "exit_code": managed.process.returncode})


def _mac_terminal_input(args: dict[str, Any]) -> str:
    action = "input"
    process_id = str(args.get("process_id") or "")
    managed = _MANAGED_PROCESSES.get(process_id)
    if managed is None or managed.process.stdin is None:
        return _process_not_found(action, process_id)
    data = str(args.get("data") or "")
    if len(data) > MAX_TERMINAL_INPUT_CHARS:
        return _error_payload(
            "mac_terminal",
            action,
            "ACTION_DENIED",
            f"Input exceeds {MAX_TERMINAL_INPUT_CHARS} character limit.",
            process_id=process_id,
        )
    policy = managed.policy or MacLocalPolicy.default()
    input_verdict = policy.classify_exec_code(data, cwd=managed.cwd or os.getcwd())
    policy_error = _terminal_policy_error(action, input_verdict, cwd=managed.cwd)
    if policy_error:
        return policy_error
    try:
        managed.process.stdin.write(data)
        managed.process.stdin.flush()
    except OSError as exc:
        return _error_payload("mac_terminal", action, "ACTION_DENIED", f"Could not write process input: {exc}", process_id=process_id)
    return json.dumps({"ok": True, "action": action, "process_id": process_id})


def handle_mac_terminal_local(args: dict[str, Any] | None = None, *, policy: MacLocalPolicy | None = None) -> str:
    """Run Mac-node side terminal actions with local-dev policy checks."""

    action = _extract_action(args)
    if action not in TERMINAL_ACTIONS:
        return _invalid_action_result("mac_terminal", action, TERMINAL_ACTIONS)
    safe_args = args if isinstance(args, dict) else {}
    active_policy = policy or MacLocalPolicy.default()
    if action == "run":
        return _mac_terminal_run(safe_args, active_policy)
    if action == "start":
        return _mac_terminal_start(safe_args, active_policy)
    if action == "poll":
        return _mac_terminal_poll(safe_args)
    if action == "wait":
        return _mac_terminal_wait(safe_args)
    if action == "kill":
        return _mac_terminal_kill(safe_args)
    if action == "input":
        return _mac_terminal_input(safe_args)
    if action == "exec_code":
        return _mac_terminal_exec_code(safe_args, active_policy)
    return _invalid_action_result("mac_terminal", action, TERMINAL_ACTIONS)


def _handle_placeholder(tool: str, args: dict[str, Any] | None, allowed_actions: list[str]) -> str:
    action = _extract_action(args)
    if action not in allowed_actions:
        return _invalid_action_result(tool, action, allowed_actions)
    if tool == "mac_system" and action == "status":
        return _mac_system_status_result(action)
    return _offline_result(tool, action)


def handle_mac_system(args: dict[str, Any] | None = None, **_: Any) -> str:
    return _handle_placeholder("mac_system", args, SYSTEM_ACTIONS)


def handle_mac_fs(args: dict[str, Any] | None = None, **_: Any) -> str:
    if os.getenv("HERMES_MAC_LOCAL_NODE_EXECUTE_LOCAL", "").lower() in {"1", "true", "yes", "on"}:
        return handle_mac_fs_local(args)
    return _handle_placeholder("mac_fs", args, FS_ACTIONS)


def handle_mac_terminal(args: dict[str, Any] | None = None, **_: Any) -> str:
    if os.getenv("HERMES_MAC_LOCAL_NODE_EXECUTE_LOCAL", "").lower() in {"1", "true", "yes", "on"}:
        return handle_mac_terminal_local(args)
    return _handle_placeholder("mac_terminal", args, TERMINAL_ACTIONS)


def handle_mac_project_context(args: dict[str, Any] | None = None, **_: Any) -> str:
    return _handle_placeholder("mac_project_context", args, PROJECT_CONTEXT_ACTIONS)


def handle_mac_ui(args: dict[str, Any] | None = None, **_: Any) -> str:
    return _handle_placeholder("mac_ui", args, UI_ACTIONS)


def handle_mac_agent(args: dict[str, Any] | None = None, **_: Any) -> str:
    return _handle_placeholder("mac_agent", args, AGENT_ACTIONS)


_SCHEMAS = get_mac_local_tool_schemas()


registry.register(
    name="mac_system",
    toolset=TOOLSET,
    schema=_SCHEMAS["mac_system"],
    handler=lambda args, **kwargs: handle_mac_system(args, **kwargs),
    check_fn=check_mac_local_node_requirements,
    emoji="🖥️",
    max_result_size_chars=50_000,
)
registry.register(
    name="mac_fs",
    toolset=TOOLSET,
    schema=_SCHEMAS["mac_fs"],
    handler=lambda args, **kwargs: handle_mac_fs(args, **kwargs),
    check_fn=check_mac_local_node_requirements,
    emoji="🖥️",
    max_result_size_chars=50_000,
)
registry.register(
    name="mac_terminal",
    toolset=TOOLSET,
    schema=_SCHEMAS["mac_terminal"],
    handler=lambda args, **kwargs: handle_mac_terminal(args, **kwargs),
    check_fn=check_mac_local_node_requirements,
    emoji="🖥️",
    max_result_size_chars=50_000,
)
registry.register(
    name="mac_project_context",
    toolset=TOOLSET,
    schema=_SCHEMAS["mac_project_context"],
    handler=lambda args, **kwargs: handle_mac_project_context(args, **kwargs),
    check_fn=check_mac_local_node_requirements,
    emoji="🖥️",
    max_result_size_chars=50_000,
)
registry.register(
    name="mac_ui",
    toolset=TOOLSET,
    schema=_SCHEMAS["mac_ui"],
    handler=lambda args, **kwargs: handle_mac_ui(args, **kwargs),
    check_fn=check_mac_local_node_requirements,
    emoji="🖥️",
    max_result_size_chars=50_000,
)
registry.register(
    name="mac_agent",
    toolset=TOOLSET,
    schema=_SCHEMAS["mac_agent"],
    handler=lambda args, **kwargs: handle_mac_agent(args, **kwargs),
    check_fn=check_mac_local_node_requirements,
    emoji="🖥️",
    max_result_size_chars=50_000,
)
