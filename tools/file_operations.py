#!/usr/bin/env python3
"""
File Operations Module

Provides file manipulation capabilities (read, write, patch, search) that work
across all terminal backends (local, docker, ssh, singularity, modal, daytona, vercel_sandbox).

The key insight is that all file operations can be expressed as shell commands,
so we wrap the terminal backend's execute() interface to provide a unified file API.

Usage:
    from tools.file_operations import ShellFileOperations
    from tools.terminal_tool import _active_environments
    
    # Get file operations for a terminal environment
    file_ops = ShellFileOperations(terminal_env)
    
    # Read a file
    result = file_ops.read_file("/path/to/file.py")
    
    # Write a file
    result = file_ops.write_file("/path/to/new.py", "print('hello')")
    
    # Search for content
    result = file_ops.search("TODO", path=".", file_glob="*.py")
"""

import os
import re
import difflib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
from tools.binary_extensions import BINARY_EXTENSIONS

from agent.file_safety import (
    build_write_denied_paths,
    build_write_denied_prefixes,
    get_safe_write_root as _shared_get_safe_write_root,
    is_write_denied as _shared_is_write_denied,
)


# ---------------------------------------------------------------------------
# Write-path deny list — blocks writes to sensitive system/credential files
# ---------------------------------------------------------------------------

_HOME = str(Path.home())

WRITE_DENIED_PATHS = build_write_denied_paths(_HOME)

WRITE_DENIED_PREFIXES = build_write_denied_prefixes(_HOME)


_OSC_SEQUENCE_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_FENCE_MARKER_RE = re.compile(r"'?\x07?__HERMES_FENCE_[A-Za-z0-9]+__\x07?'?")


def _strip_terminal_fence_leaks(text: str) -> str:
    """Strip leaked terminal fence wrappers from file read output."""
    if not text:
        return text

    cleaned_lines: List[str] = []
    for line in text.splitlines(keepends=True):
        had_terminal_wrapper = "__HERMES_FENCE_" in line or "\x1b]" in line
        cleaned = _OSC_SEQUENCE_RE.sub("", line)
        cleaned = _FENCE_MARKER_RE.sub("", cleaned)
        cleaned = cleaned.replace("\x07", "")
        if had_terminal_wrapper and cleaned.strip("'\r\n\t ") == "":
            continue
        cleaned_lines.append(cleaned)
    return "".join(cleaned_lines)


def _get_safe_write_root() -> Optional[str]:
    """Return the resolved HERMES_WRITE_SAFE_ROOT path, or None if unset.

    When set, all write_file/patch operations are constrained to this
    directory tree.  Writes outside it are denied even if the target is
    not on the static deny list.  Opt-in hardening for gateway/messaging
    deployments that should only touch a workspace checkout.
    """
    return _shared_get_safe_write_root()


def _is_write_denied(path: str) -> bool:
    """Return True if path is on the write deny list."""
    return _shared_is_write_denied(path)


# =============================================================================
# Result Data Classes
# =============================================================================

@dataclass
class ReadResult:
    """Result from reading a file."""
    content: str = ""
    total_lines: int = 0
    file_size: int = 0
    truncated: bool = False
    hint: Optional[str] = None
    is_binary: bool = False
    is_image: bool = False
    base64_content: Optional[str] = None
    mime_type: Optional[str] = None
    dimensions: Optional[str] = None  # For images: "WIDTHxHEIGHT"
    error: Optional[str] = None
    similar_files: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None and v != []}


@dataclass
class WriteResult:
    """Result from writing a file."""
    bytes_written: int = 0
    dirs_created: bool = False
    lint: Optional[Dict[str, Any]] = None
    # Semantic diagnostics from the LSP layer, when applicable.  Kept in
    # its own field (not folded into ``lint``) so the model and any
    # downstream parsers can read syntax errors and semantic errors as
    # separate signals.  ``None`` when LSP is disabled, when the file
    # isn't in a git workspace, or when no diagnostics were introduced
    # by this edit.
    lsp_diagnostics: Optional[str] = None
    error: Optional[str] = None
    warning: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class PatchResult:
    """Result from patching a file."""
    success: bool = False
    diff: str = ""
    files_modified: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    files_deleted: List[str] = field(default_factory=list)
    lint: Optional[Dict[str, Any]] = None
    # See :class:`WriteResult.lsp_diagnostics`.
    lsp_diagnostics: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        result = {"success": self.success}
        if self.diff:
            result["diff"] = self.diff
        if self.files_modified:
            result["files_modified"] = self.files_modified
        if self.files_created:
            result["files_created"] = self.files_created
        if self.files_deleted:
            result["files_deleted"] = self.files_deleted
        if self.lint:
            result["lint"] = self.lint
        if self.lsp_diagnostics:
            result["lsp_diagnostics"] = self.lsp_diagnostics
        if self.error:
            result["error"] = self.error
        return result


@dataclass
class SearchMatch:
    """A single search match."""
    path: str
    line_number: int
    content: str
    mtime: float = 0.0  # Modification time for sorting


@dataclass
class SearchResult:
    """Result from searching."""
    matches: List[SearchMatch] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    total_count: int = 0
    truncated: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        result = {"total_count": self.total_count}
        if self.matches:
            result["matches"] = [
                {"path": m.path, "line": m.line_number, "content": m.content}
                for m in self.matches
            ]
        if self.files:
            result["files"] = self.files
        if self.counts:
            result["counts"] = self.counts
        if self.truncated:
            result["truncated"] = True
        if self.error:
            result["error"] = self.error
        return result


@dataclass
class LintResult:
    """Result from linting a file."""
    success: bool = True
    skipped: bool = False
    output: str = ""
    message: str = ""
    
    def to_dict(self) -> dict:
        if self.skipped:
            return {"status": "skipped", "message": self.message}
        result = {"status": "ok" if self.success else "error", "output": self.output}
        if self.message:
            result["message"] = self.message
        return result


@dataclass
class ExecuteResult:
    """Result from executing a shell command."""
    stdout: str = ""
    exit_code: int = 0


def _parse_search_context_line(line: str) -> tuple[str, int, str] | None:
    """Parse grep/rg context output in ``path-line-content`` format.

    Context lines are ambiguous because filenames may legitimately contain
    ``-<digits>-`` segments. Prefer the rightmost numeric separator so a path
    like ``dir/file-12-name.py-8-context`` resolves to
    ``dir/file-12-name.py`` line ``8`` instead of truncating at ``file``.
    """
    if not line or line == "--":
        return None

    match = None
    for candidate in re.finditer(r'-(\d+)-', line):
        match = candidate

    if match is None:
        return None

    path = line[:match.start()]
    if not path:
        return None

    return path, int(match.group(1)), line[match.end():]


# =============================================================================
# Abstract Interface
# =============================================================================

class FileOperations(ABC):
    """Abstract interface for file operations across terminal backends."""
    
    @abstractmethod
    def read_file(self, path: str, offset: int = 1, limit: int = 500) -> ReadResult:
        """Read a file with pagination support."""
        ...

    @abstractmethod
    def read_file_raw(self, path: str) -> ReadResult:
        """Read the complete file content as a plain string.

        No pagination, no line-number prefixes, no per-line truncation.
        Returns ReadResult with .content = full file text, .error set on
        failure. Always reads to EOF regardless of file size.
        """
        ...

    @abstractmethod
    def write_file(self, path: str, content: str) -> WriteResult:
        """Write content to a file, creating directories as needed."""
        ...

    @abstractmethod
    def patch_replace(self, path: str, old_string: str, new_string: str,
                      replace_all: bool = False) -> PatchResult:
        """Replace text in a file using fuzzy matching."""
        ...

    @abstractmethod
    def patch_v4a(self, patch_content: str) -> PatchResult:
        """Apply a V4A format patch."""
        ...

    @abstractmethod
    def delete_file(self, path: str) -> WriteResult:
        """Delete a file. Returns WriteResult with .error set on failure."""
        ...

    @abstractmethod
    def move_file(self, src: str, dst: str) -> WriteResult:
        """Move/rename a file from src to dst. Returns WriteResult with .error set on failure."""
        ...

    @abstractmethod
    def search(self, pattern: str, path: str = ".", target: str = "content",
               file_glob: Optional[str] = None, limit: int = 50, offset: int = 0,
               output_mode: str = "content", context: int = 0) -> SearchResult:
        """Search for content or files."""
        ...


# =============================================================================
# Shell-based Implementation
# =============================================================================

# Image extensions (subset of binary that we can return as base64)
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.ico'}

# Shell-based linters by file extension.  Invoked via _exec() with the
# filesystem path.  Cover languages where a compile/type check needs an
# external toolchain (py_compile, node, tsc, go vet, rustfmt).
LINTERS = {
    '.py': 'python -m py_compile {file} 2>&1',
    '.js': 'node --check {file} 2>&1',
    '.ts': 'npx tsc --noEmit {file} 2>&1',
    '.go': 'go vet {file} 2>&1',
    '.rs': 'rustfmt --check {file} 2>&1',
}


def _lint_json_inproc(content: str) -> tuple[bool, str]:
    """In-process JSON syntax check.  Returns (ok, error_message)."""
    import json as _json
    try:
        _json.loads(content)
        return True, ""
    except _json.JSONDecodeError as e:
        return False, f"JSONDecodeError: {e.msg} (line {e.lineno}, column {e.colno})"
    except Exception as e:  # noqa: BLE001 — any parse failure is a lint failure
        return False, f"{type(e).__name__}: {e}"


def _lint_yaml_inproc(content: str) -> tuple[bool, str]:
    """In-process YAML syntax check.  Returns (ok, error_message).

    Skipped gracefully if PyYAML isn't installed — YAML parsing is optional.
    """
    try:
        import yaml as _yaml
    except ImportError:
        # PyYAML not available — skip silently, caller treats as no linter.
        return True, "__SKIP__"
    try:
        _yaml.safe_load(content)
        return True, ""
    except _yaml.YAMLError as e:
        return False, f"YAMLError: {e}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _lint_toml_inproc(content: str) -> tuple[bool, str]:
    """In-process TOML syntax check (stdlib tomllib, Python 3.11+)."""
    try:
        import tomllib as _toml
    except ImportError:
        # Pre-3.11 fallback via tomli, if installed.
        try:
            import tomli as _toml  # type: ignore[no-redef]
        except ImportError:
            return True, "__SKIP__"
    try:
        _toml.loads(content)
        return True, ""
    except Exception as e:  # tomllib raises TOMLDecodeError, a ValueError subclass
        return False, f"{type(e).__name__}: {e}"


def _lint_python_inproc(content: str) -> tuple[bool, str]:
    """In-process Python syntax check via ast.parse.

    Catches SyntaxError, IndentationError, and everything else the
    ast module rejects — matching py_compile's scope but with no
    subprocess overhead and no dependency on a ``python`` in PATH.
    """
    import ast as _ast
    try:
        _ast.parse(content)
        return True, ""
    except SyntaxError as e:
        loc = f" (line {e.lineno}, column {e.offset})" if e.lineno else ""
        return False, f"{type(e).__name__}: {e.msg}{loc}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


# In-process linters by file extension.  Preferred over shell linters when
# present — no subprocess overhead, microseconds per call.  Each callable
# takes file content (str) and returns (ok: bool, error: str).  An error
# string of ``"__SKIP__"`` signals the linter isn't available (missing
# dependency) and should be treated as "no linter".
LINTERS_INPROC = {
    '.py': _lint_python_inproc,
    '.json': _lint_json_inproc,
    '.yaml': _lint_yaml_inproc,
    '.yml': _lint_yaml_inproc,
    '.toml': _lint_toml_inproc,
}

# Max limits for read operations
MAX_LINES = 2000
MAX_LINE_LENGTH = 2000
MAX_FILE_SIZE = 50 * 1024  # 50KB
DEFAULT_READ_OFFSET = 1
DEFAULT_READ_LIMIT = 500
DEFAULT_SEARCH_OFFSET = 0
DEFAULT_SEARCH_LIMIT = 50


def _coerce_int(value: Any, default: int) -> int:
    """Best-effort integer coercion for tool pagination inputs."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_read_pagination(offset: Any = DEFAULT_READ_OFFSET,
                              limit: Any = DEFAULT_READ_LIMIT) -> tuple[int, int]:
    """Return safe read_file pagination bounds.

    Tool schemas declare minimum/maximum values, but not every caller or
    provider enforces schemas before dispatch. Clamp here so invalid values
    cannot leak into sed ranges like ``0,-1p``.

    The upper bound on ``limit`` comes from ``tool_output.max_lines`` in
    config.yaml (defaults to the module-level ``MAX_LINES`` constant).
    """
    from tools.tool_output_limits import get_max_lines
    max_lines = get_max_lines()
    normalized_offset = max(1, _coerce_int(offset, DEFAULT_READ_OFFSET))
    normalized_limit = _coerce_int(limit, DEFAULT_READ_LIMIT)
    normalized_limit = max(1, min(normalized_limit, max_lines))
    return normalized_offset, normalized_limit


def normalize_search_pagination(offset: Any = DEFAULT_SEARCH_OFFSET,
                                limit: Any = DEFAULT_SEARCH_LIMIT) -> tuple[int, int]:
    """Return safe search pagination bounds for shell head/tail pipelines."""
    normalized_offset = max(0, _coerce_int(offset, DEFAULT_SEARCH_OFFSET))
    normalized_limit = max(1, _coerce_int(limit, DEFAULT_SEARCH_LIMIT))
    return normalized_offset, normalized_limit


class ShellFileOperations(FileOperations):
    """
    File operations implemented via shell commands.
    
    Works with ANY terminal backend that has execute(command, cwd) method.
    This includes local, docker, singularity, ssh, modal, and daytona environments.
    """
    
    def __init__(self, terminal_env, cwd: str = None):
        """
        Initialize file operations with a terminal environment.

        Args:
            terminal_env: Any object with execute(command, cwd) method.
                         Returns {"output": str, "returncode": int}
            cwd: Optional explicit fallback cwd when the terminal env has
                 no cwd attribute (rare — most backends track cwd live).

        Note:
            Every _exec() call prefers the LIVE ``terminal_env.cwd`` over
            ``self.cwd`` so ``cd`` commands run via the terminal tool are
            picked up immediately.  ``self.cwd`` is only used as a fallback
            when the env has no cwd at all — it is NOT the authoritative
            cwd, despite being settable at init time.

            Historical bug (fixed): prior versions of this class used the
            init-time cwd for every _exec() call, which caused relative
            paths passed to patch/read/write to target the wrong directory
            after the user ran ``cd`` in the terminal.  Patches would
            claim success and return a plausible diff but land in the
            original directory, producing apparent silent failures.
        """
        self.env = terminal_env
        # Determine cwd from various possible sources.
        # IMPORTANT: do NOT fall back to os.getcwd() -- that's the HOST's local
        # path which doesn't exist inside container/cloud backends (modal, docker).
        # If nothing provides a cwd, use "/" as a safe universal default.
        self.cwd = cwd or getattr(terminal_env, 'cwd', None) or \
                   getattr(getattr(terminal_env, 'config', None), 'cwd', None) or "/"

        # Cache for command availability checks
        self._command_cache: Dict[str, bool] = {}
    
    def _exec(self, command: str, cwd: str = None, timeout: int = None,
              stdin_data: str = None) -> ExecuteResult:
        """Execute command via terminal backend.

        Args:
            stdin_data: If provided, piped to the process's stdin instead of
                        embedding in the command string. Bypasses ARG_MAX.

        Cwd resolution order (critical — see class docstring):
          1. Explicit ``cwd`` arg (if provided)
          2. Live ``self.env.cwd`` (tracks ``cd`` commands run via terminal)
          3. Init-time ``self.cwd`` (fallback when env has no cwd attribute)

        This ordering ensures relative paths in file operations follow the
        terminal's current directory — not the directory this file_ops was
        originally created in.  See test_file_ops_cwd_tracking.py.
        """
        kwargs = {}
        if timeout:
            kwargs['timeout'] = timeout
        if stdin_data is not None:
            kwargs['stdin_data'] = stdin_data

        # Resolve cwd from the live env so `cd` commands are picked up.
        # Fall through to init-time self.cwd only if the env doesn't track cwd.
        effective_cwd = cwd or getattr(self.env, 'cwd', None) or self.cwd
        result = self.env.execute(command, cwd=effective_cwd, **kwargs)
        return ExecuteResult(
            stdout=result.get("output", ""),
            exit_code=result.get("returncode", 0)
        )
    
    def _has_command(self, cmd: str) -> bool:
        """Check if a command exists in the environment (cached)."""
        if cmd not in self._command_cache:
            result = self._exec(f"command -v {cmd} >/dev/null 2>&1 && echo 'yes'")
            self._command_cache[cmd] = result.stdout.strip() == 'yes'
        return self._command_cache[cmd]
    
    def _is_likely_binary(self, path: str, content_sample: str = None) -> bool:
        """
        Check if a file is likely binary.
        
        Uses extension check (fast) + content analysis (fallback).
        """
        ext = os.path.splitext(path)[1].lower()
        if ext in BINARY_EXTENSIONS:
            return True
        
        # Content analysis: >30% non-printable chars = binary
        if content_sample:
            non_printable = sum(1 for c in content_sample[:1000]
                               if ord(c) < 32 and c not in '\n\r\t')
            return non_printable / min(len(content_sample), 1000) > 0.30
        
        return False
    
    def _is_image(self, path: str) -> bool:
        """Check if file is an image we can return as base64."""
        ext = os.path.splitext(path)[1].lower()
        return ext in IMAGE_EXTENSIONS
    
    def _add_line_numbers(self, content: str, start_line: int = 1) -> str:
        """Add line numbers to content in LINE_NUM|CONTENT format."""
        from tools.tool_output_limits import get_max_line_length
        max_line_length = get_max_line_length()
        lines = content.split('\n')
        numbered = []
        for i, line in enumerate(lines, start=start_line):
            # Truncate long lines
            if len(line) > max_line_length:
                line = line[:max_line_length] + "... [truncated]"
            numbered.append(f"{i:6d}|{line}")
        return '\n'.join(numbered)
    
    def _expand_path(self, path: str) -> str:
        """
        Expand shell-style paths like ~ and ~user to absolute paths.
        
        This must be done BEFORE shell escaping, since ~ doesn't expand
        inside single quotes.
        """
        if not path:
            return path
        
        # Handle ~ and ~user
        if path.startswith('~'):
            # Get home directory via the terminal environment
            result = self._exec("echo $HOME")
            if result.exit_code == 0 and result.stdout.strip():
                home = result.stdout.strip()
                if path == '~':
                    return home
                elif path.startswith('~/'):
                    return home + path[1:]  # Replace ~ with home
                # ~username format - extract and validate username before
                # letting shell expand it (prevent shell injection via
                # paths like "~; rm -rf /").
                rest = path[1:]  # strip leading ~
                slash_idx = rest.find('/')
                username = rest[:slash_idx] if slash_idx >= 0 else rest
                if username and re.fullmatch(r'[a-zA-Z0-9._-]+', username):
                    # Only expand ~username (not the full path) to avoid shell
                    # injection via path suffixes like "~user/$(malicious)".
                    expand_result = self._exec(f"echo ~{username}")
                    if expand_result.exit_code == 0 and expand_result.stdout.strip():
                        user_home = expand_result.stdout.strip()
                        suffix = path[1 + len(username):]  # e.g. "/rest/of/path"
                        return user_home + suffix
        
        return path
    
    def _escape_shell_arg(self, arg: str) -> str:
        """Escape a string for safe use in shell commands."""
        # Use single quotes and escape any single quotes in the string
        return "'" + arg.replace("'", "'\"'\"'") + "'"
    
    def _unified_diff(self, old_content: str, new_content: str, filename: str) -> str:
        """Generate unified diff between old and new content."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff = difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}"
        )
        return ''.join(diff)
    
    # =========================================================================
    # READ Implementation
    # =========================================================================
    
    def read_file(self, path: str, offset: int = 1, limit: int = 500) -> ReadResult:
        """
        Read a file with pagination, binary detection, and line numbers.
        
        Args:
            path: File path (absolute or relative to cwd)
            offset: Line number to start from (1-indexed, default 1)
            limit: Maximum lines to return (default 500, max 2000)
        
        Returns:
            ReadResult with content, metadata, or error info
        """
        # Expand ~ and other shell paths
        path = self._expand_path(path)
        
        offset, limit = normalize_read_pagination(offset, limit)
        
        # Check if file exists and get size (wc -c is POSIX, works on Linux + macOS)
        stat_cmd = f"wc -c < {self._escape_shell_arg(path)} 2>/dev/null"
        stat_result = self._exec(stat_cmd)
        
        if stat_result.exit_code != 0:
            # File not found - try to suggest similar files
            return self._suggest_similar_files(path)
        
        stat_output = _strip_terminal_fence_leaks(stat_result.stdout)
        try:
            file_size = int(stat_output.strip())
        except ValueError:
            file_size = 0
        
        # Check if file is too large
        if file_size > MAX_FILE_SIZE:
            # Still try to read, but warn
            pass
        
        # Images are never inlined — redirect to the vision tool
        if self._is_image(path):
            return ReadResult(
                is_image=True,
                is_binary=True,
                file_size=file_size,
                hint=(
                    "Image file detected. Automatically redirected to vision_analyze tool. "
                    "Use vision_analyze with this file path to inspect the image contents."
                ),
            )
        
        # Read a sample to check for binary content
        sample_cmd = f"head -c 1000 {self._escape_shell_arg(path)} 2>/dev/null"
        sample_result = self._exec(sample_cmd)
        sample_output = _strip_terminal_fence_leaks(sample_result.stdout)
        
        if self._is_likely_binary(path, sample_output):
            return ReadResult(
                is_binary=True,
                file_size=file_size,
                error="Binary file - cannot display as text. Use appropriate tools to handle this file type."
            )
        
        # Read with pagination using sed
        end_line = offset + limit - 1
        read_cmd = f"sed -n '{offset},{end_line}p' {self._escape_shell_arg(path)}"
        read_result = self._exec(read_cmd)
        
        if read_result.exit_code != 0:
            return ReadResult(error=f"Failed to read file: {read_result.stdout}")
        read_output = _strip_terminal_fence_leaks(read_result.stdout)
        
        # Get total line count
        wc_cmd = f"wc -l < {self._escape_shell_arg(path)}"
        wc_result = self._exec(wc_cmd)
        wc_output = _strip_terminal_fence_leaks(wc_result.stdout)
        try:
            total_lines = int(wc_output.strip())
        except ValueError:
            total_lines = 0
        
        # Check if truncated
        truncated = total_lines > end_line
        hint = None
        if truncated:
            hint = f"Use offset={end_line + 1} to continue reading (showing {offset}-{end_line} of {total_lines} lines)"
        
        return ReadResult(
            content=self._add_line_numbers(read_output, offset),
            total_lines=total_lines,
            file_size=file_size,
            truncated=truncated,
            hint=hint
        )
    
    def _suggest_similar_files(self, path: str) -> ReadResult:
        """Suggest similar files when the requested file is not found."""
        dir_path = os.path.dirname(path) or "."
        filename = os.path.basename(path)
        basename_no_ext = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1].lower()
        lower_name = filename.lower()

        # List files in the target directory
        ls_cmd = f"ls -1 {self._escape_shell_arg(dir_path)} 2>/dev/null | head -50"
        ls_result = self._exec(ls_cmd)

        scored: list = []  # (score, filepath) — higher is better
        if ls_result.exit_code == 0 and ls_result.stdout.strip():
            for f in ls_result.stdout.strip().split('\n'):
                if not f:
                    continue
                lf = f.lower()
                score = 0

                # Exact match (shouldn't happen, but guard)
                if lf == lower_name:
                    score = 100
                # Same base name, different extension (e.g. config.yml vs config.yaml)
                elif os.path.splitext(f)[0].lower() == basename_no_ext.lower():
                    score = 90
                # Target is prefix of candidate or vice-versa
                elif lf.startswith(lower_name) or lower_name.startswith(lf):
                    score = 70
                # Substring match (candidate contains query)
                elif lower_name in lf:
                    score = 60
                # Reverse substring (query contains candidate name)
                elif lf in lower_name and len(lf) > 2:
                    score = 40
                # Same extension with some overlap
                elif ext and os.path.splitext(f)[1].lower() == ext:
                    common = set(lower_name) & set(lf)
                    if len(common) >= max(len(lower_name), len(lf)) * 0.4:
                        score = 30

                if score > 0:
                    scored.append((score, os.path.join(dir_path, f)))

        scored.sort(key=lambda x: -x[0])
        similar = [fp for _, fp in scored[:5]]

        return ReadResult(
            error=f"File not found: {path}",
            similar_files=similar
        )
    
    def read_file_raw(self, path: str) -> ReadResult:
        """Read the complete file content as a plain string.

        No pagination, no line-number prefixes, no per-line truncation.
        Uses cat so the full file is returned regardless of size.
        """
        path = self._expand_path(path)
        stat_cmd = f"wc -c < {self._escape_shell_arg(path)} 2>/dev/null"
        stat_result = self._exec(stat_cmd)
        if stat_result.exit_code != 0:
            return self._suggest_similar_files(path)
        stat_output = _strip_terminal_fence_leaks(stat_result.stdout)
        try:
            file_size = int(stat_output.strip())
        except ValueError:
            file_size = 0
        if self._is_image(path):
            return ReadResult(is_image=True, is_binary=True, file_size=file_size)
        sample_result = self._exec(f"head -c 1000 {self._escape_shell_arg(path)} 2>/dev/null")
        sample_output = _strip_terminal_fence_leaks(sample_result.stdout)
        if self._is_likely_binary(path, sample_output):
            return ReadResult(
                is_binary=True, file_size=file_size,
                error="Binary file — cannot display as text."
            )
        cat_result = self._exec(f"cat {self._escape_shell_arg(path)}")
        if cat_result.exit_code != 0:
            return ReadResult(error=f"Failed to read file: {cat_result.stdout}")
        return ReadResult(
            content=_strip_terminal_fence_leaks(cat_result.stdout),
            file_size=file_size,
        )

    def delete_file(self, path: str) -> WriteResult:
        """Delete a file via rm."""
        path = self._expand_path(path)
        if _is_write_denied(path):
            return WriteResult(error=f"Delete denied: {path} is a protected path")
        result = self._exec(f"rm -f {self._escape_shell_arg(path)}")
        if result.exit_code != 0:
            return WriteResult(error=f"Failed to delete {path}: {result.stdout}")
        return WriteResult()

    def move_file(self, src: str, dst: str) -> WriteResult:
        """Move a file via mv."""
        src = self._expand_path(src)
        dst = self._expand_path(dst)
        for p in (src, dst):
            if _is_write_denied(p):
                return WriteResult(error=f"Move denied: {p} is a protected path")
        result = self._exec(
            f"mv {self._escape_shell_arg(src)} {self._escape_shell_arg(dst)}"
        )
        if result.exit_code != 0:
            return WriteResult(error=f"Failed to move {src} -> {dst}: {result.stdout}")
        return WriteResult()

    # =========================================================================
    # WRITE Implementation
    # =========================================================================

    def write_file(self, path: str, content: str) -> WriteResult:
        """
        Write content to a file, creating parent directories as needed.

        Pipes content through stdin to avoid OS ARG_MAX limits on large
        files. The content never appears in the shell command string —
        only the file path does.

        After the write, runs a post-first / pre-lazy lint check via
        ``_check_lint_delta()``.  If the new content is clean, the lint
        call is O(one parse).  If the new content has errors, the pre-write
        content is linted too and only errors newly introduced by this
        write are surfaced — pre-existing problems are filtered out so
        the agent isn't distracted chasing them.

        Args:
            path: File path to write
            content: Content to write

        Returns:
            WriteResult with bytes written, lint summary, or error.
        """
        # Expand ~ and other shell paths
        path = self._expand_path(path)

        # Block writes to sensitive paths
        if _is_write_denied(path):
            return WriteResult(error=f"Write denied: '{path}' is a protected system/credential file.")

        # Capture pre-write content for lint-delta computation.  Only do this
        # when an in-process OR shell linter exists for this extension — no
        # point paying for the read otherwise.  For in-process linters we
        # pass the content directly; for shell linters the pre-state isn't
        # useful (we'd have to re-write-read to lint the old version, which
        # defeats the purpose), so we skip the capture and accept the naive
        # "all errors" report.
        ext = os.path.splitext(path)[1].lower()
        pre_content: Optional[str] = None
        if ext in LINTERS_INPROC:
            # Best-effort read; failure (file missing, permission) leaves
            # pre_content as None which makes the delta step degrade
            # gracefully to "report all errors".
            read_cmd = f"cat {self._escape_shell_arg(path)} 2>/dev/null"
            read_result = self._exec(read_cmd)
            if read_result.exit_code == 0 and read_result.stdout:
                pre_content = read_result.stdout

        # Snapshot LSP diagnostics for this file (best-effort) so the
        # post-write LSP layer can return only diagnostics introduced
        # by this specific edit.  Mirrors claude-code's
        # ``beforeFileEdited`` pattern but wired to the local LSP
        # rather than an external IDE.
        self._snapshot_lsp_baseline(path)

        # Create parent directories
        parent = os.path.dirname(path)
        dirs_created = False

        if parent:
            mkdir_cmd = f"mkdir -p {self._escape_shell_arg(parent)}"
            mkdir_result = self._exec(mkdir_cmd)
            if mkdir_result.exit_code == 0:
                dirs_created = True

        # Write via stdin pipe — content bypasses shell arg parsing entirely,
        # so there's no ARG_MAX limit regardless of file size.
        write_cmd = f"cat > {self._escape_shell_arg(path)}"
        write_result = self._exec(write_cmd, stdin_data=content)

        if write_result.exit_code != 0:
            return WriteResult(error=f"Failed to write file: {write_result.stdout}")

        # Get bytes written (wc -c is POSIX, works on Linux + macOS)
        stat_cmd = f"wc -c < {self._escape_shell_arg(path)} 2>/dev/null"
        stat_result = self._exec(stat_cmd)

        try:
            bytes_written = int(stat_result.stdout.strip())
        except ValueError:
            bytes_written = len(content.encode('utf-8'))

        # Post-write lint with delta refinement.
        lint_result = self._check_lint_delta(path, pre_content=pre_content, post_content=content)

        # Semantic diagnostics from the LSP layer — separate channel.
        # Only fired when the syntax tier reported clean (no point asking
        # an LSP for a file that won't even parse).  Best-effort:
        # ``""`` is returned for any failure path.
        lsp_diagnostics: Optional[str] = None
        if lint_result.success or lint_result.skipped:
            block = self._maybe_lsp_diagnostics(path)
            if block:
                lsp_diagnostics = block

        return WriteResult(
            bytes_written=bytes_written,
            dirs_created=dirs_created,
            lint=lint_result.to_dict() if lint_result else None,
            lsp_diagnostics=lsp_diagnostics,
        )
    
    # =========================================================================
    # PATCH Implementation (Replace Mode)
    # =========================================================================
    
    def patch_replace(self, path: str, old_string: str, new_string: str,
                      replace_all: bool = False) -> PatchResult:
        """
        Replace text in a file using fuzzy matching.

        Args:
            path: File path to modify
            old_string: Text to find (must be unique unless replace_all=True)
            new_string: Replacement text
            replace_all: If True, replace all occurrences

        Returns:
            PatchResult with diff and lint results
        """
        # Expand ~ and other shell paths
        path = self._expand_path(path)

        # Block writes to sensitive paths
        if _is_write_denied(path):
            return PatchResult(error=f"Write denied: '{path}' is a protected system/credential file.")

        # Read current content
        read_cmd = f"cat {self._escape_shell_arg(path)} 2>/dev/null"
        read_result = self._exec(read_cmd)
        
        if read_result.exit_code != 0:
            return PatchResult(error=f"Failed to read file: {path}")
        
        content = read_result.stdout
        
        # Import and use fuzzy matching
        from tools.fuzzy_match import fuzzy_find_and_replace
        
        new_content, match_count, _strategy, error = fuzzy_find_and_replace(
            content, old_string, new_string, replace_all
        )
        
        if error or match_count == 0:
            err_msg = error or f"Could not find match for old_string in {path}"
            try:
                from tools.fuzzy_match import format_no_match_hint
                err_msg += format_no_match_hint(err_msg, match_count, old_string, content)
            except Exception:
                pass
            return PatchResult(error=err_msg)
        # Write back
        write_result = self.write_file(path, new_content)
        if write_result.error:
            return PatchResult(error=f"Failed to write changes: {write_result.error}")

        # Post-write verification — re-read the file and confirm the bytes we
        # intended to write actually landed. Catches silent persistence
        # failures (backend FS oddities, race with another task, truncated
        # pipe, etc.) that would otherwise return success-with-diff while the
        # file is unchanged on disk.
        verify_cmd = f"cat {self._escape_shell_arg(path)} 2>/dev/null"
        verify_result = self._exec(verify_cmd)
        if verify_result.exit_code != 0:
            return PatchResult(error=f"Post-write verification failed: could not re-read {path}")
        # Normalize line endings before comparing.  On Windows, Python's
        # default text-mode ``open()`` translates ``\n`` → ``\r\n`` on
        # write, so the file on disk legitimately holds CRLFs while our
        # ``new_content`` string has bare LFs.  Without this normalization
        # every patch on Windows returns a bogus "wrote 39, read 42"
        # false-negative even though the edit landed correctly.  POSIX
        # backends don't translate, so this is a no-op there.
        _verify_stdout_normalized = verify_result.stdout.replace("\r\n", "\n").replace("\r", "\n")
        _new_content_normalized = new_content.replace("\r\n", "\n").replace("\r", "\n")
        if _verify_stdout_normalized != _new_content_normalized:
            return PatchResult(error=(
                f"Post-write verification failed for {path}: on-disk content "
                f"differs from intended write "
                f"(wrote {len(_new_content_normalized)} chars, read back "
                f"{len(_verify_stdout_normalized)} chars after normalizing line endings). "
                "The patch did not persist. Re-read the file and try again."
            ))

        # Generate diff
        diff = self._unified_diff(content, new_content, path)

        # Auto-lint with delta refinement: only surface errors introduced
        # by this patch, filtering out pre-existing lint failures so the
        # agent isn't distracted by problems that were already there.
        lint_result = self._check_lint_delta(path, pre_content=content, post_content=new_content)

        return PatchResult(
            success=True,
            diff=diff,
            files_modified=[path],
            lint=lint_result.to_dict() if lint_result else None,
            # Propagate the LSP diagnostics already captured by the
            # internal ``write_file`` call.  Its baseline was the
            # pre-patch content (taken at the start of write_file via
            # ``_snapshot_lsp_baseline``) so the delta is correct for
            # the patch as a whole.  Keep the field separate from the
            # syntax-check ``lint`` so the agent can read both signals.
            lsp_diagnostics=write_result.lsp_diagnostics,
        )
    
    def patch_v4a(self, patch_content: str) -> PatchResult:
        """
        Apply a V4A format patch.
        
        V4A format:
            *** Begin Patch
            *** Update File: path/to/file.py
            @@ context hint @@
             context line
            -removed line
            +added line
            *** End Patch
        
        Args:
            patch_content: V4A format patch string
        
        Returns:
            PatchResult with changes made
        """
        # Import patch parser
        from tools.patch_parser import parse_v4a_patch, apply_v4a_operations
        
        operations, parse_error = parse_v4a_patch(patch_content)
        if parse_error:
            return PatchResult(error=f"Failed to parse patch: {parse_error}")
        
        # Apply operations
        result = apply_v4a_operations(operations, self)
        return result
    
    def _check_lint(self, path: str, content: Optional[str] = None) -> LintResult:
        """
        Run syntax check on a file after editing.

        Prefers the in-process linter for structured formats (JSON, YAML,
        TOML) when possible — those parse via the Python stdlib in
        microseconds and don't require a subprocess.  Falls back to the
        shell linter table for compiled/type-checked languages
        (py_compile, node --check, tsc, go vet, rustfmt).

        Args:
            path: File path (used to select the linter + for shell invocation).
            content: Optional file content.  If provided AND an in-process
                     linter matches the extension, we lint the content
                     directly without re-reading the file from disk.  Ignored
                     for shell linters.

        Returns:
            LintResult with status and any errors.
        """
        ext = os.path.splitext(path)[1].lower()

        # Prefer in-process linter when available.
        inproc = LINTERS_INPROC.get(ext)
        if inproc is not None:
            # Need content — either passed in or read from disk.
            if content is None:
                read_cmd = f"cat {self._escape_shell_arg(path)} 2>/dev/null"
                read_result = self._exec(read_cmd)
                if read_result.exit_code != 0:
                    return LintResult(skipped=True, message=f"Failed to read {path} for lint")
                content = read_result.stdout
            ok, err = inproc(content)
            if err == "__SKIP__":
                return LintResult(skipped=True, message=f"No linter available for {ext} (missing dependency)")
            return LintResult(success=ok, output="" if ok else err)

        # Fall back to shell linter.
        if ext not in LINTERS:
            return LintResult(skipped=True, message=f"No linter for {ext} files")

        linter_cmd = LINTERS[ext]
        # Extract the base command (first word)
        base_cmd = linter_cmd.split()[0]

        if not self._has_command(base_cmd):
            return LintResult(skipped=True, message=f"{base_cmd} not available")

        # Run linter
        cmd = linter_cmd.replace("{file}", self._escape_shell_arg(path))
        result = self._exec(cmd, timeout=30)

        return LintResult(
            success=result.exit_code == 0,
            output=result.stdout.strip() if result.stdout.strip() else ""
        )

    def _check_lint_delta(self, path: str, pre_content: Optional[str],
                          post_content: Optional[str] = None) -> LintResult:
        """
        Run post-write syntax lint with pre-write baseline comparison.

        Two-tier strategy:

        1. **Syntax check** (in-process or shell-based, microseconds).
           Catches the bug class that motivated this layer: corrupt
           writes, mashed quotes, truncated output.  Hot path.

        2. **Delta refinement against pre-write content** when the
           syntax tier reports errors.  Filter out errors that already
           existed pre-edit so the agent isn't distracted by inherited
           state.

        Semantic diagnostics from the LSP layer are fetched separately
        via :meth:`_maybe_lsp_diagnostics` and surfaced in the
        ``lsp_diagnostics`` field on :class:`WriteResult` /
        :class:`PatchResult`.  Keeping the two channels separate lets
        the agent (and any downstream parsers) read syntax errors and
        semantic errors as independent signals.

        Args:
            path: File path (for linter selection).
            pre_content: File content BEFORE the write.  Pass None for new
                         files or when the pre-state isn't available — the
                         delta refinement is skipped and all post errors
                         are returned.
            post_content: File content AFTER the write.  Optional; if None,
                          the shell linter reads from disk (same as
                          _check_lint).

        Returns:
            LintResult.  ``output`` contains either the full post-lint
            errors (no pre-state) or just the new-error lines (delta
            refinement applied).
        """
        post = self._check_lint(path, content=post_content)

        # Hot path: clean post-write syntactically.
        if post.success or post.skipped:
            return post

        # Post-write has syntax errors.  If we have pre-content, run the
        # delta refinement to filter out pre-existing errors.
        if pre_content is None:
            return post

        pre = self._check_lint(path, content=pre_content)
        if pre.success or pre.skipped or not pre.output:
            # Pre-write was clean (or we couldn't lint it) — post errors
            # are all new.  Return the full post output.
            return post

        # Both pre- and post-write had errors.  Compute the set-difference
        # on non-empty stripped lines.  Caveat: single-error parsers
        # (ast.parse, json.loads) stop at the first error and don't report
        # later ones — if the pre-existing error blocks parsing before
        # reaching the edit region, we can't prove the edit is clean.  So
        # if every post error also appeared pre-edit, we report the file
        # as still broken but annotate that this edit introduced nothing
        # new on top — the agent knows it's inherited state, not fresh
        # damage, without silently dropping the error.
        pre_lines = {ln.strip() for ln in pre.output.splitlines() if ln.strip()}
        post_lines = [ln for ln in post.output.splitlines() if ln.strip() and ln.strip() not in pre_lines]

        if not post_lines:
            # Every error in post was also in pre — this edit didn't make
            # anything obviously worse, but the file remains broken and
            # the agent should know.
            return LintResult(
                success=False,
                output=post.output,
                message="Pre-existing lint errors — this edit didn't introduce new ones but the file is still broken.",
            )

        return LintResult(
            success=False,
            output=(
                "New lint errors introduced by this edit "
                "(pre-existing errors filtered out):\n" + "\n".join(post_lines)
            )
        )

    def _lsp_local_only(self) -> bool:
        """Return True iff this FileOperations is wired to a local backend.

        LSP servers run on the host process — they need access to the
        files they're linting.  Remote/sandboxed backends (Docker,
        Modal, SSH, Daytona) keep files inside the sandbox where the
        host-side LSP server can't reach them, so we skip the LSP
        path for those entirely.
        """
        env = getattr(self, "env", None)
        if env is None:
            # Defensive: some tests construct ShellFileOperations via
            # ``__new__`` without going through ``__init__``, so
            # ``self.env`` may be missing.  No env = no LSP path.
            return False
        try:
            from tools.environments.local import LocalEnvironment
        except Exception:  # noqa: BLE001
            return False
        return isinstance(env, LocalEnvironment)

    def _snapshot_lsp_baseline(self, path: str) -> None:
        """Capture pre-edit LSP diagnostics so the post-write delta is correct.

        Best-effort.  Silent on every failure path — LSP is an
        enrichment layer and must never break a write.

        Skipped entirely on non-local backends (Docker, Modal, SSH,
        etc.) — the server can't see files inside the sandbox.
        """
        if not self._lsp_local_only():
            return
        try:
            from agent.lsp import get_service
            svc = get_service()
        except Exception:  # noqa: BLE001
            return
        if svc is None:
            return
        try:
            svc.snapshot_baseline(path)
        except Exception:  # noqa: BLE001
            pass

    def _maybe_lsp_diagnostics(self, path: str) -> str:
        """Best-effort LSP semantic diagnostics for ``path``.

        Returns a formatted ``<diagnostics>`` block, or empty string
        when LSP is unavailable / disabled / produced no errors.

        Wraps everything in a try/except so a misbehaving LSP server
        can't break a write.  This intentionally swallows all errors
        — the calling tier already returned a clean syntax result, so
        ``""`` here just means "no extra info to add".

        Skipped entirely on non-local backends (Docker, Modal, SSH,
        etc.) — same reasoning as ``_snapshot_lsp_baseline``.
        """
        if not self._lsp_local_only():
            return ""
        try:
            from agent.lsp import get_service
        except Exception:  # noqa: BLE001
            return ""
        try:
            svc = get_service()
        except Exception:  # noqa: BLE001
            return ""
        if svc is None or not svc.enabled_for(path):
            return ""
        try:
            diagnostics = svc.get_diagnostics_sync(path, delta=True)
        except Exception:  # noqa: BLE001
            return ""
        if not diagnostics:
            return ""
        try:
            from agent.lsp.reporter import report_for_file, truncate
            block = report_for_file(path, diagnostics)
            if not block:
                return ""
            return truncate("LSP diagnostics introduced by this edit:\n" + block)
        except Exception:  # noqa: BLE001
            return ""
    
    # =========================================================================
    # SEARCH Implementation
    # =========================================================================
    
    def search(self, pattern: str, path: str = ".", target: str = "content",
               file_glob: Optional[str] = None, limit: int = 50, offset: int = 0,
               output_mode: str = "content", context: int = 0) -> SearchResult:
        """
        Search for content or files.
        
        Args:
            pattern: Regex (for content) or glob pattern (for files)
            path: Directory/file to search (default: cwd)
            target: "content" (grep) or "files" (glob)
            file_glob: File pattern filter for content search (e.g., "*.py")
            limit: Max results (default 50)
            offset: Skip first N results
            output_mode: "content", "files_only", or "count"
            context: Lines of context around matches
        
        Returns:
            SearchResult with matches or file list
        """
        offset, limit = normalize_search_pagination(offset, limit)

        # Expand ~ and other shell paths
        path = self._expand_path(path)
        
        # Validate that the path exists before searching
        check = self._exec(f"test -e {self._escape_shell_arg(path)} && echo exists || echo not_found")
        if "not_found" in check.stdout:
            # Try to suggest nearby paths
            parent = os.path.dirname(path) or "."
            basename_query = os.path.basename(path)
            hint_parts = [f"Path not found: {path}"]
            # Check if parent directory exists and list similar entries
            parent_check = self._exec(
                f"test -d {self._escape_shell_arg(parent)} && echo yes || echo no"
            )
            if "yes" in parent_check.stdout and basename_query:
                ls_result = self._exec(
                    f"ls -1 {self._escape_shell_arg(parent)} 2>/dev/null | head -20"
                )
                if ls_result.exit_code == 0 and ls_result.stdout.strip():
                    lower_q = basename_query.lower()
                    candidates = []
                    for entry in ls_result.stdout.strip().split('\n'):
                        if not entry:
                            continue
                        le = entry.lower()
                        if lower_q in le or le in lower_q or le.startswith(lower_q[:3]):
                            candidates.append(os.path.join(parent, entry))
                    if candidates:
                        hint_parts.append(
                            "Similar paths: " + ", ".join(candidates[:5])
                        )
            return SearchResult(
                error=". ".join(hint_parts),
                total_count=0
            )
        
        if target == "files":
            return self._search_files(pattern, path, limit, offset)
        else:
            return self._search_content(pattern, path, file_glob, limit, offset, 
                                        output_mode, context)
    
    def _search_files(self, pattern: str, path: str, limit: int, offset: int) -> SearchResult:
        """Search for files by name pattern (glob-like)."""
        # Auto-prepend **/ for recursive search if not already present
        if not pattern.startswith('**/') and '/' not in pattern:
            search_pattern = pattern
        else:
            search_pattern = pattern.split('/')[-1]

        search_root = Path(path)
        has_hidden_path_ancestor = any(
            part not in {".", ".."} and part.startswith(".")
            for part in search_root.parts
        )

        # Prefer ripgrep: respects .gitignore, excludes hidden dirs by
        # default, and has parallel directory traversal (~200x faster than
        # find on wide trees).  Mirrors _search_content which already uses rg.
        if self._has_command('rg'):
            return self._search_files_rg(search_pattern, path, limit, offset)

        # Fallback: find (slower, no .gitignore awareness)
        if not self._has_command('find'):
            return SearchResult(
                error="File search requires 'rg' (ripgrep) or 'find'. "
                      "Install ripgrep for best results: "
                      "https://github.com/BurntSushi/ripgrep#installation"
            )

        # Exclude hidden directories (matching ripgrep's default behavior).
        hidden_exclude = "-not -path '*/.*'" if not has_hidden_path_ancestor else ""
        hidden_filter_expr = f" {hidden_exclude}" if hidden_exclude else ""

        # Use shell pagination for standard roots. For hidden roots, gather full
        # output so we can re-apply hidden-descendant filtering while allowing
        # explicit hidden-root searches.
        pagination_expr = ""
        if not has_hidden_path_ancestor:
            pagination_expr = f" | tail -n +{offset + 1} | head -n {limit}"

        cmd = f"find {self._escape_shell_arg(path)}{hidden_filter_expr} -type f -name {self._escape_shell_arg(search_pattern)} " \
              f"-printf '%T@ %p\\n' 2>/dev/null | sort -rn{pagination_expr}"

        result = self._exec(cmd, timeout=60)

        if not result.stdout.strip():
            # Try without -printf (BSD find compatibility -- macOS)
            cmd_simple = f"find {self._escape_shell_arg(path)}{hidden_filter_expr} -type f -name {self._escape_shell_arg(search_pattern)} " \
                        f"2>/dev/null | sort -rn{pagination_expr}"
            result = self._exec(cmd_simple, timeout=60)

        files = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2 and parts[0].replace('.', '').isdigit():
                files.append(parts[1])
            else:
                files.append(line)

        # For explicit hidden roots, find's path-based filtering excludes every
        # file under the hidden path. Apply descendant filtering after command
        # execution so only the explicit root ancestry is bypassed.
        if has_hidden_path_ancestor:
            normalized_root = search_root.resolve()
            filtered_files = []
            for file_path in files:
                try:
                    rel_parts = Path(file_path).resolve().relative_to(normalized_root).parts
                except ValueError:
                    rel_parts = Path(file_path).parts
                if any(part not in {".", ".."} and part.startswith(".") for part in rel_parts):
                    continue
                filtered_files.append(file_path)
            files = filtered_files[offset:offset + limit]
        # pagination for standard roots is already applied in shell

        return SearchResult(
            files=files,
            total_count=len(files)
        )

    def _search_files_rg(self, pattern: str, path: str, limit: int, offset: int) -> SearchResult:
        """Search for files by name using ripgrep's --files mode.

        rg --files respects .gitignore and excludes hidden directories by
        default, and uses parallel directory traversal for ~200x speedup
        over find on wide trees.  Results are sorted by modification time
        (most recently edited first) when rg >= 13.0 supports --sortr.
        """
        # rg --files -g uses glob patterns; wrap bare names so they match
        # at any depth (equivalent to find -name).
        if '/' not in pattern and not pattern.startswith('*'):
            glob_pattern = f"*{pattern}"
        else:
            glob_pattern = pattern

        fetch_limit = limit + offset
        # Try mtime-sorted first (rg 13+); fall back to unsorted if not supported.
        cmd_sorted = (
            f"rg --files --sortr=modified -g {self._escape_shell_arg(glob_pattern)} "
            f"{self._escape_shell_arg(path)} 2>/dev/null "
            f"| head -n {fetch_limit}"
        )
        result = self._exec(cmd_sorted, timeout=60)
        all_files = [f for f in result.stdout.strip().split('\n') if f]

        if not all_files:
            # --sortr may have failed on older rg; retry without it.
            cmd_plain = (
                f"rg --files -g {self._escape_shell_arg(glob_pattern)} "
                f"{self._escape_shell_arg(path)} 2>/dev/null "
                f"| head -n {fetch_limit}"
            )
            result = self._exec(cmd_plain, timeout=60)
            all_files = [f for f in result.stdout.strip().split('\n') if f]

        page = all_files[offset:offset + limit]

        return SearchResult(
            files=page,
            total_count=len(all_files),
            truncated=len(all_files) >= fetch_limit,
        )
    
    def _search_content(self, pattern: str, path: str, file_glob: Optional[str],
                        limit: int, offset: int, output_mode: str, context: int) -> SearchResult:
        """Search for content inside files (grep-like)."""
        # Try ripgrep first (fast), fallback to grep (slower but works)
        if self._has_command('rg'):
            return self._search_with_rg(pattern, path, file_glob, limit, offset, 
                                        output_mode, context)
        elif self._has_command('grep'):
            return self._search_with_grep(pattern, path, file_glob, limit, offset,
                                          output_mode, context)
        else:
            # Neither rg nor grep available (Windows without Git Bash, etc.)
            return SearchResult(
                error="Content search requires ripgrep (rg) or grep. "
                      "Install ripgrep: https://github.com/BurntSushi/ripgrep#installation"
            )
    
    def _search_with_rg(self, pattern: str, path: str, file_glob: Optional[str],
                        limit: int, offset: int, output_mode: str, context: int) -> SearchResult:
        """Search using ripgrep."""
        cmd_parts = ["rg", "--line-number", "--no-heading", "--with-filename"]
        
        # Add context if requested
        if context > 0:
            cmd_parts.extend(["-C", str(context)])
        
        # Add file glob filter (must be quoted to prevent shell expansion)
        if file_glob:
            cmd_parts.extend(["--glob", self._escape_shell_arg(file_glob)])
        
        # Output mode handling
        if output_mode == "files_only":
            cmd_parts.append("-l")  # Files only
        elif output_mode == "count":
            cmd_parts.append("-c")  # Count per file
        
        # Add pattern and path
        cmd_parts.append(self._escape_shell_arg(pattern))
        cmd_parts.append(self._escape_shell_arg(path))
        
        # Fetch extra rows so we can report the true total before slicing.
        # For context mode, rg emits separator lines ("--") between groups,
        # so we grab generously and filter in Python.
        fetch_limit = limit + offset + 200 if context > 0 else limit + offset
        cmd_parts.extend(["|", "head", "-n", str(fetch_limit)])
        
        cmd = " ".join(cmd_parts)
        result = self._exec(cmd, timeout=60)
        
        # rg exit codes: 0=matches found, 1=no matches, 2=error
        if result.exit_code == 2 and not result.stdout.strip():
            error_msg = result.stderr.strip() if hasattr(result, 'stderr') and result.stderr else "Search error"
            return SearchResult(error=f"Search failed: {error_msg}", total_count=0)
        
        # Parse results based on output mode
        if output_mode == "files_only":
            all_files = [f for f in result.stdout.strip().split('\n') if f]
            total = len(all_files)
            page = all_files[offset:offset + limit]
            return SearchResult(files=page, total_count=total)
        
        elif output_mode == "count":
            counts = {}
            for line in result.stdout.strip().split('\n'):
                if ':' in line:
                    parts = line.rsplit(':', 1)
                    if len(parts) == 2:
                        try:
                            counts[parts[0]] = int(parts[1])
                        except ValueError:
                            pass
            return SearchResult(counts=counts, total_count=sum(counts.values()))
        
        else:
            # Parse content matches and context lines.
            # rg match lines:   "file:lineno:content"  (colon separator)
            # rg context lines: "file-lineno-content"   (dash separator)
            # rg group seps:    "--"
            # Note: on Windows, paths contain drive letters (e.g. C:\path),
            # so naive split(":") breaks. Use regex to handle both platforms.
            _match_re = re.compile(r'^([A-Za-z]:)?(.*?):(\d+):(.*)$')
            matches = []
            for line in result.stdout.strip().split('\n'):
                if not line or line == "--":
                    continue
                
                # Try match line first (colon-separated: file:line:content)
                m = _match_re.match(line)
                if m:
                    matches.append(SearchMatch(
                        path=(m.group(1) or '') + m.group(2),
                        line_number=int(m.group(3)),
                        content=m.group(4)[:500]
                    ))
                    continue
                
                # Try context line (dash-separated: file-line-content)
                # Only attempt if context was requested to avoid false positives
                if context > 0:
                    parsed = _parse_search_context_line(line)
                    if parsed:
                        matches.append(SearchMatch(
                            path=parsed[0],
                            line_number=parsed[1],
                            content=parsed[2][:500]
                        ))
            
            total = len(matches)
            page = matches[offset:offset + limit]
            return SearchResult(
                matches=page,
                total_count=total,
                truncated=total > offset + limit
            )
    
    def _search_with_grep(self, pattern: str, path: str, file_glob: Optional[str],
                          limit: int, offset: int, output_mode: str, context: int) -> SearchResult:
        """Fallback search using grep."""
        cmd_parts = ["grep", "-rnH"]  # -H forces filename even for single-file searches
        
        # Exclude hidden directories (matching ripgrep's default behavior).
        # This prevents searching inside .hub/index-cache/, .git/, etc.
        cmd_parts.append("--exclude-dir='.*'")
        
        # Add context if requested
        if context > 0:
            cmd_parts.extend(["-C", str(context)])
        
        # Add file pattern filter (must be quoted to prevent shell expansion)
        if file_glob:
            cmd_parts.extend(["--include", self._escape_shell_arg(file_glob)])
        
        # Output mode handling
        if output_mode == "files_only":
            cmd_parts.append("-l")
        elif output_mode == "count":
            cmd_parts.append("-c")
        
        # Add pattern and path
        cmd_parts.append(self._escape_shell_arg(pattern))
        cmd_parts.append(self._escape_shell_arg(path))
        
        # Fetch generously so we can compute total before slicing
        fetch_limit = limit + offset + (200 if context > 0 else 0)
        cmd_parts.extend(["|", "head", "-n", str(fetch_limit)])
        
        cmd = " ".join(cmd_parts)
        result = self._exec(cmd, timeout=60)
        
        # grep exit codes: 0=matches found, 1=no matches, 2=error
        if result.exit_code == 2 and not result.stdout.strip():
            error_msg = result.stderr.strip() if hasattr(result, 'stderr') and result.stderr else "Search error"
            return SearchResult(error=f"Search failed: {error_msg}", total_count=0)
        
        if output_mode == "files_only":
            all_files = [f for f in result.stdout.strip().split('\n') if f]
            total = len(all_files)
            page = all_files[offset:offset + limit]
            return SearchResult(files=page, total_count=total)
        
        elif output_mode == "count":
            counts = {}
            for line in result.stdout.strip().split('\n'):
                if ':' in line:
                    parts = line.rsplit(':', 1)
                    if len(parts) == 2:
                        try:
                            counts[parts[0]] = int(parts[1])
                        except ValueError:
                            pass
            return SearchResult(counts=counts, total_count=sum(counts.values()))
        
        else:
            # grep match lines:   "file:lineno:content" (colon)
            # grep context lines: "file-lineno-content"  (dash)
            # grep group seps:    "--"
            # Note: on Windows, paths contain drive letters (e.g. C:\path),
            # so naive split(":") breaks. Use regex to handle both platforms.
            _match_re = re.compile(r'^([A-Za-z]:)?(.*?):(\d+):(.*)$')
            matches = []
            for line in result.stdout.strip().split('\n'):
                if not line or line == "--":
                    continue
                
                m = _match_re.match(line)
                if m:
                    matches.append(SearchMatch(
                        path=(m.group(1) or '') + m.group(2),
                        line_number=int(m.group(3)),
                        content=m.group(4)[:500]
                    ))
                    continue
                
                if context > 0:
                    parsed = _parse_search_context_line(line)
                    if parsed:
                        matches.append(SearchMatch(
                            path=parsed[0],
                            line_number=parsed[1],
                            content=parsed[2][:500]
                        ))

            
            total = len(matches)
            page = matches[offset:offset + limit]
            return SearchResult(
                matches=page,
                total_count=total,
                truncated=total > offset + limit
            )
