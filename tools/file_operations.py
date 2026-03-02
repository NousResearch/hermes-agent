#!/usr/bin/env python3
"""
File Operations Module

Provides file manipulation capabilities (read, write, patch, search) that work
across all terminal backends (local, docker, singularity, ssh, modal).

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
import json
import difflib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path


# ---------------------------------------------------------------------------
# Write-path deny list — blocks writes to sensitive system/credential files
# ---------------------------------------------------------------------------

_HOME = str(Path.home())

WRITE_DENIED_PATHS = {
    os.path.realpath(p) for p in [
        os.path.join(_HOME, ".ssh", "authorized_keys"),
        os.path.join(_HOME, ".ssh", "id_rsa"),
        os.path.join(_HOME, ".ssh", "id_ed25519"),
        os.path.join(_HOME, ".ssh", "config"),
        os.path.join(_HOME, ".hermes", ".env"),
        os.path.join(_HOME, ".bashrc"),
        os.path.join(_HOME, ".zshrc"),
        os.path.join(_HOME, ".profile"),
        os.path.join(_HOME, ".bash_profile"),
        os.path.join(_HOME, ".zprofile"),
        os.path.join(_HOME, ".netrc"),
        os.path.join(_HOME, ".pgpass"),
        os.path.join(_HOME, ".npmrc"),
        os.path.join(_HOME, ".pypirc"),
        "/etc/sudoers",
        "/etc/passwd",
        "/etc/shadow",
    ]
}

WRITE_DENIED_PREFIXES = [
    os.path.realpath(p) + os.sep for p in [
        os.path.join(_HOME, ".ssh"),
        os.path.join(_HOME, ".aws"),
        os.path.join(_HOME, ".gnupg"),
        os.path.join(_HOME, ".kube"),
        "/etc/sudoers.d",
    "/etc/systemd",
    ]
]

READ_DENIED_BASENAMES = {".env"}


def _is_write_denied(path: str) -> bool:
    """Return True if path is on the write deny list."""
    resolved = os.path.realpath(os.path.expanduser(path))
    if resolved in WRITE_DENIED_PATHS:
        return True
    for prefix in WRITE_DENIED_PREFIXES:
        if resolved.startswith(prefix):
            return True
    return False


def _is_env_file_path(path: str) -> bool:
    """Return True if path resolves to a file named '.env' (any directory)."""
    try:
        return os.path.basename(os.path.realpath(os.path.expanduser(path))).lower() == ".env"
    except Exception:
        return os.path.basename(path).lower() == ".env"


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
        return {
            "status": "ok" if self.success else "error",
            "output": self.output
        }


@dataclass
class ExecuteResult:
    """Result from executing a shell command."""
    stdout: str = ""
    exit_code: int = 0


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
    def search(self, pattern: str, path: str = ".", target: str = "content",
               file_glob: Optional[str] = None, limit: int = 50, offset: int = 0,
               output_mode: str = "content", context: int = 0) -> SearchResult:
        """Search for content or files."""
        ...


# =============================================================================
# Shell-based Implementation
# =============================================================================

# Binary file extensions (fast path check)
BINARY_EXTENSIONS = {
    # Images
    '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.ico', '.tiff', '.tif',
    '.svg',  # SVG is text but often treated as binary
    # Audio/Video
    '.mp3', '.mp4', '.wav', '.avi', '.mov', '.mkv', '.flac', '.ogg', '.webm',
    # Archives
    '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar',
    # Documents
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    # Compiled/Binary
    '.exe', '.dll', '.so', '.dylib', '.o', '.a', '.pyc', '.pyo', '.class',
    '.wasm', '.bin',
    # Fonts
    '.ttf', '.otf', '.woff', '.woff2', '.eot',
    # Other
    '.db', '.sqlite', '.sqlite3',
}

# Image extensions (subset of binary that we can return as base64)
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.ico'}

# Linters by file extension
LINTERS = {
    '.py': 'python -m py_compile {file} 2>&1',
    '.js': 'node --check {file} 2>&1',
    '.ts': 'npx tsc --noEmit {file} 2>&1',
    '.go': 'go vet {file} 2>&1',
    '.rs': 'rustfmt --check {file} 2>&1',
}

# Max limits for read operations
MAX_LINES = 2000
MAX_LINE_LENGTH = 2000
MAX_FILE_SIZE = 50 * 1024  # 50KB


class ShellFileOperations(FileOperations):
    """
    File operations implemented via shell commands.
    
    Works with ANY terminal backend that has execute(command, cwd) method.
    This includes local, docker, singularity, ssh, and modal environments.
    """
    
    def __init__(self, terminal_env, cwd: str = None):
        """
        Initialize file operations with a terminal environment.
        
        Args:
            terminal_env: Any object with execute(command, cwd) method.
                         Returns {"output": str, "returncode": int}
            cwd: Working directory (defaults to env's cwd or current directory)
        """
        self.env = terminal_env
        # Determine cwd from various possible sources.
        # IMPORTANT: do NOT fall back to os.getcwd() -- that's the HOST's local
        # path which doesn't exist inside container/cloud backends (modal, docker).
        # If nothing provides a cwd, use "/" as a safe universal default.
        resolved_cwd = (
            cwd
            or getattr(terminal_env, "cwd", None)
            or getattr(getattr(terminal_env, "config", None), "cwd", None)
        )
        if not resolved_cwd:
            resolved_cwd = os.path.expanduser("~") if os.name == "nt" else "/"
        # Normalize "~" style working directories on Windows so relative file
        # paths like "agents/worldview.md" resolve correctly.
        if os.name == "nt":
            resolved_cwd = str(Path(str(resolved_cwd)).expanduser())
        self.cwd = resolved_cwd
        
        # Cache for command availability checks
        self._command_cache: Dict[str, bool] = {}
    
    def _exec(self, command: str, cwd: str = None, timeout: int = None,
              stdin_data: str = None) -> ExecuteResult:
        """Execute command via terminal backend.
        
        Args:
            stdin_data: If provided, piped to the process's stdin instead of
                        embedding in the command string. Bypasses ARG_MAX.
        """
        kwargs = {}
        if timeout:
            kwargs['timeout'] = timeout
        if stdin_data is not None:
            kwargs['stdin_data'] = stdin_data
        
        result = self.env.execute(command, cwd=cwd or self.cwd, **kwargs)
        return ExecuteResult(
            stdout=result.get("output", ""),
            exit_code=result.get("returncode", 0)
        )
    
    def _has_command(self, cmd: str) -> bool:
        """Check if a command exists in the environment (cached)."""
        if cmd not in self._command_cache:
            if os.name == "nt":
                # PowerShell/cmd path: "where" is available by default.
                result = self._exec(f"where {cmd} >nul 2>nul && echo yes")
            else:
                result = self._exec(f"command -v {cmd} >/dev/null 2>&1 && echo 'yes'")
            self._command_cache[cmd] = result.stdout.strip().lower() == "yes"
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
            if not content_sample:
                return False
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
        lines = content.split('\n')
        numbered = []
        for i, line in enumerate(lines, start=start_line):
            # Truncate long lines
            if len(line) > MAX_LINE_LENGTH:
                line = line[:MAX_LINE_LENGTH] + "... [truncated]"
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
        # Native expansion on Windows local backends.
        if os.name == "nt":
            return os.path.expandvars(os.path.expanduser(path))
        
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
                # ~username format - let shell expand it
                expand_result = self._exec(f"echo {path}")
                if expand_result.exit_code == 0:
                    return expand_result.stdout.strip()
        
        return path

    @staticmethod
    def _normalize_apostrophes(s: str) -> str:
        """Normalize Unicode apostrophe variants to ASCII so path matching works across sources."""
        if not s:
            return s
        for c in ("\u2018", "\u2019", "\u201a", "\u201b", "`"):
            s = s.replace(c, "'")
        return s

    def _resolve_windows_path_apostrophe_fallback(self, path: str) -> Optional[str]:
        """
        When path does not exist, look for a file whose name matches when apostrophe
        variants are normalized (e.g. YouTube titles with '). Tries the path's
        parent dir, then cwd, workspace, and ~/.hermes/workspace. Returns the first
        matching existing path, or None.
        """
        expanded = self._expand_path(path)
        requested_name = os.path.basename(expanded) or expanded
        normalized_requested = self._normalize_apostrophes(requested_name).lower()
        parents_to_try = []
        if os.path.isabs(expanded):
            parents_to_try.append(os.path.dirname(expanded))
        else:
            if self.cwd:
                parents_to_try.append(self.cwd)
                parents_to_try.append(os.path.join(self.cwd, "workspace"))
            parents_to_try.append(os.path.normpath(os.path.expanduser(r"~/.hermes/workspace")))
        for parent in parents_to_try:
            if not parent or not os.path.isdir(parent):
                continue
            try:
                for name in os.listdir(parent):
                    if self._normalize_apostrophes(name).lower() == normalized_requested:
                        full = os.path.normpath(os.path.join(parent, name))
                        if os.path.isfile(full):
                            return full
            except OSError:
                continue
        return None

    def _resolve_windows_path(self, path: str) -> str:
        """
        Resolve a potentially relative Windows path with workspace-aware fallback.

        Resolution order for relative paths:
        1) <cwd>/<path>
        2) <cwd>/workspace/<path>
        3) ~/.hermes/workspace/<path>
        Returns the first existing path, else the first candidate.
        If none exist, tries apostrophe-normalized match in the same directory (e.g. YouTube .vtt files).
        """
        expanded = self._expand_path(path)
        if os.path.isabs(expanded):
            resolved = os.path.normpath(expanded)
        else:
            candidates = []
            if self.cwd:
                candidates.append(os.path.normpath(os.path.join(self.cwd, expanded)))
                candidates.append(os.path.normpath(os.path.join(self.cwd, "workspace", expanded)))
            hermes_ws = os.path.normpath(os.path.expanduser(r"~/.hermes/workspace"))
            candidates.append(os.path.normpath(os.path.join(hermes_ws, expanded)))

            resolved = None
            for candidate in candidates:
                if os.path.exists(candidate):
                    resolved = candidate
                    break
            if resolved is None:
                resolved = candidates[0] if candidates else os.path.normpath(expanded)

        if not os.path.exists(resolved):
            fallback = self._resolve_windows_path_apostrophe_fallback(resolved)
            if fallback is not None:
                return fallback
        return resolved

    def _is_windows_local_backend(self) -> bool:
        """True when running on Windows host with local backend file access."""
        if os.name != "nt":
            return False
        module_name = getattr(self.env.__class__, "__module__", "")
        return module_name.endswith(".local")
    
    def _escape_shell_arg(self, arg: str) -> str:
        """Escape a string for safe use in shell commands."""
        # Use single quotes and escape any single quotes in the string
        return "'" + arg.replace("'", "'\"'\"'") + "'"

    @staticmethod
    def _looks_like_file_glob(pattern: str) -> bool:
        """Heuristic: pattern appears to be a filename glob, not content regex."""
        if not pattern:
            return False
        # Common glob indicators
        has_glob = any(ch in pattern for ch in ("*", "?", "[", "]"))
        # Typical filename-ish suffixes
        has_ext_hint = "." in pattern and "/" not in pattern and "\\" not in pattern
        return has_glob and has_ext_hint
    
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
        # Block `.env` reads by basename from both POSIX and Windows backends.
        if _is_env_file_path(path):
            return ReadResult(error="Access to files named '.env' is disabled for security")

        # Windows backend: use native Python/os calls instead of POSIX shell
        # tools like stat/head/sed, which aren't available in PowerShell.
        if os.name == "nt":
            return self._read_file_windows(path, offset, limit)
        # POSIX path: use shell tools for compatibility with container/remote backends.
        # Expand ~ and other shell paths
        path = self._expand_path(path)
        
        # Clamp limit
        limit = min(limit, MAX_LINES)
        
        # Check if file exists and get size (wc -c is POSIX, works on Linux + macOS)
        stat_cmd = f"wc -c < {self._escape_shell_arg(path)} 2>/dev/null"
        stat_result = self._exec(stat_cmd)
        
        if stat_result.exit_code != 0:
            # File not found - try to suggest similar files
            return self._suggest_similar_files(path)
        
        try:
            file_size = int(stat_result.stdout.strip())
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
        
        if self._is_likely_binary(path, sample_result.stdout):
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
        
        # Get total line count
        wc_cmd = f"wc -l < {self._escape_shell_arg(path)}"
        wc_result = self._exec(wc_cmd)
        try:
            total_lines = int(wc_result.stdout.strip())
        except ValueError:
            total_lines = 0
        
        # Check if truncated
        truncated = total_lines > end_line
        hint = None
        if truncated:
            hint = f"Use offset={end_line + 1} to continue reading (showing {offset}-{end_line} of {total_lines} lines)"
        
        return ReadResult(
            content=self._add_line_numbers(read_result.stdout, offset),
            total_lines=total_lines,
            file_size=file_size,
            truncated=truncated,
            hint=hint
        )

    def _read_file_windows(self, path: str, offset: int = 1, limit: int = 500) -> ReadResult:
        """
        Windows implementation of read_file using native os/file APIs.
        
        Avoids reliance on POSIX tools like stat/head/sed which are not
        available in PowerShell-backed environments.
        """
        # Resolve to absolute path relative to the current working directory
        # of the terminal backend.
        path = self._resolve_windows_path(path)

        if _is_env_file_path(path):
            return ReadResult(error="Access to files named '.env' is disabled for security")
        
        if not os.path.exists(path):
            return self._suggest_similar_files(path)
        
        # Directories are not readable as files; return a helpful error.
        if os.path.isdir(path):
            try:
                entries = os.listdir(path)
            except OSError:
                entries = []
            # Show a small sample of children as a hint
            sample = entries[:20]
            hint = None
            if sample:
                hint = "Directory contents:\n" + "\n".join(f"- {name}" for name in sample)
            return ReadResult(
                error="Path is a directory, not a file.",
                hint=hint,
                similar_files=[os.path.join(path, name) for name in sample] if sample else [],
            )
        
        try:
            file_size = os.path.getsize(path)
        except OSError:
            file_size = 0
        
        # Clamp limit
        limit = min(limit, MAX_LINES)
        if offset < 1:
            offset = 1
        
        # Read all lines once, then slice for pagination
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception as e:
            return ReadResult(error=f"Failed to read file: {type(e).__name__}: {e}")
        
        total_lines = len(lines)
        start_idx = offset - 1
        end_idx = start_idx + limit
        page_lines = lines[start_idx:end_idx] if start_idx < total_lines else []
        truncated = end_idx < total_lines
        hint = None
        if truncated:
            hint = f"Use offset={end_idx + 1} to continue reading (showing {offset}-{end_idx} of {total_lines} lines)"
        
        content = "".join(page_lines)
        return ReadResult(
            content=self._add_line_numbers(content, offset),
            total_lines=total_lines,
            file_size=file_size,
            truncated=truncated,
            hint=hint,
        )
    
    # Images larger than this are too expensive to inline as base64 in the
    # conversation context. Return metadata only and suggest vision_analyze.
    MAX_IMAGE_BYTES = 512 * 1024  # 512 KB

    def _read_image(self, path: str) -> ReadResult:
        """Read an image file, returning base64 content."""
        # Get file size (wc -c is POSIX, works on Linux + macOS)
        stat_cmd = f"wc -c < {self._escape_shell_arg(path)} 2>/dev/null"
        stat_result = self._exec(stat_cmd)
        try:
            file_size = int(stat_result.stdout.strip())
        except ValueError:
            file_size = 0
        
        if file_size > self.MAX_IMAGE_BYTES:
            return ReadResult(
                is_image=True,
                is_binary=True,
                file_size=file_size,
                hint=(
                    f"Image is too large to inline ({file_size:,} bytes). "
                    "Use vision_analyze to inspect the image, or reference it by path."
                ),
            )
        
        # Get base64 content
        b64_cmd = f"base64 -w 0 {self._escape_shell_arg(path)} 2>/dev/null"
        b64_result = self._exec(b64_cmd, timeout=30)
        
        if b64_result.exit_code != 0:
            return ReadResult(
                is_image=True,
                is_binary=True,
                file_size=file_size,
                error=f"Failed to read image: {b64_result.stdout}"
            )
        
        # Try to get dimensions (requires ImageMagick)
        dimensions = None
        if self._has_command('identify'):
            dim_cmd = f"identify -format '%wx%h' {self._escape_shell_arg(path)} 2>/dev/null"
            dim_result = self._exec(dim_cmd)
            if dim_result.exit_code == 0:
                dimensions = dim_result.stdout.strip()
        
        # Determine MIME type from extension
        ext = os.path.splitext(path)[1].lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp',
            '.ico': 'image/x-icon',
        }
        mime_type = mime_types.get(ext, 'application/octet-stream')
        
        return ReadResult(
            is_image=True,
            is_binary=True,
            file_size=file_size,
            base64_content=b64_result.stdout,
            mime_type=mime_type,
            dimensions=dimensions
        )
    
    def _suggest_similar_files(self, path: str) -> ReadResult:
        """Suggest similar files when the requested file is not found."""
        if os.name == "nt":
            return self._suggest_similar_files_windows(path)
        # Get directory and filename
        dir_path = os.path.dirname(path) or "."
        filename = os.path.basename(path)
        
        # List files in directory
        ls_cmd = f"ls -1 {self._escape_shell_arg(dir_path)} 2>/dev/null | head -20"
        ls_result = self._exec(ls_cmd)
        
        similar = []
        if ls_result.exit_code == 0 and ls_result.stdout.strip():
            files = ls_result.stdout.strip().split('\n')
            # Simple similarity: files that share some characters with the target
            for f in files:
                # Check if filenames share significant overlap
                common = set(filename.lower()) & set(f.lower())
                if len(common) >= len(filename) * 0.5:  # 50% character overlap
                    similar.append(os.path.join(dir_path, f))
        
        return ReadResult(
            error=f"File not found: {path}",
            similar_files=similar[:5]  # Limit to 5 suggestions
        )

    def _suggest_similar_files_windows(self, path: str) -> ReadResult:
        """Windows-native similar-file suggestions (no shell dependencies)."""
        dir_path = os.path.dirname(path) or self.cwd or "."
        filename = os.path.basename(path)
        try:
            candidates = os.listdir(dir_path)
        except OSError:
            candidates = []
        similar = []
        target_lower = filename.lower()
        for name in candidates:
            common = set(target_lower) & set(name.lower())
            if filename and len(common) >= max(1, int(len(filename) * 0.5)):
                similar.append(os.path.join(dir_path, name))

        # If directory-local suggestions are empty, scan common roots to give
        # the model concrete existing paths instead of repeated blind reads.
        if not similar:
            roots = []
            if self.cwd:
                roots.append(self.cwd)
                roots.append(os.path.join(self.cwd, "workspace"))
            roots.append(os.path.expanduser(r"~/.hermes/workspace"))
            roots.append(os.path.expanduser(r"~/.hermes"))

            needle = self._normalize_apostrophes(
                (os.path.splitext(filename)[0] or filename).lower()
            )
            seen = set()
            for root in roots:
                if not root or not os.path.isdir(root):
                    continue
                try:
                    for walk_root, _, files in os.walk(root):
                        for name in files:
                            lname = self._normalize_apostrophes(name.lower())
                            if needle and needle in lname:
                                full = os.path.normpath(os.path.join(walk_root, name))
                                if full not in seen:
                                    similar.append(full)
                                    seen.add(full)
                                    if len(similar) >= 8:
                                        break
                        if len(similar) >= 8:
                            break
                except Exception:
                    continue
                if len(similar) >= 8:
                    break

        hint = None
        if similar:
            hint_lines = "\n".join(f"- {p}" for p in similar[:5])
            hint = (
                "Closest existing files:\n"
                f"{hint_lines}\n"
                "Tip: prefer these exact paths or read the containing directory first."
            )
        return ReadResult(
            error=f"File not found: {path}",
            similar_files=similar[:5],
            hint=hint,
        )
    
    # =========================================================================
    # WRITE Implementation
    # =========================================================================
    
    def write_file(self, path: str, content: str) -> WriteResult:
        """
        Write content to a file, creating parent directories as needed.

        Pipes content through stdin to avoid OS ARG_MAX limits on large
        files. The content never appears in the shell command string —
        only the file path does.

        Args:
            path: File path to write
            content: Content to write

        Returns:
            WriteResult with bytes written or error
        """
        # Windows local backend: avoid shell aliases (cat/Get-Content) and
        # write directly with Python file APIs.
        if self._is_windows_local_backend():
            return self._write_file_windows(path, content)

        # Expand ~ and other shell paths
        path = self._expand_path(path)

        # Block writes to sensitive paths
        if _is_write_denied(path):
            return WriteResult(error=f"Write denied: '{path}' is a protected system/credential file.")

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
        
        return WriteResult(
            bytes_written=bytes_written,
            dirs_created=dirs_created
        )

    def _write_file_windows(self, path: str, content: str) -> WriteResult:
        """Windows-native file write path for local backend."""
        try:
            resolved = self._resolve_windows_path(path)
            parent = os.path.dirname(resolved)
            dirs_created = False
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
                dirs_created = True

            with open(resolved, "w", encoding="utf-8", newline="") as f:
                f.write(content)
                f.flush()

            bytes_written = os.path.getsize(resolved)
            return WriteResult(bytes_written=bytes_written, dirs_created=dirs_created)
        except Exception as e:
            return WriteResult(error=f"Failed to write file: {type(e).__name__}: {e}")
    
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
        # Expand path according to platform/backend.
        if os.name == "nt":
            path = self._resolve_windows_path(path)
        else:
            path = self._expand_path(path)

        # Block writes to sensitive paths
        if _is_write_denied(path):
            return PatchResult(error=f"Write denied: '{path}' is a protected system/credential file.")

        # Read current content
        if os.name == "nt":
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception:
                return PatchResult(error=f"Failed to read file: {path}")
        else:
            read_cmd = f"cat {self._escape_shell_arg(path)} 2>/dev/null"
            read_result = self._exec(read_cmd)
            
            if read_result.exit_code != 0:
                return PatchResult(error=f"Failed to read file: {path}")
            
            content = read_result.stdout
        
        # Import and use fuzzy matching
        from tools.fuzzy_match import fuzzy_find_and_replace
        
        new_content, match_count, error = fuzzy_find_and_replace(
            content, old_string, new_string, replace_all
        )
        
        if error:
            return PatchResult(error=error)
        
        if match_count == 0:
            return PatchResult(error=f"Could not find match for old_string in {path}")
        
        # Write back
        write_result = self.write_file(path, new_content)
        if write_result.error:
            return PatchResult(error=f"Failed to write changes: {write_result.error}")
        
        # Generate diff
        diff = self._unified_diff(content, new_content, path)
        
        # Auto-lint
        lint_result = self._check_lint(path)
        
        return PatchResult(
            success=True,
            diff=diff,
            files_modified=[path],
            lint=lint_result.to_dict() if lint_result else None
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
    
    def _check_lint(self, path: str) -> LintResult:
        """
        Run syntax check on a file after editing.
        
        Args:
            path: File path to lint
        
        Returns:
            LintResult with status and any errors
        """
        ext = os.path.splitext(path)[1].lower()
        
        if ext not in LINTERS:
            return LintResult(skipped=True, message=f"No linter for {ext} files")
        
        # Check if linter command is available
        linter_cmd = LINTERS[ext]
        # Extract the base command (first word)
        base_cmd = linter_cmd.split()[0]
        
        if not self._has_command(base_cmd):
            return LintResult(skipped=True, message=f"{base_cmd} not available")
        
        # Run linter
        cmd = linter_cmd.format(file=self._escape_shell_arg(path))
        result = self._exec(cmd, timeout=30)
        
        return LintResult(
            success=result.exit_code == 0,
            output=result.stdout.strip() if result.stdout.strip() else ""
        )
    
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
        # Expand ~ and other shell paths
        path = self._expand_path(path)

        # Common model mistake: using grep with a glob pattern (e.g. "*.html").
        # Treat this as file search to avoid regex parse errors and retry loops.
        if target == "content" and not file_glob and self._looks_like_file_glob(pattern):
            target = "files"

        if os.name == "nt":
            return self._search_windows(
                pattern=pattern,
                path=path,
                target=target,
                file_glob=file_glob,
                limit=limit,
                offset=offset,
                output_mode=output_mode,
                context=context,
            )
        
        if target == "files":
            return self._search_files(pattern, path, limit, offset)
        else:
            return self._search_content(pattern, path, file_glob, limit, offset, 
                                        output_mode, context)

    def _search_windows(
        self,
        pattern: str,
        path: str,
        target: str,
        file_glob: Optional[str],
        limit: int,
        offset: int,
        output_mode: str,
        context: int,
    ) -> SearchResult:
        """Windows-native search implementation (no grep/find dependency)."""
        import fnmatch
        import re as _re

        base_path = self._resolve_windows_path(path or ".")
        if not os.path.exists(base_path):
            return SearchResult(error=f"Path not found: {base_path}")

        if _is_env_file_path(base_path):
            return SearchResult(error="Searching '.env' files is disabled for security")

        if target == "files":
            search_pattern = pattern
            if not any(ch in search_pattern for ch in ["*", "?", "[", "]"]):
                search_pattern = f"*{search_pattern}*"
            matches = []
            if os.path.isfile(base_path):
                matches = [
                    base_path
                ] if (
                    fnmatch.fnmatch(os.path.basename(base_path), search_pattern)
                    and not _is_env_file_path(base_path)
                ) else []
            else:
                for root, _, files in os.walk(base_path):
                    for name in files:
                        full = os.path.join(root, name)
                        if _is_env_file_path(full):
                            continue
                        if fnmatch.fnmatch(name, search_pattern):
                            full = os.path.join(root, name)
                            matches.append((full, os.path.getmtime(full)))
                matches.sort(key=lambda x: x[1], reverse=True)
                matches = [m[0] for m in matches]
            total = len(matches)
            page = matches[offset:offset + limit]
            return SearchResult(files=page, total_count=total, truncated=total > offset + limit)

        try:
            regex = _re.compile(pattern)
        except _re.error as e:
            return SearchResult(error=f"Invalid regex pattern: {e}")

        if os.path.isfile(base_path):
            files_to_scan = [base_path]
        else:
            files_to_scan = []
            for root, _, files in os.walk(base_path):
                for name in files:
                    if file_glob and not fnmatch.fnmatch(name, file_glob):
                        continue
                    if _is_env_file_path(os.path.join(root, name)):
                        continue
                    files_to_scan.append(os.path.join(root, name))

        content_matches: List[SearchMatch] = []
        files_only = set()
        counts: Dict[str, int] = {}

        for file_path in files_to_scan:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
            except Exception:
                continue
            hit_count = 0
            for idx, line in enumerate(lines, start=1):
                if regex.search(line):
                    hit_count += 1
                    if output_mode == "content":
                        content_matches.append(
                            SearchMatch(
                                path=file_path,
                                line_number=idx,
                                content=line.rstrip("\r\n")[:500],
                            )
                        )
            if hit_count:
                files_only.add(file_path)
                counts[file_path] = hit_count

        if output_mode == "files_only":
            all_files = sorted(files_only)
            total = len(all_files)
            page = all_files[offset:offset + limit]
            return SearchResult(files=page, total_count=total, truncated=total > offset + limit)

        if output_mode == "count":
            # Keep full counts map and total count for parity with grep mode.
            return SearchResult(counts=counts, total_count=sum(counts.values()))

        total = len(content_matches)
        page = content_matches[offset:offset + limit]
        return SearchResult(matches=page, total_count=total, truncated=total > offset + limit)
    
    def _search_files(self, pattern: str, path: str, limit: int, offset: int) -> SearchResult:
        """Search for files by name pattern (glob-like)."""
        # Check if find is available (not on Windows without Git Bash/WSL)
        if not self._has_command('find'):
            return SearchResult(
                error="File search requires 'find' command. "
                      "On Windows, use Git Bash, WSL, or install Unix tools."
            )
        
        # If the pattern is a plain token (no glob chars), treat it as
        # substring match for better UX (e.g., "worldview" -> "*worldview*").
        search_pattern = pattern.split('/')[-1]
        if not any(ch in search_pattern for ch in ["*", "?", "[", "]"]):
            search_pattern = f"*{search_pattern}*"
        
        # Use find with modification time sorting
        # -printf '%T@ %p\n' outputs: timestamp path
        # sort -rn sorts by timestamp descending (newest first)
        cmd = (
            f"find {self._escape_shell_arg(path)} -type f -name "
            f"{self._escape_shell_arg(search_pattern)} ! -name '.env' "
            f"-printf '%T@ %p\\n' 2>/dev/null | sort -rn | "
            f"tail -n +{offset + 1} | head -n {limit}"
        )
        
        result = self._exec(cmd, timeout=60)
        
        if result.exit_code != 0 and not result.stdout.strip():
            # Try without -printf (BSD find compatibility)
            cmd_simple = (
                f"find {self._escape_shell_arg(path)} -type f -name "
                f"{self._escape_shell_arg(search_pattern)} ! -name '.env' "
                f"2>/dev/null | head -n {limit + offset} | tail -n +{offset + 1}"
            )
            result = self._exec(cmd_simple, timeout=60)
        
        files = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            # Parse "timestamp path" format
            parts = line.split(' ', 1)
            if len(parts) == 2 and parts[0].replace('.', '').isdigit():
                if not _is_env_file_path(parts[1]):
                    files.append(parts[1])
            else:
                if not _is_env_file_path(line):
                    files.append(line)
        
        return SearchResult(
            files=files,
            total_count=len(files)
        )
    
    def _search_content(self, pattern: str, path: str, file_glob: Optional[str],
                        limit: int, offset: int, output_mode: str, context: int) -> SearchResult:
        """Search for content inside files (grep-like)."""
        if _is_env_file_path(path):
            return SearchResult(error="Searching '.env' files is disabled for security")

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
        cmd_parts.extend(["--glob", self._escape_shell_arg("!.env")])
        
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
        
        # Parse results based on output mode
        if output_mode == "files_only":
            all_files = [
                f for f in result.stdout.strip().split('\n')
                if f and not _is_env_file_path(f)
            ]
            total = len(all_files)
            page = all_files[offset:offset + limit]
            return SearchResult(files=page, total_count=total)
        
        elif output_mode == "count":
            counts = {}
            for line in result.stdout.strip().split('\n'):
                if ':' in line:
                    parts = line.rsplit(':', 1)
                    if len(parts) == 2:
                        if _is_env_file_path(parts[0]):
                            continue
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
            matches = []
            for line in result.stdout.strip().split('\n'):
                if not line or line == "--":
                    continue
                
                # Try match line first (colon-separated: file:line:content)
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    try:
                        if _is_env_file_path(parts[0]):
                            continue
                        matches.append(SearchMatch(
                            path=parts[0],
                            line_number=int(parts[1]),
                            content=parts[2][:500]
                        ))
                        continue
                    except ValueError:
                        pass
                
                # Try context line (dash-separated: file-line-content)
                # Only attempt if context was requested to avoid false positives
                if context > 0:
                    parts = line.split('-', 2)
                    if len(parts) >= 3:
                        try:
                            if _is_env_file_path(parts[0]):
                                continue
                            matches.append(SearchMatch(
                                path=parts[0],
                                line_number=int(parts[1]),
                                content=parts[2][:500]
                            ))
                        except ValueError:
                            pass
            
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
        
        # Add context if requested
        if context > 0:
            cmd_parts.extend(["-C", str(context)])
        
        # Add file pattern filter (must be quoted to prevent shell expansion)
        if file_glob:
            cmd_parts.extend(["--include", self._escape_shell_arg(file_glob)])
        cmd_parts.extend(["--exclude", self._escape_shell_arg(".env")])
        
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
        
        if output_mode == "files_only":
            all_files = [
                f for f in result.stdout.strip().split('\n')
                if f and not _is_env_file_path(f)
            ]
            total = len(all_files)
            page = all_files[offset:offset + limit]
            return SearchResult(files=page, total_count=total)
        
        elif output_mode == "count":
            counts = {}
            for line in result.stdout.strip().split('\n'):
                if ':' in line:
                    parts = line.rsplit(':', 1)
                    if len(parts) == 2:
                        if _is_env_file_path(parts[0]):
                            continue
                        try:
                            counts[parts[0]] = int(parts[1])
                        except ValueError:
                            pass
            return SearchResult(counts=counts, total_count=sum(counts.values()))
        
        else:
            # grep match lines:   "file:lineno:content" (colon)
            # grep context lines: "file-lineno-content"  (dash)
            # grep group seps:    "--"
            matches = []
            for line in result.stdout.strip().split('\n'):
                if not line or line == "--":
                    continue
                
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    try:
                        if _is_env_file_path(parts[0]):
                            continue
                        matches.append(SearchMatch(
                            path=parts[0],
                            line_number=int(parts[1]),
                            content=parts[2][:500]
                        ))
                        continue
                    except ValueError:
                        pass
                
                if context > 0:
                    parts = line.split('-', 2)
                    if len(parts) >= 3:
                        try:
                            if _is_env_file_path(parts[0]):
                                continue
                            matches.append(SearchMatch(
                                path=parts[0],
                                line_number=int(parts[1]),
                                content=parts[2][:500]
                            ))
                        except ValueError:
                            pass
            
            total = len(matches)
            page = matches[offset:offset + limit]
            return SearchResult(
                matches=page,
                total_count=total,
                truncated=total > offset + limit
            )
