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
import uuid
import difflib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path


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
        return {k: v for k, v in self.__dict__.items() if v is not None and v != [] and v != ""}


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
            cwd: Working directory (defaults to env's cwd or /tmp)
        """
        self.env = terminal_env
        # Determine cwd from various possible sources
        self.cwd = cwd or getattr(terminal_env, 'cwd', None) or \
                   getattr(getattr(terminal_env, 'config', None), 'cwd', None) or '/tmp'
        
        # Cache for command availability checks
        self._command_cache: Dict[str, bool] = {}
    
    def _exec(self, command: str, cwd: str = None, timeout: int = None) -> ExecuteResult:
        """Execute command via terminal backend."""
        kwargs = {}
        if timeout:
            kwargs['timeout'] = timeout
        
        result = self.env.execute(command, cwd=cwd or self.cwd, **kwargs)
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
        
        # Clamp limit
        limit = min(limit, MAX_LINES)
        
        # Check if file exists and get metadata
        stat_cmd = f"stat -c '%s' {self._escape_shell_arg(path)} 2>/dev/null"
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
        
        # Check if it's an image - return base64
        if self._is_image(path):
            return self._read_image(path)
        
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
    
    def _read_image(self, path: str) -> ReadResult:
        """Read an image file, returning base64 content."""
        # Get file size
        stat_cmd = f"stat -c '%s' {self._escape_shell_arg(path)} 2>/dev/null"
        stat_result = self._exec(stat_cmd)
        try:
            file_size = int(stat_result.stdout.strip())
        except ValueError:
            file_size = 0
        
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
    
    # =========================================================================
    # WRITE Implementation
    # =========================================================================
    
    def write_file(self, path: str, content: str) -> WriteResult:
        """
        Write content to a file, creating parent directories as needed.
        
        Uses heredoc with unique marker for safe shell execution.
        
        Args:
            path: File path to write
            content: Content to write
        
        Returns:
            WriteResult with bytes written or error
        """
        # Expand ~ and other shell paths
        path = self._expand_path(path)
        
        # Create parent directories
        parent = os.path.dirname(path)
        dirs_created = False
        
        if parent:
            mkdir_cmd = f"mkdir -p {self._escape_shell_arg(parent)}"
            mkdir_result = self._exec(mkdir_cmd)
            if mkdir_result.exit_code == 0:
                dirs_created = True
        
        # Generate unique marker for heredoc that won't appear in content
        marker = f"HERMES_EOF_{uuid.uuid4().hex[:8]}"
        while marker in content:
            marker = f"HERMES_EOF_{uuid.uuid4().hex[:8]}"
        
        # Write using heredoc with single-quoted marker (prevents all expansion)
        # The single quotes around the marker prevent variable expansion
        write_cmd = f"cat > {self._escape_shell_arg(path)} << '{marker}'\n{content}\n{marker}"
        write_result = self._exec(write_cmd)
        
        if write_result.exit_code != 0:
            return WriteResult(error=f"Failed to write file: {write_result.stdout}")
        
        # Get bytes written
        stat_cmd = f"stat -c '%s' {self._escape_shell_arg(path)} 2>/dev/null"
        stat_result = self._exec(stat_cmd)
        
        try:
            bytes_written = int(stat_result.stdout.strip())
        except ValueError:
            bytes_written = len(content.encode('utf-8'))
        
        return WriteResult(
            bytes_written=bytes_written,
            dirs_created=dirs_created
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
        
        # Read current content
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
        
        if target == "files":
            return self._search_files(pattern, path, limit, offset)
        else:
            return self._search_content(pattern, path, file_glob, limit, offset, 
                                        output_mode, context)
    
    def _search_files(self, pattern: str, path: str, limit: int, offset: int) -> SearchResult:
        """Search for files by name pattern (glob-like)."""
        # Check if find is available (not on Windows without Git Bash/WSL)
        if not self._has_command('find'):
            return SearchResult(
                error="File search requires 'find' command. "
                      "On Windows, use Git Bash, WSL, or install Unix tools."
            )
        
        # Auto-prepend **/ for recursive search if not already present
        if not pattern.startswith('**/') and '/' not in pattern:
            search_pattern = pattern
        else:
            search_pattern = pattern.split('/')[-1]
        
        # Use find with modification time sorting
        # -printf '%T@ %p\n' outputs: timestamp path
        # sort -rn sorts by timestamp descending (newest first)
        cmd = f"find {self._escape_shell_arg(path)} -type f -name {self._escape_shell_arg(search_pattern)} " \
              f"-printf '%T@ %p\\n' 2>/dev/null | sort -rn | tail -n +{offset + 1} | head -n {limit}"
        
        result = self._exec(cmd, timeout=60)
        
        if result.exit_code != 0 and not result.stdout.strip():
            # Try without -printf (BSD find compatibility)
            cmd_simple = f"find {self._escape_shell_arg(path)} -type f -name {self._escape_shell_arg(search_pattern)} " \
                        f"2>/dev/null | head -n {limit + offset} | tail -n +{offset + 1}"
            result = self._exec(cmd_simple, timeout=60)
        
        files = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            # Parse "timestamp path" format
            parts = line.split(' ', 1)
            if len(parts) == 2 and parts[0].replace('.', '').isdigit():
                files.append(parts[1])
            else:
                files.append(line)
        
        return SearchResult(
            files=files,
            total_count=len(files)
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
        cmd_parts = ["rg", "--line-number", "--no-heading"]
        
        # Add context if requested
        if context > 0:
            cmd_parts.extend(["-C", str(context)])
        
        # Add file glob filter
        if file_glob:
            cmd_parts.extend(["--glob", file_glob])
        
        # Output mode handling
        if output_mode == "files_only":
            cmd_parts.append("-l")  # Files only
        elif output_mode == "count":
            cmd_parts.append("-c")  # Count per file
        
        # Add pattern and path
        cmd_parts.append(self._escape_shell_arg(pattern))
        cmd_parts.append(self._escape_shell_arg(path))
        
        # Limit results
        cmd_parts.extend(["|", "head", "-n", str(limit + offset)])
        
        cmd = " ".join(cmd_parts)
        result = self._exec(cmd, timeout=60)
        
        # Parse results based on output mode
        if output_mode == "files_only":
            files = [f for f in result.stdout.strip().split('\n') if f][offset:]
            return SearchResult(files=files[:limit], total_count=len(files))
        
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
            # Parse content matches
            matches = []
            for line in result.stdout.strip().split('\n')[offset:]:
                if not line:
                    continue
                # Format: file:line:content
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    try:
                        matches.append(SearchMatch(
                            path=parts[0],
                            line_number=int(parts[1]),
                            content=parts[2][:500]  # Truncate long lines
                        ))
                    except ValueError:
                        # Line number not an int, skip
                        pass
            
            return SearchResult(
                matches=matches[:limit],
                total_count=len(matches),
                truncated=len(matches) > limit
            )
    
    def _search_with_grep(self, pattern: str, path: str, file_glob: Optional[str],
                          limit: int, offset: int, output_mode: str, context: int) -> SearchResult:
        """Fallback search using grep."""
        cmd_parts = ["grep", "-rn"]
        
        # Add context if requested
        if context > 0:
            cmd_parts.extend(["-C", str(context)])
        
        # Add file pattern filter
        if file_glob:
            cmd_parts.extend(["--include", file_glob])
        
        # Output mode handling
        if output_mode == "files_only":
            cmd_parts.append("-l")
        elif output_mode == "count":
            cmd_parts.append("-c")
        
        # Add pattern and path
        cmd_parts.append(self._escape_shell_arg(pattern))
        cmd_parts.append(self._escape_shell_arg(path))
        
        # Limit and offset
        cmd_parts.extend(["|", "tail", "-n", f"+{offset + 1}", "|", "head", "-n", str(limit)])
        
        cmd = " ".join(cmd_parts)
        result = self._exec(cmd, timeout=60)
        
        # Parse results (same format as rg)
        if output_mode == "files_only":
            files = [f for f in result.stdout.strip().split('\n') if f]
            return SearchResult(files=files, total_count=len(files))
        
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
            matches = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    try:
                        matches.append(SearchMatch(
                            path=parts[0],
                            line_number=int(parts[1]),
                            content=parts[2][:500]
                        ))
                    except ValueError:
                        pass
            
            return SearchResult(
                matches=matches,
                total_count=len(matches)
            )
