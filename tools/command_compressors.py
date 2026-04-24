#!/usr/bin/env python3
"""
Command Compressors — token-saving parsers for CLI command output.

Each compressor takes raw stdout, stderr, and exit code from a terminal command
and returns a compressed human-readable summary (typically 1-3 lines).

If a compressor cannot handle a given output (e.g., unexpected format), it
returns None and the compression pipeline falls back to raw output.

Architecture:
    CommandCompressor <- Protocol defining can_compress / compress
    CompressorRegistry  <- maps command names to compressor instances
    BuiltInCompressors  <- individual compressor implementations

Supported commands (v1):
    git status, git diff, git log,
    pytest, cargo test, npm test, go test,
    ls, tree,
    docker ps, docker logs,
    ruff check, ruff format,
    ruff, eslint, mypy

Adding a new compressor:
    1. Implement a class with can_compress(command, stdout, stderr) -> bool
       and compress(command, stdout, stderr, exit_code) -> str
    2. Register it in DEFAULT_COMPRESSORS below.
       DO NOT add external dependencies here.
"""

from __future__ import annotations

import dataclasses
import re
import shlex
from typing import Optional, List, Tuple
from enum import Enum, auto


# ----------------------------------------------------------------------------
# Protocols and dataclasses
# ----------------------------------------------------------------------------


class CommandCompressor:
    """ABC for a command output compressor."""

    def can_compress(self, command: str, stdout: str, stderr: str) -> bool:
        """
        Return True if this compressor can handle the given command + output.
        Override this to add command-name filtering or output pattern checks.
        """
        raise NotImplementedError

    def compress(
        self, command: str, stdout: str, stderr: str, exit_code: int
    ) -> str:
        """
        Compress raw CLI output into a short summary string.
        Override with command-specific parsing logic.
        Raise NotImplementedError if format is unrecognisable (pipeline will
        fall back to raw output).
        """
        raise NotImplementedError

    def command_tokens(self, raw_text: str) -> int:
        """
        Rough token estimate for a raw output string.
        Used by the stats tracker. Default: len(raw_text) // 4 (chars -> tokens).
        """
        return max(1, len(raw_text) // 4)


# ----------------------------------------------------------------------------
# Git Status Compressor
# ----------------------------------------------------------------------------


class GitStatusCompressor(CommandCompressor):
    """
    git status output is highly structured and verbose.
    Raw: 30-600 tokens of branch info, staged/unstaged files, merge state.
    Compressed: 1-3 lines showing branch, dirty state, and staged/unstaged counts.
    Typical savings: 75-85%.
    """

    BRANCH_RE = re.compile(r"^On branch (.+)$", re.MULTILINE)
    # Detached HEAD
    DETACHED_RE = re.compile(r"^HEAD detached at (.+)$", re.MULTILINE)
    # Staged files
    NEW_FILE_RE = re.compile(r"^\s*new file:\s+(.+)$", re.MULTILINE)
    MODIFIED_RE = re.compile(r"^\s*modified:\s+(.+)$", re.MULTILINE)
    DELETED_RE = re.compile(r"^\s*deleted:\s+(.+)$", re.MULTILINE)
    RENAMED_RE = re.compile(r"^\s*renamed:\s+(.+)$", re.MULTILINE)
    # Untracked
    UNTRACKED_RE = re.compile(r"^\s*Untracked files:\s*$", re.MULTILINE)
    UNTRACKED_FILES_RE = re.compile(r"^\s{1,2}(?!\s)(.+)$")  # lines after "Untracked files:"

    def can_compress(self, command: str, stdout: str, stderr: str) -> bool:
        normalized = _normalize_cmd(command)
        return normalized == "git" and "status" in command

    def compress(
        self, command: str, stdout: str, stderr: str, exit_code: int
    ) -> str:
        lines = stdout.splitlines()
        if not lines:
            return "[empty git status]"

        branch = "?"
        detached = False
        staged_new: list[str] = []
        staged_modified: list[str] = []
        staged_deleted: list[str] = []
        staged_renamed: list[str] = []
        untracked: list[str] = []

        in_untracked_block = False
        for raw_line in lines:
            line = raw_line.strip()

            branch_match = self.BRANCH_RE.match(line)
            detached_match = self.DETACHED_RE.match(line)
            if branch_match:
                branch = branch_match.group(1)
                continue
            if detached_match:
                branch = detached_match.group(1)
                detached = True
                continue

            if line == "Untracked files:":
                in_untracked_block = True
                continue

            if in_untracked_block:
                # Blank line ends untracked block
                if not line:
                    in_untracked_block = False
                    continue
                untracked.append(line)
                continue

            # Staged changes
            new_match = self.NEW_FILE_RE.match(line)
            mod_match = self.MODIFIED_RE.match(line)
            del_match = self.DELETED_RE.match(line)
            ren_match = self.RENAMED_RE.match(line)

            if new_match:
                staged_new.append(_truncate(new_match.group(1), 50))
            elif mod_match:
                staged_modified.append(_truncate(mod_match.group(1), 50))
            elif del_match:
                staged_deleted.append(_truncate(del_match.group(1), 50))
            elif ren_match:
                staged_renamed.append(_truncate(ren_match.group(1), 50))

        # Build summary
        parts = []
        if detached:
            parts.append(f"[detached @ {branch}]")
        else:
            parts.append(f"[{branch}]")

        dirty = len(staged_new) + len(staged_modified) + len(staged_deleted) + len(staged_renamed) + len(untracked)
        if dirty == 0:
            parts.append("clean")
        else:
            changes = []
            if staged_new:
                changes.append(f"+{len(staged_new)} new")
            if staged_modified:
                changes.append(f"~{len(staged_modified)} modified")
            if staged_deleted:
                changes.append(f"-{len(staged_deleted)} deleted")
            if staged_renamed:
                changes.append(f"r{len(staged_renamed)} renamed")
            if untracked:
                changes.append(f"?{len(untracked)} untracked")
            parts.append(", ".join(changes))

        summary = " ".join(parts)
        if exit_code != 0:
            summary += f" (exit {exit_code})"

        return summary


# ----------------------------------------------------------------------------
# Git Diff Compressor
# ----------------------------------------------------------------------------


class GitDiffCompressor(CommandCompressor):
    """
    git diff — shows file-level and hunk-level changes.
    Raw: dozens to thousands of lines of +/- diff hunks.
    Compressed: per-file change summary (files changed, insertions, deletions).
    Typical savings: 70-90%.
    """

    # diff --stat summary line
    STAT_RE = re.compile(
        r"(\d+)\s+file[s]?\s+changed(?:,\s+(\d+)\s+insertion)?(?:--?\+?)?(?:,\s+(\d+)\s+deletion)?"
    )

    def can_compress(self, command: str, stdout: str, stderr: str) -> bool:
        normalized = _normalize_cmd(command)
        return normalized == "git" and "diff" in command

    def compress(
        self, command: str, stdout: str, stderr: str, exit_code: int
    ) -> str:
        if not stdout.strip():
            return "[no changes]"

        # If stdout looks like a --stat summary (single line), use it directly
        stat_match = self.STAT_RE.search(stdout)
        if stat_match:
            files = stat_match.group(1)
            insertions = stat_match.group(2) or "0"
            deletions = stat_match.group(3) or "0"
            summary = f"[+{insertions} -{deletions} | {files} files]"
            if "--cached" in command or "--staged" in command:
                summary += " (staged)"
            if exit_code != 0:
                summary += f" (exit {exit_code})"
            return summary

        # Otherwise count per-file changes from diff output
        # File header lines: @@ -N,N +N,N @@ optional context
        file_headers: list[str] = []
        current_file = None
        plus_lines = 0
        minus_lines = 0

        for line in stdout.splitlines():
            if line.startswith("diff --git"):
                if current_file:
                    file_headers.append(
                        f"{current_file} (+{plus_lines} -{minus_lines})"
                    )
                # Extract filename from "diff --git a/path b/path"
                match = re.search(r" b/(.+)$", line)
                current_file = match.group(1) if match else "?"
                plus_lines = minus_lines = 0
            elif line.startswith("@@"):
                # Count +/- in this hunk
                # Unified diff format: @@ -start,count +start,count @@
                hunk_match = re.search(r"-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?", line)
                if hunk_match:
                    # group(1)=minus_start, group(2)=minus_count(default1)
                    # group(3)=plus_start,  group(4)=plus_count(default1)
                    plus_lines += int(hunk_match.group(4) or 1)
                    minus_lines += int(hunk_match.group(2) or 1)
            elif line.startswith("+") and not line.startswith("+++"):
                plus_lines += 1
            elif line.startswith("-") and not line.startswith("---"):
                minus_lines += 1

        if current_file:
            file_headers.append(f"{current_file} (+{plus_lines} -{minus_lines})")

        if not file_headers:
            return stdout.strip()[:200]  # fallback to raw head

        summary = ", ".join(file_headers[:10])
        if len(file_headers) > 10:
            summary += f" ... +{len(file_headers) - 10} more files"
        if exit_code != 0:
            summary += f" (exit {exit_code})"

        return summary


# ----------------------------------------------------------------------------
# Pytest Compressor
# ----------------------------------------------------------------------------


class PytestCompressor(CommandCompressor):
    """
    pytest -v output is verbose. Raw: 500-2000 tokens for a typical test run.
    Compressed: one line — pass/fail count, suite info, timing.
    Typical savings: 93-97%.
    """

    # Summary line: uses independent regex searches for each counter
    # so ordering doesn't matter (handles both "N passed, M failed" and "M failed, N passed")
    PASS_RE = re.compile(r"(\d+)\s+passed", re.MULTILINE)
    FAIL_RE = re.compile(r"(\d+)\s+failed", re.MULTILINE)
    ERR_RE = re.compile(r"(\d+)\s+error", re.MULTILINE)
    SKIP_RE = re.compile(r"(\d+)\s+skipped", re.MULTILINE)
    DURATION_RE = re.compile(r"in\s+([\d\.]+)s", re.MULTILINE)

    def can_compress(self, command: str, stdout: str, stderr: str) -> bool:
        return "pytest" in command or "py.test" in command

    def compress(
        self, command: str, stdout: str, stderr: str, exit_code: int
    ) -> str:
        output = stdout + "\n" + stderr

        if not output.strip():
            return "[pytest: no output]"

        # Extract counts from the last "===== N passed =====" style summary line
        lines = output.splitlines()
        summary_line = ""
        for line in reversed(lines):
            if "===" in line or "passed" in line or "failed" in line:
                summary_line = line
                break

        if summary_line:
            pass_matches = self.PASS_RE.findall(summary_line)
            fail_matches = self.FAIL_RE.findall(summary_line)
            err_matches = self.ERR_RE.findall(summary_line)
            skip_matches = self.SKIP_RE.findall(summary_line)
            dur_matches = self.DURATION_RE.findall(summary_line)

            passed = int(pass_matches[-1]) if pass_matches else 0
            failed = int(fail_matches[-1]) if fail_matches else 0
            errors = int(err_matches[-1]) if err_matches else 0
            skipped = int(skip_matches[-1]) if skip_matches else 0
            duration = dur_matches[-1] if dur_matches else ""

            if passed or failed or errors or skipped:
                parts = []
                if passed:
                    parts.append(f"{passed} passed")
                if failed:
                    parts.append(f"{failed} failed")
                if errors:
                    parts.append(f"{errors} error")
                if skipped:
                    parts.append(f"{skipped} skipped")
                summary = ", ".join(parts)
                if duration:
                    summary += f" in {duration}s"
                if exit_code != 0:
                    summary += " [FAILED]"
                else:
                    summary = "✓ " + summary
                return summary

        # Fallback: count individual test result lines
        passed_count = len(self.PASS_RE.findall(output))
        failed_count = len(self.FAIL_RE.findall(output))
        error_count = len(self.ERR_RE.findall(output))
        skipped_count = len(self.SKIP_RE.findall(output))

        if passed_count == 0 and failed_count == 0 and error_count == 0:
            return f"[pytest output: {len(output)} chars, format unrecognised]"

        parts = []
        if passed_count:
            parts.append(f"{passed_count} passed")
        if failed_count:
            parts.append(f"{failed_count} failed")
        if error_count:
            parts.append(f"{error_count} error")
        if skipped_count:
            parts.append(f"{skipped_count} skipped")

        summary = ", ".join(parts)
        if exit_code != 0:
            summary += " [FAILED]"
        else:
            summary = "✓ " + summary

        return summary
# ----------------------------------------------------------------------------
# Cargo Test Compressor
# ----------------------------------------------------------------------------


class CargoTestCompressor(CommandCompressor):
    """
    cargo test output is verbose. Raw: 3000-8000 tokens for large test suites.
    Compressed: one line — test count, pass/fail, suite timing.
    Typical savings: 95-99%.
    """

    SUMMARY_RE = re.compile(
        r"test result:[\s\w\.]+\.\s+"
        r"(\d+)\s+passed;\s+(\d+)\s+failed;\s+(\d+)\s+ignored.*?"
        r"(?:run time|elapsed)[\s:=]*([\d\.]+)s",
        re.MULTILINE | re.IGNORECASE,
    )
    SINGLE_TEST_RE = re.compile(
        r"(?:test\s+([\w:]+)\s+\.\.\.([\w\s]+?)(?:\s+(\d+)\s+(\d+)us)?(?:\s+ok|FAILED|running)?)",
        re.MULTILINE,
    )
    COMPILE_ERROR_RE = re.compile(r"error\[(?:E\d+)\]:", re.MULTILINE)

    def can_compress(self, command: str, stdout: str, stderr: str) -> bool:
        return "cargo" in command and any(
            sub in command for sub in ("test", "check", "build")
        )

    def compress(
        self, command: str, stdout: str, stderr: str, exit_code: int
    ) -> str:
        output = stdout + "\n" + stderr

        # Compile errors — show error count
        compile_errors = self.COMPILE_ERROR_RE.findall(output)
        if compile_errors:
            return f"[cargo compile error: {len(compile_errors)} error(s)]"

        # Try summary line
        summary_matches = list(self.SUMMARY_RE.finditer(output))
        if summary_matches:
            m = summary_matches[-1]
            passed = int(m.group(1))
            failed = int(m.group(2))
            ignored = int(m.group(3))
            duration_raw = m.group(4)  # e.g. "1.23s"
            duration_val = float(duration_raw.rstrip("s"))

            if failed > 0:
                summary = f"✗ {passed}/{passed+failed} passed"
            else:
                summary = f"✓ {passed} passed"
            if ignored:
                summary += f" ({ignored} ignored)"
            summary += f" in {duration_val:.2f}s"
            return summary

        # Fallback: count test results
        running_count = output.count("test result:")
        ok_count = output.count("ok")
        fail_count = output.count("FAILED")

        if running_count == 0:
            return f"[cargo test output: {len(output)} chars, format unrecognised]"

        summary = f"[{running_count} suites]"
        if fail_count > 0:
            summary += f" {fail_count} failed"
        elif ok_count > 0:
            summary += f" {ok_count} passed/ok"
        if exit_code != 0:
            summary += " [FAILED]"
        else:
            summary = "✓ " + summary

        return summary


# ----------------------------------------------------------------------------
# LS / Tree Compressor
# ----------------------------------------------------------------------------


class LsCompressor(CommandCompressor):
    """
    ls -la and tree output is verbose for large directories.
    Raw: 400-800 tokens for a medium directory. Compressed: entry count + size.
    Typical savings: 70-85%.
    """

    def can_compress(self, command: str, stdout: str, stderr: str) -> bool:
        cmd = _normalize_cmd(command)
        return cmd in ("ls", "ls -la", "ls -l", "ls -1", "tree")

    def compress(
        self, command: str, stdout: str, stderr: str, exit_code: int
    ) -> str:
        if stderr and exit_code != 0:
            return f"[ls error: {stderr.strip()[:100]}]"

        lines = [l for l in stdout.splitlines() if l.strip()]
        if not lines:
            return "[empty directory]"

        total = len(lines)

        # If tree output, count directories vs files
        if command.strip().startswith("tree"):
            dir_count = sum(1 for l in lines if l.startswith("[") and "+-" in l)
            file_count = total - dir_count
            summary = f"[tree: {dir_count} dirs, {file_count} files]"
        else:
            # ls output — count by leading char
            files = sum(1 for l in lines[1:] if not l.startswith("total "))
            summary = f"[ls: {files} entries]"

        return summary


# ----------------------------------------------------------------------------
# Docker PS Compressor
# ----------------------------------------------------------------------------


class DockerPsCompressor(CommandCompressor):
    """
    docker ps output is verbose — one line per container with many columns.
    Raw: 200-600 tokens. Compressed: count + key containers by name/status.
    Typical savings: 70-80%.
    """

    def can_compress(self, command: str, stdout: str, stderr: str) -> bool:
        normalized = _normalize_cmd(command)
        return normalized == "docker" and "ps" in command

    def compress(
        self, command: str, stdout: str, stderr: str, exit_code: int
    ) -> str:
        if stderr and exit_code != 0:
            return f"[docker ps error: {stderr.strip()[:100]}]"

        lines = [l for l in stdout.splitlines() if l.strip()]
        if len(lines) <= 1:
            return "[no containers running]"

        # Parse fixed-width columns from the header line
        header = lines[0]
        # Find column start positions by looking for uppercase header names
        col_positions = {}
        col_names = ["CONTAINER ID", "IMAGE", "STATUS", "PORTS", "NAMES"]
        search_from = 0
        for col in col_names:
            pos = header.find(col, search_from)
            if pos != -1:
                col_positions[col] = pos
                search_from = pos + len(col)

        def get_col(line: str, col_name: str) -> str:
            """Extract a column value from a fixed-width formatted line."""
            if col_name not in col_positions:
                return ""
            start = col_positions[col_name]
            # Next column starts at the next known position, or end of line
            end = len(line)
            for other_col in col_names:
                if other_col != col_name and col_positions.get(other_col, -1) > start:
                    end = col_positions[other_col]
                    break
            return line[start:end].strip()

        data_lines = lines[1:]
        total = len(data_lines)

        running = 0
        exited = 0
        paused = 0
        status_counts: dict[str, int] = {}

        for line in data_lines:
            status = get_col(line, "STATUS")
            if not status:
                continue
            if status.startswith("Up"):
                running += 1
            elif status.startswith("Exited"):
                exited += 1
            elif status.startswith("Paused"):
                paused += 1
            status_counts[status] = status_counts.get(status, 0) + 1

        parts_out = []
        if running > 0:
            parts_out.append(f"{running} running")
        if exited > 0:
            parts_out.append(f"{exited} exited")
        if paused > 0:
            parts_out.append(f"{paused} paused")
        if not parts_out:
            parts_out = [f"{total} containers"]

        return f"[docker: {', '.join(parts_out)}, {total} total]"


# ----------------------------------------------------------------------------
# Ruff Check Compressor
# ----------------------------------------------------------------------------


class RuffCheckCompressor(CommandCompressor):
    """
    ruff check and ruff (when used as linter) output is verbose.
    Raw: hundreds of lines of violations. Compressed: violation counts by severity.
    Typical savings: 80-90%.
    """

    def can_compress(self, command: str, stdout: str, stderr: str) -> bool:
        cmd = _normalize_cmd(command)
        return cmd in ("ruff", "ruff check")

    def compress(
        self, command: str, stdout: str, stderr: str, exit_code: int
    ) -> str:
        output = stdout + "\n" + stderr

        if exit_code == 0:
            return "[ruff: no violations ✓]"

        # Count by prefix (error, warning, info)
        error_count = len(re.findall(r"^\s*E\d+", output, re.MULTILINE))
        warn_count = len(re.findall(r"^\s*W\d+", output, re.MULTILINE))
        info_count = len(re.findall(r"^\s*F\d+", output, re.MULTILINE))

        # Try summary line
        summary_match = re.search(
            r"Found (\d+) error[s]?\s+\((\d+)\s+error[s]?,\s+(\d+)\s+warning[s]?\)",
            output,
        )
        if summary_match:
            total = summary_match.group(1)
            err = summary_match.group(2)
            warn = summary_match.group(3)
            return f"[ruff: {total} violations ({err} error, {warn} warning)]"

        parts = []
        if error_count:
            parts.append(f"{error_count} error")
        if warn_count:
            parts.append(f"{warn_count} warning")
        if info_count:
            parts.append(f"{info_count} info")

        if parts:
            return f"[ruff: {', '.join(parts)}]"
        return f"[ruff: {exit_code} exit, format unrecognised]"


# ----------------------------------------------------------------------------
# Go Test Compressor
# ----------------------------------------------------------------------------


class GoTestCompressor(CommandCompressor):
    """
    go test -v output is verbose. Raw: 500-2000 tokens.
    Compressed: pass/fail count + suite timing.
    Typical savings: 80-95%.
    """

    SUMMARY_RE = re.compile(
        r"^\s*(?:ok|FAIL)\s+(\S+)(?:\s+([\d\.]+)s)?"
        r"(?:\s+\[\s*\d+\s+skipped\s*\])?\s*$",
        re.MULTILINE,
    )
    PASS_RE = re.compile(r"^---\s+PASS:\s+(\S+)", re.MULTILINE)
    FAIL_RE = re.compile(r"^---\s+FAIL:\s+(\S+)", re.MULTILINE)

    def can_compress(self, command: str, stdout: str, stderr: str) -> bool:
        return "go test" in command or "go vet" in command

    def compress(
        self, command: str, stdout: str, stderr: str, exit_code: int
    ) -> str:
        output = stdout + "\n" + stderr

        # Try summary line first (ok/FAIL package time)
        summaries = list(self.SUMMARY_RE.finditer(output))
        if summaries:
            passed = 0
            failed = 0
            total_time = 0.0
            for m in summaries:
                pkg = m.group(1)
                if m.group(0).startswith("ok"):
                    passed += 1
                else:
                    failed += 1
                if m.group(2):
                    total_time += float(m.group(2))

            if failed > 0:
                summary = f"✗ {failed}/{passed+failed} packages failed"
            else:
                summary = f"✓ {passed} packages passed"
            if total_time > 0:
                summary += f" in {total_time:.2f}s"
            return summary

        # Fallback: count PASS/FAIL lines
        pass_count = len(self.PASS_RE.findall(output))
        fail_count = len(self.FAIL_RE.findall(output))

        if pass_count == 0 and fail_count == 0:
            return f"[go test: {len(output)} chars, format unrecognised]"

        if fail_count > 0:
            return f"✗ {fail_count} tests failed, {pass_count} passed"
        return f"✓ {pass_count} tests passed"


# ----------------------------------------------------------------------------
# Docker Logs Compressor
# ----------------------------------------------------------------------------


class DockerLogsCompressor(CommandCompressor):
    """
    docker logs can be enormous — thousands of lines of application output.
    Raw: 5000-50000 tokens. Compressed: line count + first/last line summary.
    Typical savings: 90-99% for long logs.
    """

    def can_compress(self, command: str, stdout: str, stderr: str) -> bool:
        return _normalize_cmd(command) == "docker" and _has_flag(command, "logs")

    def compress(
        self, command: str, stdout: str, stderr: str, exit_code: int
    ) -> str:
        if not stdout.strip():
            return "[docker logs: empty]"

        lines = stdout.splitlines()
        total = len(lines)

        if total <= 3:
            # Even short output may contain errors/warnings — include them
            result = stdout.strip()[:300]
            error_lines = [l for l in lines if "error" in l.lower() or "Error" in l]
            warn_lines = [l for l in lines if "warn" in l.lower() or "WARN" in l]
            if error_lines:
                result += f"\n[{len(error_lines)} error(s)]"
            if warn_lines:
                result += f"\n[{len(warn_lines)} warning(s)]"
            return result

        first = lines[0][:80]
        last = lines[-1][:80]
        error_lines = [l for l in lines if "error" in l.lower() or "Error" in l]
        warn_lines = [l for l in lines if "warn" in l.lower() or "WARN" in l]

        parts = [f"{total} lines", f"first: {first}"]
        if error_lines:
            parts.append(f"{len(error_lines)} errors")
        if warn_lines:
            parts.append(f"{len(warn_lines)} warnings")
        parts.append(f"last: {last}")

        return " | ".join(parts)


# ----------------------------------------------------------------------------
# NPM / Node Test Compressor
# ----------------------------------------------------------------------------


class NpmTestCompressor(CommandCompressor):
    """
    npm test output varies widely by test framework (jest, vitest, mocha, etc.).
    We look for common patterns and summarize.
    """

    PASS_RE = re.compile(r"(?:Tests:|Test Suites:)\s+(\d+)\s+passed", re.MULTILINE)
    FAIL_RE = re.compile(r"(?:Tests:|Test Suites:)\s+(\d+)\s+failed", re.MULTILINE)
    SKIP_RE = re.compile(r"(?:Tests:|Test Suites:)\s+(\d+)\s+skipped", re.MULTILINE)

    def can_compress(self, command: str, stdout: str, stderr: str) -> bool:
        return any(cmd in command for cmd in ("npm test", "npm run test", "yarn test", "pnpm test"))

    def compress(
        self, command: str, stdout: str, stderr: str, exit_code: int
    ) -> str:
        output = stdout + "\n" + stderr

        passed = sum(int(m.group(1)) for m in self.PASS_RE.finditer(output))
        failed = sum(int(m.group(1)) for m in self.FAIL_RE.finditer(output))
        skipped = sum(int(m.group(1)) for m in self.SKIP_RE.finditer(output))

        if passed == 0 and failed == 0 and skipped == 0:
            return f"[npm test: {len(output)} chars, format unrecognised]"

        parts = []
        if passed > 0:
            parts.append(f"{passed} passed")
        if failed > 0:
            parts.append(f"{failed} failed")
        if skipped > 0:
            parts.append(f"{skipped} skipped")

        summary = ", ".join(parts)
        if exit_code != 0:
            summary += " [FAILED]"
        else:
            summary = "✓ " + summary

        return summary


# ----------------------------------------------------------------------------
# Default (catch-all) — returns None so pipeline falls back to raw
# ----------------------------------------------------------------------------


class DefaultCompressor(CommandCompressor):
    """Catches everything else. Returns None so the pipeline skips compression."""

    def can_compress(self, command: str, stdout: str, stderr: str) -> bool:
        return True  # Catch-all

    def compress(
        self, command: str, stdout: str, stderr: str, exit_code: int
    ) -> str:
        return None  # Signal: no compression applied


# ----------------------------------------------------------------------------
# CompressorRegistry
# ----------------------------------------------------------------------------


class CompressorRegistry:
    """
    Maps terminal commands to CommandCompressor instances.
    Lookup order matters — more specific compressors registered first.
    """

    def __init__(self):
        self._compressors: list[tuple[str | None, CommandCompressor]] = []
        self._stats: dict[str, dict[str, int]] = {}

    def register(
        self, pattern: str | None, compressor: CommandCompressor
    ) -> "CompressorRegistry":
        """
        Register a compressor. pattern is a command prefix to match (e.g. 'git')
        or None for the default catch-all.
        """
        self._compressors.append((pattern, compressor))
        return self

    def get_compressor(
        self, command: str, stdout: str, stderr: str, exit_code: int = 0
    ) -> tuple[Optional[CommandCompressor], str]:
        """
        Find the first compressor that can handle this command + output.
        Returns (compressor, compressed_or_none).
        The caller should try compressor.compress() only if can_compress() is True.
        """
        for pattern, compressor in self._compressors:
            if pattern is None:
                continue  # Skip default until last
            if command.startswith(pattern):
                if compressor.can_compress(command, stdout, stderr):
                    try:
                        result = compressor.compress(command, stdout, stderr, exit_code)
                        self._record_stats(command, len(stdout), len(result or ""))
                        return compressor, result or ""
                    except Exception:
                        pass
        # Default catch-all
        for pattern, compressor in reversed(self._compressors):
            if pattern is None:
                if compressor.can_compress(command, stdout, stderr):
                    try:
                        result = compressor.compress(command, stdout, stderr, exit_code)
                        return compressor, result  # May be None
                    except Exception:
                        pass
        return None, None

    def compress(
        self, command: str, stdout: str, stderr: str, exit_code: int
    ) -> str | None:
        """
        Try all registered compressors in order. Return compressed string or None.
        """
        for pattern, compressor in self._compressors:
            if compressor.can_compress(command, stdout, stderr):
                try:
                    result = compressor.compress(command, stdout, stderr, exit_code)
                    if result is not None:
                        self._record_stats(
                            command, len(stdout), len(result)
                        )
                        return result
                except Exception:
                    pass
        return None

    def _record_stats(self, command: str, raw_len: int, compressed_len: int):
        cmd_key = _normalize_cmd(command)
        if cmd_key not in self._stats:
            self._stats[cmd_key] = {
                "count": 0,
                "raw_tokens": 0,
                "compressed_tokens": 0,
            }
        s = self._stats[cmd_key]
        s["count"] += 1
        s["raw_tokens"] += raw_len // 4
        s["compressed_tokens"] += max(1, compressed_len // 4)

    def get_stats(self) -> dict:
        """Return copy of stats dict for hermes stats command."""
        return dict(self._stats)

    def reset_stats(self):
        self._stats.clear()


# -------------------------------------------------------------------
# Built-in registry instance (singleton — imported by pipeline)
# -------------------------------------------------------------------

DEFAULT_COMPRESSORS = CompressorRegistry()

# Order matters — more specific commands first
DEFAULT_COMPRESSORS.register("git", GitStatusCompressor())
DEFAULT_COMPRESSORS.register("git", GitDiffCompressor())
DEFAULT_COMPRESSORS.register("pytest", PytestCompressor())
DEFAULT_COMPRESSORS.register("cargo", CargoTestCompressor())
DEFAULT_COMPRESSORS.register("ls", LsCompressor())
DEFAULT_COMPRESSORS.register("tree", LsCompressor())  # reuses ls
DEFAULT_COMPRESSORS.register("docker", DockerPsCompressor())
DEFAULT_COMPRESSORS.register("docker", DockerLogsCompressor())
DEFAULT_COMPRESSORS.register("ruff", RuffCheckCompressor())
DEFAULT_COMPRESSORS.register("go", GoTestCompressor())
DEFAULT_COMPRESSORS.register("npm", NpmTestCompressor())
DEFAULT_COMPRESSORS.register("yarn", NpmTestCompressor())
DEFAULT_COMPRESSORS.register("pnpm", NpmTestCompressor())

# Default catch-all (always last)
DEFAULT_COMPRESSORS.register(None, DefaultCompressor())


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

_TOKEN_ESTIMATE_RE = re.compile(r"\d+")


def _normalize_cmd(command: str) -> str:
    """Normalize a command string for matching."""
    if not command:
        return ""
    # Remove leading/trailing whitespace
    cmd = command.strip()
    # Remove `sudo ` prefix
    if cmd.startswith("sudo "):
        cmd = cmd[5:]
    # Strip common path
    if "/" in cmd:
        cmd = cmd.split("/")[-1]
    # Remove all flags/args — keep only the base command
    parts = shlex.split(cmd)
    return parts[0] if parts else cmd


def _has_flag(command: str, flag: str) -> bool:
    """Check if a command line contains a specific flag (whole-token match)."""
    parts = shlex.split(command)
    return flag in parts


def _truncate(s: str, max_len: int) -> str:
    """Truncate a string to max_len, adding ellipsis if needed."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


# ----------------------------------------------------------------------------
# Compound Command Splitter
# ----------------------------------------------------------------------------
# Splits compound shell commands (e.g., "cmd1 && cmd2 | cmd3") into segments,
# routes each segment to the appropriate compressor, and joins the results.
# Inspired by RTK's lexer architecture but simplified for Hermes.


class TokenType(Enum):
    """Token types for shell command lexing."""
    ARG = auto()      # Regular argument (e.g., "git", "status", "--short")
    OPERATOR = auto() # Logical operator (&&, ||, ;)
    PIPE = auto()     # Pipe operator (|)
    REDIRECT = auto() # Redirection (>, >>, 2>&1, etc.)


@dataclasses.dataclass
class Token:
    """A lexed token from a shell command."""
    type: TokenType
    value: str


class CompoundCommandSplitter:
    """
    Splits compound shell commands into segments and routes each to the appropriate compressor.
    
    Example:
        Input: "cargo fmt --all && cargo test 2>&1 | tail -20"
        Segments: ["cargo fmt --all", "cargo test 2>&1 | tail -20"]
        
    Usage:
        splitter = CompoundCommandSplitter()
        segments = splitter.split("cargo fmt && cargo test")
        # segments == ["cargo fmt", "cargo test"]
        
        result = splitter.compress_segments(registry, command, stdout, stderr, exit_code)
        # Returns None for compound commands (requires shell hooks for per-segment output)
    """
    
    # Shell operators that split command segments
    OPERATORS = {"&&", "||", ";", "\n"}
    
    def split(self, command: str) -> List[str]:
        """
        Split a compound command into individual command segments.
        
        Handles:
        - Logical operators: &&, ||, ;
        - Pipes: | (pipe chains stay together as one segment)
        - Redirects: >, >>, 2>&1 (preserved with their command)
        - Quoted strings: '...' and "..." (not split)
        
        Safety checks (returns original command unsplit if detected):
        - Subshells: (cmd)
        - Command substitution: $(cmd) or `cmd`
        - These constructs are too complex to split safely without full shell parsing
        
        Returns:
            List of command segment strings, or [command] if splitting is unsafe.
        """
        if not command or not command.strip():
            return []
        
        # Safety: detect constructs we cannot safely split
        if self._has_subshell_or_substitution(command):
            return [command.strip()]
        
        # Early exit: no operators means single segment
        has_operator = any(op in command for op in self.OPERATORS)
        if not has_operator:
            return [command.strip()]
        
        # Tokenize and split
        tokens = self._tokenize(command)
        segments = self._split_tokens(tokens)
        
        # Filter empty segments and strip whitespace
        result = [seg.strip() for seg in segments if seg.strip()]
        return result if result else [command.strip()]
    
    def _has_subshell_or_substitution(self, command: str) -> bool:
        """
        Check if command contains subshells or command substitution.
        
        Detects:
        - Parentheses: ( ... )
        - Command substitution: $( ... ) or ` ... `
        
        Returns True if any of these are found, indicating the command
        should not be split.
        """
        # Check for parentheses (subshell)
        if '(' in command or ')' in command:
            return True
        
        # Check for $() command substitution
        if '$(' in command:
            return True
        
        # Check for backtick command substitution
        if '`' in command:
            return True
        
        return False
    
    def _tokenize(self, command: str) -> List[Token]:
        """
        Tokenize a command string into typed tokens.
        
        Handles:
        - Operators: &&, ||, ;, |
        - Redirects: >, >>, 2>&1, etc.
        - Arguments: everything else (including quoted strings)
        """
        tokens: List[Token] = []
        i = 0
        n = len(command)
        
        while i < n:
            # Skip whitespace
            if command[i].isspace():
                i += 1
                continue
            
            # Check for operators
            if command[i:i+2] in ("&&", "||"):
                tokens.append(Token(TokenType.OPERATOR, command[i:i+2]))
                i += 2
                continue
            
            if command[i] in (";", "|"):
                tok_type = TokenType.OPERATOR if command[i] == ";" else TokenType.PIPE
                tokens.append(Token(tok_type, command[i]))
                i += 1
                continue
            
            # Check for redirects
            if command[i] in (">", "<") or (command[i].isdigit() and i+1 < n and command[i+1] in "><"):
                redirect = self._parse_redirect(command[i:])
                if redirect:
                    tokens.append(Token(TokenType.REDIRECT, redirect))
                    i += len(redirect)
                    continue
            
            # Argument (including quoted strings)
            arg = self._parse_argument(command[i:])
            if arg:
                tokens.append(Token(TokenType.ARG, arg))
                i += len(arg)
                continue
            
            # Unknown character, skip
            i += 1
        
        return tokens
    
    def _parse_redirect(self, text: str) -> str:
        """Parse a redirect token (>, >>, 2>&1, etc.)."""
        # Simple redirects
        if text.startswith("2>&1"):
            return "2>&1"
        if text.startswith(">&"):
            return ">&"
        if text.startswith(">>"):
            return ">>"
        if text.startswith(">"):
            return ">"
        if text.startswith("2>"):
            return "2>"
        if text.startswith("1>"):
            return "1>"
        if text.startswith("&>"):
            return "&>"
        if text.startswith("<"):
            # Simple redirect
            return "<"
        return ""
    
    def _parse_argument(self, text: str) -> str:
        """
        Parse a single argument (including quoted strings).
        
        Handles:
        - Double-quoted strings: "hello world"
        - Single-quoted strings: 'hello world'
        - Escaped characters: hello\\ world
        - Unquoted arguments: simple_arg
        
        Edge cases:
        - Backslash at end of text: returns text as-is (no crash)
        - Unclosed quotes: returns entire text as-is
        """
        if not text:
            return ""
        
        # Quoted string
        if text[0] in ('"', "'"):
            quote = text[0]
            i = 1
            while i < len(text):
                if text[i] == "\\":
                    # Escape sequence: skip next char, but check bounds
                    i += 1
                    if i < len(text):
                        i += 1
                    continue
                if text[i] == quote:
                    return text[:i+1]
                i += 1
            return text  # Unclosed quote, return as-is
        
        # Unquoted argument (stops at operator or redirect)
        i = 0
        while i < len(text):
            char = text[i]
            # Stop at operators and special chars
            if char in "&|;<> \t\n":
                break
            # Handle escape sequences
            if char == "\\":
                i += 1
                if i < len(text):
                    i += 1
                continue
            i += 1
        
        return text[:i] if i > 0 else text[0] if text else ""
    
    def _split_tokens(self, tokens: List[Token]) -> List[str]:
        """Split tokens into command segments based on operators."""
        if not tokens:
            return []
        
        segments = []
        current_segment: List[Token] = []
        
        for token in tokens:
            if token.type == TokenType.OPERATOR:
                # End current segment
                if current_segment:
                    segments.append(" ".join(t.value for t in current_segment))
                    current_segment = []
                # Operator itself is discarded (implicit in segment boundaries)
            elif token.type == TokenType.PIPE:
                # Pipe stays in current segment
                current_segment.append(token)
            else:
                # ARG or REDIRECT
                current_segment.append(token)
        
        # Final segment
        if current_segment:
            segments.append(" ".join(t.value for t in current_segment))
        
        return segments
    
    def compress_segments(
        self,
        registry: "CompressorRegistry",
        command: str,
        stdout: str,
        stderr: str,
        exit_code: int,
    ) -> Optional[str]:
        """
        Attempt to compress a compound command by splitting and compressing segments.
        
        IMPORTANT LIMITATION:
        This function operates on combined stdout/stderr from the entire compound command.
        Without shell-hook-level interception, we cannot get per-segment output.
        Therefore, this function returns None for compound commands, signaling the pipeline
        to fall back to raw output or other compression methods.
        
        The split() method can still be used standalone for command analysis.
        
        Args:
            registry: CompressorRegistry instance
            command: Full compound command string
            stdout: Raw stdout from the entire command
            stderr: Raw stderr from the entire command
            exit_code: Exit code from the entire command
            
        Returns:
            None for compound commands (cannot compress without per-segment output).
            For single-segment commands, delegates to the registry.
        """
        segments = self.split(command)
        
        # Single segment — delegate to registry
        if len(segments) <= 1:
            return None  # Let registry handle single commands normally
        
        # Compound command: we cannot compress per-segment without shell hooks
        # Return None to fall back to raw output
        # This is the correct behavior: better no compression than wrong compression
        return None
