#!/usr/bin/env python3
"""Diff-based surgical patching module.

Claude Code-style surgical patching uses unified diffs to compute minimal
changes and detect conflicts, only touching the exact lines that changed.
"""

import difflib
import os
from typing import Optional


class DiffPatcher:
    """
    Claude Code-style surgical patcher.
    Uses unified diffs to compute minimal changes and detect conflicts.
    """

    def compute_diff(self, old_content: str, new_content: str,
                     old_path: str = "a", new_path: str = "b",
                     context_lines: int = 3) -> str:
        """Compute a unified diff between two strings."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        # Ensure last line has newline for proper diff
        if old_lines and not old_lines[-1].endswith('\n'):
            old_lines[-1] += '\n'
        if new_lines and not new_lines[-1].endswith('\n'):
            new_lines[-1] += '\n'

        diff = difflib.unified_diff(
            old_lines, new_lines,
            fromfile=old_path, tofile=new_path,
            n=context_lines
        )
        return ''.join(diff)

    def apply_surgical(self, file_path: str, old_string: str,
                       new_string: str, strict: bool = True) -> dict:
        """
        Apply a surgical patch to a file.

        Strategy:
        1. Find OLD_STRING in the file content
        2. Compute what lines actually change (via diff)
        3. Replace only those lines (not the whole old_string)
        4. Return the result with change statistics

        Returns dict:
          - success: bool
          - applied: bool
          - new_content: str
          - changed_lines: list[int]  # line numbers that changed
          - conflict: bool
          - method: "surgical" | "fallback" | "error"
        """
        result = {
            'success': False,
            'applied': False,
            'new_content': None,
            'changed_lines': [],
            'conflict': False,
            'method': 'error',
            'error': None
        }

        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            result['error'] = f"Failed to read file: {e}"
            return result

        # Find old_string in content
        if old_string not in content:
            result['error'] = f"Could not find old_string in {file_path}"
            return result

        # Detect ambiguity - check for multiple occurrences
        matches = self.detect_ambiguous(old_string, content, max_matches=1)
        if len(matches) > 1:
            result['conflict'] = True
            result['error'] = f"Ambiguous edit: old_string appears {len(matches)} times"
            result['method'] = 'error'
            return result

        # Compute diff to find which lines actually changed
        diff_lines = self.compute_diff(old_string, new_string)
        changed_line_nums = self._get_changed_line_numbers(diff_lines, old_string)

        if changed_line_nums and not strict:
            # In non-strict mode, fall back to simple replacement
            result['method'] = 'fallback'
            new_content = content.replace(old_string, new_string, 1)
        elif changed_line_nums:
            # Surgical mode: replace only the specific lines that changed
            result['method'] = 'surgical'
            new_content = self._surgical_replace(
                content, old_string, new_string, changed_line_nums
            )
        else:
            # No lines changed (identical)
            result['method'] = 'surgical'
            new_content = content

        # Write back
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        except Exception as e:
            result['error'] = f"Failed to write file: {e}"
            return result

        result['success'] = True
        result['applied'] = True
        result['new_content'] = new_content
        result['changed_lines'] = changed_line_nums

        return result

    def detect_ambiguous(self, old_string: str, content: str,
                         max_matches: int = 1) -> list[int]:
        """
        Find all positions where old_string matches.
        If >1 match, the edit is ambiguous (could affect wrong location).
        Returns list of starting line numbers.
        """
        matches = []
        content_lines = content.splitlines(keepends=True)
        old_lines = old_string.splitlines(keepends=True)

        if not old_lines:
            return matches

        # Find all occurrences of old_string
        start = 0
        while True:
            idx = content.find(old_string, start)
            if idx == -1:
                break

            # Calculate line number (1-indexed)
            line_num = content[:idx].count('\n') + 1
            matches.append(line_num)
            start = idx + 1

        return matches

    def preview_diff(self, old_string: str, new_string: str,
                     file_path: str = None) -> str:
        """
        Return a colored/preview diff string showing what WOULD change.
        Useful for the model to see before applying.
        """
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if old_string not in content:
                    return f"Error: old_string not found in {file_path}"
            except Exception as e:
                return f"Error reading file: {e}"

        diff = self.compute_diff(old_string, new_string)
        if not diff:
            return "No changes (old_string and new_string are identical)"

        return diff

    def _get_changed_line_numbers(self, diff_output: str, old_string: str) -> list[int]:
        """
        Parse unified diff output to find which line numbers changed.
        Returns list of 1-indexed line numbers that would change.
        """
        changed_lines = []
        old_lines = old_string.splitlines(keepends=True)

        # Parse hunk headers to get line number info
        # Format: @@ -start,count +start,count @@
        import re

        # Track offset as we process hunks
        current_old_line = 0

        for line in diff_output.splitlines(keepends=True):
            if line.startswith('@@'):
                # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
                m = re.match(r'^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@', line)
                if m:
                    old_count = int(m.group(1)) if m.group(1) else 1
                    current_old_line = 0  # Will be set properly on first '-'
            elif line.startswith('-') and not line.startswith('---'):
                # Line removed from old
                current_old_line += 1
                changed_lines.append(current_old_line)
            elif line.startswith('+') and not line.startswith('+++'):
                # Line added in new (doesn't consume from old)
                pass
            elif not line.startswith('+') and not line.startswith('-') \
                 and not line.startswith('\\'):
                # Context line
                current_old_line += 1

        # If we couldn't parse proper line numbers, fall back to finding
        # which lines differ between old and new
        if not changed_lines:
            changed_lines = self._find_changed_lines_simple(old_string)

        return changed_lines

    def _find_changed_lines_simple(self, old_string: str) -> list[int]:
        """Fallback: find line numbers of all changed lines."""
        old_lines = old_string.splitlines()
        return list(range(1, len(old_lines) + 1))

    def _surgical_replace(self, content: str, old_string: str,
                          new_string: str, changed_line_nums: list[int]) -> str:
        """
        Replace only specific lines that changed.
        This preserves surrounding context exactly as it was.
        """
        old_lines = old_string.splitlines(keepends=True)
        new_lines = new_string.splitlines(keepends=True)

        # Find the position of old_string in content
        start_idx = content.find(old_string)
        if start_idx == -1:
            # Fallback to simple replace
            return content.replace(old_string, new_string, 1)

        end_idx = start_idx + len(old_string)

        # Calculate the starting line number in the file (0-indexed)
        start_line = content[:start_idx].count('\n')

        # Build the replacement lines
        result_lines = []
        old_idx = 0
        new_idx = 0

        for i, old_line in enumerate(old_lines):
            # Actual line number in the file (1-indexed for comparison with changed_line_nums)
            actual_line_num = start_line + i + 1  # +1 because changed_line_nums are 1-indexed

            # Check if this line changed: compare actual position within the block
            # changed_line_nums are 1-indexed positions within old_string (1, 2, 3...)
            if actual_line_num in changed_line_nums or (actual_line_num - start_line) in changed_line_nums:
                # This line changed - use corresponding new line
                if new_idx < len(new_lines):
                    result_line = new_lines[new_idx]
                    new_idx += 1
                else:
                    result_line = ''
                # Always consume the old line when changed, to keep indices aligned
                old_idx += 1
            else:
                # Unchanged line - use old
                result_line = old_lines[old_idx] if old_idx < len(old_lines) else ''
                old_idx += 1

            result_lines.append(result_line)

        # Handle any remaining new lines (lines added at the end of the block)
        while new_idx < len(new_lines):
            result_lines.append(new_lines[new_idx])
            new_idx += 1

        replacement = ''.join(result_lines)

        # If the replacement is empty or unchanged, fall back to standard replace
        if not replacement or (old_string in content and replacement == old_string):
            return content.replace(old_string, new_string, 1)

        # Reconstruct full content with surrounding context
        result = content[:start_idx] + replacement + content[end_idx:]

        return result
