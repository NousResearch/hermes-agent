#!/usr/bin/env python3
"""
V4A Patch Format Parser

Parses the V4A patch format used by codex, cline, and other coding agents.

V4A Format:
    *** Begin Patch
    *** Update File: path/to/file.py
    @@ optional context hint @@
     context line (space prefix)
    -removed line (minus prefix)
    +added line (plus prefix)
    *** Add File: path/to/new.py
    +new file content
    +line 2
    *** Delete File: path/to/old.py
    *** Move File: old/path.py -> new/path.py
    *** End Patch

Usage:
    from tools.patch_parser import parse_v4a_patch, apply_v4a_operations
    
    operations, error = parse_v4a_patch(patch_content)
    if error:
        print(f"Parse error: {error}")
    else:
        result = apply_v4a_operations(operations, file_ops)
"""

import difflib
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
from enum import Enum


class OperationType(Enum):
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    MOVE = "move"


@dataclass
class HunkLine:
    """A single line in a patch hunk."""
    prefix: str  # ' ', '-', or '+'
    content: str


@dataclass
class Hunk:
    """A group of changes within a file."""
    context_hint: Optional[str] = None
    lines: List[HunkLine] = field(default_factory=list)


@dataclass
class PatchOperation:
    """A single operation in a V4A patch."""
    operation: OperationType
    file_path: str
    new_path: Optional[str] = None  # For move operations
    hunks: List[Hunk] = field(default_factory=list)
    content: Optional[str] = None  # For add file operations


def parse_v4a_patch(patch_content: str) -> Tuple[List[PatchOperation], Optional[str]]:
    """
    Parse a V4A format patch.
    
    Args:
        patch_content: The patch text in V4A format
    
    Returns:
        Tuple of (operations, error_message)
        - If successful: (list_of_operations, None)
        - If failed: ([], error_description)
    """
    lines = patch_content.split('\n')
    operations: List[PatchOperation] = []
    
    # Find patch boundaries
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        if '*** Begin Patch' in line or '***Begin Patch' in line:
            start_idx = i
        elif '*** End Patch' in line or '***End Patch' in line:
            end_idx = i
            break
    
    if start_idx is None:
        # Try to parse without explicit begin marker
        start_idx = -1
    
    if end_idx is None:
        end_idx = len(lines)
    
    # Parse operations between boundaries
    i = start_idx + 1
    current_op: Optional[PatchOperation] = None
    current_hunk: Optional[Hunk] = None
    
    while i < end_idx:
        line = lines[i]
        
        # Check for file operation markers
        update_match = re.match(r'\*\*\*\s*Update\s+File:\s*(.+)', line)
        add_match = re.match(r'\*\*\*\s*Add\s+File:\s*(.+)', line)
        delete_match = re.match(r'\*\*\*\s*Delete\s+File:\s*(.+)', line)
        move_match = re.match(r'\*\*\*\s*Move\s+File:\s*(.+?)\s*->\s*(.+)', line)
        
        if update_match:
            # Save previous operation
            if current_op:
                if current_hunk and current_hunk.lines:
                    current_op.hunks.append(current_hunk)
                operations.append(current_op)
            
            current_op = PatchOperation(
                operation=OperationType.UPDATE,
                file_path=update_match.group(1).strip()
            )
            current_hunk = None
            
        elif add_match:
            if current_op:
                if current_hunk and current_hunk.lines:
                    current_op.hunks.append(current_hunk)
                operations.append(current_op)
            
            current_op = PatchOperation(
                operation=OperationType.ADD,
                file_path=add_match.group(1).strip()
            )
            current_hunk = Hunk()
            
        elif delete_match:
            if current_op:
                if current_hunk and current_hunk.lines:
                    current_op.hunks.append(current_hunk)
                operations.append(current_op)
            
            current_op = PatchOperation(
                operation=OperationType.DELETE,
                file_path=delete_match.group(1).strip()
            )
            operations.append(current_op)
            current_op = None
            current_hunk = None
            
        elif move_match:
            if current_op:
                if current_hunk and current_hunk.lines:
                    current_op.hunks.append(current_hunk)
                operations.append(current_op)
            
            current_op = PatchOperation(
                operation=OperationType.MOVE,
                file_path=move_match.group(1).strip(),
                new_path=move_match.group(2).strip()
            )
            operations.append(current_op)
            current_op = None
            current_hunk = None
            
        elif line.startswith('@@'):
            # Context hint / hunk marker
            if current_op:
                if current_hunk and current_hunk.lines:
                    current_op.hunks.append(current_hunk)
                
                # Extract context hint
                hint_match = re.match(r'@@\s*(.+?)\s*@@', line)
                hint = hint_match.group(1) if hint_match else None
                current_hunk = Hunk(context_hint=hint)
                
        elif current_op and line:
            # Parse hunk line
            if current_hunk is None:
                current_hunk = Hunk()
            
            if line.startswith('+'):
                current_hunk.lines.append(HunkLine('+', line[1:]))
            elif line.startswith('-'):
                current_hunk.lines.append(HunkLine('-', line[1:]))
            elif line.startswith(' '):
                current_hunk.lines.append(HunkLine(' ', line[1:]))
            elif line.startswith('\\'):
                # "\ No newline at end of file" marker - skip
                pass
            else:
                # Treat as context line (implicit space prefix)
                current_hunk.lines.append(HunkLine(' ', line))
        
        i += 1
    
    # Don't forget the last operation
    if current_op:
        if current_hunk and current_hunk.lines:
            current_op.hunks.append(current_hunk)
        operations.append(current_op)

    # Validate the parsed result
    if not operations:
        # Empty patch is not an error — callers get [] and can decide
        return operations, None

    parse_errors: List[str] = []
    for op in operations:
        if not op.file_path:
            parse_errors.append("Operation with empty file path")
        if op.operation == OperationType.UPDATE and not op.hunks:
            parse_errors.append(f"UPDATE {op.file_path!r}: no hunks found")
        if op.operation == OperationType.MOVE and not op.new_path:
            parse_errors.append(f"MOVE {op.file_path!r}: missing destination path (expected 'src -> dst')")

    if parse_errors:
        return [], "Parse error: " + "; ".join(parse_errors)

    return operations, None


def _count_occurrences(text: str, pattern: str) -> int:
    """Count non-overlapping occurrences of *pattern* in *text*."""
    count = 0
    start = 0
    while True:
        pos = text.find(pattern, start)
        if pos == -1:
            break
        count += 1
        start = pos + 1
    return count


def _validate_operations(
    operations: List[PatchOperation],
    file_ops: Any,
) -> List[str]:
    """Validate all operations without writing any files.

    Returns a list of error strings; an empty list means all operations
    are valid and the apply phase can proceed safely.

    For UPDATE operations, hunks are simulated in order so that later
    hunks validate against post-earlier-hunk content (matching apply order).
    """
    # Deferred import: breaks the patch_parser ↔ fuzzy_match circular dependency
    from tools.fuzzy_match import fuzzy_find_and_replace

    errors: List[str] = []

    for op in operations:
        if op.operation == OperationType.UPDATE:
            read_result = file_ops.read_file_raw(op.file_path)
            if read_result.error:
                errors.append(f"{op.file_path}: {read_result.error}")
                continue

            simulated = read_result.content
            for hunk in op.hunks:
                search_lines = [l.content for l in hunk.lines if l.prefix in {' ', '-'}]
                if not search_lines:
                    # Addition-only hunk: validate context hint uniqueness
                    if hunk.context_hint:
                        occurrences = _count_occurrences(simulated, hunk.context_hint)
                        if occurrences == 0:
                            errors.append(
                                f"{op.file_path}: addition-only hunk context hint "
                                f"'{hunk.context_hint}' not found"
                            )
                        elif occurrences > 1:
                            errors.append(
                                f"{op.file_path}: addition-only hunk context hint "
                                f"'{hunk.context_hint}' is ambiguous "
                                f"({occurrences} occurrences)"
                            )
                    continue

                search_pattern = '\n'.join(search_lines)
                replace_lines = [l.content for l in hunk.lines if l.prefix in {' ', '+'}]
                replacement = '\n'.join(replace_lines)

                new_simulated, count, _strategy, match_error = fuzzy_find_and_replace(
                    simulated, search_pattern, replacement, replace_all=False
                )
                if count == 0:
                    label = f"'{hunk.context_hint}'" if hunk.context_hint else "(no hint)"
                    msg = (
                        f"{op.file_path}: hunk {label} not found"
                        + (f" — {match_error}" if match_error else "")
                    )
                    try:
                        from tools.fuzzy_match import format_no_match_hint
                        msg += format_no_match_hint(match_error, count, search_pattern, simulated)
                    except Exception:
                        pass
                    errors.append(msg)
                else:
                    # Advance simulation so subsequent hunks validate correctly.
                    # Reuse the result from the call above — no second fuzzy run.
                    simulated = new_simulated

        elif op.operation == OperationType.DELETE:
            read_result = file_ops.read_file_raw(op.file_path)
            if read_result.error:
                errors.append(f"{op.file_path}: file not found for deletion")

        elif op.operation == OperationType.MOVE:
            if not op.new_path:
                errors.append(f"{op.file_path}: MOVE operation missing destination path")
                continue
            src_result = file_ops.read_file_raw(op.file_path)
            if src_result.error:
                errors.append(f"{op.file_path}: source file not found for move")
            dst_result = file_ops.read_file_raw(op.new_path)
            if not dst_result.error:
                errors.append(
                    f"{op.new_path}: destination already exists — move would overwrite"
                )

        # ADD: parent directory creation handled by write_file; no pre-check needed.

    return errors


def apply_v4a_operations(operations: List[PatchOperation],
                          file_ops: Any) -> 'PatchResult':
    """Apply V4A patch operations using a file operations interface.

    Uses a two-phase validate-then-apply approach:
    - Phase 1: validate all operations against current file contents without
      writing anything. If any validation error is found, return immediately
      with no filesystem changes.
    - Phase 2: apply all operations. A failure here (e.g. a race between
      validation and apply) is reported with a note to run ``git diff``.

    Args:
        operations: List of PatchOperation from parse_v4a_patch
        file_ops: Object with read_file_raw, write_file methods

    Returns:
        PatchResult with results of all operations
    """
    # Import here to avoid circular imports
    from tools.file_operations import PatchResult

    # ---- Phase 1: validate ----
    validation_errors = _validate_operations(operations, file_ops)
    if validation_errors:
        return PatchResult(
            success=False,
            error="Patch validation failed (no files were modified):\n"
                  + "\n".join(f"  • {e}" for e in validation_errors),
        )

    # ---- Phase 2: apply ----
    files_modified = []
    files_created = []
    files_deleted = []
    all_diffs = []
    errors = []

    for op in operations:
        try:
            if op.operation == OperationType.ADD:
                result = _apply_add(op, file_ops)
                if result[0]:
                    files_created.append(op.file_path)
                    all_diffs.append(result[1])
                else:
                    errors.append(f"Failed to add {op.file_path}: {result[1]}")

            elif op.operation == OperationType.DELETE:
                result = _apply_delete(op, file_ops)
                if result[0]:
                    files_deleted.append(op.file_path)
                    all_diffs.append(result[1])
                else:
                    errors.append(f"Failed to delete {op.file_path}: {result[1]}")

            elif op.operation == OperationType.MOVE:
                result = _apply_move(op, file_ops)
                if result[0]:
                    files_modified.append(f"{op.file_path} -> {op.new_path}")
                    all_diffs.append(result[1])
                else:
                    errors.append(f"Failed to move {op.file_path}: {result[1]}")

            elif op.operation == OperationType.UPDATE:
                result = _apply_update(op, file_ops)
                if result[0]:
                    files_modified.append(op.file_path)
                    all_diffs.append(result[1])
                else:
                    errors.append(f"Failed to update {op.file_path}: {result[1]}")

        except Exception as e:
            errors.append(f"Error processing {op.file_path}: {str(e)}")

    # Run lint on all modified/created files
    lint_results = {}
    for f in files_modified + files_created:
        if hasattr(file_ops, '_check_lint'):
            lint_result = file_ops._check_lint(f)
            lint_results[f] = lint_result.to_dict()

    combined_diff = '\n'.join(all_diffs)

    if errors:
        return PatchResult(
            success=False,
            diff=combined_diff,
            files_modified=files_modified,
            files_created=files_created,
            files_deleted=files_deleted,
            lint=lint_results if lint_results else None,
            error="Apply phase failed (state may be inconsistent — run `git diff` to assess):\n"
                  + "\n".join(f"  • {e}" for e in errors),
        )

    return PatchResult(
        success=True,
        diff=combined_diff,
        files_modified=files_modified,
        files_created=files_created,
        files_deleted=files_deleted,
        lint=lint_results if lint_results else None,
    )


def _apply_add(op: PatchOperation, file_ops: Any) -> Tuple[bool, str]:
    """Apply an add file operation."""
    # Extract content from hunks (all + lines)
    content_lines = []
    for hunk in op.hunks:
        for line in hunk.lines:
            if line.prefix == '+':
                content_lines.append(line.content)
    
    content = '\n'.join(content_lines)
    
    result = file_ops.write_file(op.file_path, content)
    if result.error:
        return False, result.error
    
    diff = f"--- /dev/null\n+++ b/{op.file_path}\n"
    diff += '\n'.join(f"+{line}" for line in content_lines)
    
    return True, diff


def _apply_delete(op: PatchOperation, file_ops: Any) -> Tuple[bool, str]:
    """Apply a delete file operation."""
    # Read before deleting so we can produce a real unified diff.
    # Validation already confirmed existence; this guards against races.
    read_result = file_ops.read_file_raw(op.file_path)
    if read_result.error:
        return False, f"Cannot delete {op.file_path}: file not found"

    result = file_ops.delete_file(op.file_path)
    if result.error:
        return False, result.error

    removed_lines = read_result.content.splitlines(keepends=True)
    diff = ''.join(difflib.unified_diff(
        removed_lines, [],
        fromfile=f"a/{op.file_path}",
        tofile="/dev/null",
    ))
    return True, diff or f"# Deleted: {op.file_path}"


def _apply_move(op: PatchOperation, file_ops: Any) -> Tuple[bool, str]:
    """Apply a move file operation."""
    result = file_ops.move_file(op.file_path, op.new_path)
    if result.error:
        return False, result.error

    diff = f"# Moved: {op.file_path} -> {op.new_path}"
    return True, diff


def _apply_update(op: PatchOperation, file_ops: Any) -> Tuple[bool, str]:
    """Apply an update file operation."""
    # Deferred import: breaks the patch_parser ↔ fuzzy_match circular dependency
    from tools.fuzzy_match import fuzzy_find_and_replace

    # Read current content — raw so no line-number prefixes or per-line truncation
    read_result = file_ops.read_file_raw(op.file_path)

    if read_result.error:
        return False, f"Cannot read file: {read_result.error}"

    current_content = read_result.content

    # Apply each hunk
    new_content = current_content

    for hunk in op.hunks:
        # Build search pattern from context and removed lines
        search_lines = []
        replace_lines = []

        for line in hunk.lines:
            if line.prefix == ' ':
                search_lines.append(line.content)
                replace_lines.append(line.content)
            elif line.prefix == '-':
                search_lines.append(line.content)
            elif line.prefix == '+':
                replace_lines.append(line.content)

        if search_lines:
            search_pattern = '\n'.join(search_lines)
            replacement = '\n'.join(replace_lines)

            new_content, count, _strategy, error = fuzzy_find_and_replace(
                new_content, search_pattern, replacement, replace_all=False
            )

            if error and count == 0:
                # Try with context hint if available
                if hunk.context_hint:
                    # Find the context hint location and search nearby
                    hint_pos = new_content.find(hunk.context_hint)
                    if hint_pos != -1:
                        # Search in a window around the hint
                        window_start = max(0, hint_pos - 500)
                        window_end = min(len(new_content), hint_pos + 2000)
                        window = new_content[window_start:window_end]

                        window_new, count, _strategy, error = fuzzy_find_and_replace(
                            window, search_pattern, replacement, replace_all=False
                        )
                        
                        if count > 0:
                            new_content = new_content[:window_start] + window_new + new_content[window_end:]
                            error = None
                
                if error:
                    err_msg = f"Could not apply hunk: {error}"
                    try:
                        from tools.fuzzy_match import format_no_match_hint
                        err_msg += format_no_match_hint(error, 0, search_pattern, new_content)
                    except Exception:
                        pass
                    return False, err_msg
        else:
            # Addition-only hunk (no context or removed lines).
            # Insert at the location indicated by the context hint, or at end of file.
            insert_text = '\n'.join(replace_lines)
            if hunk.context_hint:
                occurrences = _count_occurrences(new_content, hunk.context_hint)
                if occurrences == 0:
                    # Hint not found — append at end as a safe fallback
                    new_content = new_content.rstrip('\n') + '\n' + insert_text + '\n'
                elif occurrences > 1:
                    return False, (
                        f"Addition-only hunk: context hint '{hunk.context_hint}' is ambiguous "
                        f"({occurrences} occurrences) — provide a more unique hint"
                    )
                else:
                    hint_pos = new_content.find(hunk.context_hint)
                    # Insert after the line containing the context hint
                    eol = new_content.find('\n', hint_pos)
                    if eol != -1:
                        new_content = new_content[:eol + 1] + insert_text + '\n' + new_content[eol + 1:]
                    else:
                        new_content = new_content + '\n' + insert_text
            else:
                new_content = new_content.rstrip('\n') + '\n' + insert_text + '\n'
    
    # Write new content
    write_result = file_ops.write_file(op.file_path, new_content)
    if write_result.error:
        return False, write_result.error
    
    # Generate diff
    diff_lines = difflib.unified_diff(
        current_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{op.file_path}",
        tofile=f"b/{op.file_path}"
    )
    diff = ''.join(diff_lines)
    
    return True, diff
