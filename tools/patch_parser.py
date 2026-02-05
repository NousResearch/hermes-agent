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
    
    return operations, None


def apply_v4a_operations(operations: List[PatchOperation], 
                          file_ops: Any) -> 'PatchResult':
    """
    Apply V4A patch operations using a file operations interface.
    
    Args:
        operations: List of PatchOperation from parse_v4a_patch
        file_ops: Object with read_file, write_file methods
    
    Returns:
        PatchResult with results of all operations
    """
    # Import here to avoid circular imports
    from tools.file_operations import PatchResult
    
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
            error='; '.join(errors)
        )
    
    return PatchResult(
        success=True,
        diff=combined_diff,
        files_modified=files_modified,
        files_created=files_created,
        files_deleted=files_deleted,
        lint=lint_results if lint_results else None
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
    # Read file first for diff
    read_result = file_ops.read_file(op.file_path)
    
    if read_result.error and "not found" in read_result.error.lower():
        # File doesn't exist, nothing to delete
        return True, f"# {op.file_path} already deleted or doesn't exist"
    
    # Delete by writing empty and then removing
    # Use shell command via the underlying environment
    rm_result = file_ops._exec(f"rm -f {file_ops._escape_shell_arg(op.file_path)}")
    
    if rm_result.exit_code != 0:
        return False, rm_result.stdout
    
    diff = f"--- a/{op.file_path}\n+++ /dev/null\n# File deleted"
    return True, diff


def _apply_move(op: PatchOperation, file_ops: Any) -> Tuple[bool, str]:
    """Apply a move file operation."""
    # Use shell mv command
    mv_result = file_ops._exec(
        f"mv {file_ops._escape_shell_arg(op.file_path)} {file_ops._escape_shell_arg(op.new_path)}"
    )
    
    if mv_result.exit_code != 0:
        return False, mv_result.stdout
    
    diff = f"# Moved: {op.file_path} -> {op.new_path}"
    return True, diff


def _apply_update(op: PatchOperation, file_ops: Any) -> Tuple[bool, str]:
    """Apply an update file operation."""
    # Read current content
    read_result = file_ops.read_file(op.file_path, limit=10000)
    
    if read_result.error:
        return False, f"Cannot read file: {read_result.error}"
    
    # Parse content (remove line numbers)
    current_lines = []
    for line in read_result.content.split('\n'):
        if '|' in line:
            # Line format: "    123|content"
            parts = line.split('|', 1)
            if len(parts) == 2:
                current_lines.append(parts[1])
            else:
                current_lines.append(line)
        else:
            current_lines.append(line)
    
    current_content = '\n'.join(current_lines)
    
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
            
            # Use fuzzy matching
            from tools.fuzzy_match import fuzzy_find_and_replace
            new_content, count, error = fuzzy_find_and_replace(
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
                        
                        window_new, count, error = fuzzy_find_and_replace(
                            window, search_pattern, replacement, replace_all=False
                        )
                        
                        if count > 0:
                            new_content = new_content[:window_start] + window_new + new_content[window_end:]
                            error = None
                
                if error:
                    return False, f"Could not apply hunk: {error}"
    
    # Write new content
    write_result = file_ops.write_file(op.file_path, new_content)
    if write_result.error:
        return False, write_result.error
    
    # Generate diff
    import difflib
    diff_lines = difflib.unified_diff(
        current_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{op.file_path}",
        tofile=f"b/{op.file_path}"
    )
    diff = ''.join(diff_lines)
    
    return True, diff
