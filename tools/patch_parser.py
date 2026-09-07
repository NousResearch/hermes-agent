"""V4A patch parser/applier (codex, cline). ``*** Begin Patch``/``*** End Patch`` wrap ops:
``*** Update File: p`` + hunks (``@@ hint @@``, `` ctx``, ``-old``, ``+new``); ``*** Add File: n``
+ ``+`` lines; ``*** Delete File: o``; ``*** Move File: a -> b``. Entry points:
``parse_v4a_patch(text) -> (ops, error)`` and ``apply_v4a_operations(ops, file_ops)``."""

import contextlib
import difflib
import inspect
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class OperationType(Enum):
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    MOVE = "move"


@dataclass
class HunkLine:
    prefix: str  # ' ', '-', or '+'
    content: str


@dataclass
class Hunk:
    context_hint: Optional[str] = None
    lines: List[HunkLine] = field(default_factory=list)


@dataclass
class PatchOperation:
    operation: OperationType
    file_path: str
    new_path: Optional[str] = None  # MOVE only
    hunks: List[Hunk] = field(default_factory=list)


# Markers must occupy the whole line at column 0 so content lines that merely
# mention the format ("+*** End Patch") can't truncate or reset the patch.
_BEGIN_MARKER = re.compile(r'^\*\*\*\s*Begin\s+Patch\s*$')
_END_MARKER = re.compile(r'^\*\*\*\s*End\s+Patch\s*$')
_OP_MARKERS: List[Tuple[OperationType, re.Pattern]] = [
    (OperationType.UPDATE, re.compile(r'\*\*\*\s*Update\s+File:\s*(.+)')),
    (OperationType.ADD, re.compile(r'\*\*\*\s*Add\s+File:\s*(.+)')),
    (OperationType.DELETE, re.compile(r'\*\*\*\s*Delete\s+File:\s*(.+)')),
    (OperationType.MOVE, re.compile(r'\*\*\*\s*Move\s+File:\s*(.+?)\s*->\s*(.+)'))]
_HINT_RE = re.compile(r'@@\s*(.+?)\s*@@')


def parse_v4a_patch(patch_content: str) -> Tuple[List[PatchOperation], Optional[str]]:
    """-> ``(operations, None)`` (empty patch = ``[]``, no error) or ``([], "Parse error: …")``."""
    # Tolerate CRLF: a stray ``\r`` would land in every HunkLine.content and defeat the markers.
    lines = [ln[:-1] if ln.endswith('\r') else ln for ln in patch_content.split('\n')]
    start_idx = -1  # parse from the top when no Begin marker is present
    end_idx = len(lines)
    for i, line in enumerate(lines):
        if _BEGIN_MARKER.match(line):
            start_idx = i
        elif _END_MARKER.match(line):
            end_idx = i
            break
    operations: List[PatchOperation] = []
    current_op: Optional[PatchOperation] = None
    current_hunk: Optional[Hunk] = None

    def _flush_hunk() -> None:
        if current_op and current_hunk and current_hunk.lines:
            current_op.hunks.append(current_hunk)

    def _flush() -> None:
        if current_op:
            _flush_hunk()
            operations.append(current_op)

    for line in lines[start_idx + 1:end_idx]:
        op_match = next(((kind, m) for kind, rx in _OP_MARKERS if (m := rx.match(line))), None)
        if op_match:
            kind, m = op_match
            _flush()
            current_op = PatchOperation(
                operation=kind,
                file_path=m.group(1).strip(),
                new_path=m.group(2).strip() if kind is OperationType.MOVE else None)
            # UPDATE hunks start lazily ('@@' or first hunk line); ADD collects all '+' lines
            # into one hunk; DELETE/MOVE are complete.
            current_hunk = Hunk() if kind is OperationType.ADD else None
            if kind in (OperationType.DELETE, OperationType.MOVE):
                operations.append(current_op)
                current_op = None
        elif line.startswith('@@'):
            if current_op:
                _flush_hunk()
                hint_match = _HINT_RE.match(line)
                current_hunk = Hunk(context_hint=hint_match.group(1) if hint_match else None)
        elif current_op and line:
            if current_hunk is None:
                current_hunk = Hunk()
            if line[0] in '+- ':
                current_hunk.lines.append(HunkLine(line[0], line[1:]))
            elif line[0] != '\\':  # "\ No newline at end of file" marker is skipped
                current_hunk.lines.append(HunkLine(' ', line))  # implicit context line
    _flush()
    parse_errors: List[str] = []
    for op in operations:
        if not op.file_path:
            parse_errors.append("Operation with empty file path")
        if op.operation is OperationType.UPDATE and not op.hunks:
            parse_errors.append(f"UPDATE {op.file_path!r}: no hunks found")
        if op.operation is OperationType.MOVE and not op.new_path:
            parse_errors.append(
                f"MOVE {op.file_path!r}: missing destination path (expected 'src -> dst')")
    return ([], "Parse error: " + "; ".join(parse_errors)) if parse_errors else (operations, None)


def _count_occurrences(text: str, pattern: str) -> int:
    """Count occurrences of *pattern* in *text*, advancing one char per hit (overlaps count)."""
    return sum(1 for i in range(len(text) + 1) if text.startswith(pattern, i))


def _split_hunk(hunk: Hunk) -> Tuple[List[str], List[str]]:
    """``(search_lines, replace_lines)``: context+removed vs context+added."""
    return ([l.content for l in hunk.lines if l.prefix != '+'],
            [l.content for l in hunk.lines if l.prefix != '-'])


def _no_match_hint(error: Optional[str], search_pattern: str, content: str) -> str:
    """Best-effort 'Did you mean...' suffix; never lets a hint failure mask the real error."""
    with contextlib.suppress(Exception):
        from tools.fuzzy_match import format_no_match_hint
        return format_no_match_hint(error, 0, search_pattern, content)
    return ""


def _hint_ambiguity(content: str, hint: str, tail: str = "") -> Tuple[int, str]:
    """(occurrences, error) for an addition-only hunk's context hint; error is '' when unique."""
    n = _count_occurrences(content, hint)
    return n, f"context hint '{hint}' is ambiguous ({n} occurrences){tail}" if n > 1 else ""


def _validate_operations(operations: List[PatchOperation], file_ops: Any) -> List[str]:
    """Dry-run every operation -> error strings (empty = safe). UPDATE hunks are simulated in
    order so later hunks see post-earlier-hunk content, exactly as apply will."""
    from tools.fuzzy_match import fuzzy_find_and_replace, is_already_applied
    errors: List[str] = []
    real_change_count = 0
    # Overlay so inter-op state validates (a MOVE creating the path a later UPDATE targets).
    pending_content: dict = {}
    removed_paths: set = set()

    def _read(path: str) -> Tuple[Optional[str], Optional[str]]:
        if path in pending_content:
            return pending_content[path], None
        if path in removed_paths:
            return None, "file not found"
        r = file_ops.read_file_raw(path)
        return (None, r.error) if r.error else (r.content, None)

    def _validate_update(op: PatchOperation) -> None:
        nonlocal real_change_count
        simulated, read_err = _read(op.file_path)
        if read_err:
            errors.append(f"{op.file_path}: {read_err}")
            return
        for hunk_index, hunk in enumerate(op.hunks, start=1):
            search_lines, replace_lines = _split_hunk(hunk)
            if search_lines == replace_lines:
                # Context-only anchor hunks (models emit these between changes) are inert; identical
                # -/+ lines are skipped by apply as a no-op — neither may fail validation.
                real_change_count += any(l.prefix in '-+' for l in hunk.lines)
                continue
            real_change_count += 1
            if not search_lines:  # addition-only: the context hint must be unique
                if hunk.context_hint:
                    occurrences, ambiguous = _hint_ambiguity(simulated, hunk.context_hint)
                    if occurrences == 0:
                        errors.append(f"{op.file_path}: addition-only hunk context hint "
                                      f"'{hunk.context_hint}' not found")
                    elif ambiguous:
                        errors.append(f"{op.file_path}: addition-only hunk {ambiguous}")
                continue
            search_pattern, replacement = '\n'.join(search_lines), '\n'.join(replace_lines)
            new_simulated, count, _strategy, match_error = fuzzy_find_and_replace(
                simulated, search_pattern, replacement, replace_all=False)
            if count:
                simulated = new_simulated
            elif not is_already_applied(simulated or "", search_pattern, replacement):
                # Already-applied hunks are no-ops (apply performs the same skip).
                label = f"'{hunk.context_hint}'" if hunk.context_hint else "(no hint)"
                errors.append(
                    f"{op.file_path}: hunk {hunk_index} {label} not found"
                    + (f" — {match_error}" if match_error else "")
                    + _no_match_hint(match_error, search_pattern, simulated))
        pending_content[op.file_path] = simulated

    def _remove(path: str) -> None:
        removed_paths.add(path)
        pending_content.pop(path, None)

    for op in operations:
        if op.operation == OperationType.UPDATE:
            _validate_update(op)
            continue
        real_change_count += 1
        if op.operation == OperationType.DELETE:
            if _read(op.file_path)[1]:
                errors.append(f"{op.file_path}: file not found for deletion")
            else:
                _remove(op.file_path)
        elif op.operation == OperationType.MOVE:
            if not op.new_path:
                errors.append(f"{op.file_path}: MOVE operation missing destination path")
                continue
            src_content, src_err = _read(op.file_path)
            if src_err:
                errors.append(f"{op.file_path}: source file not found for move")
            if not _read(op.new_path)[1]:
                errors.append(f"{op.new_path}: destination already exists — move would overwrite")
            elif not src_err:  # only a cleanly-validated move updates the overlay
                pending_content[op.new_path] = src_content if src_content is not None else ""
                _remove(op.file_path)
        # ADD: write_file creates parent directories; no pre-check needed.
    if not errors and real_change_count == 0:
        errors.append("Patch contains no changes (only context lines were provided)")
    return errors


# Every _apply_* returns (success, diff_or_error, lsp_diagnostics, lint_result).
ApplyResult = Tuple[bool, str, Optional[str], Optional[dict]]


def _fail(error: str) -> ApplyResult:
    return False, error, None, None


def _written(result: Any, diff: str) -> ApplyResult:
    """Outcome of a write: its error, else success with LSP/lint propagated from the WriteResult."""
    if result.error:
        return _fail(result.error)
    return True, diff, getattr(result, "lsp_diagnostics", None), getattr(result, "lint", None)


def _unified_diff(path: str, old: str, new: Optional[str]) -> str:
    """Unified diff ``a/path`` -> ``b/path`` (``new=None`` = deletion, ``/dev/null``)."""
    return ''.join(difflib.unified_diff(
        old.splitlines(keepends=True), [] if new is None else new.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile="/dev/null" if new is None else f"b/{path}"))


def apply_v4a_operations(operations: List[PatchOperation], file_ops: Any) -> 'PatchResult':
    """Two-phase: validate everything, then apply (atomic on validation failure). A phase-2
    failure (validate/apply race) carries a ``git diff`` note since state may be inconsistent.
    ``file_ops`` needs read_file_raw/write_file/delete_file/move_file."""
    from tools.file_operations_common import PatchResult  # avoid circular import

    def _bullets(errs: List[str]) -> str:
        return "\n".join(f"  • {e}" for e in errs)

    if errors := _validate_operations(operations, file_ops):
        return PatchResult(
            success=False,
            error="Patch validation failed (no files were modified):\n" + _bullets(errors))
    files: Dict[str, List[str]] = {"created": [], "deleted": [], "modified": []}
    all_diffs: List[str] = []
    # V4A bypasses write_file's WriteResult plumbing: LSP diagnostics and lint propagate per file.
    lsp_blocks: List[str] = []
    lint_results: Dict[str, dict] = {}
    for op in operations:
        handler, verb, bucket = _APPLY_DISPATCH[op.operation]
        try:
            ok, payload, lsp, lint = handler(op, file_ops)
        except Exception as e:
            ok, payload = None, str(e)
        if not ok:
            prefix = f"Failed to {verb}" if ok is False else "Error processing"
            errors.append(f"{prefix} {op.file_path}: {payload}")
            continue
        is_move = op.operation is OperationType.MOVE
        files[bucket].append(f"{op.file_path} -> {op.new_path}" if is_move else op.file_path)
        all_diffs.append(payload)
        if lsp:
            lsp_blocks.append(lsp)
        if lint:
            lint_results[op.file_path] = lint
    # Each LSP block carries its own <diagnostics file="..."> header; joining keeps attribution.
    return PatchResult(
        success=not errors,
        error=("Apply phase failed (state may be inconsistent — run `git diff` to assess):\n"
               + _bullets(errors)) if errors else None,
        diff='\n'.join(all_diffs),
        files_modified=files["modified"], files_created=files["created"], files_deleted=files["deleted"],
        lint=lint_results or None, lsp_diagnostics="\n\n".join(lsp_blocks) or None)


def _write_file_accepts_pre_content(file_ops: Any) -> bool:
    """Whether ``file_ops.write_file`` accepts ``pre_content`` — read from the signature, not by
    catching TypeError around the call, so a TypeError raised *inside* it can't double-write."""
    try:
        params = inspect.signature(file_ops.write_file).parameters
    except (TypeError, ValueError):
        return False
    return "pre_content" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _apply_add(op: PatchOperation, file_ops: Any) -> ApplyResult:
    """Create a file from the hunks' '+' lines."""
    content_lines = [line.content for hunk in op.hunks for line in hunk.lines if line.prefix == '+']
    result = file_ops.write_file(op.file_path, '\n'.join(content_lines))
    diff = f"--- /dev/null\n+++ b/{op.file_path}\n" + '\n'.join(f"+{line}" for line in content_lines)
    return _written(result, diff)


def _apply_delete(op: PatchOperation, file_ops: Any) -> ApplyResult:
    """Delete a file, producing a real unified diff of the removed content."""
    read_result = file_ops.read_file_raw(op.file_path)  # re-read guards validate/apply races
    if read_result.error:
        return _fail(f"Cannot delete {op.file_path}: file not found")
    result = file_ops.delete_file(op.file_path)
    diff = _unified_diff(op.file_path, read_result.content, None) or f"# Deleted: {op.file_path}"
    return _fail(result.error) if result.error else (True, diff, None, None)


def _apply_move(op: PatchOperation, file_ops: Any) -> ApplyResult:
    result = file_ops.move_file(op.file_path, op.new_path)
    return _fail(result.error) if result.error else (
        True, f"# Moved: {op.file_path} -> {op.new_path}", None, None)


def _insert_addition_only(new_content: str, hunk: Hunk, insert_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Place an addition-only hunk after its context hint (or at EOF). Returns (content, error)."""
    if hunk.context_hint:
        occurrences, ambiguous = _hint_ambiguity(
            new_content, hunk.context_hint, " — provide a more unique hint")
        if ambiguous:
            return None, f"Addition-only hunk: {ambiguous}"
        if occurrences == 1:
            eol = new_content.find('\n', new_content.find(hunk.context_hint))
            if eol == -1:
                return new_content + '\n' + insert_text, None
            return new_content[:eol + 1] + insert_text + '\n' + new_content[eol + 1:], None
    # No hint / hint not found — append at end as a safe fallback.
    return new_content.rstrip('\n') + '\n' + insert_text + '\n', None


def _apply_update(op: PatchOperation, file_ops: Any) -> ApplyResult:
    """Apply each hunk via fuzzy replace, then write once."""
    from tools.fuzzy_match import fuzzy_find_and_replace, is_already_applied
    read_result = file_ops.read_file_raw(op.file_path)  # raw: no line numbers / truncation
    if read_result.error:
        return _fail(f"Cannot read file: {read_result.error}")
    current_content = new_content = read_result.content
    for hunk in op.hunks:
        search_lines, replace_lines = _split_hunk(hunk)
        if search_lines and search_lines == replace_lines:
            continue
        search_pattern, replacement = '\n'.join(search_lines), '\n'.join(replace_lines)
        if not search_lines:
            new_content, err = _insert_addition_only(new_content, hunk, replacement)
            if err:
                return _fail(err)
            continue
        new_content, count, _strategy, error = fuzzy_find_and_replace(
            new_content, search_pattern, replacement, replace_all=False)
        if not (error and count == 0):
            continue
        # Retry inside a window around the context hint, if any.
        hint_pos = new_content.find(hunk.context_hint) if hunk.context_hint else -1
        if hint_pos != -1:
            window_start = max(0, hint_pos - 500)
            window_end = min(len(new_content), hint_pos + 2000)
            window_new, count, _strategy, error = fuzzy_find_and_replace(
                new_content[window_start:window_end], search_pattern, replacement, replace_all=False)
            if count > 0:
                new_content = new_content[:window_start] + window_new + new_content[window_end:]
                error = None
        if error:
            # Mirror validation's already-applied skip, else the two phases disagree and fail here.
            if is_already_applied(new_content, search_pattern, replacement):
                continue
            hint = _no_match_hint(error, search_pattern, new_content)
            return _fail(f"Could not apply hunk: {error}" + hint)
    # Pass pre_content to skip a redundant re-read inside write_file when supported.
    extra = {"pre_content": current_content} if _write_file_accepts_pre_content(file_ops) else {}
    write_result = file_ops.write_file(op.file_path, new_content, **extra)
    return _written(write_result, _unified_diff(op.file_path, current_content, new_content))


# operation -> (handler, verb for error text, files_* bucket)
_APPLY_DISPATCH: Dict[OperationType, Tuple[Callable[[PatchOperation, Any], ApplyResult], str, str]] = {
    OperationType.ADD: (_apply_add, "add", "created"),
    OperationType.DELETE: (_apply_delete, "delete", "deleted"),
    OperationType.MOVE: (_apply_move, "move", "modified"),
    OperationType.UPDATE: (_apply_update, "update", "modified"),
}
