"""V4A patch parser/applier (codex, cline). ``*** Begin Patch``/``*** End Patch`` wrap ops:
``*** Update File: p`` + hunks (``@@ hint @@``, `` ctx``, ``-old``, ``+new``); ``*** Add File: n``
+ ``+`` lines; ``*** Delete File: o``; ``*** Move File: a -> b``. Entry points:
``parse_v4a_patch(text) -> (ops, error)`` and ``apply_v4a_operations(ops, file_ops)``."""

import difflib
import inspect
import ntpath
import posixpath
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple, Optional, Tuple

if TYPE_CHECKING:  # annotations only; the real import is per-call in apply_v4a_operations
    from tools.file_operations_common import PatchResult

from tools.file_operations_common import PatchResult


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
    lines: list[HunkLine] = field(default_factory=list)


@dataclass
class PatchOperation:
    operation: OperationType
    file_path: str
    new_path: Optional[str] = None  # MOVE only
    hunks: list[Hunk] = field(default_factory=list)


# Markers must occupy the whole line at column 0 so content lines that merely
# mention the format ("+*** End Patch") can't truncate or reset the patch.
_BEGIN_MARKER = re.compile(r'^\*\*\*\s*Begin\s+Patch\s*$')
_END_MARKER = re.compile(r'^\*\*\*\s*End\s+Patch\s*$')
_OP_MARKERS: list[tuple[OperationType, re.Pattern]] = [
    (OperationType.UPDATE, re.compile(r'\*\*\*\s*Update\s+File:\s*(.+)')),
    (OperationType.ADD, re.compile(r'\*\*\*\s*Add\s+File:\s*(.+)')),
    (OperationType.DELETE, re.compile(r'\*\*\*\s*Delete\s+File:\s*(.+)')),
    (OperationType.MOVE, re.compile(r'\*\*\*\s*Move\s+File:\s*(.+?)\s*->\s*(.+)'))]
_HINT_RE = re.compile(r'@@\s*(.+?)\s*@@')


def parse_v4a_patch(patch_content: str) -> tuple[list[PatchOperation], Optional[str]]:
    """-> ``(operations, None)`` (empty patch = ``[]``, no error) or ``([], "Parse error: …")``."""
    # Tolerate CRLF: a stray ``\r`` would land in every HunkLine.content and defeat the markers.
    lines = [ln.removesuffix('\r') for ln in patch_content.split('\n')]
    start_idx = -1  # parse from the top when no Begin marker is present
    end_idx = len(lines)
    for i, line in enumerate(lines):
        if _BEGIN_MARKER.match(line):
            start_idx = i
        elif _END_MARKER.match(line):
            end_idx = i
            break
    operations: list[PatchOperation] = []
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
    parse_errors: list[str] = []
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


# Preserve the existing hint window; its boundary remains an open specification.
_HINT_WINDOW_BEFORE, _HINT_WINDOW_AFTER = 500, 2000


class _Plan(NamedTuple):
    """An Update's hunks applied in memory: ``notes`` are the escape-drift corrections made,
    ``strategies`` the fuzzy strategies that located the edits (for the boundary note)."""
    content: str
    errors: list[str]
    changes: int
    notes: list[str]
    strategies: list[str]


def _plan_hunks(content: str, hunks: list[Hunk]) -> _Plan:
    """Use the same provenance planner for direct callers and patch application."""
    return _plan_file(content, hunks)


def _overlay_key(path: str, file_ops: Any) -> str:
    """One validation-overlay key per backend file, however the patch spells it.

    Lexical only (the parser cannot resolve paths on a remote backend): ``x``, ``./x``,
    ``a//x`` and ``<cwd>/x`` share a key, so a second op on an alias validates against the
    first op's simulated result instead of the stale on-disk file. Never merge two real files:
    on POSIX ``..`` is kept, because the kernel resolves it AFTER following symlinks
    (``sub/link/../f`` is ``real/f`` when ``link`` points into ``real/``); Windows collapses
    ``..`` lexically and is case-insensitive, so only a Windows backend gets ntpath folding.
    Keys the path the backend opens: the backend's own ``_expand_path`` (``~`` -> its HOME, a
    host path under a docker workspace mount -> the container path) runs first. A trailing
    ``/`` or ``/.`` needs a directory, so it keeps a marker and never shares the file's key.
    """
    env = getattr(file_ops, "env", None)
    cwd = getattr(env, "cwd", None) or getattr(file_ops, "cwd", None)
    cwd = cwd if isinstance(cwd, str) else ""
    path = path.strip()
    expand = getattr(file_ops, "_expand_path", None)
    expanded = expand(path) if callable(expand) else None
    path = expanded if isinstance(expanded, str) and expanded else path
    if _backend_is_windows(env, cwd):
        tail = "\\" if re.search(r"[\\/]\.?[\\/]*$", path) else ""
        return ntpath.normcase(ntpath.normpath(ntpath.join(cwd, path))) + tail
    raw = posixpath.join(cwd, path) if cwd.startswith("/") else path
    segments = [seg for seg in raw.split("/") if seg not in ("", ".")]
    tail = "/" if segments and raw.split("/")[-1] in ("", ".") else ""
    return ("/" if raw.startswith("/") else "") + "/".join(segments) + tail


def _backend_is_windows(env: Any, cwd: str) -> bool:
    """A local env on a Windows host, or a backend whose cwd is a drive path. Remote
    sandboxes (docker, ssh, modal ...) are POSIX shells even when Hermes runs on Windows."""
    from tools.environments.local import _IS_WINDOWS
    return bool(_IS_WINDOWS and getattr(env, "is_local", False)) or bool(
        re.match(r"^[A-Za-z]:[\\/]", cwd))


class _Commit(NamedTuple):
    """One mutation whose bytes were chosen before any write. Commit must not replan it."""
    op: PatchOperation
    snapshot: Optional[str]
    content: Optional[str]
    read_sha: Optional[str]
    notes: tuple[str, ...]
    strategies: tuple[str, ...]


class _Snap(NamedTuple):
    """Disk observation the plan depended on. Re-read this, do not select against a new read."""
    path: str
    content: Optional[str]
    absent: bool
    sha: Optional[str]


class _Resolution(NamedTuple):
    errors: list[str]
    commits: list[_Commit]
    snaps: dict[str, _Snap]


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value]


def _redact_planner_errors(errors: list) -> list[str]:
    """Drop source excerpts an untrusted planner may echo. Keep the diagnostic shape."""
    return [
        re.sub(r"(?m)^  L(\d+):[^\n]*", r"  L\1: [source omitted]", str(error))
        for error in errors
    ]


def _plan_file(content: Optional[str], hunks: list[Hunk], state=None, *, finalize=True) -> _Plan:
    """Run the provenance planner once. A raised or unusable result is a refusal, not a write."""
    from tools.v4a_planner import plan_hunks
    if not isinstance(content, str):
        return _Plan("", ["hunk planning failed"], 0, [], [])
    try:
        raw = plan_hunks(content, hunks, state, finalize=finalize)
    except Exception:  # health: allow BLE001 -- planner boundary: fail closed with a fixed code; exception text can hold source
        return _Plan(content, ["hunk planning failed"], 0, [], [])
    planned = getattr(raw, "content", None)
    errors = getattr(raw, "errors", None)
    changes = getattr(raw, "changes", None)
    if not isinstance(planned, str) or not isinstance(errors, list) or type(changes) is not int:
        return _Plan(content, ["hunk planning failed"], 0, [], [])
    return _Plan(
        planned, _redact_planner_errors(errors), changes,
        _as_str_list(getattr(raw, "notes", ())),
        _as_str_list(getattr(raw, "strategies", ())))


def _note_disk_read(snaps: dict[str, _Snap], key: str, path: str, result: Any) -> None:
    """Record the first disk observation for ``key``. Overlay hits must not call this."""
    if key in snaps:
        return
    if getattr(result, "error", None):
        if getattr(result, "not_found", False):
            snaps[key] = _Snap(path, None, True, None)
        return
    snaps[key] = _Snap(path, result.content, False, getattr(result, "_content_sha256", None))


def _remember_update(commits: list[_Commit], op: PatchOperation, snapshot: Optional[str],
                     sha: Optional[str], plan: _Plan) -> None:
    """Keep a successful Update's planned bytes. A refusal is not a commit."""
    if plan.errors or not isinstance(snapshot, str) or not isinstance(plan.content, str):
        return
    commits.append(_Commit(op, snapshot, plan.content, sha, tuple(plan.notes), tuple(plan.strategies)))


def _drift_reason(snap: _Snap, again: Any) -> list[str]:
    """Why ``again`` is not the snapshot the plan used, or empty when it still matches."""
    if snap.absent:
        if getattr(again, "not_found", False):
            return []
        if getattr(again, "error", None):
            return [f"{snap.path}: READ_ERROR"]
        return [f"{snap.path}: source changed after it was read"]
    if getattr(again, "error", None):
        return [f"{snap.path}: READ_ERROR"]
    if again.content != snap.content:
        return [f"{snap.path}: source changed after it was read"]
    return []


def _snapshot_drift(snaps: dict[str, _Snap], file_ops: Any) -> list[str]:
    """Re-read every plan-time disk snapshot. A mismatch refuses; it does not replan."""
    drifted: list[str] = []
    for snap in snaps.values():
        try:
            again = file_ops.read_file_raw(snap.path)
        except Exception:  # health: allow BLE001 -- re-read boundary: fail closed with a fixed code; exception text can hold paths/secrets
            drifted.append(f"{snap.path}: could not re-read source before write")
            continue
        drifted.extend(_drift_reason(snap, again))
    return drifted


def _fail_closed(errors: list[str]) -> PatchResult:
    detail = "\n".join(f"  • {error}" for error in errors)
    return PatchResult(
        success=False,
        error="Patch validation failed (no files were modified):\n" + detail)


def _read_raw(file_ops: Any, path: str) -> Any:
    """Backend read for planning; a raised exception is a fixed-code read error, never its text."""
    try:
        return file_ops.read_file_raw(path)
    except Exception:  # health: allow BLE001 -- plan-read boundary: exception text can hold paths or secrets
        return SimpleNamespace(error="READ_ERROR", content=None, not_found=False)


class _Overlay:
    """In-memory file state shared by one plan. Disk reads are snapshotted; nothing is written."""

    def __init__(self, file_ops: Any) -> None:
        self.file_ops = file_ops
        self.errors: list[str] = []
        self.commits: list[_Commit] = []
        self.snaps: dict[str, _Snap] = {}
        self.pending: dict = {}
        self.shas: dict = {}
        self.removed: set = set()
        self.planners: dict = {}
        self.all_planners: list = []
        self.changes = 0

    def read(self, path: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        key = _overlay_key(path, self.file_ops)
        if key in self.pending:
            return self.pending[key], None, self.shas.get(key)
        if key in self.removed:
            return None, "file not found", None
        result = _read_raw(self.file_ops, path)
        _note_disk_read(self.snaps, key, path, result)
        sha = getattr(result, "_content_sha256", None)
        return (None, "READ_ERROR", None) if result.error else (result.content, None, sha)

    def occupied(self, path: str) -> Optional[str]:
        """Why an Add target or Move destination is not free, or None.

        Only a read that reports the path absent (``not_found``) frees it. A read that
        FAILED says nothing about what is there; treating it as free would overwrite
        the file the check exists to protect.
        """
        key = _overlay_key(path, self.file_ops)
        if key in self.pending:
            return "exists"
        if key in self.removed:
            return None
        result = _read_raw(self.file_ops, path)
        _note_disk_read(self.snaps, key, path, result)
        if not result.error:
            return "exists"
        if getattr(result, "not_found", False):
            return None
        return "READ_ERROR"

    def remove(self, path: str) -> None:
        key = _overlay_key(path, self.file_ops)
        self.removed.add(key)
        self.pending.pop(key, None)


def _resolve_update(op: PatchOperation, overlay: _Overlay) -> None:
    """Plan this Update once against the overlay. A refusal is recorded, not written."""
    simulated, read_err, sha = overlay.read(op.file_path)
    if read_err or not isinstance(simulated, str):
        overlay.errors.append(f"{op.file_path}: {read_err or 'READ_ERROR'}")
        return
    from tools.v4a_planner import PlannerState

    key = _overlay_key(op.file_path, overlay.file_ops)
    if key not in overlay.planners:
        state = PlannerState.from_content(simulated)
        overlay.planners[key] = state
        overlay.all_planners.append((op.file_path, state))
    plan = _plan_file(simulated, op.hunks, overlay.planners[key], finalize=False)
    overlay.changes += plan.changes
    overlay.errors.extend(f"{op.file_path}: {error}" for error in plan.errors)
    overlay.pending[key] = plan.content
    overlay.shas.pop(key, None)
    _remember_update(overlay.commits, op, simulated, sha, plan)


def _resolve_delete(op: PatchOperation, overlay: _Overlay) -> None:
    deleted, delete_err, delete_sha = overlay.read(op.file_path)
    if delete_err:
        overlay.errors.append(f"{op.file_path}: file not found for deletion")
        return
    overlay.remove(op.file_path)
    overlay.planners.pop(_overlay_key(op.file_path, overlay.file_ops), None)
    overlay.commits.append(_Commit(op, deleted, None, delete_sha, (), ()))


def _resolve_move(op: PatchOperation, overlay: _Overlay) -> None:
    if not op.new_path:
        overlay.errors.append(f"{op.file_path}: MOVE operation missing destination path")
        return
    src_content, src_err, src_sha = overlay.read(op.file_path)
    if src_err:
        overlay.errors.append(f"{op.file_path}: source file not found for move")
    dst_taken = overlay.occupied(op.new_path)
    if dst_taken == "exists":
        overlay.errors.append(f"{op.new_path}: destination already exists — move would overwrite")
        return
    if dst_taken:
        overlay.errors.append(f"{op.new_path}: {dst_taken}")
        return
    if src_err:  # only a cleanly-validated move updates the overlay
        return
    overlay.pending[_overlay_key(op.new_path, overlay.file_ops)] = src_content if src_content is not None else ""
    src_key = _overlay_key(op.file_path, overlay.file_ops)
    dst_key = _overlay_key(op.new_path, overlay.file_ops)
    overlay.shas[dst_key] = overlay.shas.pop(src_key, src_sha)
    if src_key in overlay.planners:
        overlay.planners[dst_key] = overlay.planners.pop(src_key)
    overlay.remove(op.file_path)
    overlay.commits.append(_Commit(op, src_content, None, src_sha, (), ()))


def _added_text(op: PatchOperation) -> str:
    return "\n".join(line.content for hunk in op.hunks for line in hunk.lines if line.prefix == "+")


def _resolve_add(op: PatchOperation, overlay: _Overlay) -> None:
    """An Add creates a new file. An existing target is a refusal, not a clobber.

    Overlay-aware: an Add after a Delete of the same path stays legal, and the added
    text enters the overlay so a later Update validates against it.
    """
    add_taken = overlay.occupied(op.file_path)
    if add_taken == "exists":
        overlay.errors.append(f"{op.file_path}: file already exists — use Update File, not Add File")
        return
    if add_taken:
        overlay.errors.append(f"{op.file_path}: {add_taken}")
        return
    key = _overlay_key(op.file_path, overlay.file_ops)
    overlay.removed.discard(key)
    overlay.pending[key] = _added_text(op)
    overlay.commits.append(_Commit(op, None, None, "", (), ()))


_RESOLVE_OP = {
    OperationType.DELETE: _resolve_delete,
    OperationType.MOVE: _resolve_move,
    OperationType.ADD: _resolve_add,
}


def _resolve_operations(operations: list[PatchOperation], file_ops: Any) -> _Resolution:
    """Plan every op against one overlay. The planner runs once per Update; nothing is written."""
    overlay = _Overlay(file_ops)
    for op in operations:
        if op.operation == OperationType.UPDATE:
            _resolve_update(op, overlay)
            continue
        overlay.changes += 1
        _RESOLVE_OP[op.operation](op, overlay)
    from tools.v4a_planner import finish_state

    for path, state in overlay.all_planners:
        try:
            errors = _redact_planner_errors(finish_state(state))
        except Exception:  # health: allow BLE001 -- planner boundary: fail closed with a fixed code; exception text can hold source
            errors = ["hunk planning failed"]
        overlay.errors.extend(f"{path}: {error}" for error in errors)
    if not overlay.errors and overlay.changes == 0:
        overlay.errors.append("Patch contains no changes (only context lines were provided)")
    return _Resolution(overlay.errors, overlay.commits, overlay.snaps)


class ApplyResult(NamedTuple):
    """Outcome of one ``_apply_*`` handler. ``write`` is the PatchResult._writes entry of a file
    write (None for Delete/Move and failures); ``notes`` tell the caller how the edit was matched
    (escape-drift corrections, hunks located approximately)."""
    ok: bool
    payload: str  # the diff on success, the error text on failure
    lsp: Optional[str] = None
    lint: Optional[dict] = None
    write: Optional[tuple] = None
    notes: tuple[str, ...] = ()


def _fail(error: str) -> ApplyResult:
    return ApplyResult(False, error)


def _written(result: Any, diff: str, path: str, read_sha256: Optional[str]) -> ApplyResult:
    """Outcome of a write: its error, else success with LSP/lint propagated from the WriteResult
    and the ``(path, read_sha256, written_sha256)`` record of the bytes it replaced and wrote."""
    if result.error:
        return _fail(result.error)
    write = (path, read_sha256, getattr(result, "_content_sha256", None))
    return ApplyResult(True, diff, getattr(result, "lsp_diagnostics", None), getattr(result, "lint", None), write)


def _unified_diff(path: str, old: str, new: Optional[str]) -> str:
    """Unified diff ``a/path`` -> ``b/path`` (``new=None`` = deletion, ``/dev/null``)."""
    return ''.join(difflib.unified_diff(
        old.splitlines(keepends=True), [] if new is None else new.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile="/dev/null" if new is None else f"b/{path}"))


def apply_v4a_operations(operations: list[PatchOperation], file_ops: Any) -> PatchResult:
    """Plan every file, refuse before any write, then commit the planned bytes.

    The provenance planner runs once per Update. A refused hunk writes nothing in any
    file. Commit does not select or plan again: it re-reads the snapshots and writes
    only when they still match. A later I/O error is reported, not rolled back.
    ``file_ops`` needs read_file_raw/write_file/delete_file/move_file.
    """
    resolved = _resolve_operations(operations, file_ops)
    if resolved.errors:
        return _fail_closed(resolved.errors)
    if drift := _snapshot_drift(resolved.snaps, file_ops):
        return _fail_closed(drift)
    return _commit_resolved(resolved.commits, file_ops)


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
    """Create a file from the hunks' '+' lines. Fails closed when the target already
    exists: validation confirmed the path was free (or freed by an earlier DELETE in
    this patch, which has already applied by now), so an existing file here is a
    validate/apply race — never clobber."""
    read_back = file_ops.read_file_raw(op.file_path)
    if not read_back.error:
        return _fail(f"{op.file_path}: file already exists — use Update File, not Add File")
    if not getattr(read_back, "not_found", False):
        # The read FAILED; it did not report an absent path. Treating that as "the path is free"
        # writes the Add payload over whatever is actually there.
        return _fail(f"{op.file_path}: could not confirm the path is free — {read_back.error}")
    content_lines = [line.content for hunk in op.hunks for line in hunk.lines if line.prefix == '+']
    result = file_ops.write_file(op.file_path, '\n'.join(content_lines))
    diff = f"--- /dev/null\n+++ b/{op.file_path}\n" + '\n'.join(f"+{line}" for line in content_lines)
    return _written(result, diff, op.file_path, "")


def _apply_delete(op: PatchOperation, file_ops: Any) -> ApplyResult:
    """Delete a file, producing a real unified diff of the removed content."""
    read_result = file_ops.read_file_raw(op.file_path)  # re-read guards validate/apply races
    if read_result.error:
        return _fail(f"Cannot delete {op.file_path}: file not found")
    result = file_ops.delete_file(op.file_path)
    diff = _unified_diff(op.file_path, read_result.content, None) or f"# Deleted: {op.file_path}"
    return _fail(result.error) if result.error else ApplyResult(True, diff)


def _apply_move(op: PatchOperation, file_ops: Any) -> ApplyResult:
    """Move, re-checking the destination first: validation's answer is stale once earlier ops of
    this patch have applied, and ``mv`` replaces whatever is there."""
    dst = file_ops.read_file_raw(op.new_path)
    if not dst.error:
        return _fail(f"{op.new_path}: destination already exists — move would overwrite")
    if not getattr(dst, "not_found", False):
        return _fail(f"{op.new_path}: could not confirm the destination is free — {dst.error}")
    result = file_ops.move_file(op.file_path, op.new_path)
    return _fail(result.error) if result.error else ApplyResult(
        True, f"# Moved: {op.file_path} -> {op.new_path}")


def _commit_planned_update(commit: _Commit, file_ops: Any) -> ApplyResult:
    """Write the bytes the planner already chose. Never read the file to select a site."""
    from tools.fuzzy_match import _note_edit_match, boundary_note
    op = commit.op
    if not isinstance(commit.snapshot, str) or not isinstance(commit.content, str):
        return _fail("planned update is missing content")
    # pre_content skips a redundant re-read inside write_file. It is the snapshot, not a new selection.
    extra = {"pre_content": commit.snapshot} if _write_file_accepts_pre_content(file_ops) else {}
    write_result = file_ops.write_file(op.file_path, commit.content, **extra)
    result = _written(
        write_result, _unified_diff(op.file_path, commit.snapshot, commit.content),
        op.file_path, commit.read_sha)
    if not result.ok:
        return result
    for strategy in commit.strategies:
        _note_edit_match(strategy)  # shared metrics: which strategy landed each hunk
    notes = list(commit.notes)
    if note := boundary_note(commit.strategies, "the hunk's", f" in {op.file_path}"):
        notes.append(note)
    return result._replace(notes=tuple(notes))


def _commit_one(commit: _Commit, file_ops: Any) -> tuple[ApplyResult, str, str]:
    """Dispatch a planned mutation. Update never re-enters the planner."""
    op = commit.op
    if op.operation is OperationType.UPDATE:
        return _commit_planned_update(commit, file_ops), "update", "modified"
    handler, verb, bucket = _APPLY_DISPATCH[op.operation]
    return handler(op, file_ops), verb, bucket


# Backend denials that name the policy knob, not source text. Anything else is IO_ERROR.
_KNOWN_IO_DENIALS = {"HERMES_WRITE_SAFE_ROOT": "WRITE_DENIED (HERMES_WRITE_SAFE_ROOT)"}


def _io_code(backend_error: Any) -> str:
    """Fixed code for a backend write failure; the backend text itself is never echoed."""
    text = backend_error if isinstance(backend_error, str) else ""
    for marker, code in _KNOWN_IO_DENIALS.items():
        if marker in text:
            return code
    return "IO_ERROR"


def _commit_resolved(commits: list[_Commit], file_ops: Any) -> PatchResult:
    """Write planned bytes in order. An I/O failure is not a selection refusal and is not rolled back."""
    errors: list[str] = []
    files: dict[str, list[str]] = {"created": [], "deleted": [], "modified": []}
    all_diffs: list[str] = []
    notes: list[str] = []
    lsp_blocks: list[str] = []
    lint_results: dict[str, dict] = {}
    writes: list[tuple] = []
    written_sha: dict[str, Optional[str]] = {}
    for commit in commits:
        op = commit.op
        key = _overlay_key(op.file_path, file_ops)
        if commit.read_sha is None and written_sha.get(key):
            # A later Update of a file this patch already wrote was planned on the earlier
            # op's bytes; its baseline is what that earlier write produced.
            commit = commit._replace(read_sha=written_sha[key])
        try:
            outcome, verb, bucket = _commit_one(commit, file_ops)
            ok, payload, lsp, lint, write, op_notes = outcome
        except Exception:  # health: allow BLE001 -- commit boundary: report a fixed code; exception text can hold source or secrets
            ok, payload, lsp, lint, write, op_notes = None, "INTERNAL_ERROR", None, None, None, ()
            verb, bucket = "process", "modified"
        if not ok:
            prefix = f"Failed to {verb}" if ok is False else "Error processing"
            code = 'INTERNAL_ERROR' if ok is None else _io_code(payload)
            errors.append(f"{prefix} {op.file_path}: {code}")
            continue
        if write:
            written_sha[key] = write[2]
        if op.operation is OperationType.MOVE and op.new_path:
            written_sha[_overlay_key(op.new_path, file_ops)] = written_sha.pop(key, None)
        is_move = op.operation is OperationType.MOVE
        files[bucket].append(f"{op.file_path} -> {op.new_path}" if is_move else op.file_path)
        all_diffs.append(payload)
        if write:
            writes.append(write)
        notes.extend(op_notes)
        if lsp:
            lsp_blocks.append(lsp)
        if lint:
            lint_results[op.file_path] = lint
    detail = "\n".join(f"  • {error}" for error in errors)
    return PatchResult(
        success=not errors,
        error=("Apply phase failed (state may be inconsistent — run `git diff` to assess):\n"
               + detail) if errors else None,
        diff='\n'.join(all_diffs),
        files_modified=files["modified"], files_created=files["created"], files_deleted=files["deleted"],
        lint=lint_results or None, lsp_diagnostics="\n\n".join(lsp_blocks) or None, _writes=writes,
        note="\n".join(dict.fromkeys(notes)) or None)


# operation -> (handler, verb for error text, files_* bucket). Update is not here: commit
# writes the planned bytes and must not select a site.
_APPLY_DISPATCH: dict[OperationType, tuple[Callable[[PatchOperation, Any], ApplyResult], str, str]] = {
    OperationType.ADD: (_apply_add, "add", "created"),
    OperationType.DELETE: (_apply_delete, "delete", "deleted"),
    OperationType.MOVE: (_apply_move, "move", "modified"),
}
