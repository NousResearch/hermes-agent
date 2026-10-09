"""V4A patch parser/applier (codex, cline). ``*** Begin Patch``/``*** End Patch`` wrap ops:
``*** Update File: p`` + hunks (``@@ hint @@``, `` ctx``, ``-old``, ``+new``); ``*** Add File: n``
+ ``+`` lines; ``*** Delete File: o``; ``*** Move File: a -> b``. Entry points:
``parse_v4a_patch(text) -> (ops, error)`` and ``apply_v4a_operations(ops, file_ops)``."""

import contextlib
import difflib
import inspect
import itertools
import ntpath
import posixpath

import re
from dataclasses import dataclass, field
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


def _split_hunk(hunk: Hunk) -> tuple[list[str], list[str]]:
    """``(search_lines, replace_lines)``: context+removed vs context+added."""
    return ([l.content for l in hunk.lines if l.prefix != '+'],
            [l.content for l in hunk.lines if l.prefix != '-'])


def _no_match_hint(error: Optional[str], search_pattern: str, content: str) -> str:
    """Best-effort 'Did you mean...' suffix; never lets a hint failure mask the real error."""
    with contextlib.suppress(Exception):
        from tools.fuzzy_match import format_no_match_hint
        return format_no_match_hint(error, 0, search_pattern, content)
    return ""


def _hint_ambiguity(content: str, hint: str, tail: str = "") -> tuple[int, str]:
    """(occurrences, error) for an addition-only hunk's context hint; error is '' when unique."""
    n = _count_occurrences(content, hint)
    return n, f"context hint '{hint}' is ambiguous ({n} occurrences){tail}" if n > 1 else ""


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


# codex seek_sequence tiers: a hunk's lines match exactly, then ignoring trailing whitespace,
# then ignoring all surrounding whitespace.
_LINE_TIERS: tuple[Callable[[str], str], ...] = (lambda line: line, str.rstrip, str.strip)
_HINT_WINDOW_BEFORE, _HINT_WINDOW_AFTER = 500, 2000  # chars around an @@ hint @@ it narrows to


def _line_spans(content: str, search_lines: list[str]) -> list[list[tuple[int, int]]]:
    """Per tier, every ``(start, end)`` char span whose lines equal ``search_lines``. The caller
    picks the tier inside each scope it searches, so an exact match elsewhere cannot hide a
    whitespace-equivalent one in the scope the patch's order points at."""
    lines = content.split('\n')
    starts = list(itertools.accumulate((len(line) + 1 for line in lines[:-1]), initial=0))
    n = len(search_lines)
    tiers = []
    for tier in _LINE_TIERS:
        wanted = [tier(line) for line in search_lines]
        tiers.append([(starts[i], starts[i + n - 1] + len(lines[i + n - 1]))
                      for i in range(len(lines) - n + 1)
                      if [tier(line) for line in lines[i:i + n]] == wanted])
    return tiers


def _positions(content: str, needle: str) -> list[int]:
    """Start of every (overlapping) occurrence of ``needle``."""
    return [m.start() for m in re.finditer(f"(?={re.escape(needle)})", content)] if needle else []


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _hint_windows(content: str, hint: Optional[str]) -> list[tuple[int, int]]:
    """The window around EVERY occurrence of an @@ hint @@ (a repeated hint proves nothing)."""
    return [(max(0, pos - _HINT_WINDOW_BEFORE), min(len(content), pos + _HINT_WINDOW_AFTER))
            for pos in _positions(content, hint or "")]


def _distinct(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """One span per physical site: spans that overlap are the same site seen twice."""
    sites: list[tuple[int, int]] = []
    for span in spans:
        if not any(_overlaps(span, site) for site in sites):
            sites.append(span)
    return sites


def _select_span(content: str, pattern: str, hunk: Hunk, cursor: int,
                 *, ordered: bool) -> tuple[Optional[tuple[int, int]], int]:
    """``(span, candidates)``: the one source site this hunk may edit, else ``(None, candidates)``.

    A site is a whole-line match under the first tier that matches inside the scope searched,
    or a raw substring hit elsewhere (a raw hit overlapping a line match is that same site), so
    neither normalization nor whole-line matching hides a competing site. Scope comes first:
    after a previous hunk only the rest of the file is searched, and the whole file only when
    nothing matches there. Competing sites are resolved only by order the patch states: an
    ``ordered`` hunk (the first of several, or one of a run of identical hunks) takes the first
    site at or after ``cursor``; any other hunk needs exactly one site, or exactly one site
    across the windows around every occurrence of its @@ hint @@."""
    tiers = _line_spans(content, pattern.split('\n'))
    raw = [(pos, pos + len(pattern)) for pos in _positions(content, pattern)]

    def sites(lo: int, hi: int) -> list[tuple[int, int]]:
        inside = lambda span: lo <= span[0] and span[1] <= hi
        lines = next((found for tier in tiers if (found := [s for s in tier if inside(s)])), [])
        extra = [r for r in raw if inside(r) and not any(_overlaps(r, line) for line in lines)]
        return sorted(lines + _distinct(extra))

    whole = sites(0, len(content))
    if not any(tiers):
        return None, len(whole)  # no whole-line site anywhere: the fuzzy chain decides
    after = sites(cursor, len(content)) if cursor else whole
    if len(after) == 1 or (after and ordered):
        return after[0], len(after)
    pool = after or whole  # nothing after the previous hunk: a unique earlier site still counts
    if len(pool) == 1:
        return pool[0], 1
    picks = [sites(lo, hi) for lo, hi in _hint_windows(content, hunk.context_hint)]
    if picks and all(len(p) <= 1 for p in picks) and len(_distinct([p[0] for p in picks if p])) == 1:
        return next(p[0] for p in picks if p), len(pool)
    return None, len(pool)


def _v4a_advice(hunk: Hunk) -> str:
    """V4A cannot set ``replace_all``; ambiguity recovery must name what the model can change."""
    return ("include unique context lines in its search text" if hunk.context_hint else
            "add a unique @@ hint @@ to this hunk or include unique context lines in its search text")


def _fuzzy_spans(text: str, pattern: str) -> list[tuple[int, int]]:
    """The spans the fuzzy strategy chain matches (its first matching strategy), in ``text``."""
    from tools.fuzzy_match import STRATEGIES
    return next((spans for _name, find in STRATEGIES if (spans := find(text, pattern))), [])


def _fuzzy_replace_hunk(content: str, hunk: Hunk, pattern: str, replacement: str,
                        cursor: int) -> tuple[str, int, Optional[str]]:
    """No whole-line site: the fuzzy strategy chain over the whole file (it refuses its own
    ambiguity), then inside the windows around the @@ hint @@, which must all agree on one
    site. -> (content, cursor, error); the cursor is the end of the text actually written,
    never a search for the replacement (that text may already exist earlier in the file)."""
    from tools.fuzzy_match import fuzzy_find_and_replace
    new, count, _strategy, error = fuzzy_find_and_replace(content, pattern, replacement)
    if count:
        return new, len(new) - (len(content) - _fuzzy_spans(content, pattern)[0][1]), None
    windows = [(lo, hi, _fuzzy_spans(content[lo:hi], pattern))
               for lo, hi in _hint_windows(content, hunk.context_hint)] if error else []
    targets = _distinct([(lo + s, lo + e) for lo, _hi, spans in windows for s, e in spans])
    if len(targets) > 1 or any(len(spans) > 1 for *_w, spans in windows):
        return content, cursor, (f"context hint '{hunk.context_hint}' selects {len(targets)} "
                                 f"different matches — {_v4a_advice(hunk)}")
    if targets:
        lo, hi, spans = next(w for w in windows if w[2])
        window, count, _strategy, error = fuzzy_find_and_replace(content[lo:hi], pattern, replacement)
        if count:
            return (content[:lo] + window + content[hi:],
                    lo + len(window) - (hi - lo - spans[0][1]), None)
    replace_all_advice = "Provide more context to make it unique, or use replace_all=True."
    return content, cursor, error and error.replace(replace_all_advice, _v4a_advice(hunk).capitalize() + ".")


def _plan_hunks(content: str, hunks: list[Hunk]) -> tuple[str, list[str], int]:
    """Apply an Update's ``hunks`` in order to ``content`` -> ``(new_content, errors, changes)``.

    The ONE selection policy for an Update: validation runs it on the source it read and the
    apply phase runs it again on the bytes it actually reads, so a source that changed between
    the two phases is decided again (a site that became ambiguous refuses) instead of apply
    taking a first match that validation never admitted."""
    from tools.fuzzy_match import fuzzy_find_and_replace, is_already_applied
    errors: list[str] = []
    changes = cursor = 0
    changed_patterns = ['\n'.join(s) for s, r in map(_split_hunk, hunks) if s and s != r]
    for index, hunk in enumerate(hunks, start=1):
        search_lines, replace_lines = _split_hunk(hunk)
        label = f"hunk {index} " + (f"'{hunk.context_hint}'" if hunk.context_hint else "(no hint)")
        if search_lines == replace_lines:
            # Context-only anchor hunks (models emit these between changes) are inert; identical
            # -/+ lines are a no-op. An anchor that selects exactly one site moves the cursor.
            changes += any(line.prefix in '-+' for line in hunk.lines)
            if search_lines:
                span, _ = _select_span(content, '\n'.join(search_lines), hunk, cursor, ordered=False)
                cursor = span[1] if span else cursor
            continue
        changes += 1
        replacement = '\n'.join(replace_lines)
        if not search_lines:  # addition-only: placed after a unique context hint, else at EOF
            if hunk.context_hint and not _count_occurrences(content, hunk.context_hint):
                errors.append(f"addition-only hunk context hint '{hunk.context_hint}' not found")
                continue
            new_content, error, at = _insert_addition_only(content, hunk, replacement)
            if error:
                errors.append(error)
                continue
            # The cursor keeps marking the same site: text inserted before it shifts that site.
            if at < cursor:
                cursor = min(len(new_content), cursor + len(new_content) - len(content))
            content = new_content
            continue
        pattern = '\n'.join(search_lines)
        # Order the patch itself states: the first hunk of several starts at the top of the file,
        # and a run of identical hunks edits successive sites.
        ordered = (index == 1 and len(hunks) > 1) or changed_patterns.count(pattern) > 1
        span, candidates = _select_span(content, pattern, hunk, cursor, ordered=ordered)
        if span is not None:
            start, end = span
            window, count, _strategy, error = fuzzy_find_and_replace(
                content[start:end], pattern, replacement)
            if count:
                content, cursor = content[:start] + window + content[end:], start + len(window)
            else:
                errors.append(f"{label} not found" + (f" — {error}" if error else ""))
            continue
        if candidates > 1:
            errors.append(f"{label} is ambiguous ({candidates} matches"
                          + (" after the previous hunk" if cursor else "") + f") — {_v4a_advice(hunk)}")
            continue
        content, cursor, error = _fuzzy_replace_hunk(content, hunk, pattern, replacement, cursor)
        # Already-applied hunks are a no-op; only reached when no site of the search text remains.
        if error and not is_already_applied(content, pattern, replacement):
            errors.append(f"{label} not found — {error}" + _no_match_hint(error, pattern, content))
    return content, errors, changes


def _validate_operations(operations: list[PatchOperation], file_ops: Any) -> list[str]:
    """Dry-run every operation -> error strings (empty = safe). UPDATE hunks are simulated in
    order so later hunks see post-earlier-hunk content, exactly as apply will."""
    errors: list[str] = []
    real_change_count = 0
    # Overlay so inter-op state validates (a MOVE creating the path a later UPDATE targets),
    # keyed by _overlay_key so path aliases share one entry.
    pending_content: dict = {}
    removed_paths: set = set()

    def _read_checked(path: str) -> tuple[Optional[str], Optional[str], bool]:
        """(content, error, too_large): too_large means the file exists on disk but is past
        read_file_raw's cap; the patch's own overlay always answers first."""
        key = _overlay_key(path, file_ops)
        if key in pending_content:
            return pending_content[key], None, False
        if key in removed_paths:
            return None, "file not found", False
        r = file_ops.read_file_raw(path)
        if r.error:
            return None, r.error, bool(getattr(r, "_too_large", False))
        return r.content, None, False

    def _read(path: str) -> tuple[Optional[str], Optional[str]]:
        return _read_checked(path)[:2]

    def _occupied(path: str) -> Optional[str]:
        """Why an Add target or Move destination is not free, or None. Only a read that reports
        the path absent (``not_found``) frees it: a read that FAILED (no byte transport, a
        directory, an unreadable file) says nothing about what is there, and taking it as free
        writes over the file the check exists to protect."""
        key = _overlay_key(path, file_ops)
        if key in pending_content:
            return "exists"
        if key in removed_paths:
            return None
        r = file_ops.read_file_raw(path)
        if not r.error:
            return "exists"
        return None if getattr(r, "not_found", False) else f"could not confirm the path is free — {r.error}"

    def _validate_update(op: PatchOperation) -> None:
        nonlocal real_change_count
        simulated, read_err = _read(op.file_path)
        if read_err:
            errors.append(f"{op.file_path}: {read_err}")
            return
        simulated, hunk_errors, changes = _plan_hunks(simulated, op.hunks)
        real_change_count += changes
        errors.extend(f"{op.file_path}: {error}" for error in hunk_errors)
        pending_content[_overlay_key(op.file_path, file_ops)] = simulated

    def _remove(path: str) -> None:
        key = _overlay_key(path, file_ops)
        removed_paths.add(key)
        pending_content.pop(key, None)

    for op in operations:
        if op.operation == OperationType.UPDATE:
            _validate_update(op)
            continue
        real_change_count += 1
        if op.operation == OperationType.DELETE:
            # A file over read_file_raw's cap exists; Delete needs no content, so it stays legal.
            _content, read_err, too_large = _read_checked(op.file_path)
            if read_err and not too_large:
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
            dst_taken = _occupied(op.new_path)
            if dst_taken == "exists":
                errors.append(f"{op.new_path}: destination already exists — move would overwrite")
            elif dst_taken:
                errors.append(f"{op.new_path}: {dst_taken}")
            elif not src_err:  # only a cleanly-validated move updates the overlay
                pending_content[_overlay_key(op.new_path, file_ops)] = src_content if src_content is not None else ""
                _remove(op.file_path)
        elif op.operation == OperationType.ADD:
            # An Add must create a NEW file. If the target already exists, write_file
            # would clobber it with only the patch's '+' lines and report success,
            # silently destroying the original contents (models frequently confuse Add
            # with Update). Reject it here so the two-phase contract holds, mirroring
            # the MOVE destination guard. Overlay-aware: an Add after a Delete of the
            # same path in this patch stays legal, and the added content enters the
            # overlay so later hunks against it validate.
            add_taken = _occupied(op.file_path)
            if add_taken == "exists":
                errors.append(f"{op.file_path}: file already exists — use Update File, not Add File")
            elif add_taken:
                errors.append(f"{op.file_path}: {add_taken}")
            else:
                removed_paths.discard(_overlay_key(op.file_path, file_ops))
                pending_content[_overlay_key(op.file_path, file_ops)] = '\n'.join(
                    line.content for hunk in op.hunks for line in hunk.lines if line.prefix == '+')
    if not errors and real_change_count == 0:
        errors.append("Patch contains no changes (only context lines were provided)")
    return errors


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
    """Two-phase: validate everything, then apply (atomic on validation failure). A phase-2
    failure (validate/apply race) carries a ``git diff`` note since state may be inconsistent.
    ``file_ops`` needs read_file_raw/write_file/delete_file/move_file."""

    def _bullets(errs: list[str]) -> str:
        return "\n".join(f"  • {e}" for e in errs)

    if errors := _validate_operations(operations, file_ops):
        return PatchResult(
            success=False,
            error="Patch validation failed (no files were modified):\n" + _bullets(errors))
    files: dict[str, list[str]] = {"created": [], "deleted": [], "modified": []}
    all_diffs: list[str] = []
    notes: list[str] = []
    # V4A bypasses write_file's WriteResult plumbing: LSP diagnostics and lint propagate per file.
    lsp_blocks: list[str] = []
    lint_results: dict[str, dict] = {}
    writes: list[tuple] = []
    for op in operations:
        handler, verb, bucket = _APPLY_DISPATCH[op.operation]
        try:
            ok, payload, lsp, lint, write, op_notes = handler(op, file_ops)
        except Exception as e:
            ok, payload = None, str(e)
        if not ok:
            prefix = f"Failed to {verb}" if ok is False else "Error processing"
            errors.append(f"{prefix} {op.file_path}: {payload}")
            continue
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
    # Each LSP block carries its own <diagnostics file="..."> header; joining keeps attribution.
    return PatchResult(
        success=not errors,
        error=("Apply phase failed (state may be inconsistent — run `git diff` to assess):\n"
               + _bullets(errors)) if errors else None,
        diff='\n'.join(all_diffs),
        files_modified=files["modified"], files_created=files["created"], files_deleted=files["deleted"],
        lint=lint_results or None, lsp_diagnostics="\n\n".join(lsp_blocks) or None, _writes=writes,
        note="\n".join(dict.fromkeys(notes)) or None)


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
    too_large = getattr(read_result, "_too_large", False)  # exists, just past the read cap: no diff
    if read_result.error and not too_large:
        return _fail(f"Cannot delete {op.file_path}: file not found")
    result = file_ops.delete_file(op.file_path)
    diff = (None if too_large else _unified_diff(op.file_path, read_result.content, None)) \
        or f"# Deleted: {op.file_path}"
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


def _insert_addition_only(new_content: str, hunk: Hunk,
                          insert_text: str) -> tuple[Optional[str], Optional[str], int]:
    """Place an addition-only hunk after its context hint (or at EOF).
    Returns (content, error, offset in the old content where the text went)."""
    if hunk.context_hint:
        occurrences, ambiguous = _hint_ambiguity(
            new_content, hunk.context_hint, " — provide a more unique hint")
        if ambiguous:
            return None, f"Addition-only hunk: {ambiguous}", 0
        if occurrences == 1:
            eol = new_content.find('\n', new_content.find(hunk.context_hint))
            if eol == -1:
                return new_content + '\n' + insert_text, None, len(new_content)
            return new_content[:eol + 1] + insert_text + '\n' + new_content[eol + 1:], None, eol + 1
    # No hint / hint not found — append at end as a safe fallback.
    kept = new_content.rstrip('\n')
    return kept + '\n' + insert_text + '\n', None, len(kept)


def _apply_update(op: PatchOperation, file_ops: Any) -> ApplyResult:
    """Re-run validation's hunk selection on the bytes read now, then write once. A source that
    changed since validation is decided again; a hunk that no longer selects one site fails
    the operation before anything is written."""
    read_result = file_ops.read_file_raw(op.file_path)  # raw: no line numbers / truncation
    if read_result.error:
        return _fail(f"Cannot read file: {read_result.error}")
    from tools.fuzzy_match import boundary_note
    current_content = read_result.content
    notes: list[str] = []
    strategies: list[str] = []
    new_content, hunk_errors, _changes = _plan_hunks(current_content, op.hunks)
    if hunk_errors:
        return _fail("Could not apply hunk (file not written): " + "; ".join(hunk_errors))
    # Pass pre_content to skip a redundant re-read inside write_file when supported.
    extra = {"pre_content": current_content} if _write_file_accepts_pre_content(file_ops) else {}
    write_result = file_ops.write_file(op.file_path, new_content, **extra)
    result = _written(write_result, _unified_diff(op.file_path, current_content, new_content),
                      op.file_path, getattr(read_result, "_content_sha256", None))
    if not result.ok:
        return result
    # A hunk found by a similarity strategy may have landed a line off; say so, per file.
    if note := boundary_note(strategies, "the hunk's", f" in {op.file_path}"):
        notes.append(note)
    return result._replace(notes=tuple(notes))


# operation -> (handler, verb for error text, files_* bucket)
_APPLY_DISPATCH: dict[OperationType, tuple[Callable[[PatchOperation, Any], ApplyResult], str, str]] = {
    OperationType.ADD: (_apply_add, "add", "created"),
    OperationType.DELETE: (_apply_delete, "delete", "deleted"),
    OperationType.MOVE: (_apply_move, "move", "modified"),
    OperationType.UPDATE: (_apply_update, "update", "modified"),
}
