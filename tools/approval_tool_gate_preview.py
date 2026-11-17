"""What the full-scope approval prompt shows for a file edit: the change the tool will make.

The preview is rendered from the tool's own pre-write plan (``ShellFileOperations.plan_patch_replace``
/ ``plan_write``) on the same file-ops backend and the same resolved path, so the matcher,
escape corrections, BOM stripping and CRLF / BOM preservation are the ones that will run, on the
host or in a sandbox. It is shaped for a narrow panel that shows only the first rows: the path and
line on the first row, then only the changed lines, no diff headers.
"""

import difflib
from typing import Optional

_MAX_LINES = 40
_MAX_CHARS = 4000
_HUNK_GAP = "⋯"
_NEWLINE_ADDED = "⏎ final newline added"
_NEWLINE_REMOVED = "⏎ final newline removed"
_DIFF_WORK = 250_000  # changed-region old x new lines above which the region is shown whole, not diffed


def _clip(lines: list[str]) -> str:
    shown = "\n".join(lines[:_MAX_LINES])
    if len(shown) > _MAX_CHARS:
        shown = shown[:_MAX_CHARS] + "…"
    if len(lines) > _MAX_LINES:
        shown += f"\n… ({len(lines) - _MAX_LINES} more lines)"
    return shown


def _final_newline_note(old: str, new: str) -> str:
    """A change to the final newline alone is invisible in ``-``/``+`` lines, so it gets a row."""
    had, has = old.endswith(("\n", "\r")), new.endswith(("\n", "\r"))
    if not old or not new or had == has:
        return ""
    return _NEWLINE_ADDED if has else _NEWLINE_REMOVED


def _opcodes(old_lines: list[str], new_lines: list[str]) -> tuple[int, list[tuple]]:
    """``(offset, opcodes)`` for the region between the common head and tail. The head and tail
    are trimmed in linear time; a region still too large to diff quickly is shown as one hunk,
    so the card never waits on a quadratic comparison."""
    limit = min(len(old_lines), len(new_lines))
    head = 0
    while head < limit and old_lines[head] == new_lines[head]:
        head += 1
    tail = 0
    while tail < limit - head and old_lines[-1 - tail] == new_lines[-1 - tail]:
        tail += 1
    a, b = old_lines[head:len(old_lines) - tail], new_lines[head:len(new_lines) - tail]
    if len(a) * len(b) > _DIFF_WORK:
        return head, [("replace", 0, len(a), 0, len(b))]
    return head, difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()


def _changes(old: str, new: str) -> tuple[Optional[int], list[str]]:
    """``(first_changed_line, changed_lines)``: ``-``/``+`` lines only, hunks separated by ``⋯``."""
    old_lines, new_lines = old.splitlines(), new.splitlines()
    head, opcodes = _opcodes(old_lines, new_lines)
    old_lines, new_lines = old_lines[head:], new_lines[head:]
    first, out = None, []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        if first is None:
            first = head + i1 + 1
        elif out:
            out.append(_HUNK_GAP)
        out += [f"-{line}" for line in old_lines[i1:i2]] + [f"+{line}" for line in new_lines[j1:j2]]
    note = _final_newline_note(old, new)
    if note:
        first = first or head + len(old_lines) or 1
        out.append(note)
    return first, out


def _size(changes: list[str]) -> str:
    """``N lines changed: -a +b`` for the header, so a panel that shows only the first rows still
    tells the user how much is behind "show full"; N counts the larger side of the change."""
    removed = sum(1 for line in changes if line.startswith("-"))
    added = sum(1 for line in changes if line.startswith("+"))
    count = max(removed, added)
    if count == 0:
        return "only the final newline changes"
    return f"{count} line{'s' if count != 1 else ''} changed: -{removed} +{added}"


def _kept(line_ending: Optional[str], bom: bool) -> str:
    notes = (["CRLF line endings kept"] if line_ending == "\r\n" else []) + (["BOM kept"] if bom else [])
    return f", {', '.join(notes)}" if notes else ""


def _backend(path: str, task_id: str):
    """``(file_ops, target, refusal)``: the tool's backend and resolved target, or the hard
    refusal the tool will return first (checked before anything is read)."""
    from tools.file_tools import _get_file_ops, _resolve_or_none
    from tools.file_tools_write_guards import _check_sensitive_path
    refusal = _check_sensitive_path(path, task_id)
    if refusal:
        return None, path, refusal
    return _get_file_ops(task_id), _resolve_or_none(path, task_id) or path, None


def write_file_preview(args: dict, task_id: str) -> str:
    from tools.file_operations_common import WriteResult, _strip_bom
    path = args.get("path") if isinstance(args.get("path"), str) else "?"
    content = args.get("content") if isinstance(args.get("content"), str) else ""
    file_ops, target, refusal = _backend(path, task_id)
    if refusal:
        return f"write_file {target} — the tool will refuse this:\n{refusal}"
    plan = file_ops.plan_write(target, content, want_pre=True)
    if isinstance(plan, WriteResult):
        return f"write_file {target} — the tool will refuse this:\n{plan.error}"
    new_text, _ = _strip_bom(plan.content)
    count = len(new_text.splitlines())
    if plan.pre_content is None:
        return _clip([f"write_file {plan.path} (new or empty file, {count} lines{_kept(plan.line_ending, plan.bom)})",
                      *new_text.splitlines()])
    old_text, _ = _strip_bom(plan.pre_content)
    first, changes = _changes(old_text, new_text)
    where = f", line {first}, {_size(changes)}" if first else ", content unchanged"
    header = (f"write_file {plan.path} (overwrites {len(old_text.splitlines())} → {count} lines{where}"
              f"{_kept(plan.line_ending, plan.bom)})")
    return _clip([header, *changes])


def patch_preview(args: dict, task_id: str) -> str:
    from tools.file_operations_common import PatchResult
    if args.get("mode") == "patch":
        patch = args.get("patch") if isinstance(args.get("patch"), str) else ""
        return _clip(["patch (V4A, as written by the model)", *patch.splitlines()])
    path = args.get("path") if isinstance(args.get("path"), str) else "?"
    old = args.get("old_string") if isinstance(args.get("old_string"), str) else ""
    new = args.get("new_string") if isinstance(args.get("new_string"), str) else ""
    file_ops, target, refusal = _backend(path, task_id)
    if refusal:
        return f"patch {target} — the tool will refuse this:\n{refusal}"
    plan = file_ops.plan_patch_replace(target, old, new, bool(args.get("replace_all")))
    if isinstance(plan, PatchResult):
        outcome = (f"no change, the edit is already in the file: {plan.note}" if plan.success
                   else f"the patch will fail, nothing is written: {plan.error}")
        requested = [f"requested old_string: {line}" for line in old.splitlines()[:3]]
        return _clip([f"patch {target} — {outcome.splitlines()[0]}", *requested])
    first, changes = _changes(plan.content, plan.new_content)
    count = f"{plan.match_count} replacement{'s' if plan.match_count != 1 else ''}"
    bom = plan.raw_content != plan.content
    header = f"patch {plan.path}, line {first} ({count}, {_size(changes)}{_kept(plan.line_ending, bom)})"
    return _clip([header, *changes, *(f"note: {note}" for note in plan.correction_notes)])
