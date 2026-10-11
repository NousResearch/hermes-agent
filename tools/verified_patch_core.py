"""Verified anchored line patching.

This module is an experimental pure-Python algorithm layer for safer LLM file
writes.  It combines the mature ideas behind ``git apply``/unified-diff context
relocation with JSON Patch's ``test``-then-write semantics:

* the patch carries the old target lines as a precondition;
* nearby context is used only to relocate/disambiguate the target;
* stale context is never written back to disk;
* semantic target drift rejects instead of guessing.

Unlike whole-file hash guards, unrelated edits elsewhere in the file are allowed
when the intended target can still be uniquely verified.
"""

from __future__ import annotations

import io
import tokenize
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence


class VerifiedPatchError(Exception):
    """Raised when a verified patch cannot be applied safely."""


@dataclass(frozen=True)
class VerifiedOperation:
    """One verified line edit.

    ``start``/``end`` are 1-based line numbers from the model's snapshot.
    ``old`` is the expected target content from that snapshot.  ``before`` and
    ``after`` are context lines from the snapshot; they are used to find a unique
    target when line numbers shift or the target text is duplicated.
    """

    kind: str
    start: int
    end: int
    old: Sequence[str] = field(default_factory=tuple)
    new: Sequence[str] = field(default_factory=tuple)
    before: Sequence[str] = field(default_factory=tuple)
    after: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class PlannedEdit:
    start: int  # 0-based inclusive
    end: int  # 0-based exclusive
    new: Sequence[str]


def split_lines(text: str) -> List[str]:
    """Split into logical lines without a trailing-newline ghost element."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if text.endswith("\n"):
        text = text[:-1]
    if text == "":
        return []
    return text.split("\n")


def join_lines(lines: Sequence[str], trailing_newline: bool) -> str:
    out = "\n".join(lines)
    if trailing_newline:
        out += "\n"
    return out


def make_replace_operation(
    snapshot: str,
    start: int,
    end: int,
    new: Iterable[str],
    *,
    context: int = 2,
) -> VerifiedOperation:
    """Build a replace operation from a previously read file snapshot."""

    lines = split_lines(snapshot)
    if start < 1 or end < start or end > len(lines):
        raise VerifiedPatchError(f"replace {start}..{end} outside snapshot")
    before_start = max(0, start - 1 - context)
    after_end = min(len(lines), end + context)
    return VerifiedOperation(
        kind="replace",
        start=start,
        end=end,
        old=tuple(lines[start - 1 : end]),
        new=tuple(new),
        before=tuple(lines[before_start : start - 1]),
        after=tuple(lines[end:after_end]),
    )


def _leading_ws(text: str) -> str:
    return text[: len(text) - len(text.lstrip(" \t"))]


def _token_key(text: str) -> tuple[tuple[int, str], ...] | None:
    """Return a single-line Python token key.

    Only horizontal spacing *inside* a line is ignored. Leading indentation and
    line boundaries are checked outside this helper because they can be semantic
    in Python, YAML, shell heredocs, Markdown, and other unknown file types.
    """

    try:
        tokens = tokenize.generate_tokens(io.StringIO(text.strip() + "\n").readline)
        ignored = {
            tokenize.ENCODING,
            tokenize.ENDMARKER,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.NL,
            tokenize.NEWLINE,
        }
        return tuple((tok.type, tok.string) for tok in tokens if tok.type not in ignored)
    except (IndentationError, tokenize.TokenError):
        return None


def _line_formatting_equivalent(current: str, expected: str) -> bool:
    if current == expected:
        return True
    # Indentation is a local precondition. Treating it as cosmetic can silently
    # move Python statements between blocks or change YAML/Markdown structure.
    if _leading_ws(current) != _leading_ws(expected):
        return False

    current_key = _token_key(current)
    expected_key = _token_key(expected)
    if current_key is not None and expected_key is not None:
        return current_key == expected_key

    # Conservative fallback for non-tokenizable snippets: allow trailing-space
    # drift only. Interior whitespace can be semantic in strings, docs, shell,
    # YAML, or unknown languages.
    return current.rstrip(" \t") == expected.rstrip(" \t")


def _formatting_equivalent(current: Sequence[str], expected: Sequence[str]) -> bool:
    if len(current) != len(expected):
        return False
    return all(
        _line_formatting_equivalent(got, want)
        for got, want in zip(current, expected)
    )


def _block_matches(current: Sequence[str], expected: Sequence[str]) -> tuple[bool, bool]:
    """Return ``(matches, exact)`` for one candidate block."""

    if len(current) != len(expected):
        return False, False
    if list(current) == list(expected):
        return True, True
    if _formatting_equivalent(current, expected):
        return True, False
    return False, False


def _context_score(lines: Sequence[str], start: int, end: int, op: VerifiedOperation) -> int:
    score = 0
    # Compare the suffix of before-context immediately preceding the target.
    before = list(op.before)
    if before:
        available = list(lines[max(0, start - len(before)) : start])
        for got, want in zip(reversed(available), reversed(before)):
            if got == want:
                score += 8
            elif _formatting_equivalent([got], [want]):
                score += 5
            else:
                break
    after = list(op.after)
    if after:
        available = list(lines[end : end + len(after)])
        for got, want in zip(available, after):
            if got == want:
                score += 8
            elif _formatting_equivalent([got], [want]):
                score += 5
            else:
                break
    return score


def _find_target(lines: Sequence[str], op: VerifiedOperation) -> PlannedEdit:
    if op.kind not in {"replace", "delete"}:
        raise VerifiedPatchError(f"unsupported operation kind: {op.kind}")
    if op.start < 1 or op.end < op.start:
        raise VerifiedPatchError(f"invalid range {op.start}..{op.end}")

    width = len(op.old)
    if width == 0:
        raise VerifiedPatchError("replace/delete requires non-empty old precondition")
    if width != op.end - op.start + 1:
        raise VerifiedPatchError("old precondition length does not match operation range")
    if width > len(lines):
        raise VerifiedPatchError("target longer than current file")

    candidates: list[tuple[int, int, int]] = []
    for start in range(0, len(lines) - width + 1):
        end = start + width
        matches, exact = _block_matches(lines[start:end], op.old)
        if not matches:
            continue
        # Target precondition is mandatory, but context is the primary
        # relocation/disambiguation signal. An exact duplicate without matching
        # context must not beat the intended location whose target only differs
        # by safe formatting drift.
        context_score = _context_score(lines, start, end, op)
        exact_bonus = 1 if exact else 0
        candidates.append((context_score, exact_bonus, start))

    if not candidates:
        raise VerifiedPatchError(
            f"target precondition for lines {op.start}..{op.end} no longer matches"
        )

    candidates.sort(reverse=True)
    if len(candidates) > 1 and candidates[0][:2] == candidates[1][:2]:
        raise VerifiedPatchError(
            f"ambiguous target for lines {op.start}..{op.end}: multiple verified matches"
        )

    _, _exact_bonus, start = candidates[0]
    end = start + width
    replacement: Sequence[str] = () if op.kind == "delete" else op.new
    return PlannedEdit(start=start, end=end, new=tuple(replacement))


def plan_operations(content: str, ops: Sequence[VerifiedOperation]) -> list[PlannedEdit]:
    lines = split_lines(content)
    planned = [_find_target(lines, op) for op in ops]
    planned_sorted = sorted(planned, key=lambda edit: (edit.start, edit.end))
    previous_end = -1
    for edit in planned_sorted:
        if edit.start < previous_end:
            raise VerifiedPatchError("overlapping verified edits are not supported")
        previous_end = edit.end
    return planned


def apply_operations(content: str, ops: Sequence[VerifiedOperation]) -> str:
    """Apply verified operations atomically to one file content string."""

    trailing_newline = content.endswith("\n") or content.endswith("\r\n")
    lines = split_lines(content)
    planned = plan_operations(content, ops)
    for edit in sorted(planned, key=lambda item: item.start, reverse=True):
        lines[edit.start : edit.end] = list(edit.new)
    return join_lines(lines, trailing_newline)
