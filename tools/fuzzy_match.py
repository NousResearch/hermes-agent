#!/usr/bin/env python3
"""Fuzzy find-and-replace for LLM-generated edits.

An ordered chain of increasingly permissive strategies (``STRATEGIES``) lets
whitespace, indentation, escaping and Unicode drift in tool-call arguments
still land on the intended region::

    new_content, match_count, strategy, error = fuzzy_find_and_replace(
        content, old_string, new_string, replace_all=False)
"""

import re
from difflib import SequenceMatcher
from typing import Callable, Optional

Span = tuple[int, int]

IDENTICAL_STRINGS_ERROR = (
    "No edit was applied because old_string and new_string are identical. "
    "Provide the existing text to replace in old_string and the changed "
    "replacement text in new_string.")

UNICODE_MAP = {
    "\u201c": '"', "\u201d": '"',  # smart double quotes
    "\u2018": "'", "\u2019": "'",  # smart single quotes
    "\u2014": "--", "\u2013": "-",  # em/en dashes
    "\u2026": "...", "\u00a0": " ",  # ellipsis and non-breaking space
    "\u2212": "-",  # typographic minus (math/scientific docs)
    # Space-separator family (Zs): otherwise such files fall to the similarity fallback.
    "\u2000": " ", "\u2001": " ", "\u2002": " ", "\u2003": " ",
    "\u2004": " ", "\u2005": " ", "\u2006": " ", "\u2007": " ",
    "\u2008": " ", "\u2009": " ", "\u200a": " ", "\u202f": " ",
    "\u205f": " ", "\u3000": " "}


def _unicode_normalize(text: str) -> str:
    """Map typographic Unicode variants to ASCII equivalents."""
    for char, repl in UNICODE_MAP.items():
        text = text.replace(char, repl)
    return text


# ── Position helpers ─────────────────────────────────────────────────────

def _calculate_line_positions(content_lines: list[str], start_line: int,
                              end_line: int, content_length: int) -> Span:
    """Character span covering ``content_lines[start_line:end_line]`` (end exclusive)."""
    start_pos = sum(len(line) + 1 for line in content_lines[:start_line])
    end_pos = sum(len(line) + 1 for line in content_lines[:end_line]) - 1
    return start_pos, min(content_length, end_pos)


def _window_spans(content: str, content_lines: list[str], n: int,
                  accept: Callable[[int], bool]) -> list[Span]:
    """Spans of every ``n``-line window starting at ``i`` for which ``accept(i)``."""
    return [
        _calculate_line_positions(content_lines, i, i + n, len(content))
        for i in range(len(content_lines) - n + 1) if accept(i)]


def _match_transformed_lines(content: str, pattern: str,
                             transform: Callable[[list[str]], list[str]]) -> list[Span]:
    """Match ``pattern`` against ``content`` after applying ``transform`` to each line block."""
    content_lines = content.split('\n')
    pattern_norm = transform(pattern.split('\n'))
    n = len(pattern_norm)
    return _window_spans(content, content_lines, n,
                         lambda i: transform(content_lines[i:i + n]) == pattern_norm)


def _strip_boundary(lines: list[str]) -> list[str]:
    """Strip whitespace on the first and last lines only."""
    lines = list(lines)
    lines[0] = lines[0].strip()
    if len(lines) > 1:
        lines[-1] = lines[-1].strip()
    return lines


def _build_orig_to_norm_map(original: str) -> list[int]:
    """Map each original index to its index in ``_unicode_normalize(original)``
    (replacements can expand one char into several). Length ``len(original)+1``;
    the last entry is a sentinel one past the final character."""
    result: list[int] = []
    norm_pos = 0
    for char in original:
        result.append(norm_pos)
        repl = UNICODE_MAP.get(char)
        norm_pos += len(repl) if repl is not None else 1
    result.append(norm_pos)
    return result


def _invert_norm_map(orig_to_norm: list[int]) -> dict[int, int]:
    """norm_pos -> first original position mapping to it."""
    inverted: dict[int, int] = {}
    for orig_pos, norm_pos in enumerate(orig_to_norm[:-1]):
        inverted.setdefault(norm_pos, orig_pos)
    return inverted


def _norm_end_to_orig(orig_to_norm: list[int], orig_start: int, norm_end: int) -> int:
    """Walk from ``orig_start`` until the mapped position reaches ``norm_end``."""
    orig_len = len(orig_to_norm) - 1
    orig_end = orig_start
    while orig_end < orig_len and orig_to_norm[orig_end] < norm_end:
        orig_end += 1
    return orig_end


def _map_positions_norm_to_orig(orig_to_norm: list[int], norm_matches: list[Span]) -> list[Span]:
    """Convert spans in the normalised string to original-string spans."""
    norm_to_orig_start = _invert_norm_map(orig_to_norm)
    results: list[Span] = []
    for norm_start, norm_end in norm_matches:
        if norm_start in norm_to_orig_start:
            orig_start = norm_to_orig_start[norm_start]
            results.append((orig_start, _norm_end_to_orig(orig_to_norm, orig_start, norm_end)))
    return results


def _map_normalized_positions(original: str, normalized: str,
                              normalized_matches: list[Span]) -> list[Span]:
    """Best-effort span mapping for ``[ \\t]+`` -> ``' '`` whitespace collapsing."""
    orig_to_norm = []  # orig_to_norm[i] = position in normalized
    orig_idx = norm_idx = 0
    while orig_idx < len(original) and norm_idx < len(normalized):
        if original[orig_idx] == normalized[norm_idx]:
            orig_to_norm.append(norm_idx)
            orig_idx += 1
            norm_idx += 1
        elif original[orig_idx] in ' \t' and normalized[norm_idx] == ' ':
            # Collapsed run: advance norm_idx only once the run is consumed.
            orig_to_norm.append(norm_idx)
            orig_idx += 1
            if orig_idx < len(original) and original[orig_idx] not in ' \t':
                norm_idx += 1
        else:
            # Extra whitespace in original, or a mismatch normalization should
            # never produce — either way, pin to the current norm_idx.
            orig_to_norm.append(norm_idx)
            orig_idx += 1
    orig_to_norm.extend([len(normalized)] * (len(original) - orig_idx))

    norm_to_orig_start = {}
    norm_to_orig_end = {}
    for orig_pos, norm_pos in enumerate(orig_to_norm):
        norm_to_orig_start.setdefault(norm_pos, orig_pos)
        norm_to_orig_end[norm_pos] = orig_pos

    original_matches = []
    for norm_start, norm_end in normalized_matches:
        if norm_start in norm_to_orig_start:
            orig_start = norm_to_orig_start[norm_start]
        else:
            orig_start = min(i for i, n in enumerate(orig_to_norm) if n >= norm_start)
        if norm_end - 1 in norm_to_orig_end:
            orig_end = norm_to_orig_end[norm_end - 1] + 1
        else:
            orig_end = orig_start + (norm_end - norm_start)
        # Absorb trailing collapsed whitespace only when the normalized match
        # itself ended in a space; otherwise the first whitespace after the
        # match is a word boundary that must survive.
        if norm_end < len(normalized) and normalized[norm_end - 1] == ' ':
            while orig_end < len(original) and original[orig_end] in ' \t':
                orig_end += 1
        original_matches.append((orig_start, min(orig_end, len(original))))
    return original_matches


# ── Strategies ───────────────────────────────────────────────────────────
# Each takes ``(content, pattern)`` and returns ``(start, end)`` spans in the
# ORIGINAL content.

def _strategy_exact(content: str, pattern: str) -> list[Span]:
    """Strategy 1: exact, non-overlapping occurrences (str.replace semantics —
    overlapping spans would corrupt the file under replace_all)."""
    return [m.span() for m in re.finditer(re.escape(pattern), content)]


def _strategy_line_trimmed(content: str, pattern: str) -> list[Span]:
    """Strategy 2: strip each line before comparing."""
    return _match_transformed_lines(content, pattern, lambda ls: [l.strip() for l in ls])


def _strategy_whitespace_normalized(content: str, pattern: str) -> list[Span]:
    """Strategy 3: collapse runs of spaces/tabs to a single space."""
    def normalize(s):
        return re.sub(r'[ \t]+', ' ', s)

    content_normalized = normalize(content)
    matches_in_normalized = _strategy_exact(content_normalized, normalize(pattern))
    if not matches_in_normalized:
        return []
    return _map_normalized_positions(content, content_normalized, matches_in_normalized)


def _strategy_indentation_flexible(content: str, pattern: str) -> list[Span]:
    """Strategy 4: ignore leading indentation entirely."""
    return _match_transformed_lines(content, pattern, lambda ls: [l.lstrip() for l in ls])


def _strategy_escape_normalized(content: str, pattern: str) -> list[Span]:
    """Strategy 5: treat literal ``\\n``/``\\t``/``\\r`` in the pattern as control chars."""
    pattern_unescaped = pattern.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
    if pattern_unescaped == pattern:
        return []
    return _strategy_exact(content, pattern_unescaped)


def _strategy_trimmed_boundary(content: str, pattern: str) -> list[Span]:
    """Strategy 6: strip whitespace on the first and last lines only."""
    return _match_transformed_lines(content, pattern, _strip_boundary)


def _strategy_unicode_normalized(content: str, pattern: str) -> list[Span]:
    """Strategy 7: exact/line-trimmed match after Unicode->ASCII normalisation of both sides."""
    norm_pattern = _unicode_normalize(pattern)
    norm_content = _unicode_normalize(content)
    if norm_content == content and norm_pattern == pattern:
        return []
    norm_matches = (_strategy_exact(norm_content, norm_pattern)
                    or _strategy_line_trimmed(norm_content, norm_pattern))
    if not norm_matches:
        return []
    return _map_positions_norm_to_orig(_build_orig_to_norm_map(content), norm_matches)


def _strategy_block_anchor(content: str, pattern: str) -> list[Span]:
    """Strategy 8: anchor on first+last lines, similarity-score the middle."""
    pattern_lines = _unicode_normalize(pattern).split('\n')
    if len(pattern_lines) < 2:
        return []
    first_line = pattern_lines[0].strip()
    last_line = pattern_lines[-1].strip()
    n = len(pattern_lines)

    # Match on normalized lines; compute offsets from the ORIGINAL lines so
    # multi-char expansions (em-dash -> '--') don't shift positions.
    norm_content_lines = _unicode_normalize(content).split('\n')
    potential_matches = {
        i for i in range(len(norm_content_lines) - n + 1)
        if norm_content_lines[i].strip() == first_line
        and norm_content_lines[i + n - 1].strip() == last_line}
    # Looser thresholds (0.10/0.30) matched unrelated blocks; these are the safe floor.
    threshold = 0.50 if len(potential_matches) == 1 else 0.70
    pattern_middle = '\n'.join(pattern_lines[1:-1])

    def similar(i: int) -> bool:
        if i not in potential_matches:
            return False
        if n <= 2:
            return True
        content_middle = '\n'.join(norm_content_lines[i + 1:i + n - 1])
        return SequenceMatcher(None, content_middle, pattern_middle).ratio() >= threshold

    return _window_spans(content, content.split('\n'), n, similar)


def _strategy_context_aware(content: str, pattern: str) -> list[Span]:
    """Strategy 9 (last resort): anchored per-line similarity, every non-blank line >= 0.80.
    The anchor pre-filter bounds the scan; the all-lines rule stops coincidental matches."""
    pattern_lines = pattern.split('\n')
    content_lines = content.split('\n')
    n = len(pattern_lines)
    if n > len(content_lines):
        return []
    first_pat = pattern_lines[0].strip()
    last_pat = pattern_lines[-1].strip()

    def _sim(a: str, b: str) -> float:
        return 1.0 if a == b else SequenceMatcher(None, a, b).ratio()

    def accept(i: int) -> bool:
        block_lines = content_lines[i:i + n]
        if _sim(first_pat, block_lines[0].strip()) < 0.80:
            return False
        if _sim(last_pat, block_lines[-1].strip()) < 0.80:
            return False
        return all(
            not p_line.strip() or _sim(p_line.strip(), c_line.strip()) >= 0.80
            for p_line, c_line in zip(pattern_lines, block_lines))

    return _window_spans(content, content_lines, n, accept)


# Ordered chain: precise strategies first, similarity-based last.
STRATEGIES: list[tuple[str, Callable[[str, str], list[Span]]]] = [
    ("exact", _strategy_exact),
    ("line_trimmed", _strategy_line_trimmed),
    ("whitespace_normalized", _strategy_whitespace_normalized),
    ("indentation_flexible", _strategy_indentation_flexible),
    ("escape_normalized", _strategy_escape_normalized),
    ("trimmed_boundary", _strategy_trimmed_boundary),
    ("unicode_normalized", _strategy_unicode_normalized),
    ("block_anchor", _strategy_block_anchor),
    ("context_aware", _strategy_context_aware)]

# Matches from these only *approximately* resemble old_string — fine for one
# unique replacement, never safe under replace_all.
SIMILARITY_STRATEGIES = frozenset({"block_anchor", "context_aware"})


# ── Orchestrator ─────────────────────────────────────────────────────────

def is_already_applied(content: str, old_string: str, new_string: str) -> bool:
    """True when the edit is already present (re-sent edit -> success-shaped no-op).
    Conservative: new_string non-trivial (>= 8 chars) and present EXACTLY; old_string gone."""
    if not new_string or len(new_string.strip()) < 8 or new_string not in content:
        return False
    return old_string == new_string or old_string not in content


def _matched_regions(content: str, matches: list[Span]) -> str:
    return "".join(content[start:end] for start, end in matches)


def _format_match_locations(content: str, matches: list[Span], cap: int = 5) -> str:
    """Render up to ``cap`` match positions as 'L<line>: <snippet>' rows."""
    rows = []
    for start, _end in matches[:cap]:
        line_no = content.count("\n", 0, start) + 1
        line_start = content.rfind("\n", 0, start) + 1
        line_end = content.find("\n", line_start)
        if line_end == -1:
            line_end = len(content)
        snippet = content[line_start:line_end].strip()
        if len(snippet) > 80:
            snippet = snippet[:77] + "..."
        rows.append(f"  L{line_no}: {snippet}")
    extra = len(matches) - cap
    if extra > 0:
        rows.append(f"  ... and {extra} more")
    return "\n".join(rows)


def fuzzy_find_and_replace(content: str, old_string: str, new_string: str,
                           replace_all: bool = False) -> tuple[str, int, Optional[str], Optional[str]]:
    """Find and replace via the strategy chain.

    Returns ``(new_content, match_count, strategy_name, error)``; on failure
    ``(content, 0, None, error)``.
    """
    if not old_string:
        return content, 0, None, "old_string cannot be empty"
    if not old_string.strip():
        # Whitespace-only anchors match trivially and mass-replace or
        # ambiguity-error; never meaningful.
        return content, 0, None, "old_string is only whitespace — provide non-blank text to match"
    if old_string == new_string:
        return content, 0, None, IDENTICAL_STRINGS_ERROR

    for strategy_name, strategy_fn in STRATEGIES:
        matches = strategy_fn(content, old_string)
        if not matches:
            continue

        if len(matches) > 1 and not replace_all:
            locations = _format_match_locations(content, matches)
            return content, 0, None, (
                f"Found {len(matches)} matches for old_string. "
                f"Provide more context to make it unique, or use replace_all=True. "
                f"Matches:\n{locations}")
        if replace_all and len(matches) > 1 and strategy_name in SIMILARITY_STRATEGIES:
            return content, 0, None, (
                f"Found {len(matches)} approximate matches via the "
                f"'{strategy_name}' strategy; replace_all only applies to exact "
                f"matches. Provide the precise text (whitespace included) so an "
                f"exact/line-trimmed match can be made.")

        # Non-exact matches came through some normalization, so new_string may
        # carry serialization drift the file doesn't have.
        if strategy_name != "exact":
            drift_err = _detect_escape_drift(content, matches, old_string, new_string)
            if drift_err:
                return content, 0, None, drift_err

        effective_new = _maybe_unescape_new_string(new_string, content, matches)
        if strategy_name == "unicode_normalized":
            effective_new = _preserve_unicode_in_replacement(content, matches, old_string, effective_new)
        new_content = _apply_replacements(
            content, matches, effective_new,
            old_string=old_string if strategy_name != "exact" else None)
        return new_content, len(matches), strategy_name, None

    return content, 0, None, "Could not find a match for old_string in the file"


# ── Escape-drift guards ──────────────────────────────────────────────────

def _detect_escape_drift(content: str, matches: list[Span],
                         old_string: str, new_string: str) -> Optional[str]:
    """Error string when new_string carries tool-call escape artifacts, else None:
    ``\\'``/``\\"`` in both strings but not the matched region, or doubled backslash runs."""
    has_quote_suspects = "\\'" in new_string or '\\"' in new_string
    if not has_quote_suspects and "\\" not in old_string:
        return None

    matched_regions = _matched_regions(content, matches)
    if has_quote_suspects:
        for suspect in ("\\'", '\\"'):
            if suspect in new_string and suspect in old_string and suspect not in matched_regions:
                plain = suspect[1]
                return (
                    f"Escape-drift detected: old_string and new_string contain "
                    f"the literal sequence {suspect!r} but the matched region of "
                    f"the file does not. This is almost always a tool-call "
                    f"serialization artifact where an apostrophe or quote got "
                    f"prefixed with a spurious backslash. Re-read the file with "
                    f"read_file and pass old_string/new_string without "
                    f"backslash-escaping {plain!r} characters.")
    return _detect_backslash_doubling(matched_regions, old_string, new_string)


def _backslash_runs(s: str) -> list[int]:
    """Lengths of maximal backslash runs in ``s``, in order."""
    return [len(run) for run in re.findall(r"\\+", s)]


def _detect_backslash_doubling(matched_regions: str, old_string: str,
                               new_string: str) -> Optional[str]:
    """Detect old_string whose every backslash run is exactly 2x the file's (arguments
    JSON-escaped one extra time). Requires the same run count, a non-trivial signal
    (a run >= 2 or 2+ runs), and new_string not already matching the file's counts."""
    old_runs = _backslash_runs(old_string)
    file_runs = _backslash_runs(matched_regions)
    if (not old_runs or not file_runs or len(old_runs) != len(file_runs)
            or old_runs == file_runs
            or any(o != f * 2 for o, f in zip(old_runs, file_runs))
            or not (any(f >= 2 for f in file_runs) or len(file_runs) >= 2)
            or _backslash_runs(new_string) == file_runs):
        return None
    return (
        "Escape-drift detected: every backslash run in old_string is exactly "
        "twice as long as in the matched region of the file (e.g. the file "
        "has `\\\\` where old_string has `\\\\\\\\`). The tool-call arguments "
        "were JSON-escaped one extra time; applying new_string verbatim would "
        "double every backslash in the file. Re-read the file with read_file "
        "and resend old_string/new_string with the backslash counts exactly "
        "as they appear in the file.")


def _maybe_unescape_new_string(new_string: str, content: str, matches: list[Span]) -> str:
    """Convert literal ``\\t``/``\\r`` in new_string to control chars, per sequence, only
    when the matched region already contains the real control char (so ``sep = "\\t"`` files
    are left alone). ``\\n`` is excluded: rewriting it would mangle source escape literals."""
    if "\\t" not in new_string and "\\r" not in new_string:
        return new_string
    matched_regions = _matched_regions(content, matches)
    for literal, control in (("\\t", "\t"), ("\\r", "\r")):
        if literal in new_string and control in matched_regions:
            new_string = new_string.replace(literal, control)
    return new_string


# ── Replacement shaping ──────────────────────────────────────────────────

def _leading_whitespace(line: str) -> str:
    return line[:len(line) - len(line.lstrip(" \t"))]


def _first_meaningful_line(text: str) -> Optional[str]:
    return next((line for line in text.split("\n") if line.strip()), None)


def _reindent_replacement(file_region: str, old_string: str, new_string: str) -> str:
    """Re-anchor ``new_string``'s indentation onto the file's actual base indent after a
    non-exact match: swap the LLM base prefix (first non-blank old_string line) for the
    file's, preserving relative nesting; shallower lines anchor to the file base."""
    if not new_string:
        return new_string
    old_first = _first_meaningful_line(old_string)
    file_first = _first_meaningful_line(file_region)
    if old_first is None or file_first is None:
        return new_string
    old_indent = _leading_whitespace(old_first)
    file_indent = _leading_whitespace(file_first)
    if old_indent == file_indent:
        return new_string

    out_lines: list[str] = []
    for line in new_string.split("\n"):
        if not line.strip():
            out_lines.append(line)
        elif _leading_whitespace(line).startswith(old_indent):
            out_lines.append(file_indent + line[len(old_indent):])
        else:
            out_lines.append(file_indent + line.lstrip(" \t"))
    return "\n".join(out_lines)


def _preserve_unicode_in_replacement(content: str, matches: list[Span],
                                     old_string: str, new_string: str) -> str:
    """Apply only the old->new edits onto the file's original (Unicode) text, so a
    unicode_normalized match doesn't flatten the file's em-dashes/smart quotes."""
    file_region = _matched_regions(content, matches)
    norm_old = _unicode_normalize(old_string)
    if norm_old != _unicode_normalize(file_region):
        return new_string  # strategy shouldn't have fired; fall back

    file_orig_to_norm = _build_orig_to_norm_map(file_region)
    file_norm_to_orig = _invert_norm_map(file_orig_to_norm)

    result_parts: list[str] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, norm_old, new_string).get_opcodes():
        if tag == "equal":
            orig_start = file_norm_to_orig.get(i1, 0)
            orig_end = _norm_end_to_orig(file_orig_to_norm, orig_start, i2)
            result_parts.append(file_region[orig_start:orig_end])
        elif tag != "delete":
            result_parts.append(new_string[j1:j2])
    return "".join(result_parts)


def _apply_replacements(content: str, matches: list[Span],
                        new_string: str, old_string: Optional[str] = None) -> str:
    """Splice ``new_string`` over each span (end-to-start so offsets stay valid);
    ``old_string`` non-None (non-exact match) re-indents it per region."""
    result = content
    for start, end in sorted(matches, key=lambda x: x[0], reverse=True):
        adjusted = new_string
        if old_string is not None:
            adjusted = _reindent_replacement(content[start:end], old_string, new_string)
        result = result[:start] + adjusted + result[end:]
    return result


# ── "Did you mean?" diagnostics ──────────────────────────────────────────

def _visualize_whitespace(line: str) -> str:
    """Render the leading whitespace run visibly (→ = tab, · = space)."""
    stripped = line.lstrip(" \t")
    prefix = line[:len(line) - len(stripped)]
    return prefix.replace("\t", "→").replace(" ", "·") + stripped


def find_closest_lines(old_string: str, content: str, context_lines: int = 2, max_results: int = 3) -> str:
    """Numbered snippets of the lines most similar to old_string's anchor line, or ''."""
    if not old_string or not content:
        return ""
    old_lines = old_string.splitlines()
    content_lines = content.splitlines()
    if not old_lines or not content_lines:
        return ""

    anchor = old_lines[0].strip() or next((l.strip() for l in old_lines if l.strip()), "")
    if not anchor:
        return ""

    scored = sorted(((SequenceMatcher(None, anchor, line.strip()).ratio(), i)
                     for i, line in enumerate(content_lines) if line.strip()), key=lambda x: -x[0])
    top = [s for s in scored if s[0] > 0.3][:max_results]
    if not top:
        return ""

    parts = []
    seen_ranges = set()
    for _, line_idx in top:
        start = max(0, line_idx - context_lines)
        end = min(len(content_lines), line_idx + len(old_lines) + context_lines)
        if (start, end) in seen_ranges:
            continue
        seen_ranges.add((start, end))
        parts.append("\n".join(
            f"{start + j + 1:4d}| {content_lines[start + j]}" for j in range(end - start)))
    result = "\n---\n".join(parts)

    # Whitespace-shaped miss: best line equals the anchor once stripped. Show
    # both with visible leading whitespace so the model copies the file's.
    best_line = content_lines[top[0][1]]
    if best_line.strip() == anchor and best_line != old_lines[0]:
        result += (
            "\n\nWhitespace difference detected (→ = tab, · = space):\n"
            f"  file has: {_visualize_whitespace(best_line)}\n"
            f"  you sent: {_visualize_whitespace(old_lines[0])}\n"
            "Use the exact whitespace shown in 'file has'.")
    return result


def format_no_match_hint(error: Optional[str], match_count: int,
                         old_string: str, content: str) -> str:
    """'\\n\\nDid you mean...' snippet for plain no-match errors only, else '' (ambiguous /
    escape-drift / identical errors also have ``match_count == 0`` but a hint would mislead)."""
    if match_count != 0 or not error or not error.startswith("Could not find"):
        return ""
    hint = find_closest_lines(old_string, content)
    return "\n\nDid you mean one of these sections?\n" + hint if hint else ""


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import List  # noqa: F401,E402
from typing import Tuple  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
