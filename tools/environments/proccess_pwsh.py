"""Transform PowerShell 7.x syntax to PowerShell 5.1 compatible syntax.

PowerShell 7 introduced several expression-level operators that do not exist in
PowerShell 5.1:

  * Ternary:          $cond ? $true_expr : $false_expr
  * Null-coalescing:  $a ?? $fallback
  * Null-assign:      $a ??= $default
  * Pipeline chains:  cmd1 && cmd2   /   cmd1 || cmd2
  * Null-conditional: $obj?.Property / $obj?[0]

This module performs a *source-to-source* transformation.  It operates on raw
text rather than an AST because the target environment (5.1) cannot parse the
new syntax in the first place.
"""

from __future__ import annotations

import re
from bisect import bisect_right


# ===========================================================================
# Constants
# ===========================================================================

_PS_KEYWORDS = frozenset({
    "begin", "break", "catch", "class", "continue", "data", "define", "do",
    "dynamicparam", "else", "elseif", "end", "enum", "exit", "filter", "finally",
    "for", "foreach", "from", "function", "hidden", "if", "in", "param",
    "process", "return", "static", "switch", "throw", "trap", "try", "until",
    "using", "var", "while",
})

_VAR_FIRST_CHARS = '$(["\'@0123456789'

_EXPR_STOP = "=;|&,"

_DEPTH_OPEN = "([{"
_DEPTH_CLOSE = ")]}"


# ===========================================================================
# Low-level scanners — skip over strings, comments, subexpressions
# ===========================================================================

def _scan_single_quoted(code: str, i: int) -> int:
    """Skip a single-quoted string starting at *i*; return index after it."""
    i += 1
    n = len(code)
    while i < n:
        if code[i] == "'":
            if i + 1 < n and code[i + 1] == "'":
                i += 2          # escaped '' → literal single-quote
            else:
                return i + 1     # closing quote
        else:
            i += 1
    return i


def _scan_double_quoted(code: str, i: int) -> int:
    """Skip a double-quoted string starting at *i*; return index after it."""
    i += 1
    n = len(code)
    while i < n:
        ch = code[i]
        if ch == "`" and i + 1 < n:
            i += 2              # backtick-escaped char
        elif ch == '"':
            return i + 1         # closing quote
        elif ch == "$" and i + 1 < n and code[i + 1] == "(":
            i = _skip_subexpression(code, i)
        else:
            i += 1
    return i


def _scan_block_comment(code: str, i: int) -> int:
    """Skip a block comment ``<# ... #>`` starting at *i*; return index after it."""
    depth = 1
    i += 2
    n = len(code)
    while i < n and depth:
        if code[i] == "<" and i + 1 < n and code[i + 1] == "#":
            depth += 1
            i += 2
        elif code[i] == "#" and i + 1 < n and code[i + 1] == ">":
            depth -= 1
            i += 2
        else:
            i += 1
    return i


def _skip_subexpression(code: str, start: int) -> int:
    """Skip past a ``$(...)`` sub-expression starting at *start*.

    Returns the index *after* the closing ``)``.
    """
    assert code[start] == "$"
    i = start + 2
    depth = 1
    n = len(code)
    while i < n and depth:
        c = code[i]
        if c == "(":
            depth += 1
            i += 1
        elif c == ")":
            depth -= 1
            i += 1
        elif c == "'":
            i = _scan_single_quoted(code, i)
        elif c == '"':
            i = _scan_double_quoted(code, i)
        elif c == "$" and i + 1 < n and code[i + 1] == "(":
            i = _skip_subexpression(code, i)
        else:
            i += 1
    return i


# ===========================================================================
# Region finders  (strings, comments, here-strings)
# ===========================================================================

def _find_regions(code: str, *, here_strings: bool = True) -> list[tuple[int, int]]:
    """Return intervals covering all string/comment regions in *code*.

    When *here_strings* is False, here-strings are not detected (used for
    line-level scanning where here-strings cannot be reliably identified).
    """
    regions: list[tuple[int, int]] = []
    i = 0
    n = len(code)
    while i < n:
        c = code[i]
        if c == "<" and i + 1 < n and code[i + 1] == "#":
            start = i
            i = _scan_block_comment(code, i)
            regions.append((start, i))
        elif c == "#":
            start = i
            if here_strings:
                while i < n and code[i] != "\n":
                    i += 1
            else:
                i = n  # rest of line is comment
            regions.append((start, i))
        elif here_strings and c == "@" and i + 1 < n and code[i + 1] in ("'", '"'):
            j = i + 2
            while j < n and code[j] in " \t\r":
                j += 1
            if j < n and code[j] != "\n":
                i += 1
                continue
            start = i
            quote_char = code[i + 1]
            i += 2
            while i < n:
                if code[i] == quote_char and i + 1 < n and code[i + 1] == "@":
                    line_start = code.rfind("\n", 0, i)
                    line_start = 0 if line_start == -1 else line_start + 1
                    if code[line_start:i].strip() == "":
                        i += 2
                        break
                i += 1
            regions.append((start, i))
        elif c == "'":
            start = i
            i = _scan_single_quoted(code, i)
            regions.append((start, i))
        elif c == '"':
            start = i
            i = _scan_double_quoted(code, i)
            regions.append((start, i))
        else:
            i += 1
    return regions


# ===========================================================================
# Fast region mask: O(1) per-character lookups
# ===========================================================================

def _region_mask(length: int, regions: list[tuple[int, int]]) -> bytearray:
    """Return a bytearray where 1 means outside regions and 0 means inside.

    This replaces repeated ``_outside_regions`` calls (O(log r) each) with
    a simple index lookup (O(1)).
    """
    mask = bytearray(b"\x01" * length)
    for start, end in regions:
        if start >= length:
            break
        end = min(end, length)
        mask[start:end] = b"\x00" * (end - start)
    return mask


def _line_mask(line: str) -> bytearray:
    """Return a region mask for *line* (here-strings disabled)."""
    return _region_mask(len(line), _find_regions(line, here_strings=False))


# ===========================================================================
# Depth tracking  (for matching ternary colon)
# ===========================================================================

def _compute_depths(line: str, mask: bytearray) -> list[int]:
    """Return nesting depth of ``()``, ``{}``, ``[]`` before each character."""
    depths: list[int] = []
    depth = 0
    for i, ch in enumerate(line):
        depths.append(depth)
        if mask[i]:
            if ch in _DEPTH_OPEN:
                depth += 1
            elif ch in _DEPTH_CLOSE:
                depth -= 1
    depths.append(depth)
    return depths


# ===========================================================================
# Pre-processing: backtick line continuation
# ===========================================================================

def _join_continuation_lines(code: str) -> str:
    """Collapse backtick line-continuations into single logical lines."""
    regions = _find_regions(code)
    n = len(code)
    mask = _region_mask(n, regions)
    result: list[str] = []
    i = 0
    while i < n:
        if code[i] == "`" and mask[i]:
            j = i + 1
            while j < n and code[j] in " \t\r":
                j += 1
            if j < n and code[j] == "\n":
                j += 1
                while j < n and code[j] in " \t\r":
                    j += 1
                result.append(" ")
                i = j
                continue
        result.append(code[i])
        i += 1
    return "".join(result)


# ===========================================================================
# Assignment detection
# ===========================================================================

_ASSIGN_RE = re.compile(r"(.*?)(\$\w+(?::\w+)?(?:\.\w+)*)\s*=\s*$")
_COMMAND_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*\s+")


def _match_assignment(before: str) -> tuple[str, str] | None:
    """Match an assignment prefix like ``$var = `` at the end of *before*."""
    m = _ASSIGN_RE.match(before)
    if m:
        return m.group(1), m.group(2)
    return None


def _build_replacement(prefix: str, inner: str) -> str:
    """Build replacement string, preserving an assignment if one is detected."""
    assign = _match_assignment(prefix.rstrip())
    if assign:
        p, var = assign
        return f"{p}{var} = {inner}"
    return f"{prefix}{inner}"


def _strip_command_prefix(expr: str, start: int) -> tuple[str, int]:
    """Strip a leading command name (e.g. ``Write-Output ``) from *expr*.

    Returns ``(stripped_expr, adjusted_start)``.
    PowerShell keywords (if, foreach, …) are never stripped.
    """
    m = _COMMAND_PREFIX_RE.match(expr)
    if m:
        cmd = m.group(0).strip().lower()
        if cmd not in _PS_KEYWORDS:
            expr_part = expr[m.end():]
            if expr_part and expr_part[0] in _VAR_FIRST_CHARS:
                return expr_part, start + m.end()
    return expr, start


# ===========================================================================
# $? adjacency check — shared by all transforms
# ===========================================================================

def _after_dollar_question(line: str, op_idx: int) -> bool:
    """Return True if *op_idx* is immediately after ``$?``.

    When True the first ``?`` of the operator at *op_idx* is actually the
    ``?`` of the ``$?`` automatic variable, so the operator should be skipped.

    Does NOT match ``$$?.`` — ``$$`` is a separate automatic variable.
    """
    return (
        op_idx > 0
        and line[op_idx - 1] == "$"
        and not (op_idx > 1 and line[op_idx - 2] == "$")
    )


# ===========================================================================
# Shared colon-context check
# ===========================================================================

def _is_scope_colon(line: str, i: int) -> bool:
    """Return ``True`` if the colon at *i* belongs to a ``$scope:var`` prefix."""
    if i > 0 and (line[i - 1].isalnum() or line[i - 1] == "_"):
        j = i - 1
        while j > 0 and (line[j].isalnum() or line[j] == "_"):
            j -= 1
        if line[j] == "$":
            return True
    return False


# ===========================================================================
# Expression boundary helpers
# ===========================================================================

def _find_expr_start(line: str, end: int, mask: bytearray,
                     extra_stop: str = "") -> int:
    """Scan backwards from *end* to locate the start of the expression.

    *extra_stop* can contain additional delimiter characters (e.g. ``"?:``
    for null-conditional base scanning).
    """
    depth = 0
    stop_set = _EXPR_STOP + extra_stop if extra_stop else _EXPR_STOP
    for i in range(end - 1, -1, -1):
        if not mask[i]:
            continue
        c = line[i]
        if c in _DEPTH_CLOSE:
            depth += 1
        elif c in _DEPTH_OPEN:
            depth -= 1
            if depth < 0:
                return i + 1
        elif depth == 0 and c in stop_set:
            if c == "?":
                if i + 1 < len(line) and line[i + 1] == ".":
                    continue  # ?. is null-conditional
                if _after_dollar_question(line, i):
                    continue  # $? auto var (but not $$)
            elif c == ":":
                if i + 1 < len(line) and line[i + 1] == ":":
                    continue  # :: static member
                if i > 0 and line[i - 1] == ":":
                    continue  # :: static member (right side)
                if _is_scope_colon(line, i):
                    continue  # $scope:var
            return i + 1
    return 0


def _find_expr_end(line: str, start: int, mask: bytearray) -> int:
    """Scan forwards from *start* to locate the end of the expression."""
    depth = 0
    # Track mask value at the previous position to detect region boundaries.
    prev_mask = 1 if start == 0 else mask[start - 1]
    for i in range(start, len(line)):
        c = line[i]
        cur_mask = mask[i]
        if not cur_mask:
            # If we just stepped from outside (mask=1) into a region and the
            # character is '#', it is the start of a line comment — stop here.
            if prev_mask and c == "#":
                return i
            prev_mask = cur_mask
            continue
        prev_mask = cur_mask
        if c in _DEPTH_OPEN:
            depth += 1
        elif c in _DEPTH_CLOSE:
            depth -= 1
            if depth < 0:
                return i
        elif depth == 0:
            # Keep the direct '#' check as a defensive fallback (e.g. if mask
            # handling changes in the future).
            if c == "#":
                return i
            if c in _EXPR_STOP:
                return i
    return len(line)


def _expr_left(line: str, pos: int, mask: bytearray,
               extra_stop: str = "") -> tuple[int, int]:
    """Return (start, end) of the expression immediately left of *pos*."""
    end = pos
    while end > 0 and line[end - 1] == " ":
        end -= 1
    start = _find_expr_start(line, end, mask, extra_stop)
    return start, end


def _expr_right(line: str, pos: int, mask: bytearray) -> tuple[int, int]:
    """Return (start, end) of the expression immediately right of *pos*."""
    start = pos
    while start < len(line) and line[start] == " ":
        start += 1
    end = _find_expr_end(line, start, mask)
    return start, end


# ===========================================================================
# Generic operator transform engine
# ===========================================================================

def _find_next_op(line: str, op: str, mask: bytearray,
                  after_dq_check: bool = False, start: int = 0) -> int:
    """Find the next occurrence of *op* in *line* that is outside regions.

    If *after_dq_check* is True, skip positions immediately after ``$?``.
    *start* restricts the search to indices >= *start*.
    """
    idx = line.find(op, start)
    while idx != -1:
        if mask[idx] and (not after_dq_check or not _after_dollar_question(line, idx)):
            return idx
        idx = line.find(op, idx + 1)
    return -1


def _transform_operator(
    line: str,
    op: str,
    builder: callable,
    *,
    right_scanner: callable | None = None,
    skip_dq: bool = True,
) -> tuple[str, list[str]]:
    """Generic single-operator line transformer.

    *op*          — the operator string to search for (e.g. ``"??"``).
    *builder*     — callable(left_expr, right_expr, right_extra) → inner_str.
    *right_scanner* — optional callable(line, pos, mask) → (extra, end_pos).
    *skip_dq*    — when True, skip positions immediately after ``$?``.

    Returns ``(transformed_line, warnings)``.
    """
    warnings: list[str] = []
    op_len = len(op)
    search = 0
    while True:
        mask = _line_mask(line)
        idx = _find_next_op(line, op, mask, skip_dq, search)
        if idx == -1:
            break

        left_start, left_end = _expr_left(line, idx, mask)
        left_expr = line[left_start:left_end].strip()
        left_expr, left_start = _strip_command_prefix(left_expr, left_start)

        if right_scanner is not None:
            right_extra, right_end = right_scanner(line, idx + op_len, mask)
            right_expr = line[idx + op_len:right_end].strip()
        else:
            right_start, right_end = _expr_right(line, idx + op_len, mask)
            right_expr = line[right_start:right_end].strip()
            right_extra = None

        if not left_expr or not right_expr:
            search = idx + 1
            continue

        inner = builder(left_expr, right_expr, right_extra)
        warnings.append(f"{op} operator `{left_expr} {op} {right_expr}` rewritten to `{inner}`")
        line = _build_replacement(line[:left_start], inner) + line[right_end:]
        search = left_start
    return line, warnings


# ===========================================================================
# Transform: null-coalescing assignment  (??=)
# ===========================================================================

def _transform_nca_line(line: str) -> tuple[str, list[str]]:
    """Rewrite null-coalescing assignment ``$var ??= value``.

    Uses the generic expression scanner so braced variables (``${foo}``),
    property chains (``$obj.Name``) and indexed targets (``$arr[0]``) are
    all handled correctly.
    """
    warnings: list[str] = []
    search = 0
    while True:
        mask = _line_mask(line)
        idx = _find_next_op(line, "??=", mask, True, search)
        if idx == -1:
            break

        left_start, left_end = _expr_left(line, idx, mask)
        var = line[left_start:left_end].strip()
        if not var:
            search = idx + 1
            continue

        val_start, val_end = _expr_right(line, idx + 3, mask)
        value = line[val_start:val_end].strip()
        before = line[:left_start].rstrip()
        prefix = f"{before} " if before else ""
        new_inner = f"if ($null -eq {var}) {{ {var} = {value} }}"
        warnings.append(
            f"null-coalescing assignment `{var} ??= {value}` "
            f"rewritten to `{new_inner}`"
        )
        line = f"{prefix}{new_inner}" + line[val_end:]
        search = 0
    return line, warnings


# ===========================================================================
# Transform: null-coalescing  (??)
# ===========================================================================

def _transform_nc_line(line: str) -> tuple[str, list[str]]:
    """Transform every ``??`` on *line* into PS 5.1 compatible ``if`` form."""

    def _builder(left: str, right: str, _extra: None) -> str:
        return f"if ($null -ne {left}) {{ {left} }} else {{ {right} }}"

    return _transform_operator(line, "??", _builder)


# ===========================================================================
# Transform: ternary  (? :)
# ===========================================================================

def _find_matching_colon(line: str, start: int, mask: bytearray,
                         depth_arr: list[int]) -> int:
    """Find the colon that separates the true/false branches of a ternary."""
    for i in range(start, len(line)):
        if line[i] != ":" or depth_arr[i] != 0 or not mask[i]:
            continue
        if (i > 0 and line[i - 1] == ":") or (i + 1 < len(line) and line[i + 1] == ":"):
            continue
        if _is_scope_colon(line, i):
            continue
        return i
    return -1


def _transform_ternary_line(line: str) -> tuple[str, list[str]]:
    """Rewrite ternary ``$cond ? $true : $false`` into an ``if`` statement."""
    warnings: list[str] = []
    mask = _line_mask(line)
    depth_arr = _compute_depths(line, mask)
    pos = 0
    while pos < len(line):
        if (
            line[pos] == "?"
            and mask[pos]
            and not (pos > 0 and line[pos - 1] == "$")
        ):
            colon_pos = _find_matching_colon(line, pos + 1, mask, depth_arr)
            if colon_pos != -1:
                cond_start, cond_end = _expr_left(line, pos, mask)
                condition = line[cond_start:cond_end].strip()
                true_expr = line[pos + 1:colon_pos].strip()
                false_start, false_end = _expr_right(line, colon_pos + 1, mask)
                false_expr = line[false_start:false_end].strip()
                if not _match_assignment(line[:cond_start].rstrip()) and condition:
                    m = _COMMAND_PREFIX_RE.match(condition)
                    if m:
                        expr_part = condition[m.end():]
                        if expr_part and expr_part[0] in _VAR_FIRST_CHARS:
                            cond_start += m.end()
                            condition = expr_part
                inner = f"if ({condition}) {{ {true_expr} }} else {{ {false_expr} }}"
                warnings.append(
                    f"ternary operator `{condition} ? {true_expr} : {false_expr}` "
                    f"rewritten to `{inner}`"
                )
                suffix = line[false_end:]
                line = _build_replacement(line[:cond_start], inner) + suffix
                mask = _line_mask(line)
                depth_arr = _compute_depths(line, mask)
                pos = len(line) - len(suffix)
                continue
        pos += 1
    return line, warnings


# ===========================================================================
# Transform: pipeline chain operators  (&& / ||)
# ===========================================================================

def _transform_chain_line(line: str) -> tuple[str, list[str]]:
    """Rewrite pipeline chain operators ``&&`` and ``||``.

    Uses rightmost-first to maintain correct right-associative semantics.
    """
    warnings: list[str] = []
    while True:
        mask = _line_mask(line)
        best_pos = -1
        best_op = ""
        for op in ("&&", "||"):
            pos = line.rfind(op)
            while pos != -1:
                if mask[pos]:
                    if pos > best_pos:
                        best_pos = pos
                        best_op = op
                    break
                pos = line.rfind(op, 0, pos)
        if best_pos == -1:
            break
        condition = "$?" if best_op == "&&" else "-not $?"
        left = line[:best_pos].strip()
        right_start = best_pos + 2
        # Separate a trailing line comment from the right-hand expression
        # so it stays outside the generated ``{ }`` block (a ``#`` inside
        # braces would comment out the closing ``}``).
        right_raw_end = len(line)
        comment = ""
        prev_m = 1 if right_start == 0 else mask[right_start - 1]
        for ri in range(right_start, len(line)):
            if mask[ri] == 0 and prev_m == 1 and line[ri] == "#":
                right_raw_end = ri
                comment = line[ri:]
                break
            prev_m = mask[ri]
        right = line[right_start:right_raw_end].strip()
        new_line = f"{left}; if ({condition}) {{ {right} }}{comment}"
        warnings.append(
            f"pipeline chain `{left} {best_op} {right}` "
            f"rewritten to `{new_line}`"
        )
        line = new_line
    return line, warnings


# ===========================================================================
# Null-conditional helpers
# ===========================================================================

def _scan_member_name(line: str, ms: int, mask: bytearray) -> int:
    """Scan a member name starting at *ms*; return the index after it."""
    if ms >= len(line):
        return ms
    c0 = line[ms]
    if c0 == "$":
        me = ms + 1
        if me < len(line) and line[me] == "{":
            bd = 1
            me += 1
            while me < len(line) and bd > 0:
                if line[me] == "{":
                    bd += 1
                elif line[me] == "}":
                    bd -= 1
                me += 1
        elif me < len(line) and line[me] in "?$^":
            me += 1  # single-char automatic variables ($? $$ $^)
        else:
            while me < len(line) and (line[me].isalnum() or line[me] in "_:"):
                me += 1
        return me
    if c0 == "'":
        return _scan_single_quoted(line, ms)
    if c0 == '"':
        return _scan_double_quoted(line, ms)
    me = ms
    while me < len(line) and (line[me].isalnum() or line[me] == "_"):
        me += 1
    return me


def _scan_method_args(line: str, start: int, mask: bytearray) -> tuple[str, int]:
    """If *start* points to ``(``, scan the method argument list.

    Returns ``(args_string, index_after_closing_paren)``.
    """
    j = start
    while j < len(line) and line[j] == " ":
        j += 1
    if j >= len(line) or line[j] != "(":
        return "", start
    d = 1
    k = j + 1
    while k < len(line) and d > 0:
        if mask[k]:
            if line[k] == "(":
                d += 1
            elif line[k] == ")":
                d -= 1
        k += 1
    return line[j:k], k


# ===========================================================================
# Transform: null-conditional  (?. and ?[)
# ===========================================================================

def _transform_null_conditional_line(line: str, op: str) -> tuple[str, list[str]]:
    """Rewrite null-conditional member access (``?.``) or index access (``?[``)."""
    warnings: list[str] = []
    is_dot = op == "?."
    op_len = len(op)
    while True:
        mask = _line_mask(line)
        matched = False
        search = 0
        while True:
            idx = _find_next_op(line, op, mask, True, search)
            if idx == -1:
                break

            expr_start, expr_end = _expr_left(line, idx, mask, "?:")
            base = line[expr_start:expr_end].strip()
            base, expr_start = _strip_command_prefix(base, expr_start)
            if not base:
                search = idx + op_len
                continue

            if is_dot:
                chain: list[tuple[str, str, int]] = []
                cur = idx
                while cur < len(line) - 1 and line[cur:cur + 2] == "?.":
                    ms = cur + 2
                    while ms < len(line) and line[ms] == " ":
                        ms += 1
                    me = _scan_member_name(line, ms, mask)
                    if me == ms:
                        break
                    mem = line[ms:me]
                    args, me = _scan_method_args(line, me, mask)
                    chain.append((mem, args, me))
                    cur = me
                if not chain:
                    search = idx + 2
                    continue
                paths = [base]
                orig_parts = [base]
                for mem, args, _ in chain:
                    paths.append(f"{paths[-1]}.{mem}{args}")
                    orig_parts.append(f"{mem}{args}")
                inner = paths[-1]
                for p in reversed(paths[:-1]):
                    inner = f"if ($null -ne {p}) {{ {inner} }}"
                inner = f"$({inner})"
                end_pos = chain[-1][2]
                orig_expr = "?.".join(orig_parts)
            else:
                bracket_depth = 1
                bracket_end = idx + 2
                while bracket_end < len(line) and bracket_depth > 0:
                    c = line[bracket_end]
                    if mask[bracket_end]:
                        if c == "[":
                            bracket_depth += 1
                        elif c == "]":
                            bracket_depth -= 1
                    bracket_end += 1
                index_expr = line[idx + 2:bracket_end - 1]
                inner = f"$(if ($null -ne {base}) {{ {base}[{index_expr}] }})"
                end_pos = bracket_end
                orig_expr = f"{base}?[{index_expr}]"

            kind = "member access" if is_dot else "index"
            warnings.append(
                f"null-conditional {kind} `{orig_expr}` "
                f"rewritten to `{inner}`"
            )
            line = _build_replacement(line[:expr_start], inner) + line[end_pos:]
            matched = True
            break
        if not matched:
            break
    return line, warnings


# ===========================================================================
# Transform dispatch
# ===========================================================================

_TRANSFORMS = (
    _transform_nca_line,
    lambda line: _transform_null_conditional_line(line, "?."),
    lambda line: _transform_null_conditional_line(line, "?["),
    _transform_nc_line,
    _transform_ternary_line,
    _transform_chain_line,
)


# ===========================================================================
# Public API
# ===========================================================================

def pwsh_transform(code: str) -> tuple[str, list[str]]:
    """Transform PowerShell 7.x syntax into PowerShell 5.1 compatible syntax.

    Returns ``(transformed_code, warnings)`` where *warnings* is a list of
    human-readable messages describing each transformation that was applied.
    """
    code = _join_continuation_lines(code)
    lines = code.split("\n")
    regions = _find_regions(code)

    line_offsets = [0]
    for ln in lines[:-1]:
        line_offsets.append(line_offsets[-1] + len(ln) + 1)

    multi: set[int] = set()
    for s, e in regions:
        if "\n" not in code[s:e]:
            continue
        first = bisect_right(line_offsets, s) - 1
        last = bisect_right(line_offsets, e) - 1
        multi.update(range(first, last + 1))

    result: list[str] = []
    all_warnings: list[str] = []

    for i, line in enumerate(lines):
        if i in multi:
            result.append(line)
            continue
        for xform in _TRANSFORMS:
            line, w = xform(line)
            if w:
                all_warnings.extend(f"Line {i + 1}: {msg}" for msg in w)
        result.append(line)

    return "\n".join(result), all_warnings


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = sys.stdin.read()
    result, warnings = pwsh_transform(text)
    for w in warnings:
        print(f"[WARNING] {w}", file=sys.stderr)
    print(result)
