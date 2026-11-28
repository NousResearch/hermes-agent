"""Effect-based `rm` rules: what a deletion REMOVES, matched with the shell lexer (#151542).

Sibling of :mod:`tools.approval_detection` for two reasons. The rules need the lexer that module
owns, and the lines they add would push it past the code-health file-size target (2000 lines), so
they live next to it, the shape this package already uses for approval_floors, approval_wmic and
the rest.

The regex rules in the parent key on the FIRST CHARACTER of the argument, so the gate depends on how
a deletion is spelled: `rm /srv/app/x/*.log.gz` prompts while `cd /srv/app/x && rm
logs/archive/*.log.gz` — the same files — does not. These classify the operand list instead: a
shell-expanded glob, a `..` path component, or more than one target.

Operand semantics need a lexer, not a regex over raw text: `rm "a.txt" "b.txt"` is two targets,
`rm a.txt -v` is one target and a flag, `rm '../secret'` is still a traversal when quoted,
`rm -f "$DIR"/*.log` is an unquoted glob while `rm "*.log"` is a literal name the shell will not
expand. Reusing the parent's command-word spans also gives command position for free: `then rm`,
`$(rm)`, `F=1 rm`, `sudo -u root rm`, `nice -n 10 rm` and `builtin rm` are found, `echo rm a b` is
not.

The parent imports this module while it is still initialising, so this module must never import it
at module level. The lexer helpers it needs are imported inside the functions that use them,
at call time.
"""

from __future__ import annotations

import functools
import os
import re


# Effect-based rm rules (#151542). The regex rules below key on the FIRST CHARACTER of the
# argument, so the gate depends on how a deletion is spelled: `rm /srv/app/x/*.log.gz` prompts
# while `cd /srv/app/x && rm logs/archive/*.log.gz` — the same files — does not. These describe
# what rm REMOVES instead: a glob, a `..` traversal, or more than one target.
#
# They are classified by walking the operands with the lexer, not by a regex, because operand
# semantics need one.
# A regex over raw text cannot tell `rm "a.txt" "b.txt"` (two targets) from `rm a.txt -v` (a flag),
# `rm "../secret"` (a traversal) from prose, or `rm -f "$DIR"/*.log` (an unquoted glob) from
# `rm "*.log"` (a literal name the shell will not expand). Reusing _iter_shell_command_word_spans
# also gives command position for free: `then rm`, `$(rm)`, `F=1 rm`, `sudo -u root rm`,
# `nice -n 10 rm` and `builtin rm` are all found, while `echo rm a b` is not.
#
# The p strings ARE compiled into the parent's DANGEROUS_PATTERNS_COMPILED, but _scanned_patterns()
# drops these rows from every scan, so they are never matched: they exist so _PATTERN_KEY_ALIASES
# derives a DISTINCT legacy key per rule. They must therefore keep their
# `\b` — a shared 20-character prefix would collapse the three onto one alias, and since
# _approval_scope accepts any alias, one stored approval would satisfy all three.
RM_GLOB_KEY = "delete with a glob"
# Named for the SPELLING, not the effect on the tree: `rm ./a/../b` and `rm ../etc/passwd` both
# walk through a parent, and only one leaves the working directory. The rule cannot resolve paths
# without executing the shell, so the key must not claim "outside the working directory".
RM_DOTDOT_KEY = "delete with a parent-directory (..) path"
RM_MULTI_KEY = "delete of multiple paths"
_RM_EFFECT_KEYS = (RM_GLOB_KEY, RM_DOTDOT_KEY, RM_MULTI_KEY)

# A `..` that is a PATH COMPONENT. `{1..50}.log` is a brace range, not a traversal.
_RM_DOTDOT_COMPONENT_RE = re.compile(r"(?:^|/)\.\.(?:/|$)")


def _collapse_arithmetic(text: str) -> str:
    """`$((...))` expands to ONE word, so its parentheses are not command structure; leaving them
    in made the operand run stop at the first `)`, so `rm $((1+1)) *.log` hid its real targets.

    A command substitution nested inside the arithmetic still RUNS (`echo $(( $(rm a b) + 1 ))`
    deletes), so a span containing `$(` is left untouched rather than collapsed away.
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith("$((", i):
            depth, k = 2, i + 3
            while k < n and depth:
                depth += (text[k] == "(") - (text[k] == ")")
                k += 1
            if depth == 0 and "$(" not in text[i + 3:k - 2]:
                out.append("0")  # one word, content irrelevant to the operand count
                i = k
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _rm_scan_text(command: str) -> str:
    """The command as the shell would SEE it, for operand purposes.

    A backslash-newline is a line continuation the shell deletes, so `r\\<newline>m a b` really
    runs `rm`; keeping the newline inside the word hid the command. Escaped SPACES are left alone:
    `rm a\\ b` is genuinely one operand.

    Heredoc bodies are deliberately NOT blanked here. Trying to treat `<<` as a heredoc header
    also swallowed here-strings (`<<<`), a `<<` inside quotes or a `#` comment, and an arithmetic
    shift (`1<<2`) — and each of those blanked a real command further down, letting deletions run
    unguarded. Over-prompting on heredoc data is the safe direction; missing a delete is not.
    """
    return _collapse_arithmetic(command.replace("\\\r\n", "").replace("\\\n", ""))


def _unquoted_view(word: str) -> str:
    """The part of a shell word the shell will EXPAND: quoted and escaped characters drop out.

    Quoting and escaping both suppress expansion, so `rm "*.log"` and `rm \\*.log` are literal
    names while `rm "$DIR"/*.log` and `rm backup{,.bak}` are not. Testing expansion on the raw
    text gets the quoted case wrong; testing it on the de-quoted text gets the escaped case wrong.
    The shell's own lexer answers both.
    """
    from tools.approval_detection import _scan_shell

    return "".join(word[i] for kind, i, _, quote in _scan_shell(word)
                   if kind == "char" and quote is None)


def _word_expands_to_a_glob(word: str) -> bool:
    """True when the shell would expand this word as a glob.

    An unmatched `[` is a literal character to the shell (`rm a[b` deletes one file), so a `[`
    only counts once a `]` follows it.
    """
    view = _unquoted_view(word)
    if "*" in view or "?" in view:
        return True
    bracket = view.find("[")
    return bracket >= 0 and "]" in view[bracket + 1:]


def _split_top_level(text: str) -> list[str]:
    """Split on commas that are not inside a nested brace."""
    parts, depth, start = [], 0, 0
    for index, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return parts


def _word_brace_expands(word: str) -> bool:
    """True when an UNQUOTED brace expression in the word expands to more than one word.

    Balanced, so a nested list (`{x{a}b,y}`) counts. `{lock}` and `{,,}` expand to at most one
    word / only empty fields, so they do not.
    """
    view = _unquoted_view(word)
    stack: list[int] = []
    for index, ch in enumerate(view):
        if ch == "{" and not (index and view[index - 1] == "$"):
            stack.append(index)
        elif ch == "}" and stack:
            inner = view[stack.pop() + 1:index]
            if "," not in inner:
                low, _, high = inner.partition("..")
                # A sequence needs BOTH endpoints and more than one member: `{1..50}` expands,
                # `{a..}` is literal and `{1..1}` is a single word.
                if (".." in inner and low.strip() and high.strip()
                        and low.strip() != high.strip()):
                    return True
            alternatives = _split_top_level(inner)
            if len(alternatives) > 1 and any(a.strip() for a in alternatives):
                return True
    return False


def _rm_effect(command: str, pos: int) -> str | None:
    """Classify what the `rm` whose argv starts at *pos* removes. Returns one of _RM_EFFECT_KEYS."""
    return _rm_effect_and_end(command, pos)[0]


def _rm_effect_and_end(command: str, pos: int) -> tuple[str | None, int]:
    """``(effect key, end of this rm's operand run)`` for the `rm` whose argv starts at *pos*.

    The end position is what tells one invocation's own rule from text that merely mentions it:
    `rm ../secret; echo rm -rf` runs one `rm`, and the `rm -rf` after it is echo's argument, not
    this deletion."""
    from tools.approval_detection import (
        _SHELL_REDIRECTION_RE, _is_shell_comment_start, _read_shell_word, _skip_shell_whitespace,
        _strip_shell_word_syntax,
    )

    operands: list[str] = []
    glob_seen = False
    brace_seen = False
    end_of_options = False
    while pos < len(command):
        gap_end = _skip_shell_whitespace(command, pos)
        # A newline before the next word ends the command. Checked BEFORE skipping whitespace,
        # which would swallow it.
        if "\n" in command[pos:gap_end]:
            break
        pos = gap_end
        if pos >= len(command):
            break
        # A command separator ends the operand run too.
        if command[pos] in ";|&()" or _is_shell_comment_start(command, pos):
            break
        redirect = _SHELL_REDIRECTION_RE.match(command, pos)
        if redirect:
            # The target of a redirection is not an operand; read past it.
            _, pos, _ = _read_shell_word(command, redirect.end())
            continue
        start, pos, raw = _read_shell_word(command, pos)
        if start == pos:
            break
        value = _strip_shell_word_syntax(raw)
        if not end_of_options:
            if value == "--":
                end_of_options = True
                continue
            if value.startswith("-") and value != "-":
                # rm exits after printing help/version and deletes nothing, whatever follows.
                if value in ("--help", "--version"):
                    return None, pos
                continue  # a flag, never an operand, however it is positioned
        if not value:
            continue
        operands.append(value)
        if _word_expands_to_a_glob(raw):
            glob_seen = True
        if _word_brace_expands(raw):
            brace_seen = True
    if glob_seen:
        return RM_GLOB_KEY, pos
    if any(_RM_DOTDOT_COMPONENT_RE.search(op) for op in operands):
        return RM_DOTDOT_KEY, pos
    # A brace range/list becomes several words before rm sees them.
    if brace_seen or len(operands) > 1:
        return RM_MULTI_KEY, pos
    return None, pos


@functools.lru_cache(maxsize=8)
def _rm_invocations(command: str) -> tuple[tuple[int, int, str | None], ...]:
    """One ``(start, operands_end, effect)`` per `rm` COMMAND WORD the command runs: the span of the
    invocation in the scan text, and its effect key (or ``None`` when it deletes nothing this module
    classifies: a single relative file, `--help`).

    Span and count matter as much as the keys. One invocation's effect key describes the same act as
    that invocation's own verb rule; a SECOND `rm` on the line is a different deletion; and a verb
    rule that matched outside this span is describing text, not this deletion, so it must not carry
    this invocation's key away.
    """
    from tools.approval_detection import (
        _deobfuscate_shell_word_for_detection, _iter_shell_command_word_spans,
    )

    text = _rm_scan_text(command)
    invocations: list[tuple[int, int, str | None]] = []
    for start, word_end, word in _iter_shell_command_word_spans(text):
        name = os.path.basename(_deobfuscate_shell_word_for_detection(word)).lower()
        # `rm.exe` is the spelling a Windows host actually runs; the module normalises `.exe` the
        # same way when it decides whether a word may run curl or wget.
        if name.removesuffix(".exe") == "rm":
            effect, operands_end = _rm_effect_and_end(text, word_end)
            invocations.append((start, operands_end, effect))
    return tuple(invocations)


def _rm_invocation_count(command: str) -> int:
    """How many `rm` command words the command runs."""
    return len(_rm_invocations(command))


def _rm_effects(command: str) -> frozenset[str]:
    """Every effect key this command's rm invocations produce.

    ``_rm_invocations`` holds the ``lru_cache``: it owns the lexer walk, and its two callers (the
    fallback yield and the consent exemption) would otherwise walk the same text twice in one scan.
    ``lru_cache`` rather than a mutable module dict: a plain dict let any in-process code disable
    all three rules by assigning an empty entry. Bounded by entry count, so the retained commands
    are at most eight.
    """
    return frozenset(effect for _, _, effect in _rm_invocations(command) if effect is not None)


def _inside_a_quote(command: str, pos: int) -> bool:
    """Whether *pos* sits inside a quoted string the shell will not expand as syntax.

    ``_scan_shell`` marks the delimiters ``kind == "quote"`` and the content between them as plain
    characters carrying the quote in the fourth field, so the field is what says "inside"."""
    from tools.approval_detection import _scan_shell

    return any(quote and start <= pos and (end is None or pos < end)
               for _kind, start, end, quote in _scan_shell(command))


def _rm_verb_match_belongs_to_the_single_invocation(command: str, pattern_key: str) -> bool:
    """Whether *pattern_key*, an rm verb rule, describes the one `rm` invocation this command runs.

    True only when the command runs exactly one `rm`, that rule's FIRST match starts INSIDE that
    invocation's own span (the command word through its last operand), and the match is on unquoted
    text. A rule that matched anywhere else is describing something that is not this deletion:
    `grep -r "rm -rf" docs; rm ../secret` reports `recursive delete` from the grep argument, and
    `rm ../secret; echo rm -rf` reports it from echo's, while `rm a b "rm -rf"` reports it from a
    quoted operand. In each case the real deletion's own key must survive.
    """
    text = _rm_scan_text(command)
    invocations = _rm_invocations(command)
    if len(invocations) != 1:
        return False
    from tools.approval_detection import DANGEROUS_PATTERNS_COMPILED

    span_start, span_end, _effect = invocations[0]
    for pattern_re, description in DANGEROUS_PATTERNS_COMPILED:
        if description != pattern_key:
            continue
        # Same text the spans were measured on: _rm_scan_text collapses `$((...))` to one
        # character, so a raw-text offset would be compared against a shifted span and a decoy
        # mention could land inside it.
        match = pattern_re.search(text)
        if match is None:
            return False
        return span_start <= match.start() < span_end and not _inside_a_quote(text, match.start())
    return False


def _scanned_patterns(only: frozenset[str] | None):
    """The rules one scan matches INLINE. The effect-based rm rules are never among them: they are
    yielded once, as a fallback, by _rm_effect_keys_for(), so that a delete is never the key
    reported for a command that also runs an interpreter."""
    from tools.approval_detection import DANGEROUS_PATTERNS_COMPILED

    return [
        (pattern_re, description) for pattern_re, description in DANGEROUS_PATTERNS_COMPILED
        if description not in _RM_EFFECT_KEYS and (only is None or description in only)
    ]


def _rm_effect_keys_for(command: str, only: frozenset[str] | None) -> list[str]:
    """Effect-based rm keys for the fallback yield, restricted to *only* when the caller named a set,
    and empty when the caller asked for rules that are not among them."""
    if only is not None and not any(key in only for key in _RM_EFFECT_KEYS):
        return []
    effects = _rm_effects(command)
    return [key for key in _RM_EFFECT_KEYS if key in effects and (only is None or key in only)]


# The three table rows the parent appends LAST to DANGEROUS_PATTERNS. They are listed there only so
# each rule has a pattern text to derive its approval key from: _scanned_patterns() drops these rows
# out of every scan, and _iter_dangerous_descriptions yields their keys once, as a FALLBACK after
# every other rule and the execution-flag sweep. That ordering matters: approvals are stored per
# key, so a delete must never be the key reported for a command that also runs an interpreter or a
# nested shell, because a stored grant for the delete key would then authorize that other thing too.
#
# The pattern text only seeds a DISTINCT key per rule: exactly one `\b` each, with a different
# remainder, so _PATTERN_KEY_ALIASES cannot collapse the three onto a shared alias (which would let
# one stored approval cover all).
RM_PATTERN_ROWS = [
    (r'\brm[^\n;|&]*[*?[]', RM_GLOB_KEY),
    (r'\brm[^\n;|&]*\.\.(?:/|\s|$)', RM_DOTDOT_KEY),
    # `rm a.txt b.txt` removes both. Counting, flag handling, quoting and command position all
    # live in _rm_effect; this entry exists for the key.
    (r'\brm[^\n;|&]*\s+[^-\s][^\n;|&]*\s+[^-\s]', RM_MULTI_KEY),
]
