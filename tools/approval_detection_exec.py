"""Execution-flag findings: interpreter -e/-c, shell -c payloads, read-tool program flags.

``bash -c``, ``python -c``, ``deno eval`` and ``rg --pre`` are parsed structurally so a
regex over the raw command cannot miss a payload or treat quoted prose as code. Command
substitution bodies and quoted heredocs are walked here because they change what is
executable. The shell lexer lives in ``approval_detection`` and is late-imported
(``_facade()``) so this sibling never forms a module-level cycle.
"""

import os
import re
import shlex

from tools.approval_detection_awk import AWK_EXEC_DESCRIPTION, AWK_NAMES, awk_program_runs_shell


def _facade():
    from tools import approval_detection

    return approval_detection


_INTERPRETER_NAME_RES = tuple((family, re.compile(pattern)) for family, pattern in (
    ("python", r"py(?:\.exe)?|python[23]?(?:\.\d+)*(?:\.exe)?"), ("node", r"node(?:js)?(?:\.exe)?"),
    ("perl", r"perl[0-9]*(?:\.\d+)*(?:\.exe)?"), ("ruby", r"ruby[0-9.]*(?:\.exe)?"), ("php", r"php(?:\.exe)?"),
    ("powershell", r"powershell(?:\.exe)?|pwsh(?:\.exe)?"),
    ("bun", r"bun(?:\.exe)?"), ("deno", r"deno(?:\.exe)?"),
))
_INTERPRETER_EXEC_FLAGS = {
    "python": {"-c"}, "node": {"-e", "--eval", "-p", "--print"}, "perl": {"-e", "--eval"}, "ruby": {"-e"},
    "php": {"-r"}, "powershell": {"-command", "-c", "-file", "-f"},
    "bun": {"-e", "--eval"}, "deno": {"eval", "-e", "--eval"},
}
_INTERPRETER_WITH_ARG = {
    "python": {"-W", "-X", "--check-hash-based-pycs"},
    "node": {"-C", "--conditions", "--cpu-prof-dir", "--diagnostic-dir", "--icu-data-dir", "--import", "--loader",
             "--openssl-config", "--require", "--title"},
    "perl": {"-0", "-F", "-I", "-M", "-m", "-x"}, "ruby": {"-C", "-E", "-F", "-I", "-K", "-r"},
    "php": {"-c", "-d", "-z"},
    "powershell": {"-configurationname", "-custompipename", "-executionpolicy", "-inputformat", "-outputformat",
                   "-settingsfile", "-version", "-windowstyle", "-workingdirectory"},
    # Deno's inline-script entry is the bare `eval` subcommand; the only global option that
    # may precede it and take a separate value is `-L/--log-level <level>` (`--env-file[=v]`
    # binds with `=`; the `--unstable-*` and `--ext` flags belong after `eval`).
    "bun": {"--config", "--cwd", "--env-file", "--preload", "--require"}, "deno": {"-L", "--log-level"},
}
_READ_TOOL_EXEC_FLAGS = {
    "sort": {"--compress-program"}, "rg": {"--pre", "--hostname-bin"}, "ag": {"--pager"},
    "man": {"--pager", "--html", "-P", "-H"},
}
# Required-argument options are ownership boundaries: an option-looking next token is data, not another option. These
# sets mirror the invocation grammar of the supported binaries (ripgrep 14, GNU sort, man-db, and ag 2.2).
_READ_TOOL_LONG_OPTIONS_WITH_ARG = {
    "rg": {
        "--after-context", "--before-context", "--color", "--colors", "--context", "--context-separator",
        "--dfa-size-limit", "--encoding", "--engine", "--field-context-separator", "--field-match-separator",
        "--file", "--generate", "--glob", "--hostname-bin", "--hyperlink-format", "--iglob", "--ignore-file",
        "--max-columns", "--max-count", "--max-depth", "--max-filesize", "--path-separator", "--pre", "--pre-glob",
        "--regex-size-limit", "--regexp", "--replace", "--sort", "--sortr", "--threads", "--type", "--type-add",
        "--type-clear", "--type-not",
    },
    "sort": {
        "--batch-size", "--buffer-size", "--compress-program", "--field-separator", "--files0-from", "--key",
        "--output", "--parallel", "--random-source", "--sort", "--temporary-directory",
    },
    "man": {
        "--config-file", "--encoding", "--extension", "--locale", "--manpath", "--pager", "--preprocessor",
        "--prompt", "--recode", "--sections", "--systems",
    },
    "ag": {
        "--ackmate-dir-filter", "--color-line-number", "--color-match", "--color-path", "--depth",
        "--filename-pattern", "--file-search-regex", "--ignore", "--ignore-dir", "--max-count", "--pager",
        "--path-to-ignore", "--width", "--workers",
    },
}
_READ_TOOL_SHORT_OPTIONS_WITH_ARG = {
    "rg": frozenset("efEmjgdtTABCMr"), "sort": frozenset("koStT"), "man": frozenset("CRLmMSserEPp"),
    "ag": frozenset("gGmpW"),
}
_BASH_OPTIONS_WITH_ARG = {"-O", "+O", "-o", "+o", "--init-file", "--rcfile"}
_BASH_SHORT_OPTION_LETTERS = frozenset("ilrsDcabefhkmnptuvxBCEHPTOo")


def _interpreter_family(executable: str) -> str | None:
    name = os.path.basename(executable).lower()
    return next((family for family, name_re in _INTERPRETER_NAME_RES if name_re.fullmatch(name)), None)


def _shell_segment_tokens(segment: str, start: int) -> list[str] | None:
    """Tokenize an already-bounded command segment. ``None`` distinguishes malformed quoting from
    an empty segment so callers can fail closed for a program-bearing option rather than silently
    skip it."""
    try:
        lexer = shlex.shlex(segment[start:], posix=True, punctuation_chars="<>")
        lexer.whitespace_split, lexer.commenters = True, ""
        return list(lexer)
    except ValueError:
        return None


def _is_redirection_operator_char(text: str, i: int, previous) -> bool:
    """Whether the unquoted ``&`` / ``|`` at *i* belongs to a redirection operator (``>&``, ``<&``,
    ``&>``, ``&>>``, ``>|``) rather than separating commands. *previous* is the prior ``_scan_shell``
    step; only an unquoted plain ``<`` / ``>`` char there can own the ``&`` / ``|``. Treating the
    ``&`` in ``bash 2>&1 -c 'reboot'`` as a separator cut the command before ``-c``."""
    before = text[previous[1]] if previous and previous[0] == "char" and previous[3] is None else ""
    if text[i] == "|":
        return before == ">"
    if text[i] != "&":
        return False
    return before in ("<", ">") or (text.startswith(">", i + 1) and before not in ("&", "|"))


_REDIRECTION_OPERATOR_RE = re.compile(r"&>>?|<<<|<<-?|<>|>>|>&|<&|>\||[<>]")
_REDIRECTION_FD_PREFIX_RE = re.compile(r"(?:[0-9]+|\{[A-Za-z_][A-Za-z0-9_]*\})$")


def _blank_shell_redirections(text: str, start: int) -> str:
    """Replace every unquoted redirection (optional fd / ``{var}`` prefix, operator, target word) in
    ``text[start:]`` with spaces, keeping offsets. Redirections may appear anywhere in a simple
    command, and ``bash 2>/dev/null -c 'reboot'`` otherwise leaves ``2 > /dev/null`` in argv where
    option parsing stops at the first non-option before ``-c``. Substitutions are skipped as units
    and process substitutions ``<(...)`` / ``>(...)`` are words, not redirections."""
    edits, skip_to, previous = [], start, None
    for step in _facade()._scan_shell(text, start, subst="uq"):
        kind, i, _, quote = step
        if i < skip_to:
            continue
        match = None
        if kind == "char" and quote is None and not text.startswith(("<(", ">("), i):
            match = _REDIRECTION_OPERATOR_RE.match(text, i)
            if match and match.group().startswith("&") and previous and previous[0] == "char" \
                    and previous[3] is None and text[previous[1]] in "&|":
                match = None  # `&&>x` / `|&>x`: the `&` is a separator, the `>` is matched next
        if match is None:
            previous = step
            continue
        op_start = i
        if not match.group().startswith("&"):
            # Bounded lookback for an fd (`2>`) or `{var}>` prefix that forms its own word.
            prefix = _REDIRECTION_FD_PREFIX_RE.search(text, max(start, i - 64), i)
            if prefix and (prefix.start() == start or text[prefix.start() - 1].isspace()):
                op_start = prefix.start()
        _, target_end, _ = _facade()._read_shell_word(text, match.end())
        edits.append((op_start, target_end, " " * (target_end - op_start)))
        skip_to, previous = target_end, None
    return _facade()._splice(text, edits) if edits else text


def _top_level_shell_segments(command: str, subst: str, brace: bool = False) -> tuple[list[str], bool]:
    """Split *command* at unquoted separators -> (segments, closed) where *closed* says the final
    quote state is balanced."""
    segments, start, state, previous = [], 0, None, None
    for step in _facade()._scan_shell(command, subst=subst, brace=brace, comments=True):
        kind, i, j, quote = step
        state = (None if quote else command[i]) if kind == "quote" else quote
        separator = (kind == "char" and quote is None and command[i] in ";&|\n"
                     and not _is_redirection_operator_char(command, i, previous))
        previous = step
        if kind == "comment" or separator:
            if start < i:
                segments.append(command[start:i])
            start = j
    if start < len(command):
        segments.append(command[start:])
    return segments, state is None


def _iter_top_level_shell_segments(command: str):
    """Yield top-level command segments in one left-to-right pass."""
    # A "$(...)" inside double quotes starts a fresh quote context, so its own quotes must not
    # toggle the outer state: `echo "$(grep -c "a|b" f)"` would otherwise expose the pattern's
    # `|` as a top-level separator and split grep into an unterminated (fail-closed) fragment.
    segments, closed = _top_level_shell_segments(command, "uq", brace=True)
    if not closed:
        # An outer quote left open after skipping a substitution (`echo "$(grep "a|b" f)`) is not
        # valid shell, so the fresh-context reading has no authority. Fall back to the flat quote
        # parity, which reports the grep fragment as malformed (fail-closed) exactly as before.
        segments, _ = _top_level_shell_segments(command, "")
    yield from segments


def _substitution_body_spans(command: str) -> list[tuple[int, int]]:
    """``(body_start, body_end)`` of every ``$(...)`` / backtick body, nested ones included, found
    with the same quote-aware recursion as ``_iter_shell_command_starts`` (an unterminated one runs
    to the end of its enclosing span)."""
    spans: list[tuple[int, int]] = []

    def scan(start: int, end: int) -> None:
        for kind, i, j, _ in _facade()._scan_shell(command, start, end, subst="uq", stop_unterminated=True,
                                         comments=True):
            if kind == "subst":
                inner = i + (1 if command[i] == "`" else 2)
                body_end = end if j is None else j - 1
                spans.append((inner, body_end))
                scan(inner, body_end)

    scan(0, len(command))
    return spans


def _simple_command_end(command: str, start: int, spans: list[tuple[int, int]]) -> int:
    """End offset of the simple command starting at *start*: bounded by the innermost substitution
    body that contains it, then by the first unquoted separator or closing ``)`` at its own nesting
    level. Nested ``$(...)`` / backtick operands are skipped as units, so their quotes and
    separators never leak into (or truncate) the enclosing command's words — without this,
    ``echo "$(grep "a|b" f; bash -c 'reboot')"`` tokenized ``bash`` through the enclosing ``)"``,
    lexed as malformed, and the shell payload was silently skipped."""
    limit = len(command)
    enclosing = max((span for span in spans if span[0] <= start < span[1]), default=None)
    if enclosing is not None:
        limit = enclosing[1]
    depth = 0  # unquoted `(`: subshells, process substitutions, arrays
    previous = None
    for step in _facade()._scan_shell(command, start, limit, subst="uq", brace=True, comments=True):
        kind, i, _, quote = step
        prior, previous = previous, step
        if kind == "comment":
            return i
        if kind != "char" or quote is not None:
            continue
        ch = command[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                return i
            depth -= 1
        elif ch == "`" or (depth == 0 and ch in ";&|\n"
                           and not _is_redirection_operator_char(command, i, prior)):
            # A backtick left as a char has no closer: it ends the command it sits inside.
            return i
    return limit


def _interpreter_exec_flag(family: str, args: list[str]) -> str | None:
    """Return an execution-bearing interpreter option, if present."""
    flags, with_arg = _INTERPRETER_EXEC_FLAGS[family], _INTERPRETER_WITH_ARG[family]
    powershell = family == "powershell"
    skip_value = False
    for token in args:
        if skip_value:
            skip_value = False
            continue
        if token == "--" or (not powershell and not token.startswith("-")):
            # Deno evaluates inline scripts via a bare `eval` subcommand rather than a dash
            # flag: the FIRST positional token, after any global options (`deno -q eval ...`);
            # a later positional `eval` (`deno run eval.ts`) stays data.
            if family == "deno" and token.lower() == "eval":
                return "eval"
            break
        option, equals, _ = token.partition("=")
        comparable = option.lower() if powershell else option
        if comparable in flags:
            return comparable
        # `-Wonce` and `ruby -rjson` attach an option value; they are not short-option bundles containing an execution
        # flag. PowerShell's normal long options also use one dash, so bundle parsing never applies to that family.
        has_attached_option_value = any(
            option.startswith(short) and len(option) > len(short)
            for short in with_arg if short.startswith("-") and not short.startswith("--")
        )
        if not powershell and not option.startswith("--") and len(option) > 2 and not has_attached_option_value:
            bundled = next((f"-{char}" for char in option[1:] if f"-{char}" in flags), None)
            if bundled:
                return bundled
        skip_value = comparable in with_arg and not equals
    return None


def _bash_exec_payload(args: list[str]) -> tuple[bool, str | None]:
    """Return whether Bash ``-c`` occurs and the command string it owns.
    Bash's O/o options consume the following argument even when they precede a later ``-c`` or
    share its short-option bundle; the two startup-file long options own their next token.
    Parsing those first prevents both missed payloads and false ``-c`` hits."""
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--" or not token.startswith(("-", "+")):
            break
        if token in _BASH_OPTIONS_WITH_ARG:
            index += 2
            continue
        chars = token[1:]
        # Bash option letters are case-sensitive; restricting to the documented alphabet
        # preserves invalid controls such as `-Wc`.
        if token.startswith("--") or not set(chars) <= _BASH_SHORT_OPTION_LETTERS:
            index += 1
            continue
        consumed_option_arg = int("O" in chars or "o" in chars)
        if "c" in chars:
            payload_index = index + 1 + consumed_option_arg
            return True, (args[payload_index] if payload_index < len(args) else None)
        index += 1 + consumed_option_arg
    return False, None


def _read_tool_exec_flag(tool: str, args: list[str]) -> tuple[str, str] | None:
    """Return (option, program) for a read-only tool's program-running flag."""
    flags = _READ_TOOL_EXEC_FLAGS[tool]
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--":
            break
        option, equals, payload = token.partition("=")
        payload = payload if equals else None
        matched = option if option in flags else None
        if tool == "man" and token.startswith(("-P", "-H")) and len(token) > 2:
            matched, payload = token[:2], token[2:]
        if matched:
            if payload is None and index + 1 < len(args):
                payload = args[index + 1]
            # The option owns its program argument regardless of spelling; the real binaries
            # execute a '-'-prefixed payload rather than reparsing it.
            if payload:
                return matched, payload
            index += 2 if payload is not None and "=" not in token else 1
        elif option in _READ_TOOL_LONG_OPTIONS_WITH_ARG[tool] and payload is None:
            index += 2
        elif token.startswith("-") and not token.startswith("--") and len(token) > 1:
            # In a short bundle, the first argument-taking option owns the rest of the token, or
            # the following token when it occurs last.
            shorts = _READ_TOOL_SHORT_OPTIONS_WITH_ARG[tool]
            owner = next((k for k, char in enumerate(token[1:], start=1) if char in shorts), None)
            index += 2 if owner == len(token) - 1 else 1
        else:
            index += 1
    return None


def _substitution_body(text: str, i: int, j: int) -> str:
    """Body of the ``$(...)`` / backtick span ``text[i:j]`` found by ``_scan_shell``."""
    return text[i + (1 if text[i] == "`" else 2):j - 1]


def _quoted_heredoc_body_spans(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` of each heredoc body whose delimiter word is quoted or escaped
    (``<<'EOF'``, ``<<"EOF"``, ``<<\\EOF``, ``<<E'O'F``). The shell performs no expansion there, so a
    ``$(...)`` in such a body is data. Bodies are read line by line, not through the quote scanner:
    an apostrophe in heredoc text is not a quote."""
    spans, pos, n = [], 0, len(text)
    while pos < n:
        pending, body_start = [], None
        for kind, i, _, quote in _facade()._scan_shell(text, pos, subst="uq", comments=True):
            if kind != "char" or quote is not None:
                continue
            if text.startswith("<<", i) and not text.startswith("<<<", i) and (i == 0 or text[i - 1] != "<"):
                strip = text.startswith("<<-", i)
                pending.append((_facade()._read_shell_word(text, i + 2 + strip)[2], strip))
            elif text[i] == "\n" and pending:
                body_start = i + 1
                break
        if body_start is None:
            break
        pos = body_start
        for word, strip in pending:  # bodies follow the line in operator order
            delimiter, start = _facade()._strip_shell_word_syntax(word), pos
            while pos < n:
                newline = text.find("\n", pos)
                line_end = n if newline < 0 else newline
                line = text[pos:line_end]
                end, pos = pos, line_end + 1
                if (line.lstrip("\t") if strip else line) == delimiter:
                    break
            else:
                end = n
            if any(ch in word for ch in "'\"\\"):
                spans.append((start, end))
    return spans


def _blank_nested_substitutions(text: str) -> tuple[str, list[str]]:
    """Return *text* with each complete ``$(...)`` / backtick span (quoted or unquoted) blanked to a
    same-length empty ``$(   )``, plus the blanked bodies. Offsets are preserved. A literal
    substitution (``$(echo bash)``) stays so command-word deobfuscation still resolves it. Callers
    scan each body separately, so every nesting level is walked once instead of once per ancestor.
    Substitutions inside a quoted heredoc body never run and are left as text."""
    literal = _quoted_heredoc_body_spans(text) if "<<" in text else []
    scanned = _facade()._splice(text, [(s, e, re.sub(r"[^\n]", " ", text[s:e])) for s, e in literal])
    parts, bodies = [], []
    for kind, i, j, _ in _facade()._scan_shell(scanned, subst="uq", comments=True):
        if kind == "subst" and j is not None and j - i >= 3:
            body = _substitution_body(text, i, j)
            if _facade()._literal_command_substitution_output(body) is None:
                parts.append("$(" + " " * (j - i - 3) + ")")
                bodies.append(body)
                continue
        parts.append(text[i:j])
    return "".join(parts), bodies


def _iter_command_substitution_bodies(command: str):
    """Yield every non-literal ``$(...)`` / backtick body at any nesting depth, each with its own
    nested substitutions blanked (they are yielded separately)."""
    pending = [command]
    while pending:
        _, bodies = _blank_nested_substitutions(pending.pop())
        for body in bodies:
            blanked, _ = _blank_nested_substitutions(body)
            yield blanked
            pending.append(body)


def _execution_flag_findings(command: str):
    """Yield scoped execution mechanisms and any executable payloads.
    ``$(...)`` / backtick bodies (quoted or unquoted) are scanned as commands of their own: a
    command inside a double-quoted substitution otherwise lexes with the outer closing quote
    attached, fails to tokenize, and its ``bash -c`` payload went unseen."""
    pending = [command]
    while pending:
        yield from _execution_flag_findings_flat(pending.pop(0), pending)


def _execution_flag_findings_flat(command: str, bodies: list[str]):
    for segment in _iter_top_level_shell_segments(command):
        # Command words are located on the blanked copy (same offsets), so words inside a
        # substitution are left to that body's own pass; tokens still come from the real text.
        blanked, nested = _blank_nested_substitutions(segment)
        bodies.extend(nested)
        spans = _substitution_body_spans(blanked)
        for start, _, word in _facade()._iter_shell_command_word_spans(blanked):
            executable = _facade()._deobfuscate_shell_word_for_detection(word)
            executable_name = os.path.basename(executable).lower()
            family = _interpreter_family(executable)
            relevant = (family is not None or executable_name in _READ_TOOL_EXEC_FLAGS
                        or executable_name in _facade()._SHELL_NAMES or executable_name in AWK_NAMES)
            if not relevant:
                continue
            bounded = segment[:_simple_command_end(blanked, start, spans)]
            tokens = _shell_segment_tokens(bounded, start)
            if tokens:
                # Option parsing runs on argv as the program sees it: redirections are removed by the
                # shell wherever they sit. Heredoc detection still reads the raw tokens below.
                argv = _shell_segment_tokens(_blank_shell_redirections(bounded, start), start)
                if argv is None:
                    tokens = None
            if tokens is None:
                yield (_facade()._MALFORMED_EXEC_DESCRIPTION, None)
                continue
            if not tokens:
                continue
            args = argv[1:]
            if family and _interpreter_exec_flag(family, args):
                yield ("script execution via -e/-c flag", None)
            elif family and any(token.startswith("<<") for token in tokens[1:]):
                yield ("script execution via heredoc", None)
            else:
                if executable_name in _facade()._SHELL_NAMES:
                    found, payload = _bash_exec_payload(args)
                    if found:
                        yield ("shell command via -c/-lc flag", payload)
                if executable_name in _READ_TOOL_EXEC_FLAGS:
                    finding = _read_tool_exec_flag(executable_name, args)
                    if finding:
                        yield (f"arbitrary program execution via {executable_name} {finding[0]}", finding[1])
                if executable_name in AWK_NAMES and awk_program_runs_shell(args):
                    yield (AWK_EXEC_DESCRIPTION, None)
