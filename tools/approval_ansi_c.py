"""Rewrite bash/zsh ANSI-C quoting (``$'...'``) into plain single quotes for approval detection.

Every detection scanner lexes ``$'`` as ``$`` plus an ordinary single-quoted string. That leaves the
escapes encoded (``$'\\x72\\x6d'`` never reads as ``rm``) and lets ``\\'`` end the quote early, so the
scanner's quote state drifts from the shell's and a following command is read as quoted data. Rewriting
each span to the single-quoted literal the shell would produce lets the existing quote-aware machinery
see the real words. Non-executing: only escape sequences are decoded.
"""

import re

_ESCAPE_RE = re.compile(
    r"\\(?:x([0-9A-Fa-f]{1,2})|([0-7]{1,3})|u([0-9A-Fa-f]{1,4})|U([0-9A-Fa-f]{1,8})|c(.)|(.))", re.DOTALL)
_NAMED = {"a": "\a", "b": "\b", "e": "\x1b", "E": "\x1b", "f": "\f", "n": "\n", "r": "\r", "t": "\t",
          "v": "\v", "\\": "\\", "'": "'", '"': '"', "?": "?"}


def _decode_escape(match: re.Match) -> str:
    hex_digits, octal, short_u, long_u, control, named = match.groups()
    if hex_digits or short_u or long_u:
        code = int(hex_digits or short_u or long_u, 16)
        # bash emits surrogates as raw bytes, never a real character; keep them encoded.
        return chr(code) if code <= 0x10FFFF and not 0xD800 <= code <= 0xDFFF else match.group(0)
    if octal:
        return chr(int(octal, 8) & 0xFF)
    if control:
        return chr(ord(control) & 0x1F)
    return _NAMED.get(named or "", match.group(0))


def _ansi_c_span_end(text: str, start: int) -> int | None:
    """Offset just past the closing quote of the ``$'...'`` at *start*; ``None`` if unterminated."""
    match = re.compile(r"(?:\\.|[^'\\])*'", re.DOTALL).match(text, start + 2)
    return match.end() if match else None


def _single_quoted(body: str) -> str:
    # bash ends the string at a decoded NUL. A literal `'` is spelled `"'"`, not `'\''`: detection
    # normalization strips backslashes, which would turn `'\''` into an unbalanced quote again.
    literal = _ESCAPE_RE.sub(_decode_escape, body).split("\0", 1)[0]
    return "'" + literal.replace("'", "'\"'\"'") + "'"


def expand_ansi_c_quotes(command: str) -> str:
    """Return *command* with each shell-recognized ``$'...'`` span rewritten as ``'decoded'``.

    ``$'`` is ANSI-C quoting only outside quotes, including inside a ``$(...)`` / backtick
    substitution nested in double quotes; inside double quotes it is a literal ``$'``. ``$"..."``
    (locale translation) is a double-quoted string, so its ``$`` is dropped. An unterminated
    ``$'`` is left as is."""
    if "$" not in command:
        return command
    out: list[str] = []
    # Contexts: "u" (unquoted), '"' (double quotes), "(" ($(...) opened inside double quotes, with
    # its own paren depth) and "`" (backtick opened inside double quotes).
    stack: list[list] = [["u", 0]]
    i, n = 0, len(command)
    while i < n:
        i = _step(command, i, stack, out)
    return "".join(out)


def _is_shell_comment_start(command: str, index: int) -> bool:
    """Whether the ``#`` at *index* starts a comment: it must begin a word. An escaped separator
    (``echo \\ #``) is part of the previous word, so its ``#`` is literal and what follows runs."""
    if command[index] != "#" or index == 0:
        return command[index] == "#"
    prev = command[index - 1]
    if not (prev.isspace() or prev in ";&|()<>"):
        return False
    j = index - 2
    while j >= 0 and command[j] == "\\":
        j -= 1
    return (index - 2 - j) % 2 == 0


def _skip_continuations(command: str, i: int) -> int:
    while command.startswith("\\\n", i):
        i += 2
    return i


def _step(command: str, i: int, stack: list[list], out: list[str]) -> int:
    ch, context = command[i], stack[-1]
    if command.startswith("\\\n", i):
        # A line continuation is removed before tokenizing (also inside double quotes), so it can
        # sit between `$` and the quote that makes it `$'`.
        return i + 2
    if ch == "\\":
        out.append(command[i:i + 2])
        return i + 2
    # Offset of the character the shell sees right after a `$`.
    after = _skip_continuations(command, i + 1) if ch == "$" else i + 1
    following = command[after:after + 1]
    if ch == "$" and following == "$":
        # `$$` is the PID; a `'`/`"`/`(` after it starts a new token, not `$'`/`$"`/`$(`.
        out.append("$$")
        return after + 1
    if context[0] == '"':
        if ch == '"':
            stack.pop()
        elif (ch == "$" and following == "(") or ch == "`":
            stack.append(["(" if ch == "$" else "`", 0])
            out.append("$(" if ch == "$" else "`")
            return after + 1 if ch == "$" else i + 1
        out.append(ch)
        return i + 1
    if ch == "$" and following == "'" and (end := _ansi_c_span_end(command, after - 1)) is not None:
        out.append(_single_quoted(command[after + 1:end - 1]))
        return end
    if ch == "$" and following == '"':
        return after
    if ch == "'":
        end = command.find("'", i + 1)
        end = len(command) if end < 0 else end + 1
        out.append(command[i:end])
        return end
    if ch == "#" and _is_shell_comment_start(command, i):
        end = command.find("\n", i)
        end = len(command) if end < 0 else end
        out.append(command[i:end])
        return end
    if ch == '"':
        stack.append(['"', 0])
    elif context[0] == "`" and ch == "`":
        stack.pop()
    elif context[0] == "(" and ch == "(":
        context[1] += 1
    elif context[0] == "(" and ch == ")":
        if context[1]:
            context[1] -= 1
        else:
            stack.pop()
    out.append(ch)
    return i + 1
