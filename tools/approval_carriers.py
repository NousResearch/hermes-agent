"""Literal shell command carriers used by approval detection."""

import os
import re
import shlex

from tools.approval_detection import (
    _ENV_ASSIGNMENT_RE,
    _PARAM_DEFAULT_RE,
    _PARAM_REPLACEMENT_RE,
    _bash_exec_payload,
    _deobfuscate_shell_word_for_detection,
    _iter_shell_command_starts,
    _literal_command_substitution_output,
    _read_shell_word,
    _scan_backtick_end,
    _scan_dollar_paren_end,
)

_MAX_COMMAND_PREFIX_WORDS = 64
_MAX_COMMAND_PREFIX_WORK = 8_192

class _CommandScanLimitExceeded(Exception):
    """Structured command scanning exceeded its bounded work budget."""

def _replace_command_word_expansions(word: str) -> str:
    """Resolve supported literal expansions unless single-quote protected."""
    chars: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(word):
        char = word[i]
        if char == "\\" and quote != "'" and i + 1 < len(word):
            chars.extend((char, word[i + 1]))
            i += 2
            continue
        if char == "'" and quote != '"':
            quote = None if quote == "'" else "'"
            chars.append(char)
            i += 1
            continue
        if char == '"' and quote != "'":
            quote = None if quote == '"' else '"'
            chars.append(char)
            i += 1
            continue
        if quote != "'" and word.startswith("$(", i):
            end = _scan_dollar_paren_end(word, i)
            if end is not None:
                replacement = _literal_command_substitution_output(
                    word[i + 2:end - 1]
                )
                if replacement is not None:
                    chars.append(replacement)
                    i = end
                    continue
        if quote != "'" and char == "`":
            end = _scan_backtick_end(word, i)
            if end is not None:
                replacement = _literal_command_substitution_output(
                    word[i + 1:end - 1]
                )
                if replacement is not None:
                    chars.append(replacement)
                    i = end
                    continue
        if quote != "'" and word.startswith("${", i):
            end = word.find("}", i + 2)
            if end != -1:
                token = word[i:end + 1]
                replacement = _PARAM_REPLACEMENT_RE.fullmatch(token)
                if replacement is not None:
                    chars.append(replacement.group("replacement"))
                    i = end + 1
                    continue
                default = _PARAM_DEFAULT_RE.fullmatch(token)
                if default is not None:
                    chars.append(default.group("default"))
                    i = end + 1
                    continue
        chars.append(char)
        i += 1
    return "".join(chars)

def _decode_ansi_c_shell_quote(word: str, start: int) -> tuple[int, str] | None:
    """Decode a literal Bash ``$'...'`` word fragment without executing it."""
    chars: list[str] = []
    escapes = {
        "a": "\a", "b": "\b", "e": "\x1b", "E": "\x1b", "f": "\f",
        "n": "\n", "r": "\r", "t": "\t", "v": "\v", "\\": "\\",
        "'": "'", '"': '"',
    }
    i = start + 2
    while i < len(word):
        char = word[i]
        if char == "'":
            return i + 1, "".join(chars)
        if char != "\\" or i + 1 >= len(word):
            chars.append(char)
            i += 1
            continue
        escaped = word[i + 1]
        if escaped == "\n":
            i += 2
            continue
        if escaped in escapes:
            chars.append(escapes[escaped])
            i += 2
            continue
        if escaped in "01234567":
            end = i + 2
            while end < len(word) and end < i + 4 and word[end] in "01234567":
                end += 1
            value = int(word[i + 1:end], 8)
            if value:
                chars.append(chr(value))
            i = end
            continue
        if escaped == "c" and i + 2 < len(word):
            control = word[i + 2]
            value = 127 if control == "?" else ord(control.upper()) & 31
            if value:
                chars.append(chr(value))
            i += 3
            continue
        if escaped in {"x", "u", "U"}:
            limit = {"x": 2, "u": 4, "U": 8}[escaped]
            end = i + 2
            while (
                end < len(word)
                and end < i + 2 + limit
                and word[end] in "0123456789abcdefABCDEF"
            ):
                end += 1
            if end > i + 2:
                value = int(word[i + 2:end], 16)
                if value == 0:
                    i = end
                    continue
                if value <= 0x10FFFF:
                    chars.append(chr(value))
                    i = end
                    continue
                # Bash accepts extended values that Python cannot represent as
                # Unicode. Preserve the source escape; it cannot form an ASCII
                # option name and must never crash approval detection.
                chars.append(word[i:end])
                i = end
                continue
        # Bash preserves the backslash for an unrecognized ANSI-C escape.
        chars.extend(("\\", escaped))
        i += 2
    return None

def _strip_shell_word_syntax(word: str) -> str:
    chars: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(word):
        ch = word[i]
        if quote:
            if ch == "\\" and quote == '"' and i + 1 < len(word):
                escaped = word[i + 1]
                if escaped in {'$', '`', '"', "\\"}:
                    chars.append(escaped)
                    i += 2
                    continue
                if escaped == "\n":
                    i += 2
                    continue
                # Inside double quotes, backslash is literal before every
                # other character. GNU env must receive `\_` intact for its
                # split-string separator grammar.
                chars.append(ch)
                i += 1
                continue
            if ch == quote:
                quote = None
                i += 1
                continue
            chars.append(ch)
            i += 1
            continue
        if word.startswith("$'", i):
            decoded = _decode_ansi_c_shell_quote(word, i)
            if decoded is not None:
                i, value = decoded
                chars.append(value)
                continue
        if ch in ("'", '"'):
            quote = ch
            i += 1
            continue
        if ch == "\\" and i + 1 < len(word):
            chars.append(word[i + 1])
            i += 2
            continue
        chars.append(ch)
        i += 1
    return "".join(chars)

def _shell_argv_word_for_detection(word: str) -> str:
    """Decode literal shell quoting and escapes into an approximate argv word.

    Runtime expansions stay opaque here. Expanding before quote parsing would
    incorrectly activate `$()` and parameter syntax protected by single quotes.
    """
    return _strip_shell_word_syntax(word)

_SHELL_C_CARRIERS = frozenset({
    "sh", "bash", "dash", "zsh", "ksh", "ash", "mksh", "sash",
    "su", "runuser",
})
# How deep to follow carriers nested inside carriers (`sh -c 'sh -c reboot'`).
# A small bound keeps a crafted deeply-nested string from spinning.
_EMBEDDED_COMMAND_MAX_DEPTH = 4

# Standard command wrappers a carrier can hide behind. A carrier reached
# through any composition of these, with their options, option operands, bare
# numeric or duration arguments, and NAME=VALUE assignments, is still the same
# carried-command class (`sudo -u root sh -c 'reboot'`, `env -i sh -c 'reboot'`,
# `timeout 5 sh -c 'reboot'`), so the whole prefix is skipped before looking for
# the carrier program. This is the wrapper set the command-position rules treat
# the same way, kept here so carrier detection does not fall behind it.
_CARRIER_WRAPPER_WORDS = frozenset({
    "sudo", "doas", "env", "exec", "nohup", "setsid", "time",
    "command", "builtin", "nice", "ionice", "stdbuf", "timeout",
})
# A bare numeric or duration argument (`5`, `1.5s`, `30m`). `timeout` takes one
# as its leading positional, so it is part of the prefix there.
_WRAPPER_NUMERIC_ARG_RE = re.compile(r"\d+(?:\.\d+)?[A-Za-z]*")

# Wrapper options that consume a following STRING operand (a username, host,
# signal name, variable, chdir path). Skipping the operand is what lets the real
# program after it be reached. A `--opt=value` form carries its own operand and
# needs no entry. Any option NOT listed for its wrapper is treated as taking no
# operand, so a no-operand flag (`sudo -E`, `sudo -n`, `timeout --foreground`)
# can never swallow the wrapper's actual program word.
_WRAPPER_OPTIONS_STR_ARG = {
    "sudo": {
        "-u", "--user", "-g", "--group", "-h", "--host", "-p", "--prompt",
        "-r", "--role", "-t", "--type", "-R", "--chroot", "-D", "--chdir",
        "-T", "--command-timeout", "-U", "--other-user",
    },
    "doas": {"-u", "-C", "-a"},
    "env": {"-a", "--argv0", "-u", "--unset", "-C", "--chdir"},
    "timeout": {"-s", "--signal", "-k", "--kill-after"},
    "stdbuf": {"-i", "--input", "-o", "--output", "-e", "--error"},
    "exec": {"-a"},
}
# Wrapper options whose operand is numeric (a priority, class, pid, fd), so a
# non-numeric following token is the program, not the operand (`nice -n echo` is
# `echo` run at default niceness, not niceness `echo`).
_WRAPPER_OPTIONS_NUM_ARG = {
    "nice": {"-n", "--adjustment"},
    "ionice": {"-n", "--classdata", "-p", "--pid", "-P", "--pgid", "-u", "--uid"},
    "sudo": {"-C", "--close-from"},
}
# `ionice -c` takes either a numeric class or a class name.
_IONICE_CLASS_OPTIONS = {"-c", "--class"}
_IONICE_CLASS_NAMES = {"none", "realtime", "best-effort", "idle"}


def _carrier_prefix_word(token: str) -> str:
    """Normalise a token to its lowercase basename for carrier matching."""
    word = _strip_optional_shell_quotes(
        _deobfuscate_shell_word_for_detection(token)
    )
    return word.lower().rsplit("/", 1)[-1]


def _wrapper_option_takes_operand(wrapper: str, option: str, operand: str) -> bool:
    """Whether ``option`` of ``wrapper`` consumes ``operand`` as its value."""
    if wrapper == "env" and option.startswith("--"):
        resolved = _env_long_option(option[2:])
        if resolved in _ENV_LONG_OPTIONS_REQUIRED_ARG:
            return True
    if option in _WRAPPER_OPTIONS_STR_ARG.get(wrapper, ()):
        return True
    if option in _WRAPPER_OPTIONS_NUM_ARG.get(wrapper, ()):
        return bool(_WRAPPER_NUMERIC_ARG_RE.fullmatch(operand))
    if wrapper == "ionice" and option in _IONICE_CLASS_OPTIONS:
        return bool(_WRAPPER_NUMERIC_ARG_RE.fullmatch(operand)) or (
            operand.lower() in _IONICE_CLASS_NAMES
        )
    return False


def _skip_wrapper_arguments(tokens: list, idx: int, wrapper: str) -> int:
    """Advance past a wrapper's options, option operands, leading duration, and
    assignments, stopping at the next program candidate.

    Option parsing is arity-aware: a flag consumes one operand only when that
    wrapper's option is known to take one, so `sudo -u root sh -c` skips the
    `root` operand and reaches `sh`, while a no-operand flag (`sudo -E sh -c`,
    `timeout --foreground echo ...`) leaves the following word as the program.
    A carrier or another wrapper is never consumed as an operand either, so a
    carrier is always reached rather than skipped.
    """
    env_options_ended = False
    while idx < len(tokens):
        word = _strip_optional_shell_quotes(
            _deobfuscate_shell_word_for_detection(tokens[idx])
        )
        is_assignment = (
            "=" in word if wrapper == "env" else bool(_ENV_ASSIGNMENT_RE.fullmatch(word))
        )
        if is_assignment:
            idx += 1
            env_options_ended = env_options_ended or wrapper == "env"
            continue
        if wrapper == "env" and not env_options_ended and word in {"-", "--"}:
            idx += 1
            env_options_ended = True
            continue
        if word.startswith("-") and word != "-" and not env_options_ended:
            option = word.split("=", 1)[0]
            env_bundle = (
                _env_short_option_bundle(word) if wrapper == "env" else None
            )
            idx += 1
            if env_bundle and env_bundle[0]:
                _, value_option, attached, _ = env_bundle
                if (
                    value_option in {"a", "u", "C"}
                    and attached is None
                    and idx < len(tokens)
                ):
                    idx += 1
                continue
            if "=" not in word and idx < len(tokens):
                nxt = tokens[idx]
                stripped_nxt = _strip_optional_shell_quotes(
                    _deobfuscate_shell_word_for_detection(nxt)
                )
                takes_operand = _wrapper_option_takes_operand(
                    wrapper, option, stripped_nxt
                )
                if takes_operand:
                    idx += 1
            continue
        # `timeout DURATION command` carries a bare duration positional before
        # the command it runs.
        if wrapper == "timeout" and _WRAPPER_NUMERIC_ARG_RE.fullmatch(word):
            idx += 1
            continue
        break
    return idx


def _read_same_command_word(command: str, pos: int) -> tuple[int, int, str]:
    """Read a word without crossing a comment or command-line boundary."""
    while pos < len(command) and command[pos].isspace() and command[pos] not in "\r\n":
        pos += 1
    if pos >= len(command) or command[pos] in "\r\n#":
        return (pos, pos, "")
    return _read_shell_word(command, pos)


def _read_command_tokens(
    command: str,
    start: int,
    work: list[int],
) -> tuple[list, bool]:
    """Read a bounded command prefix and report whether more words remain."""
    tokens: list = []
    pos = start
    carrier: str | None = None
    carrier_rest = 0
    while len(tokens) < _MAX_COMMAND_PREFIX_WORDS:
        work[0] += 1
        if work[0] > _MAX_COMMAND_PREFIX_WORK:
            raise _CommandScanLimitExceeded
        word_start, word_end, word = _read_same_command_word(command, pos)
        if word_start == word_end:
            return tokens, False
        tokens.append(word)
        pos = word_end

        if carrier is None:
            program, rest, prefix_incomplete = _carrier_program(tokens)
            if program is None and not prefix_incomplete:
                return tokens, False
            if program is not None:
                carrier = program
                carrier_rest = rest

        if carrier is not None and carrier not in {"env", "su", "runuser"}:
            if _dash_c_payload(carrier, tokens[carrier_rest:]) is not None:
                return tokens, False

    work[0] += 1
    if work[0] > _MAX_COMMAND_PREFIX_WORK:
        raise _CommandScanLimitExceeded
    word_start, word_end, _ = _read_same_command_word(command, pos)
    return tokens, word_start != word_end


def _carrier_program(tokens: list) -> tuple:
    """Return ``(program, rest_index, prefix_incomplete)`` for a carrier.

    Skips any composition of the leading exec wrappers in
    ``_CARRIER_WRAPPER_WORDS`` together with their options, option operands,
    numeric or duration arguments, and ``NAME=VALUE`` assignments (the same
    prefix a command-position verb is read through), so a carrier reached
    behind `sudo -u root ...`, `env -i ...`, `nice ...`, or `timeout 5 ...`
    still resolves. Bails on any other leading token so a payload is only ever
    extracted when the program really is a carrier.
    """
    idx = 0
    while idx < len(tokens):
        base = _carrier_prefix_word(tokens[idx])
        if _ENV_ASSIGNMENT_RE.fullmatch(
            _strip_optional_shell_quotes(
                _deobfuscate_shell_word_for_detection(tokens[idx])
            )
        ):
            idx += 1
            continue
        if base == "env":
            # `env` is a carrier only when it carries `-S` / `--split-string`.
            # Otherwise it is a pass-through wrapper (`env FOO=1 sh -c ...`,
            # `env -i sh -c ...`), so skip its options and assignments and keep
            # looking for the real program.
            if _env_split_string_payload(tokens[idx + 1:]) is not None:
                return ("env", idx + 1, False)
            idx = _skip_wrapper_arguments(tokens, idx + 1, "env")
            continue
        if base in _SHELL_C_CARRIERS:
            return (base, idx + 1, False)
        if base in _CARRIER_WRAPPER_WORDS:
            idx = _skip_wrapper_arguments(tokens, idx + 1, base)
            continue
        return (None, 0, False)
    return (None, 0, True)


_SU_LONG_OPTIONS_WITH_ARG = frozenset({
    "command", "session-command", "shell", "group", "supp-group", "user",
    "whitelist-environment",
})
_SU_LONG_OPTIONS_NO_ARG = frozenset({
    "fast", "login", "preserve-environment", "pty", "no-pty", "help",
    "version",
})
_SU_SHORT_OPTIONS_WITH_ARG = frozenset("cgGsuw")
_SU_SHORT_OPTIONS_NO_ARG = frozenset("flmpPThV")
_SU_COMMAND_LONG_OPTIONS = frozenset({"command", "session-command"})


def _su_long_option(name: str) -> str | None:
    """Resolve an exact or uniquely abbreviated util-linux su/runuser option."""
    options = _SU_LONG_OPTIONS_WITH_ARG | _SU_LONG_OPTIONS_NO_ARG
    if name in options:
        return name
    matches = [option for option in options if option.startswith(name)]
    return matches[0] if len(matches) == 1 else None


def _su_command_payload(program: str, args: list) -> str | None:
    """Parse util-linux su/runuser options and return the effective command."""
    payload: str | None = None
    runuser_user = False
    preserve_environment = False
    whitelist_environment = False
    i = 0
    while i < len(args):
        token = _deobfuscate_shell_word_for_detection(args[i])
        if token == "--":
            break
        if token == "-" or not token.startswith("-"):
            # GNU getopt permutes options past positional words by default.
            i += 1
            continue
        if token.startswith("--"):
            option_token, attached = _split_option(token[2:])
            option = _su_long_option(option_token)
            if option is None:
                return None
            if option in _SU_LONG_OPTIONS_NO_ARG:
                if attached is not None or option in {"help", "version"}:
                    return None
                preserve_environment = preserve_environment or option == "preserve-environment"
                i += 1
                continue
            if attached is None:
                if i + 1 >= len(args):
                    return None
                value = _deobfuscate_shell_word_for_detection(args[i + 1])
                i += 2
            else:
                value = attached
                i += 1
            if option == "user":
                if program != "runuser":
                    return None
                runuser_user = True
            elif option in _SU_COMMAND_LONG_OPTIONS:
                payload = value or None
            elif option == "whitelist-environment":
                whitelist_environment = True
            continue

        letters = token[1:]
        pos = 0
        while pos < len(letters):
            option = letters[pos]
            if option in _SU_SHORT_OPTIONS_NO_ARG:
                if option in {"h", "V"}:
                    return None
                preserve_environment = preserve_environment or option in {"m", "p"}
                pos += 1
                continue
            if option not in _SU_SHORT_OPTIONS_WITH_ARG:
                return None
            if pos + 1 < len(letters):
                value = letters[pos + 1:]
                i += 1
            elif i + 1 < len(args):
                value = _deobfuscate_shell_word_for_detection(args[i + 1])
                i += 2
            else:
                return None
            if option == "u":
                if program != "runuser":
                    return None
                runuser_user = True
            elif option == "c":
                payload = value or None
            elif option == "w":
                whitelist_environment = True
            break
        else:
            i += 1
            continue
        continue

    # In runuser's -u mode, command-string options are mutually exclusive and
    # the remaining argv is executed directly rather than passed to a shell.
    # util-linux also rejects preserve-environment combined with a whitelist.
    if runuser_user or (preserve_environment and whitelist_environment):
        return None
    return payload


def _dash_c_payload(program: str, args: list) -> str | None:
    """The command string a shell or account-switching carrier runs via `-c`.

    Shell invocation options are flags: in `sh -cx cmd`, both `c` and `x` are
    options and the next word is the command string. `su` and `runuser` use
    getopt-style options, parsed by `_su_command_payload`. Only programs in
    `_SHELL_C_CARRIERS` reach this helper.
    """
    if program in {"su", "runuser"}:
        return _su_command_payload(program, args)
    found, payload = _bash_exec_payload(args)
    return _strip_optional_shell_quotes(payload) if found and payload else None


_ENV_SHORT_OPTIONS_NO_ARG = frozenset("i0v")
_ENV_SHORT_OPTIONS_WITH_ARG = frozenset("auCS")
_ENV_LONG_OPTIONS_REQUIRED_ARG = frozenset({
    "argv0", "unset", "chdir", "split-string",
})
_ENV_LONG_OPTIONS_OPTIONAL_ARG = frozenset({
    "default-signal", "ignore-signal", "block-signal",
})
_ENV_LONG_OPTIONS_NO_ARG = frozenset({
    "ignore-environment", "null", "list-signal-handling", "debug", "help",
    "version",
})


def _env_long_option(name: str) -> str | None:
    """Resolve an exact or uniquely abbreviated GNU env long option."""
    options = (
        _ENV_LONG_OPTIONS_REQUIRED_ARG
        | _ENV_LONG_OPTIONS_OPTIONAL_ARG
        | _ENV_LONG_OPTIONS_NO_ARG
    )
    if name in options:
        return name
    matches = [option for option in options if option.startswith(name)]
    return matches[0] if len(matches) == 1 else None


def _env_short_option_bundle(
    token: str,
) -> tuple[bool, str | None, str | None, bool]:
    """Parse a GNU env short-option bundle.

    Returns ``(valid, value_option, attached_value, has_null)``. The first
    option that takes a value owns the rest of the token, matching getopt
    semantics; later letters are therefore data, not more options.
    """
    if not token.startswith("-") or token.startswith("--") or len(token) < 2:
        return False, None, None, False
    letters = token[1:]
    has_null = False
    for index, letter in enumerate(letters):
        if letter in _ENV_SHORT_OPTIONS_NO_ARG:
            has_null = has_null or letter == "0"
            continue
        if letter in _ENV_SHORT_OPTIONS_WITH_ARG:
            attached = letters[index + 1:] or None
            return True, letter, attached, has_null
        return False, None, None, False
    return True, None, None, has_null


_ENV_SPLIT_ESCAPES = {
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
    "#": "#",
    "$": "$",
    '"': '"',
    "'": "'",
    "\\": "\\",
}


def _split_gnu_env_string(value: str) -> list[str] | None:
    """Split a GNU ``env -S`` value without applying shell semantics."""
    words: list[str] = []
    current: list[str] = []
    quote: str | None = None
    word_started = False
    index = 0

    def finish_word() -> None:
        nonlocal current, word_started
        if word_started:
            words.append("".join(current))
            current = []
            word_started = False

    while index < len(value):
        char = value[index]
        if quote is None:
            if char.isspace():
                finish_word()
                index += 1
                continue
            if char in {"'", '"'}:
                quote = char
                word_started = True
                index += 1
                continue
            if char == "#" and not word_started:
                break
            if char == "$":
                # ${NAME} expands from env's runtime environment, so it is
                # outside this literal carrier scanner's ceiling.
                return None
            if char == "\\":
                if index + 1 >= len(value):
                    return None
                escaped = value[index + 1]
                if escaped == "c":
                    break
                if escaped == "_":
                    finish_word()
                elif escaped in _ENV_SPLIT_ESCAPES:
                    current.append(_ENV_SPLIT_ESCAPES[escaped])
                    word_started = True
                else:
                    return None
                index += 2
                continue
            current.append(char)
            word_started = True
            index += 1
            continue

        if char == quote:
            quote = None
            index += 1
            continue
        if char == "$" and quote == '"':
            return None
        if char == "\\":
            if index + 1 >= len(value):
                return None
            escaped = value[index + 1]
            if quote == "'" and escaped not in {"'", "\\"}:
                current.append("\\")
                word_started = True
                index += 1
                continue
            if escaped == "c" and quote == '"':
                return None
            if escaped == "_" and quote == '"':
                current.append(" ")
            elif escaped in _ENV_SPLIT_ESCAPES:
                current.append(_ENV_SPLIT_ESCAPES[escaped])
            else:
                return None
            word_started = True
            index += 2
            continue
        current.append(char)
        word_started = True
        index += 1

    if quote is not None:
        return None
    finish_word()
    return words


def _env_split_effective_command(payload: str, trailing_args: list) -> str | None:
    """Build env's effective argv from a split string and later operands."""
    split_args = _split_gnu_env_string(payload)
    if split_args is None:
        return None
    effective_argv = split_args + [
        _shell_argv_word_for_detection(arg) for arg in trailing_args
    ]
    # GNU env inserts the split words back into its own argv and reprocesses
    # them as options, assignments, and finally the command to execute.
    return f"env {shlex.join(effective_argv)}" if effective_argv else None


def _env_split_string_payload(args: list) -> str | None:
    """The command string GNU `env` would run for `-S` / `--split-string`.

    Only env's own option prefix is scanned. GNU env uses leading ``+`` getopt
    mode, so parsing stops at ``--`` and at the first non-option word, including
    a ``NAME=VALUE`` assignment. Searching every remaining token would
    treat data for that program as env options, so
    ``env -- echo --split-string reboot`` and ``env echo --split-string reboot``
    would hardline-block on a carried ``reboot`` even though ``echo`` runs with
    those words as arguments.
    """
    i = 0
    null_output = False
    while i < len(args):
        stripped = _shell_argv_word_for_detection(args[i])
        if stripped in {"--", "-"}:
            return None
        if not stripped.startswith("-"):
            # GNU env's leading '+' getopt mode stops at the first assignment
            # or program word. Later option-looking words are program data.
            return None
        if stripped.startswith("--"):
            option_token, attached = _split_option(stripped[2:])
            option = _env_long_option(option_token)
            if option is None:
                return None
            if option in _ENV_LONG_OPTIONS_NO_ARG:
                if attached is not None or option in {"help", "version"}:
                    return None
                null_output = null_output or option == "null"
                i += 1
                continue
            if option in _ENV_LONG_OPTIONS_OPTIONAL_ARG:
                # getopt_long accepts an optional long-option operand only in
                # the attached `=VALUE` form, so no following argv is consumed.
                i += 1
                continue
            if attached is None:
                if i + 1 >= len(args):
                    return None
                value = _shell_argv_word_for_detection(args[i + 1])
                i += 2
            else:
                value = attached
                i += 1
            if option == "split-string":
                if null_output:
                    return None
                return _env_split_effective_command(value, args[i:])
            continue

        valid_bundle, value_option, attached, bundle_null = (
            _env_short_option_bundle(stripped)
        )
        if not valid_bundle:
            return None
        if value_option == "S":
            if null_output or bundle_null:
                return None
            if attached is not None:
                return _env_split_effective_command(attached, args[i + 1:])
            if i + 1 < len(args):
                return _env_split_effective_command(
                    _shell_argv_word_for_detection(args[i + 1]),
                    args[i + 2:],
                )
            return None
        null_output = null_output or bundle_null
        i += 2 if value_option in {"a", "u", "C"} and attached is None else 1
    return None


def _direct_embedded_commands(command: str):
    """Yield the literal command strings command-carrying wrappers would run."""
    work = [0]
    for start in _iter_shell_command_starts(command):
        tokens, truncated = _read_command_tokens(command, start, work)
        if not tokens:
            continue
        program, rest, prefix_incomplete = _carrier_program(tokens)
        if program is None:
            if truncated and prefix_incomplete:
                raise _CommandScanLimitExceeded
            continue
        args = tokens[rest:]
        payload = (
            _env_split_string_payload(args)
            if program == "env"
            else _dash_c_payload(program, args)
        )
        if truncated and (program in {"env", "su", "runuser"} or payload is None):
            raise _CommandScanLimitExceeded
        if payload:
            yield payload


def _iter_embedded_commands(command: str, _depth: int = 0):
    """Yield carried command strings, following carriers nested in carriers."""
    if _depth >= _EMBEDDED_COMMAND_MAX_DEPTH:
        return
    for payload in _direct_embedded_commands(command):
        yield payload
        yield from _iter_embedded_commands(payload, _depth + 1)



def _split_option(token: str) -> tuple[str, str | None]:
    if "=" in token:
        option, value = token.split("=", 1)
        return option, value
    return token, None


def _strip_optional_shell_quotes(word: str) -> str:
    if len(word) >= 2 and word[0] == word[-1] and word[0] in ("'", '"'):
        return word[1:-1]
    return word
