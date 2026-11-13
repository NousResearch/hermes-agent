"""Command-position normalization for the hardline floor (detection-only rewrites).

The ``_CMDPOS``-anchored hardline rules see a verb only where the shell would run it. Two bounded
projections put a verb that IS run back on that anchor:

* ``strip_wrapper_prefixes`` blanks a leading chain of exec wrappers from a fixed set (``nice``,
  ``timeout 5``, ``command``, ``doas -u x`` ...), parsing each wrapper's own option grammar so an
  option operand is never mistaken for the program (``timeout -s KILL 5 reboot``).
* ``carried_command_variants`` re-scans the LITERAL command string a carrier would run: GNU
  ``env -S`` / ``--split-string`` and the ``-c`` / ``--command`` argument of ``su`` / ``runuser``,
  plus wrapper-peeled spellings of ``sh -c`` payloads. Computed strings (``sh -c "$(...)"``,
  ``echo cmd | sh``) and argv grammars such as ``find -exec`` are out of reach for a static scan.

This is heuristic hardening of the in-process approval floor, not a security boundary.
"""

import re

# Adapted from Adolanium's wrapper-operand parser in #58643.
_WRAPPER_WORDS = frozenset({
    "sudo", "doas", "env", "exec", "nohup", "setsid", "time",
    "command", "builtin", "nice", "ionice", "stdbuf", "timeout", "xargs",
})

_WRAPPER_STR_OPERAND = {
    "sudo": {"-u", "-g", "-h", "-p", "-U", "-D", "-R", "-T",
             "--user", "--group", "--host", "--prompt", "--chdir",
             "--role", "--type"},
    "doas": {"-u", "-C"},
    # -S / --split-string carry the command itself: carried_command_variants scans that payload.
    "env": {"-u", "-C", "-a", "-S",
            "--unset", "--chdir", "--argv0", "--split-string"},
    "exec": {"-a"},
    "ionice": {"-c", "-p", "--class", "--pid"},
    "stdbuf": {"-i", "-o", "-e", "--input", "--output", "--error"},
    "time": {"-f", "-o", "--format", "--output"},
    "timeout": {"-s", "-k", "--signal", "--kill-after"},
    "xargs": {"-a", "-d", "-E", "-I", "-L", "-n", "-P", "-s",
              "--arg-file", "--delimiter", "--max-args", "--max-procs", "--max-chars",
              "--process-slot-var"},
}

_WRAPPER_NUM_OPERAND = {
    "sudo": {"-C", "--close-from"},
    "nice": {"-n", "--adjustment"},
    "ionice": {"-n", "--classdata"},
}

# Lookup / query modes name a program without running it (`command -v reboot`, `ionice -p 1`).
_WRAPPER_QUERY_OPTIONS = {
    "command": {"-v", "-V"}, "ionice": {"-p", "--pid", "--pgid", "--uid"},
    "sudo": {"-l", "--list", "-v", "--validate", "-k", "-K", "--reset-timestamp", "--remove-timestamp"},
}

_WRAPPER_DURATION_RE = re.compile(r"[+-]?\d+(?:\.\d+)?[A-Za-z]*")

# su / runuser: options that take a separate value (util-linux su-common.c), and the command option.
_SU_VALUE_SHORT = frozenset("sgGwu")
_SU_VALUE_LONG = frozenset({"--shell", "--group", "--supp-group", "--whitelist-environment", "--user"})
_SU_COMMAND_LONG = frozenset({"--command", "--session-command"})

_MAX_CARRIER_DEPTH = 3


def _option_kind(base: str, word: str) -> str | None:
    """'str' / 'num' when *word* is an option of *base* owning the NEXT word, else None."""
    str_opts = _WRAPPER_STR_OPERAND.get(base, ())
    num_opts = _WRAPPER_NUM_OPERAND.get(base, ())
    if word in str_opts or (re.fullmatch(r"-[a-zA-Z]+", word) and "-" + word[-1] in str_opts):
        return "str"
    if word in num_opts or (re.fullmatch(r"-[a-zA-Z]+", word) and "-" + word[-1] in num_opts):
        return "num"
    return None


def _is_query_option(base: str, word: str) -> bool:
    queries = _WRAPPER_QUERY_OPTIONS.get(base, set())
    if word.split("=", 1)[0] in queries:
        return True
    # Short bundles: `command -pv reboot` still only looks reboot up.
    return bool(re.fullmatch(r"-[a-zA-Z]+", word)) and any("-" + ch in queries for ch in word[1:])


def _wrapper_prefix_end(command: str, start: int) -> int:
    """Offset where the real program after any leading wrappers begins (*start* when none)."""
    from tools.approval_detection import (
        _ENV_ASSIGNMENT_RE, _deobfuscate_shell_word_for_detection, _read_shell_word,
    )

    pos, seen_prefix = start, False
    while True:
        ws, we, word = _read_shell_word(command, pos)
        if ws == we:
            return start  # prefix with no program: nothing executes
        if _ENV_ASSIGNMENT_RE.fullmatch(word):
            pos, seen_prefix = we, True  # `X=1 nice reboot`
            continue
        # Quote/escape-spelled wrappers (`n\\ice`, `'nice'`) are still the wrapper.
        base = re.split(r"[\\/]", _deobfuscate_shell_word_for_detection(word))[-1].lower()
        if base not in _WRAPPER_WORDS:
            return ws if seen_prefix else start
        seen_prefix, pos = True, we
        while True:  # this wrapper's options / assignments / leading positional
            ows, owe, oword = _read_shell_word(command, pos)
            if ows == owe or oword == "--":
                pos = owe
                break
            if oword.startswith("-") and oword != "-":
                if _is_query_option(base, oword):
                    return start
                if "=" in oword:
                    pos = owe  # --opt=value carries its own operand
                    continue
                kind = _option_kind(base, oword)  # case-sensitive: `sudo -H` vs `sudo -h host`
                if kind is None:
                    pos = owe  # no-operand flag
                    continue
                vws, vwe, value = _read_shell_word(command, owe)
                if vws == vwe:
                    pos = owe
                    continue
                if kind == "num" and not _WRAPPER_DURATION_RE.fullmatch(value):
                    return start  # invalid operand: the wrapper errors here
                pos = vwe
                continue
            if base == "env" and _ENV_ASSIGNMENT_RE.fullmatch(oword):
                pos = owe
                continue
            if base == "timeout" and _WRAPPER_DURATION_RE.fullmatch(oword):
                pos = owe  # leading duration positional
                continue
            break


def strip_wrapper_prefixes(command: str) -> str:
    """Blank every leading wrapper chain so the real verb sits at the ``_CMDPOS`` anchor."""
    from tools.approval_detection import _iter_shell_command_starts

    cuts = []
    for start in _iter_shell_command_starts(command):
        end = _wrapper_prefix_end(command, start)
        if end > start:
            cuts.append((start, end))
    out = command
    for s, e in reversed(cuts):
        out = out[:s] + " " + out[e:]
    return out


def _su_command_payload(args: list[str]) -> str | None:
    """The ``-c`` / ``--command`` string of a ``su`` / ``runuser`` argv (getopt permutes, so it may
    follow the user name), or None."""
    index = 0
    while index < len(args):
        token = args[index]
        index += 1
        if token == "--":
            return None
        option, equals, value = token.partition("=")
        if option in _SU_COMMAND_LONG:
            return value if equals else (args[index] if index < len(args) else None)
        if option in _SU_VALUE_LONG:
            index += 0 if equals else 1
            continue
        if not re.fullmatch(r"-[a-zA-Z].*", token):
            continue
        for offset, char in enumerate(token[1:], start=2):
            if char == "c":
                return token[offset:] or (args[index] if index < len(args) else None)
            if char in _SU_VALUE_SHORT:
                index += 0 if token[offset:] else 1
                break
    return None


def _carried_payloads(command: str):
    """Literal command strings that env -S, su -c, runuser -c or a shell -c would run."""
    from tools.approval_detection import (
        _deobfuscate_shell_word_for_detection, _env_split_payload, _execution_flag_findings,
        _iter_shell_command_word_spans, _iter_top_level_shell_segments, _shell_segment_tokens,
    )

    for _, payload in _execution_flag_findings(command):
        if payload:
            yield payload
    for segment in _iter_top_level_shell_segments(command):
        for start, _, word in _iter_shell_command_word_spans(segment):
            name = re.split(r"[\\/]", _deobfuscate_shell_word_for_detection(word))[-1].lower()
            if name not in ("env", "su", "runuser"):
                continue
            tokens = _shell_segment_tokens(segment, start)
            if not tokens:
                continue
            payload = _env_split_payload(tokens) if name == "env" else _su_command_payload(tokens[1:])
            if payload:
                yield payload


def carried_command_variants(command: str):
    """Detection variants for literal carried command strings, recursing a bounded depth; each is
    wrapper-peeled and command-start marked like the top-level command."""
    from tools.approval_detection import _command_detection_variants

    seen, frontier = {command}, [command]
    for _ in range(_MAX_CARRIER_DEPTH):
        nested = []
        for source in frontier:
            for payload in _carried_payloads(source):
                if payload in seen:
                    continue
                seen.add(payload)
                nested.append(payload)
                yield from _command_detection_variants(payload)
                yield from _command_detection_variants(strip_wrapper_prefixes(payload))
        frontier = nested
