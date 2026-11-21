"""Launcher option table and sink patterns of the pipe-to-shell rule in :mod:`tools.approval_detection`
(moved from there)."""

import hermes_re as re


# Launchers that run their trailing words as a command, with the short/long options that take a
# SEPARATE value word. One shared table made a bare flag (`time -p`, `exec -a x`) swallow the shell
# name as its value; per-launcher rows let `-p` be bare for time/command and valued for sudo.
_PIPE_LAUNCHER_VALUE_FLAGS = {
    "sudo": ("ugUCDhpRrtT", ("user", "group", "other-user", "close-from", "chdir", "host", "prompt",
                             "role", "type", "command-timeout", "chroot")),
    "doas": ("uC", ()),
    # `-S STRING` is the command itself (split into words), never a swallowed value.
    "env": ("uCa", ("unset", "chdir", "argv0")),
    "exec": ("a", ()),
    "command": ("", ()),
    "nohup": ("", ()),
    "nice": ("n", ("adjustment",)),
    "time": ("fo", ("format", "output")),
    "timeout": ("sk", ("signal", "kill-after")),
    "stdbuf": ("ioe", ("input", "output", "error")),
    "busybox": ("", ()),
    "setsid": ("", ()),
    "ionice": ("cnpPu", ("class", "classdata", "pid", "pgid", "uid")),
}


def _not_any_of(words: tuple) -> str:
    """Option characters (possibly none) that spell none of `words`."""
    heads = sorted({w[0] for w in words if w})
    tails = [h + _not_any_of(tuple(w[1:] for w in words if w[:1] == h)) for h in heads]
    other = rf"[^\s|;&<>{''.join(heads)}][^\s|;&<>]*"
    run = rf"(?:{'|'.join([other, *tails])})"
    return run if "" in words else run + "?"


def _pipe_launcher_re(name: str, short: str, long_opts: tuple, ws: str = r"\s") -> str:
    # One parse per argument word without an atomic group: the alternatives are spelled disjoint. A
    # bare flag is a dash word that is NOT a valued spelling (`-u`, `-Eu`, `--user`), so the word after
    # a valued flag is always its value (`sudo -u bash cat` reads `bash` as the user).
    valued = [rf"(?-i:-[^\W{short}\d-]*[{short}]){ws}+[^\s|;&<>]+"] if short else []
    valued += [rf"--(?:{'|'.join(long_opts)}){ws}+[^\s|;&<>]+"] if long_opts else []
    bare = [rf"--{_not_any_of(long_opts)}"]
    bare += [rf"(?-i:-(?:[^\W{short}\d-]+(?:(?:[{short}][^\s|;&<>]|\d|[^\w\s|;&<>])[^\s|;&<>]*)?"
             rf"|[{short}][^\s|;&<>]+|(?:\d|[^\w\s|;&<>-])[^\s|;&<>]*))"] if short else [r"-[^\s|;&<>-][^\s|;&<>]*"]
    duration = r"(?:infinity|\d+(?:\.\d+)?(?:e\d+)?|\.\d+)[smhd]?"
    args = "|".join([*valued, *bare, r"\w+=[^\s|;&<>]*", duration])
    return rf"{name}(?:{ws}+(?:{args}))*"


_PIPE_PATH_PREFIX_RE = r"(?:(?:[a-zA-Z]:)?(?:[.\w~-]*[/\\])*)"
# The path prefix of a word that follows a launcher: it never starts with `-` (a dash word there is
# one of the launcher's flags, `sudo -x/bash`).
_PIPE_PATH_PREFIX_AFTER_LAUNCHER_RE = r"(?:(?:[a-zA-Z]:|(?:[.\w~][.\w~-]*)?[/\\])(?:[.\w~-]*[/\\])*)?"


def _pipe_wrapped_re(word, ws: str = r"\s") -> str:
    """`word(path_prefix)` behind zero or more launchers (`sudo -u root -E`, `/usr/bin/env A=1`, `timeout 60`)."""
    launchers = "|".join(_pipe_launcher_re(n, *f, ws) for n, f in _PIPE_LAUNCHER_VALUE_FLAGS.items())
    after = _PIPE_PATH_PREFIX_AFTER_LAUNCHER_RE
    return (
        rf"(?:{_PIPE_PATH_PREFIX_RE}(?:{launchers}){ws}+(?:{after}(?:{launchers}){ws}+)*{word(after)}"
        rf"|{word(_PIPE_PATH_PREFIX_RE)})"
    )


# The launcher run alone, for pattern text that is never compiled; a word after it may wrongly start with `-`.
_PIPE_WRAPPERS_RE = (
    rf"(?:{_PIPE_PATH_PREFIX_RE}(?:"
    + "|".join(_pipe_launcher_re(n, *f) for n, f in _PIPE_LAUNCHER_VALUE_FLAGS.items())
    + r")\s+)*"
)


# The shell members of _SHELL_CARRIER_NAMES, as one tuple plus alternation shared by every
# pipe/decode/process-substitution/heredoc pattern and the structural -c payload scan, so
# the shell-name list cannot drift between them again (the drift let `curl url | zsh` and
# `dash -c` through while bash/sh were flagged).
_SHELL_NAMES = ("bash", "sh", "zsh", "ksh", "dash")
_SHELL_NAMES_RE = "|".join(_SHELL_NAMES)
# Pipe-to-shell sink: the shell may sit behind launcher words (`sudo -u root -E`, `env -i A=1`,
# `/usr/bin/env`, `timeout 60`, `busybox`), be path-qualified (/bin/sh, ./zsh,
# C:\\msys64\\...\\bash.exe), quoted, .exe-suffixed, or followed by a redirect instead of
# whitespace. The lookahead terminator keeps `shred`, `shellcheck` and `bash-helper` out.
# fish/csh/tcsh read a script on stdin too, so they count here (they take no part in the -c
# payload scan). Launcher tokens fall in disjoint classes (a flag that takes a value -- sudo/env
# `-u USER`, `-g`, `-C` ... -- a bare flag, VAR=value, a duration), spelled so that no word fits two
# of them: each token has one parse and the scan stays linear; `sudo -u bash cat` reads `bash`
# as the user, not the shell.
_PIPE_SHELL_NAMES_RE = "|".join((*_SHELL_NAMES, "fish", "csh", "tcsh", "ash"))
_PIPE_SHELL_NAME_RE = rf"(?:{_PIPE_SHELL_NAMES_RE})(?:\.exe)?(?=['\"\s;&|<>)`]|$)"
_PIPE_SHELL_TOKEN_RE = rf"['\"]?{_PIPE_PATH_PREFIX_RE}{_PIPE_SHELL_NAME_RE}"
# A stdin-reading interpreter with no script argument (or `-`) runs the piped text as code;
# `python3 -m json.tool` / `node script.js` name their own program and stay out. Output redirects
# (`> out.txt`, `2>&1`, `&>> log`) leave stdin alone; an input redirect replaces it and stays out.
_PIPE_REDIRECT_OPERATOR_RE = r"\s*(?:[0-9]*>>?|&>>?)\s*"
_PIPE_REDIRECT_WORD_RE = r"[^\s;&|<>()`]+"
# What may follow the interpreter and its redirects. Every redirect is taken whole, so a `&` ends the
# command only where it does not open a `&>` redirect, and a `}` only where it is not part of a
# redirect's target word (`> a}b`).
_PIPE_BACKGROUND_RE = r"&(?:[^>]|$|>(?:>?\s*(?:[;|<()`]|&(?:[^0-9-]|$)|$)|>\s*>|\s+>))"
_PIPE_COMMAND_END_RE = rf"\s*(?:[;|)`]|{_PIPE_BACKGROUND_RE}|$)"
_PIPE_INTERPRETER_NAME_RE = (
    r"(?:python[0-9.]*|node|perl|ruby|php)(?:\.exe)?(?:\s+-)?"
    rf"(?:(?:{_PIPE_REDIRECT_OPERATOR_RE}(?:&[0-9-]|{_PIPE_REDIRECT_WORD_RE}))*"
    rf"(?:{_PIPE_REDIRECT_OPERATOR_RE}&[0-9-](?=\s*}}|{_PIPE_COMMAND_END_RE})"
    rf"|{_PIPE_REDIRECT_OPERATOR_RE}{_PIPE_REDIRECT_WORD_RE}(?=\s+}}|{_PIPE_COMMAND_END_RE}))"
    rf"|(?=}}|\s+}}|{_PIPE_COMMAND_END_RE}))"
)


def _pipe_sink_word(prefix: str) -> str:
    """The shell or interpreter word of a pipe sink behind path `prefix` (a quoted shell keeps the plain prefix)."""
    return (
        rf"(?:(?:['\"]{_PIPE_PATH_PREFIX_RE}|{prefix}){_PIPE_SHELL_NAME_RE}"
        rf"|{prefix}{_PIPE_INTERPRETER_NAME_RE})"
    )
# Lookup/list modes name a shell without running it (`command -v bash`, `sudo -l bash`). The `l`
# must be a bare flag: in `sudo -pl bash` / `-ul` it is the VALUE of -p / -u and bash runs.
_PIPE_LOOKUP_ONLY_RE = re.compile(
    rf"{_PIPE_PATH_PREFIX_RE}(?:command\s+-[a-zA-Z]*[vV]|sudo\s+(?:-(?-i:[^\W\dugUCDhpRrtT])*l|--list))\b",
    re.IGNORECASE,
)
# `| (bash)` / `| { bash; }`: a subshell or group opener in front of the sink still feeds it stdin.
_PIPE_SINK_RE = re.compile(
    rf"(?:\|&?\s*|>\(\s*)(?:[({{]\s*)*{_pipe_wrapped_re(_pipe_sink_word)}",
    re.IGNORECASE,
)
_PIPE_FETCH_RE = re.compile(r"\b(?:curl|wget)\b", re.IGNORECASE)
_PIPE_OPERATOR_RE = re.compile(r"\|&?\s*|>\(\s*")
# The widened sink skips `||` (an or-list, not a pipe); main's raw sink keeps the old operator.
_PIPE_ONLY_OPERATOR_RE = re.compile(r"(?<!\|)\|(?:&\s*|\s+|(?!\|))|>\(\s*")


# main's original sink (bare or slash-prefixed shell), scanned on the RAW text: a quoted payload handed
# to any carrier (su -c, ssh, xargs, docker exec ...) stays flagged exactly as before.
_PIPE_BARE_SINK_RE = re.compile(rf"\|\s*(?:[/\w]*/)?(?:{_SHELL_NAMES_RE})(?:\s|$|-c)", re.IGNORECASE)
