"""Askpass delivery of the sudo password for the terminal tool.

Where a backend's sudo supports an askpass helper, the password is handed to each counted
sudo invocation through a helper script instead of the stdin shared by the whole command.
``build_prologue`` emits the shell text that runs before the rewritten command: it reads the
single password line the caller prepends to stdin, keeps it in a detached writer that serves
it through a FIFO once per counted sudo, and tears everything down when the command's shell
(and every child holding fd 19) exits, or a timeout after that shell is gone. The per-environment
``CAPABILITY_PROBE`` decides whether a backend can use it; ``has_own_stdin_or_askpass`` keeps
commands that already manage sudo's input on the stdin form.
"""

import logging
import platform
import re
import shlex
import subprocess

logger = logging.getLogger("tools.terminal_tool")

# Placeholder for the watchdog's ``read -t`` seconds. ``_prepare_command`` cannot know the
# command's timeout, so ``BaseEnvironment.execute`` fills it via ``fill_timeout_token``.
SUDO_ASKPASS_TIMEOUT_TOKEN = "__HERMES_SUDO_ASKPASS_T__"

# Extra seconds the watchdog waits past the command's own timeout.
_TIMEOUT_SLACK_S = 30
# Used when the effective timeout is not a number (the execute backstop uses 120 s then).
_FALLBACK_TIMEOUT_S = 120

SUDO_ASKPASS_PREFIX = 'SUDO_ASKPASS="$__hermes_sudo_dir/a" '
# Expands to ``-A`` once the helper is set up, and to ``-S -p ''`` when setup failed at run time
# (the command then reads its password lines from stdin exactly as the stdin form does).
SUDO_ASKPASS_SUFFIX = ' "${__hermes_sudo_o[@]}"'

# bash 3.2 syntax only. ``\command`` / ``\builtin`` keep aliases and functions from the session
# snapshot from shadowing the commands the prologue relies on; ``exec`` is written ``\exec``
# (alias-proof) because bash 3.2's ``command exec`` keeps saved copies of the redirected fds
# open, so the detached watchdog would hold the command's output pipe. A temp path holding a
# character that is special inside double quotes (the helper embeds it in one) counts as a
# setup failure.
_PROLOGUE_HEAD = "__hermes_sudo_pw=; __hermes_sudo_o=(-S -p '')\n"
_PROLOGUE_TEMPLATE = (
    _PROLOGUE_HEAD
    + '__hermes_sudo_dir=$(\\builtin umask 077 && \\command mktemp -d "${TMPDIR:-/tmp}/hermes-sudo.XXXXXXXXXX"'
    " 2>/dev/null) || __hermes_sudo_dir=\n"
    'case $__hermes_sudo_dir in *[\\"\\\\\\$\\`]*) \\command rm -rf -- "$__hermes_sudo_dir" 2>/dev/null;'
    " __hermes_sudo_dir= ;; esac\n"
    # Setup comes before any read of stdin: when a step fails the password lines stay on stdin
    # and every counted sudo keeps ``-S -p ''``, which is the stdin form.
    'if \\builtin test -n "$__hermes_sudo_dir" && \\command mkfifo "$__hermes_sudo_dir/p" "$__hermes_sudo_dir/l"'
    " 2>/dev/null &&\n"
    # The helper takes a lock file for its whole exchange, so helpers of sudos that ask at the same
    # time are served one after another instead of sharing one line. The lock is an O_EXCL create
    # (``set -C``), not ``mkdir``: uutils' mkdir (Ubuntu 25.10+) lets two racing callers both
    # succeed, so two helpers read one serving and a sudo gets an empty password.
    "  \\builtin printf '#!/bin/sh\\nd=\"%s\"\\n"
    'until (set -C; : > "$d/k") 2>/dev/null; do [ -d "$d" ] || exit 1; /bin/sleep 0.1 2>/dev/null || /bin/sleep 1; done\\n'
    '/bin/cat -- "$d/p" 2>/dev/null; r=$?\\n/bin/rm -f "$d/k"\\nexit $r\\n\' "$__hermes_sudo_dir"'
    ' > "$__hermes_sudo_dir/a" 2>/dev/null && \\command chmod 700 "$__hermes_sudo_dir/a" 2>/dev/null &&\n'
    '  \\exec 19<>"$__hermes_sudo_dir/l"; then\n'
    # One password line per counted sudo is on stdin; the helper form needs the first only.
    "  IFS= \\builtin read -r __hermes_sudo_pw || \\builtin :\n"
    "  __hermes_sudo_i=1\n"
    "  while \\builtin test \"$__hermes_sudo_i\" -lt @COUNT@; do\n"
    "    IFS= \\builtin read -r __hermes_sudo_x || \\builtin break; __hermes_sudo_i=$((__hermes_sudo_i + 1))\n"
    "  done\n"
    "  \\builtin unset __hermes_sudo_x __hermes_sudo_i\n"
    "  __hermes_sudo_o=(-A)\n"
    '( ( \\exec 8<"$__hermes_sudo_dir/l" 19>&- </dev/null >/dev/null 2>&1\n'
    "    \\builtin trap '' HUP PIPE; \\builtin trap '\\builtin :' TERM\n"
    "    __hermes_sudo_n=@COUNT@\n"
    "    { __hermes_sudo_i=0\n"
    '  while \\builtin test "$__hermes_sudo_i" -lt "$__hermes_sudo_n"; do\n'
    "    __hermes_sudo_i=$((__hermes_sudo_i + 1))\n"
    # Each external command of this background group runs in its own subshell: Apple's bash 3.2
    # replaces a background ``{ ...; } &`` group with the first ``command <program>`` it meets
    # (no fork), so nothing after it would run.
    '    { if \\builtin test "$__hermes_sudo_i" -lt "$__hermes_sudo_n"; then\n'
    '        ( \\command mkfifo "$__hermes_sudo_dir/q" ) && ( \\command mv -f "$__hermes_sudo_dir/q" "$__hermes_sudo_dir/p" )\n'
    '      else ( \\command rm -f "$__hermes_sudo_dir/p" ); fi >/dev/null 2>&1\n'
    "      \\builtin printf '%s\\n' \"$__hermes_sudo_pw\"; } > \"$__hermes_sudo_dir/p\" || \\builtin break\n"
    "  done; } 8<&- &\n"
    "    __hermes_sudo_w=$!\n"
    "    \\builtin unset __hermes_sudo_pw\n"
    # The wait ends at EOF on fd 19 (the shell and every child holding it are gone). It is
    # checked every two seconds: while the command's shell ($$) runs, as after a yield to the
    # background, serving continues; once the shell is gone it stops at once, unless a counted
    # sudo was put in the background (@BG@ = 1), which is then served for up to @TIMEOUT@ more
    # seconds. A timed-out read is status > 128 on bash 4+ and status 1 after the wait on bash
    # 3.2, where EOF returns 1 at once (SECONDS counts whole seconds, hence ``-lt 1``).
    "    __hermes_sudo_t=@TIMEOUT@; __hermes_sudo_g=0\n"
    "    while \\builtin :; do\n"
    "      SECONDS=0; \\builtin read -r -t 2 __hermes_sudo_x <&8; __hermes_sudo_r=$?\n"
    '      if \\builtin test "$__hermes_sudo_r" -le 128 && \\builtin test "$SECONDS" -lt 1; then \\builtin break; fi\n'
    "      if \\builtin kill -0 $$ 2>/dev/null; then \\builtin continue; fi\n"
    "      if \\builtin test @BG@ -eq 0; then \\builtin break; fi\n"
    "      __hermes_sudo_g=$((__hermes_sudo_g + 2))\n"
    '      if \\builtin test "$__hermes_sudo_g" -ge "$__hermes_sudo_t"; then \\builtin break; fi\n'
    "    done\n"
    '\\builtin kill "$__hermes_sudo_w" 2>/dev/null\n'
    '\\exec 7<>"$__hermes_sudo_dir/p" 2>/dev/null\n'
    '\\command rm -rf "$__hermes_sudo_dir"\n'
    "\\exec 7>&-\n"
    "  ) & )\n"
    "else\n"
    '  \\builtin test -z "$__hermes_sudo_dir" || \\command rm -rf -- "$__hermes_sudo_dir" 2>/dev/null\n'
    "  __hermes_sudo_dir=\n"
    "fi\n"
    "\\builtin unset __hermes_sudo_pw\n"
)


def build_prologue(count: int, timeout_token: str, background: bool = True) -> str:
    """Shell text that serves the password read from stdin to *count* askpass helper calls.

    Stdin carries *count* password lines, as for the stdin form. *timeout_token* is how long a
    backgrounded sudo is still served after the command's shell has exited: a literal number,
    or ``SUDO_ASKPASS_TIMEOUT_TOKEN`` for ``fill_timeout_token`` to replace later. With
    *background* false (no counted sudo runs in the background) serving stops with the shell.
    """
    return (_PROLOGUE_TEMPLATE.replace("@COUNT@", str(int(count))).replace("@TIMEOUT@", timeout_token)
            .replace("@BG@", "1" if background else "0"))


def fill_timeout_token(command: str, timeout) -> str:
    """Replace the prologue's timeout token with ``int(timeout) + 30``.

    Only a command that starts with the prologue is touched, and only its one token, so any
    other command text stays byte-for-byte as given.
    """
    if not isinstance(command, str) or not command.startswith(_PROLOGUE_HEAD):
        return command
    try:
        seconds = int(timeout)
    except (TypeError, ValueError, OverflowError):
        seconds = _FALLBACK_TIMEOUT_S
    return command.replace(SUDO_ASKPASS_TIMEOUT_TOKEN, str(seconds + _TIMEOUT_SLACK_S), 1)


# sudo options that take a separate value word (sudo 1.9 and sudo-rs).
_VALUE_SHORT_OPTIONS = frozenset("ugpCDrtUTRhc")
_VALUE_LONG_OPTIONS = frozenset({
    "--user", "--group", "--prompt", "--close-from", "--chdir", "--chroot", "--role", "--type",
    "--other-user", "--command-timeout", "--host", "--login-class",
})
_REDIRECTION_RE = re.compile(r"^[0-9]*(?:[<>]|&>)")
# A redirection operator written apart from its target (``> out``, ``2>``, ``<<``): the next
# word is that target, not sudo's command.
_BARE_REDIRECTION_RE = re.compile(r"^[0-9]*(?:<<<|<<-?|<>|>>|>\||<|>)$")


def _option_words_carry_own_input(words: list[str]) -> bool:
    """True when sudo's option words (up to its command) include a stdin or askpass option."""
    skip_value = False
    for raw in words:
        if skip_value:
            skip_value = False
            continue
        if _REDIRECTION_RE.match(raw):
            skip_value = bool(_BARE_REDIRECTION_RE.match(raw))
            continue
        if "$" in raw or "`" in raw:
            return True  # an expansion could produce the option: keep the stdin form
        try:
            parsed = shlex.split(raw)
        except ValueError:
            return True
        if len(parsed) != 1:
            return True
        word = parsed[0]
        if word in ("--stdin", "--askpass"):
            return True
        if word == "--" or not word.startswith("-") or word == "-":
            return False
        if word.startswith("--"):
            skip_value = word in _VALUE_LONG_OPTIONS
            continue
        for index, char in enumerate(word[1:], start=1):
            if char in "SA":
                return True
            if char in _VALUE_SHORT_OPTIONS:
                skip_value = index == len(word) - 1
                break
    return False


def has_own_stdin_or_askpass(command: str | None) -> bool:
    """True when any counted sudo already carries its own stdin or askpass option.

    Single-letter clusters count (``-nS``); values of value-taking options are skipped
    (``-u S``, ``-pS``). Such a command keeps the stdin form, since sudo rejects ``-A`` with
    ``-S``.
    """
    from tools.terminal_tool_sudo import _rewrite_real_sudo_invocations, _scan_shell

    if not command:
        return False
    marked, count = _rewrite_real_sudo_invocations(command, askpass=True)
    if not count:
        return False
    # Each counted sudo now reads ``<prefix><sudo word> -A <its options...>``. Text that happens
    # to contain the prefix only adds a chunk to check, which can only select the stdin form.
    for chunk in marked.split(SUDO_ASKPASS_PREFIX)[1:]:
        words: list[str] = []
        for kind, start, end, _ in _scan_shell(chunk):
            if kind == "op" and chunk[start:end] == "&" and (
                chunk[start - 1:start] in ("<", ">") or chunk[end:end + 1] == ">"
            ):
                continue  # ``2>&1`` / ``&> out``: part of a redirection, not the end of the command
            if kind in ("op", "comment") or (kind == "ws" and chunk[start] == "\n"):
                break
            if kind == "word":
                words.append(chunk[start:end])
        if _option_words_carry_own_input(words[2:]):  # [0] sudo word, [1] the added option word
            return True
    return False


def has_background_sudo(command: str | None) -> bool:
    """True when a counted sudo may still ask after the command's shell has exited.

    That is a counted sudo followed by a background ``&`` in the same list, or one inside a
    group, subshell or function body with a background ``&`` anywhere after it. A command with
    a ``case`` or ``coproc`` word counts too, because the list tracking below does not model
    them. It errs toward True: a miss would stop serving a backgrounded sudo that the stdin
    form still feeds.
    """
    from tools.terminal_tool_sudo import _rewrite_real_sudo_invocations, _scan_shell

    if not command:
        return False
    marked, count = _rewrite_real_sudo_invocations(command, askpass=True)
    if not count:
        return False
    events = list(_scan_shell(marked))
    if any(kind == "word" and marked[start:end] in ("case", "coproc") for kind, start, end, _ in events):
        return True
    prefix_word = SUDO_ASKPASS_PREFIX.strip()
    depth = 0
    in_list = in_group = False
    for kind, start, end, _ in events:
        text = marked[start:end]
        if kind == "op":
            if text == "(":
                depth += 1
            elif text == ")":
                depth = max(0, depth - 1)
            elif text == "&":
                if marked[start - 1:start] in ("<", ">") or marked[end:end + 1] == ">":
                    continue  # part of a redirection
                if in_list or in_group:
                    return True
            elif text in (";", ";;") and depth == 0:
                in_list = False
        elif kind == "ws" and text == "\n" and depth == 0:
            in_list = False
        elif kind == "word":
            if text == "{":
                depth += 1
            elif text == "}":
                depth = max(0, depth - 1)
            elif text == prefix_word:
                in_list = True
                in_group = in_group or depth > 0
    return False


# Exit 0 only when sudo accepts ``-A`` (``-V`` authenticates nothing) and a 0700 helper in a
# ``mktemp -d`` under ${TMPDIR:-/tmp} runs, next to a FIFO the shell can open read+write.
CAPABILITY_PROBE = (
    "SUDO_ASKPASS=/bin/false sudo -A -V >/dev/null 2>&1 || exit 1\n"
    'for t in /bin/cat /bin/rm /bin/sleep; do [ -x "$t" ] || exit 1; done\n'
    'd=$(umask 077 && mktemp -d "${TMPDIR:-/tmp}/hermes-sudo.XXXXXXXXXX" 2>/dev/null) || exit 1\n'
    'case $d in *[\\"\\\\\\$\\`]*) rm -rf -- "$d"; exit 1 ;; esac\n'
    "r=1\n"
    "if mkfifo \"$d/l\" 2>/dev/null && printf '#!/bin/sh\\nexec /bin/cat -- /dev/null\\n' > \"$d/a\""
    ' && chmod 700 "$d/a"; then\n'
    '  ( exec 3<>"$d/l" ) 2>/dev/null && "$d/a" </dev/null >/dev/null 2>&1 && r=0\n'
    "fi\n"
    'rm -rf -- "$d"\n'
    "exit $r\n"
)

_PROBE_ERRORS = (OSError, RuntimeError, ValueError, TypeError, AttributeError, subprocess.SubprocessError)


def sudo_askpass_supported_once(env) -> bool:
    """Run ``CAPABILITY_PROBE`` once per environment object and cache the answer.

    False without probing when the backend does not opt in to in-backend sudo probes
    (``_passwordless_sudo_probe_supported``) or is the local backend on Windows; False on any
    error or timeout. Bound as ``BaseEnvironment._sudo_askpass_supported``.
    """
    cached = getattr(env, "_sudo_askpass_ok", None)
    if cached is not None:
        return cached
    supported = False
    if getattr(env, "_passwordless_sudo_probe_supported", False) and not (
        getattr(env, "is_local", False) and platform.system() == "Windows"
    ):
        timeout = getattr(env, "_SUDO_PROBE_TIMEOUT_S", 3)
        try:
            proc = env._run_bash(CAPABILITY_PROBE, timeout=timeout)
            supported = env._wait_for_process(proc, timeout=timeout).get("returncode") == 0
        except _PROBE_ERRORS:
            logger.debug("sudo askpass capability probe failed", exc_info=True)
            supported = False
    env._sudo_askpass_ok = supported
    return supported
