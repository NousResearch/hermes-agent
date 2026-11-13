"""macOS ``open`` frontmost raise-ladder for the terminal tool.

Split out of ``tools/terminal_tool.py`` (PR #95286, Jackal991): a pure,
deterministic string rewrite — no subprocess calls at transform time — that
appends a verified raise-ladder to file-opening ``open`` invocations so the
opened document lands in front of the Hermes window instead of behind it.

The ladder's Apple Events are routed through HermesTerminalHelper
(``native/hermes_terminal_helper/``), a named, signed .app bundle installed
next to the launcher. macOS TCC attributes an automation prompt to the
process that SENDS the event, so sending the events from the bundle makes
the prompt read "Hermes Terminal Helper wants to control System Events"
with the purpose string from its Info.plist — instead of the generic,
unexplained "osascript". Every rung keeps its exact ``osascript`` form as a
runtime fallback, so hosts without the helper (fresh installs, CI,
non-checkout layouts) behave exactly as before.
"""

import os
import platform
import shlex
from pathlib import Path

# ---------------------------------------------------------------------------
# macOS `open` frontmost raise-ladder
#
# On macOS, when the agent opens a file via a tool call (e.g. `open -a Preview
# file.pdf` or `open file.pdf`), the command returns exit 0 and genuinely
# succeeds — the document loads and a window is created — but the window opens
# BEHIND the Hermes desktop window (Hermes is typically maximised, so the file
# is completely hidden). Because the exit code is 0, the agent reports success
# while the user sees nothing happen.
#
# Root cause: the tool shell runs as a child process of the Electron app; macOS
# does not let a non-frontmost process hand activation to another app, so the
# implicit activation that normally accompanies `open` is silently ignored.
# `osascript -e 'tell application "X" to activate'` alone is NOT reliable (it
# also returns 0 while focus lands elsewhere, e.g. a racing Chrome).
#
# Fix: VERIFY, don't trust the exit code. After an `open`, check `frontmost`
# and escalate via a raise-ladder, then settle ~0.8s, re-check, and re-assert
# once. Each rung is verified before moving on; the report comes from the final
# observed state, never from "a rung returned 0".
# ---------------------------------------------------------------------------

# `open` flags that do NOT open a file in a new frontmost window and must never
# be transformed (conservative: only transform when it clearly opens a file).
_MACOS_OPEN_NO_TRANSFORM_FLAGS = frozenset(
    {"-R", "-e", "-t", "-f", "-W", "-n", "-g", "-h", "-b", "-D"}
)

# Shell splice that breaks out of a single-quoted osascript argument so the
# shell expands $_H_APP (handles app names containing spaces, e.g. "Google
# Chrome"). Renders as:  " '" $_H_APP "' "
_APP_SPLICE = '"\'"$_H_APP"\'"'

# ---------------------------------------------------------------------------
# Hermes Terminal Helper routing
#
# HermesTerminalHelper is a named, signed .app bundle that sends the ladder's
# Apple Events in-process (NSAppleScript), so TCC automation prompts attribute
# to "Hermes Terminal Helper" instead of the generic "osascript". It ships in
# the checkout at .hermes/bin/ (next to the launcher) and installs to the
# shared ~/.hermes/bin. The ladder emits a runtime probe rather than resolving
# the path at transform time, keeping this module a pure string rewrite with
# identical output on every host.
# ---------------------------------------------------------------------------

_CHECKOUT_ROOT = Path(__file__).resolve().parent.parent
_HELPER_BUNDLE_BIN = ".hermes/bin/HermesTerminalHelper.app/Contents/MacOS/HermesTerminalHelper"


def terminal_helper_binary() -> str | None:
    """Absolute path to HermesTerminalHelper when installed, else None.

    Runtime resolution (filesystem stat) for callers that exec the helper
    directly — unlike the ladder, which embeds a runtime probe instead so
    its rewrite stays a pure string.
    """
    for candidate in (
        _CHECKOUT_ROOT / _HELPER_BUNDLE_BIN,
        Path.home() / ".hermes/bin/HermesTerminalHelper.app/Contents/MacOS/HermesTerminalHelper",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _helper_probe() -> str:
    """Shell preamble locating HermesTerminalHelper at runtime.

    Emits ``_H_HELPER=<path>`` when an executable helper exists (checkout
    copy first, then the shared install bin), else ``_H_HELPER=""``. Every
    rung below checks ``-n "$_H_HELPER"`` and takes its osascript fallback
    when empty.
    """
    checkout = shlex.quote(str(_CHECKOUT_ROOT / _HELPER_BUNDLE_BIN))
    install = "$HOME/.hermes/bin/HermesTerminalHelper.app/Contents/MacOS/HermesTerminalHelper"
    return (
        '_H_HELPER=""; '
        f'for _p in {checkout} "{install}"; do '
        'if [ -x "$_p" ]; then _H_HELPER="$_p"; break; fi; done'
    )


def _helper_or(helper_args: str, fallback: str) -> str:
    """Statement-level rung: helper when installed, else the osascript form."""
    return (
        f'if [ -n "$_H_HELPER" ]; then "$_H_HELPER" {helper_args} 2>/dev/null; '
        f"else {fallback}; fi"
    )


def _helper_or_query(helper_args: str, fallback: str) -> str:
    """Query rung wrapped in a subshell so ``if ! <query>`` keeps working.

    The subshell is a simple command, so the ``!`` negation applies to its
    exit status — which is the selected branch's exit status.
    """
    return f"( {_helper_or(helper_args, fallback)} )"


def _osascript_app(script: str) -> str:
    """Wrap an AppleScript snippet in an `osascript -e` invocation.

    This is the fallback form of every helper-routed rung (and the whole
    ladder on hosts without the helper). ``script`` may reference the target
    app via the ``_APP_SPLICE`` shell splice (``$_H_APP``). Errors are
    suppressed so a swallowed AppleEvent simply falls through to the next
    rung of the ladder.
    """
    return f"osascript -e '{script}' 2>/dev/null"


def _build_macos_open_raise_ladder(app: str | None, file: str) -> str:
    """Build the shell raise-ladder appended after a macOS ``open`` command.

    The ladder verifies the target app is frontmost and escalates through
    progressively more forceful activation mechanisms, then settles ~0.8s,
    re-checks ``frontmost``, and re-asserts once to absorb a queued activation
    from another app. Each rung is verified before moving on.

    Args:
        app: The target application name (from ``open -a <App>``), or None for
            a bare ``open <file>`` where the app is resolved at runtime from
            the frontmost process.
        file: The file path that was opened.
    """
    if app is not None:
        app_assign = f"_H_APP={shlex.quote(app)}"
    else:
        # Bare `open <file>`: the target app is unknown at transform time, so
        # resolve it at runtime from whichever process is frontmost after the
        # open (the app that actually received the document).
        app_assign = (
            '_H_APP="$('
            + _helper_or(
                "getName",
                _osascript_app(
                    'tell application "System Events" to get name of '
                    "first application process whose frontmost is true"
                ),
            )
            + ')"'
        )
    file_q = shlex.quote(file)

    activate = _helper_or(
        'activate "$_H_APP"',
        _osascript_app("tell application " + _APP_SPLICE + " to activate"),
    )
    frontmost = _helper_or_query(
        'frontmost "$_H_APP"',
        _osascript_app(
            'tell application "System Events" to get frontmost of application process '
            + _APP_SPLICE
        ),
    )
    unminimise = _helper_or(
        "unhide-all",
        _osascript_app(
            'tell application "System Events" to set visible of every process to true'
        ),
    )
    # lsappinfo is a different subsystem and works even when AppleEvents are
    # swallowed. `find` returns the app's ASN; `setfront` raises it.
    asn = 'lsappinfo find "$_H_APP" 2>/dev/null | awk \'{print $1}\''
    setfront = '[ -n "$_H_ASN" ] && lsappinfo setfront "$_H_ASN" 2>/dev/null'
    # Re-issuing `open -a <App> <file>` on the already-open document is treated
    # by macOS as pure activation.
    reopen = f'open -a "$_H_APP" {file_q} 2>/dev/null'

    # Outcome reporting from the FINAL observed frontmost state. The two
    # branches are deliberately distinct so a focus race can never read as
    # success: the tool output tells the agent (and the user) whether the
    # opened document actually ended up in front, and if not, which
    # application holds focus. ``file_q`` is a self-contained quoted token, so
    # the messages survive paths with spaces or shell metacharacters.
    frontmost_name = _helper_or(
        "getName",
        _osascript_app(
            'tell application "System Events" to get name of '
            "first application process whose frontmost is true"
        ),
    )
    report = (
        f'if {frontmost}; then echo open: {file_q} is now frontmost; '
        f'else _H_FOCUS=$({frontmost_name}); '
        f'echo open: {file_q} opened but another application holds focus; '
        f'echo "frontmost: $_H_FOCUS"; fi'
    )

    return (
        f"{_helper_probe()}; "
        f"{app_assign}; "
        f"if ! {frontmost}; then "
        f"{activate}; sleep 0.2; "
        f"if ! {frontmost}; then "
        f"_H_ASN=$({asn}); {setfront}; sleep 0.2; "
        f"if ! {frontmost}; then "
        f"{reopen}; sleep 0.2; {unminimise}; {activate}; "
        f"fi; "
        f"fi; "
        f"fi; "
        f"sleep 0.8; "
        f"if ! {frontmost}; then {activate}; sleep 0.4; fi; "
        f"{report}"
    )


def _transform_macos_open_command(command: str | None, system: str | None = None) -> str | None:
    """Append a frontmost raise-ladder to a macOS ``open`` command.

    On macOS, ``open <file>`` / ``open -a <App> <file>`` returns exit 0 and
    genuinely opens the document, but the window can land BEHIND the Hermes
    desktop window (Hermes is typically maximised), so the user sees nothing
    happen while the agent reports success. This transform appends a verified
    raise-ladder that brings the opened app to the front.

    The transform is a pure, deterministic string rewrite (no subprocess calls
    at transform time) so it is unit-testable on any OS by passing ``system``
    as data (``"Darwin"`` to exercise the ladder, ``"Linux"`` to prove the
    no-op). The actual helper/osascript/lsappinfo calls happen at runtime
    inside the rewritten shell command.

    Conservative: only transforms a bare ``open <file>`` or
    ``open -a <App> <file>``. Never transforms ``open`` with no args, the
    non-file-opening flags (``-R``/``-e``/``-t``/``-f``/``-W``/``-n``/``-g``/
    ``-h``/``-b``/``-D``), ``open -a <App>`` with no file, compound commands,
    or any non-``open`` command. The original ``open`` runs exactly as written.
    """
    if (system if system is not None else platform.system()) != "Darwin":
        return command
    if command is None:
        return command
    try:
        tokens = shlex.split(command)
    except ValueError:
        return command
    if not tokens or tokens[0] != "open":
        return command

    args = tokens[1:]
    if not args:
        return command
    first = args[0]

    if first.startswith("-"):
        if first in _MACOS_OPEN_NO_TRANSFORM_FLAGS:
            return command
        if first == "-a":
            # open -a <App> <file>
            if len(args) < 3:
                return command
            app, file = args[1], args[2]
            if file.startswith("-"):
                return command
            if len(tokens) != 4:
                return command
        else:
            return command
    else:
        # bare open <file>
        file = first
        app = None
        if len(tokens) != 2:
            return command

    ladder = _build_macos_open_raise_ladder(app, file)
    return f"{command}; {ladder}"
