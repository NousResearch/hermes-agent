"""AppleScript that can discard the operator's unsaved work always reaches a human (#141346).

Smart approval auto-approved `osascript` that closed the user's own unsaved PowerPoint deck with
`saving no`, then `rm -f` of its only export. Three things are pinned here, because any one alone
leaves the hole open:

* detection: every osascript run whose program can close, quit, save over, delete, send keystrokes
  or run a shell is flagged, and so is every run whose program this command line does not show
  (script file, pipe, carrier, expansion). Read-only literal programs stay unflagged;
* the gate: in smart mode those keys never reach the guardian LLM, even when it would APPROVE;
* scope: the human answer covers one command, so "session" cannot pre-approve the next close.
"""

import pytest

from tools import approval as approval_module
from tools import approval_context
from tools.approval import check_all_command_guards, detect_dangerous_command
from tools.approval_applescript import OSASCRIPT_OPAQUE_DESCRIPTION, OSASCRIPT_RISKY_DESCRIPTION

RISKY, OPAQUE = OSASCRIPT_RISKY_DESCRIPTION, OSASCRIPT_OPAQUE_DESCRIPTION
REPORTED = (
    'osascript -e \'tell application "Microsoft PowerPoint"\n'
    '  open POSIX file "/Users/me/traz.pptx"\n'
    '  save active presentation in POSIX file "/Users/me/traz.pdf" as save as PDF\n'
    '  close active presentation saving no\n'
    'end tell\''
)
FRONTMOST = ("osascript -e 'tell application \"System Events\" to get name of first application "
             "process whose frontmost is true'")


@pytest.mark.parametrize("command", [
    REPORTED,
    'osascript -e \'tell application "Microsoft PowerPoint" to close presentation 1 saving no\'',
    'osascript <<\'EOF\'\ntell application "Microsoft PowerPoint" to close active presentation saving no\nEOF',
    'osascript <<< \'tell application "Keynote" to quit\'',
    'osascript -e \'tell application "Keynote" to quit\'',
    'osascript -e \'tell application "Pages" to quit without saving\'',
    'osascript -e \'tell application "Word" to close every document saving no\'',
    'osascript -e \'tell application "Keynote" to save front document\'',
    'osascript -e \'tell application "TextEdit" to set text of front document to ""\'',
    'osascript -e \'tell application "Finder" to delete POSIX file "/Users/me/traz.pdf"\'',
    # Runs a shell command: must not surface as a guardian-reviewable `rm` rule first.
    'osascript -e \'do shell script "rm -f /Users/me/traz.pdf"\'',
    'osascript -e \'tell application "System Events" to keystroke "w" using command down\'',
    'osascript -e \'tell application "System Events" to key code 12 using command down\'',
    'osascript -e \'tell app "Keynote" to «event coreclos» front document\'',
    # JXA can assemble any method name at run time, so every JXA program counts.
    'osascript -l JavaScript -e \'Application("Keynote").documents[0].close({saving: "no"})\'',
    'osascript -l JavaScript -e \'Application("Keynote")["clo" + "se"]()\'',
    # Executable identity: paths, wrappers, quote/escape spellings, a second run after a read.
    '/usr/bin/osascript -e \'tell app "Word" to quit saving no\'',
    'sudo nice -n 5 /usr/bin/osascript -e \'tell app "Word" to quit saving no\'',
    'command osascript -e \'quit app "Pages"\'',
    "osa''script -e 'quit app \"Pages\"'",
    FRONTMOST + ' && osascript -e \'tell app "Keynote" to close front document saving no\'',
    # Review rows: a quoted or attached `-l`, a backslash-newline inside the name, a second command
    # on a heredoc's opening line, and `<<- EOF` with a blank line in the body.
    'osascript \'-l\' JavaScript -e \'Application("Keynote").documents[0]["cl"+"ose"]({saving:"no"})\'',
    'osascript -lJavaScript -e \'Application("Keynote").documents.length\'',
    'osa\\\nscript -e \'tell app "Keynote" to quit\'',
    'osascript <<\'EOF\'; osascript -e \'tell app "Keynote" to quit\'\nget name of me\nEOF',
    'osascript <<- EOF\nget name of me\n\ntell application "Keynote" to quit\nEOF',
    'osascript <<-EOF\n\tget name of me\n\ttell application "Keynote" to quit\n\tEOF',
])
def test_risky_program_is_flagged(command):
    assert detect_dangerous_command(command)[1] == RISKY, command


@pytest.mark.parametrize("command", [
    # Carriers: the program reaches osascript through another command's arguments.
    'sh -c "osascript -e \'tell app \\"X\\" to get name\'"',
    'eval "osascript -e \'quit\'"',
    'env -S "osascript -e \'quit\'"',
    'ssh mac "osascript -e \'quit app Keynote\'"',
    'python3 -c "import os; os.system(\'osascript -e quit\')"',
    'xargs -I{} osascript -e {} < cmds.txt',
    "find . -name '*.scpt' -exec osascript {} \\;",
    "sh -c 'o\\sascript -e quit'",
    # The program text is not on this command line.
    'printf \'tell app "Keynote" to quit\' | osascript',
    'echo cXVpdA== | base64 -d | osascript',
    'cat x.applescript | osascript -',
    'osascript /Users/me/cleanup.scpt',
    'o\\sascript /Users/me/cleanup.scpt',
    'osascript - < cleanup.applescript',
    'osascript -e "$SCRIPT"',
    'osascript -e "tell app \\"$APP\\" to get name"',
    'osascript <<EOF\ntell app "$APP" to get name\nEOF',
    "X=osascript; $X -e 'quit'",
    "$(which osascript) -e 'quit'",
    # Interpreters the detector already knows (bun/deno) carry osascript like python/node do.
    'bun -e \'require("child_process").execFileSync("osascript", ["-e", "tell application \\"Keynote\\" to quit"])\'',
    'deno eval \'new Deno.Command("osascript", {args: ["-e", "quit app \\"Keynote\\""]}).outputSync()\'',
])
def test_unreviewable_program_is_flagged(command):
    assert detect_dangerous_command(command)[1] == OPAQUE, command


@pytest.mark.parametrize("command", [
    FRONTMOST,
    # The exact read Hermes' own macOS `open` raise-ladder runs, inside $(...) and with a redirect.
    '_H_APP="$(' + FRONTMOST + ' 2>/dev/null)"',
    "osascript -e 'tell application \"System Events\" to get frontmost of application process \"Safari\"' 2>/dev/null",
    "osascript -e 'tell application \"Finder\" to get the name of every disk'",
    "osascript -e 'tell application \"Keynote\" to activate'",
    "osascript -e 'display notification \"Build done\" with title \"Hermes\"'",
    "osascript -e 'get volume settings' >/dev/null 2>&1 || true",
    "env LANG=C osascript -e 'tell application \"Music\" to get name of current track'",
    # Mentions that run no AppleScript.
    'git commit -m "osascript quit the app with saving no"',
    'echo "osascript closed it"',
    "grep -rn osascript tools/",
    "find . -name '*osascript*'",
    "which osascript",
    # A read-only heredoc whose body merely contains the word in prose, and an explicit AppleScript `-l`.
    "osascript <<'EOF'\nget name of me\nEOF",
    "osascript -l AppleScript -e 'get name of me'",
])
def test_read_only_and_unrelated_commands_are_unchanged(command):
    assert detect_dangerous_command(command) == (False, None, None), command


@pytest.fixture
def smart_interactive(monkeypatch):
    """Interactive smart mode with a guardian that would APPROVE everything and records its calls."""
    session_key = "test-141346"
    monkeypatch.setenv("HERMES_SESSION_KEY", session_key)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    for name in ("HERMES_EXEC_ASK", "HERMES_GATEWAY_SESSION", "HERMES_CRON_SESSION", "HERMES_YOLO_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
    guardian_calls = []

    def guardian(command, *_args, **_kwargs):
        guardian_calls.append(command)
        return "approve"

    monkeypatch.setattr(approval_module, "_smart_verdict", guardian)
    approval_module.clear_session(session_key)
    approval_module._permanent_approved.clear()
    yield guardian_calls
    approval_module.clear_session(session_key)


@pytest.mark.parametrize("command", [
    REPORTED,
    'osascript -e \'do shell script "rm -f /Users/me/traz.pdf"\'',
    'printf \'tell app "Keynote" to quit\' | osascript',
])
def test_smart_mode_asks_the_human_not_the_guardian(command, smart_interactive):
    prompts = []

    def human(cmd, desc, **kwargs):
        prompts.append(kwargs)
        return "deny"

    result = check_all_command_guards(command, "local", approval_callback=human)

    assert smart_interactive == []  # the guardian was never asked
    assert len(prompts) == 1
    assert prompts[0].get("allow_permanent") is False and prompts[0].get("allow_session") is False
    assert result["approved"] is False
    assert not result.get("smart_approved")


def test_a_session_answer_does_not_preapprove_the_next_close(smart_interactive):
    answers = []

    def human(cmd, desc, **kwargs):
        answers.append(cmd)
        return "session"  # an older client may still return a broader scope

    assert check_all_command_guards(REPORTED, "local", approval_callback=human)["approved"] is True
    assert check_all_command_guards(REPORTED, "local", approval_callback=human)["approved"] is True
    assert len(answers) == 2  # asked both times
    assert smart_interactive == []


def test_other_dangerous_commands_still_use_the_guardian(smart_interactive):
    result = check_all_command_guards("rm -rf ./build", "local", approval_callback=lambda *a, **k: "deny")
    assert result.get("smart_approved") is True
    assert smart_interactive == ["rm -rf ./build"]
