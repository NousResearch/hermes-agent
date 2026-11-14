"""Hardline floor: wrapped / path-qualified commands must not newly block ordinary work.

Covers carriers bounded inside a quoted substitution, sudo's no-command modes in the carrier scan,
quoted rm operands that continue past their closing quote, and literal (quoted-delimiter) heredoc
bodies copied by cat / tee.
"""
import pytest

from tools.approval_detection import detect_hardline_command

_Q = '"'

_BLOCKED = [
    # A carrier inside a double-quoted substitution: its argv ends at the `)` (#58642 review).
    "echo " + _Q + "$(su -c 'reboot')" + _Q,
    "x=" + _Q + "$(runuser -u a -c 'reboot') tail" + _Q,
    # sudo -k still runs its command.
    "sudo -k su -c reboot",
    # Quoted roots joined to `.` / `/` / `*` or a protected dir still collapse onto the root.
    "nice rm -rf " + _Q + "/" + _Q,
    "nice rm -rf " + _Q + "/" + _Q + "*",
    "nice rm -rf " + _Q + "/" + _Q + "./*",
    "nice rm -rf " + _Q + "/" + _Q + "../",
    "nice rm -rf " + _Q + "$HOME" + _Q + "/*",
    "nice rm -rf " + _Q + "/" + _Q + "etc",
    "rm -rf " + _Q + "/etc" + _Q + "/*",
    # A trailing slash after a quoted (or bare) system dir is still that dir.
    "rm -rf " + _Q + "/etc" + _Q + "/",
    "nice rm -rf '/usr'/",
    "rm -rf " + _Q + "/" + _Q + "etc/",
    "rm -rf /etc/",
    "rm -rf /usr//",
    # `..` segments before the glob climb out to the root's children.
    "rm -rf " + _Q + "/etc" + _Q + "/../*",
    "rm -rf " + _Q + "/etc" + _Q + "/../../*",
    "nice rm -rf '/usr'/../*",
    # Heredoc bodies that a shell (or a shell behind a pipe / ssh) executes stay scanned.
    "bash <<'EOF'\ntrue; " + _Q + "reboot" + _Q + "\nEOF",
    "cat <<'EOF' | sh\ntrue; " + _Q + "reboot" + _Q + "\nEOF",
    "ssh host <<'EOF'\ntrue; " + _Q + "reboot" + _Q + "\nEOF",
    # A command after the delimiter runs.
    "cat <<'EOF'\nx\nEOF\ntrue; " + _Q + "reboot" + _Q,
    # An unquoted delimiter expands the body, so it is not literal.
    "cat <<EOF\ntrue; " + _Q + "reboot" + _Q + "\nEOF",
    "ls; " + _Q + "reboot" + _Q,
]

_ALLOWED = [
    # sudo -l lists what the carrier would run; nothing runs.
    "sudo -l su -c reboot",
    # A quoted prefix of a subtree is the subtree.
    "nice rm -rf " + _Q + "$HOME" + _Q + "/project/build",
    "/bin/rm -rf " + _Q + "/" + _Q + "tmp/build",
    "rm -rf " + _Q + "$HOME" + _Q + "/project/build",
    "rm -rf " + _Q + "/etc" + _Q + "/nginx/old",
    "rm -rf /etcetera/",
    "rm -rf /etc/../tmp/x",
    # Literal heredoc text written by cat / tee.
    "cat <<'EOF'\ntrue; " + _Q + "reboot" + _Q + "\nEOF",
    "cat <<'EOF' > notes.md\ntrue; " + _Q + "reboot" + _Q + "\nEOF",
    "tee docs.md <<" + _Q + "EOF" + _Q + "\nls; " + _Q + "shutdown" + _Q + " -h now\nEOF",
]


@pytest.mark.parametrize("command", _BLOCKED)
def test_runs_the_verb_is_hardline(command):
    assert detect_hardline_command(command)[0] is True, command


@pytest.mark.parametrize("command", _ALLOWED)
def test_ordinary_work_is_not_hardline(command):
    assert detect_hardline_command(command) == (False, None), command
