"""Shell-variable indirection vs the guards (a real incident).

An agent's test script sent ``X="rm -rf /home"; $X`` through the REAL terminal tool, under a
profile with ``approvals.single_query_mode: approve``. The hardline floor saw only a quoted
assignment value (data) and a ``$X`` command word, so it returned (False, None). Approve mode
auto-approved the dangerous-pattern and Tirith findings, and the command wiped most of the home
directory.

These tests pin two layers:

* the hardline floor resolves same-command ``NAME=value`` assignments (``;``/``&&`` chains, export,
  arrays, chained variables) and re-checks the substituted command;
* unattended approve mode no longer auto-approves a ``$VAR`` command word the command never
  assigns, a command substitution leader with destructive arguments, any Tirith block combined
  with ``delete in root path``, or a Tirith blast-radius block combined with another destructive
  pattern. A later reassignment never hides an earlier destructive use (resolution is per use).

TEST HYGIENE: every assertion here stops at ``check_all_command_guards`` or a pure detector. Nothing
in this file (and nothing anywhere in the suite) may send a must-block control through the real
terminal executor. See CONTRIBUTING.md, "Never execute a must-BLOCK control".
"""

from unittest.mock import patch

import pytest

import tools.approval as approval_module
from gateway.session_context import reset_session_vars
from tools.approval import check_all_command_guards
from tools.approval_detection import detect_dangerous_command, detect_hardline_command
from tools.approval_detection_assignments import (
    opaque_command_leaders, resolve_shell_assignment_variants, resolve_shell_assignments,
    uninspectable_reasons,
)
from tools.approval_unattended_floor import tirith_blast_rules, tirith_block_rules, unattended_approve_refusal

INCIDENT_COMMANDS = [
    'X="rm -rf /home"; $X',
    'true && X="rm -rf /home" && $X',
]
_ALLOW_TIRITH = {"action": "allow", "findings": [], "summary": ""}


@pytest.fixture(autouse=True)
def _clean_approval_state():
    approval_module._permanent_approved.clear()
    approval_module.clear_session("default")
    reset_session_vars()
    yield
    approval_module._permanent_approved.clear()
    approval_module.clear_session("default")
    reset_session_vars()


@pytest.fixture
def single_query(monkeypatch):
    """A -q run (nobody to answer a prompt) with the mode chosen per test."""
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    for name in ("HERMES_GATEWAY_SESSION", "HERMES_EXEC_ASK", "HERMES_YOLO_MODE", "HERMES_CRON_SESSION"):
        monkeypatch.delenv(name, raising=False)

    def with_mode(mode):
        return patch("tools.approval_context._get_single_query_approval_mode", return_value=mode)
    return with_mode


# ---- hardline floor --------------------------------------------------------------------------

@pytest.mark.parametrize("command", INCIDENT_COMMANDS)
def test_incident_commands_are_hardline(command):
    assert detect_hardline_command(command) == (True, "recursive delete of system directory")


@pytest.mark.parametrize(("command", "description"), [
    ('X=rm; "$X" -rf /', "recursive delete of root filesystem"),
    ('A=rm; B="$A -rf /etc"; $B', "recursive delete of system directory"),
    ('export C="rm -rf /"; ${C}', "recursive delete of root filesystem"),
    ('declare -x C="rm -rf /usr"; $C', "recursive delete of system directory"),
    ("c=(rm -rf /usr); \"${c[@]}\"", "recursive delete of system directory"),
    ('D=/; rm -rf $D', "recursive delete of root filesystem"),
    ('H=/home; rm -rf "$H"', "recursive delete of system directory"),
    ('P=rm; P+=" -rf /boot"; $P', "recursive delete of system directory"),
    ('X="rm -rf /home"\n$X', "recursive delete of system directory"),
    ("bash -c 'X=\"rm -rf /home\"; $X'", "recursive delete of system directory"),
    ("X=\"shutdown -h now\"; $X", "system shutdown/reboot"),
    ("bash <<'EOF'\nX=\"rm -rf /home\"; $X\nEOF", "recursive delete of system directory"),
    ('X="dd if=/dev/zero of=/dev/sda"; $X', "dd to raw block device"),
])
def test_indirect_spellings_are_hardline(command, description):
    assert detect_hardline_command(command) == (True, description)


@pytest.mark.parametrize("command", [
    'X="rm -rf /home"; echo $X',          # the value is printed, never executed
    'X="rm -rf /home"',                   # assignment only
    "X='rm -rf /home'; echo '$X'",        # single quotes: no expansion
    'T=/tmp/x; rm -rf $T',
    'W=/home/user/ws; cd $W && ls',
    'PY=python3; $PY -c "print(1)"',
    'echo "X=rm -rf /home; \\$X"',
])
def test_benign_assignments_stay_off_the_floor(command):
    assert detect_hardline_command(command) == (False, None)


def test_resolution_reports_unchanged_commands_as_none():
    assert resolve_shell_assignments("ls -la") is None
    assert resolve_shell_assignments("echo $HOME") is None
    assert resolve_shell_assignments('X=1; echo $X') == 'X=1; echo 1'
    # `C=3 cmd $C`: $C expands before the prefix assignment applies, so it keeps its environment
    # value; the prefix value is only a possible one.
    assert resolve_shell_assignment_variants('export A=1 B="two words"; C=3 cmd $A $B $C') == [
        'export A=1 B="two words"; C=3 cmd 1 two words $C', 'export A=1 B="two words"; C=3 cmd 1 two words 3']


# ---- combined guard under single_query_mode: approve (the acceptance criterion) --------------

@pytest.mark.parametrize("command", INCIDENT_COMMANDS)
def test_incident_commands_blocked_by_combined_guard_in_approve_mode(single_query, command):
    with single_query("approve"):
        result = check_all_command_guards(command, "local")
    assert result["approved"] is False
    assert "hardline" in result["message"].lower()


@pytest.mark.parametrize("command", INCIDENT_COMMANDS)
def test_incident_commands_blocked_even_under_yolo(monkeypatch, command):
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", True)
    assert check_all_command_guards(command, "local")["approved"] is False


def test_unresolved_leader_with_destructive_args_refused_in_approve_mode(single_query):
    # `X=rm; $X -rf ~/.local` resolves to `rm -rf ~/.local`, which is only dangerous (recoverable
    # from backup), so it is the Tirith+destructive rule or the opaque-leader rule that must hold.
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        result = check_all_command_guards("$X -rf ~/.local", "local")
    assert result["approved"] is False
    assert "never fixes" in result["description"]
    assert "single_query_mode" in result["message"]


def test_tirith_blast_block_plus_destructive_pattern_refused_in_approve_mode(single_query):
    blast = {"action": "block", "summary": "", "findings": [
        {"rule_id": "blast_writes_system_path", "severity": "HIGH", "title": "system path"}]}
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=blast):
        result = check_all_command_guards("X=rm; $X -rf ~/.local", "local")
    assert result["approved"] is False
    assert "blast_writes_system_path" in result["description"]


@pytest.mark.parametrize("command", [
    "rm -rf build/",
    "rm -rf /tmp/stuff",
    'T=/tmp/x; rm -rf $T',
    'for p in a b; do $p --version; done',
    'PY=python3; $PY -c "print(1)"',
    'rm -f /tmp/why197.py; for f in *.pdf; do echo "$f"; done',
])
def test_routine_commands_still_auto_approve_in_approve_mode(single_query, command):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        assert check_all_command_guards(command, "local")["approved"] is True


@pytest.mark.parametrize("command", ["rm -rf build/", "for f in a b; do rm -r \"$f.tmp\"; done"])
def test_relative_recursive_delete_with_incomplete_tirith_still_approves(single_query, command):
    # analysis_incomplete + a recursive delete that is NOT rooted stays advisory (see floor module doc).
    incomplete = {"action": "block", "summary": "", "findings": [{"rule_id": "analysis_incomplete"}]}
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=incomplete):
        assert check_all_command_guards(command, "local")["approved"] is True


def test_refusal_is_not_a_floor_yolo_still_bypasses(monkeypatch, single_query):
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", True)
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        assert check_all_command_guards("$X -rf ~/.local", "local")["approved"] is True


def test_deny_mode_still_blocks_first(single_query):
    with single_query("deny"):
        result = check_all_command_guards("X=rm; $X -rf ~/.local", "local")
    assert result["approved"] is False
    assert "Command flagged as dangerous" in result["message"]


# ---- pure classifiers ------------------------------------------------------------------------

def _opaque_words(command):
    return [word for *_, word in opaque_command_leaders(command)]


def test_opaque_leaders_skip_resolved_and_literal_words():
    assert _opaque_words("X=ls; $X -la") == []
    assert _opaque_words("for p in python3 node; do $p --version; done") == []
    assert _opaque_words("$CMD -rf x") == ["$CMD"]
    assert _opaque_words('"${TOOL}" --version') == ['"${TOOL}"']
    assert _opaque_words("$(which rm) -fr x") == ["$(which rm)"]
    # Only bindings made BEFORE the use resolve it: the first $X comes from the environment.
    assert _opaque_words("$X; X=ls") == ["$X"]


@pytest.mark.parametrize(("command", "refused"), [
    # An unassigned variable in command position is refused whatever follows it.
    ("$X -rf ~/.local", True),
    ("$X /home", True),
    ("$DD if=/dev/zero of=/dev/sdb", True),
    ("$X", True),
    ("${X}", True),
    ('"$X"', True),
    ("true && $X", True),
    ("$PY -m pytest -q", True),
    ("for f in *.sh; do $f; done", True),        # a glob loop cannot be enumerated
    # A command substitution in command position is refused only with destructive arguments.
    ("$(which rm) -fr ~", True),
    ("$(which rm) /home", True),
    ("$(which python3) --version", False),
    # Resolved in the same command: judged by what it resolves to.
    ("X=echo; $X hello", False),
    ('PY=python3; $PY -c "print(1)"', False),
    ("for p in python3 node; do $p --version; done", False),
    # Expansions in argument positions or quoted data are not command words.
    ('echo "$X"', False),
    ("ls $HOME", False),
    ("echo '$X'", False),
    # Data that only LOOKS like a command word once a value is spliced back in, or heredoc text.
    ("B=$(printf '%s' '<?php $c=1; if(!f($c)){exit;}' | base64); ssh h \"echo $B | php\"", False),
    ("cat > /tmp/x.php <<'PHP'\n<?php\n$config = 1;\nPHP\nphp /tmp/x.php", False),
    ("FILES=\"a b\"\nfor f in $FILES; do grep -c x \"$f\"; done", False),
    # A variable whose same-command value is itself an environment variable stays opaque.
    ("P=$__TEST_PYTHON; $P -m pytest", True),
])
def test_opaque_leader_classifier(command, refused):
    reason = unattended_approve_refusal(command, dangerous_description=None, tirith_result=None)
    assert (reason is not None) is refused


def test_tirith_rule_extraction():
    assert tirith_blast_rules({"action": "block", "findings": [{"rule_id": "analysis_incomplete"}]}) == []
    assert tirith_blast_rules({"action": "warn", "findings": [{"rule_id": "blast_find_delete"}]}) == []
    assert tirith_blast_rules({"action": "block", "findings": [
        {"rule_id": "analysis_incomplete"}, {"rule_id": "blast_deletes_outside_repo"}]}) == ["blast_deletes_outside_repo"]
    assert tirith_blast_rules(None) == []
    assert tirith_block_rules({"action": "block", "findings": [{"rule_id": "analysis_incomplete"}]}) == [
        "analysis_incomplete"]
    assert tirith_block_rules({"action": "block", "findings": []}) == ["block"]
    assert tirith_block_rules({"action": "warn", "findings": [{"rule_id": "x"}]}) == []


_INCOMPLETE = {"action": "block", "summary": "", "findings": [{"rule_id": "analysis_incomplete"}]}


@pytest.mark.parametrize(("description", "tirith", "refused"), [
    # ANY Tirith block + 'delete in root path' is refused (the minimum).
    ("delete in root path", _INCOMPLETE, True),
    ("delete in root path", {"action": "block", "findings": [{"rule_id": "blast_writes_system_path"}]}, True),
    ("delete in root path", {"action": "warn", "findings": [{"rule_id": "blast_writes_system_path"}]}, False),
    ("delete in root path", _ALLOW_TIRITH, False),
    # The other destructive classes still need a blast-radius block.
    ("recursive delete", _INCOMPLETE, False),
    ("recursive delete", {"action": "block", "findings": [{"rule_id": "blast_deletes_outside_repo"}]}, True),
    ("find -delete", {"action": "block", "findings": [{"rule_id": "blast_find_delete"}]}, True),
    # Not destructive: a Tirith block stays advisory.
    ("pipe remote content to shell", {"action": "block", "findings": [{"rule_id": "blast_x"}]}, False),
])
def test_tirith_block_plus_destructive_classifier(description, tirith, refused):
    reason = unattended_approve_refusal("true", dangerous_description=description, tirith_result=tirith)
    assert (reason is not None) is refused


# ---- regressions, set 1: combined guard, approve mode, guard-only ------------------

# P1 #1: a later reassignment must not erase an earlier destructive use or copy.
REASSIGNED_COMMANDS = [
    'X="rm -rf /home"; $X; X=echo',
    'X="rm -rf /home" && $X && X=echo',
    'X="rm -rf /home"; Y=$X; X=echo; $Y',
    'X="rm -rf /home"; $X\nX=ls; $X',
    'for c in echo "rm -rf /home"; do $c; done',
]


@pytest.mark.parametrize("command", REASSIGNED_COMMANDS)
def test_later_reassignment_does_not_hide_earlier_destructive_use(command):
    assert detect_hardline_command(command) == (True, "recursive delete of system directory")


@pytest.mark.parametrize("command", REASSIGNED_COMMANDS)
def test_later_reassignment_blocked_by_combined_guard_in_approve_mode(single_query, command):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_INCOMPLETE):
        result = check_all_command_guards(command, "local")
    assert result["approved"] is False
    assert "hardline" in result["message"].lower()


def test_resolution_is_per_use():
    assert resolve_shell_assignment_variants("X=a; echo $X; X=b; echo $X") == ["X=a; echo a; X=b; echo b"]
    assert resolve_shell_assignment_variants("X=a; Y=$X; X=b; echo $Y") == ["X=a; Y=a; X=b; echo a"]


# P1 #2: an unresolved whole-command variable is not auto-approved, with or without arguments.
@pytest.mark.parametrize("command", ["$X", "${X}", '"$X"', "cd /tmp && $X", "$CMD --help"])
def test_unresolved_variable_command_refused_in_approve_mode(single_query, command):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        result = check_all_command_guards(command, "local")
    assert result["approved"] is False
    assert "never fixes" in result["description"]
    assert "single_query_mode" in result["message"]


@pytest.mark.parametrize("command", [
    "X=echo; $X hello", 'X="rm -rf /home"; echo "$X"', "echo $X", "printf '%s' \"$X\"",
])
def test_resolved_or_printed_variables_still_approve(single_query, command):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        assert check_all_command_guards(command, "local")["approved"] is True


# P1 #3: any Tirith block + 'delete in root path' is refused, not only a blast_* block.
def test_non_blast_tirith_block_plus_delete_in_root_path_refused(single_query):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_INCOMPLETE):
        result = check_all_command_guards("rm -f /tmp/review-data", "local")
    assert result["approved"] is False
    assert "analysis_incomplete" in result["description"]
    assert "delete in root path" in result["description"]


def test_delete_in_root_path_without_tirith_block_still_approves(single_query):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        assert check_all_command_guards("rm -f /tmp/review-data", "local")["approved"] is True


def test_dangerous_detector_sees_the_resolved_command():
    assert detect_dangerous_command("X=rm; $X -rf ~/.local")[0] is True


# ---- regressions, set 2: combined guard, approve mode, guard-only ------------------

# P1 #1: a binding that may not run, or not in this shell, must not replace an earlier destructive
# value. Each of these must stay HARDLINE, not only blocked by the Tirith layer.
UNPROVEN_OVERWRITE_COMMANDS = [
    'X="rm -rf /home"; false && X=echo; $X',
    'X="rm -rf /home"; true || X=echo; $X',
    'X="rm -rf /home"; (X=echo); $X',
    'X="rm -rf /home"; X=echo | cat; $X',
    'X="rm -rf /home"; X=echo & $X',
    'X="rm -rf /home"; if test -e f; then X=echo; fi; $X',
    'X="rm -rf /home"; f() { X=echo; }; $X',
    'X="rm -rf /home"; X=echo true; $X',
    'X="rm -rf /home"; echo "$(X=echo)"; $X',
]


@pytest.mark.parametrize("command", UNPROVEN_OVERWRITE_COMMANDS)
def test_unproven_reassignment_keeps_destructive_value_hardline(command):
    assert detect_hardline_command(command) == (True, "recursive delete of system directory")


@pytest.mark.parametrize("command", UNPROVEN_OVERWRITE_COMMANDS)
def test_unproven_reassignment_hardline_under_combined_guard(single_query, command):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        result = check_all_command_guards(command, "local")
    assert result["approved"] is False
    assert "hardline" in result["message"].lower()


# A temporary, clobbered or conditional binding does not prove what a later $X runs.
@pytest.mark.parametrize("command", [
    "X=echo true; $X",                 # prefix assignment: only true's environment
    "X=echo; read X; $X",              # read replaces X with unknown input
    "X=echo; unset X; $X",
    "X=echo; mapfile X < f; $X",
    "X=echo; printf -v X '%s' \"$Y\"; $X",
    "X=echo; eval \"$Y\"; $X",           # eval can set any name
    "X=echo; . ./env.sh; $X",
    "false && X=echo; $X",
    "(X=echo); $X",
    "for X; do $X; done",               # positional parameters
])
def test_unproven_binding_leader_refused_in_approve_mode(single_query, command):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        result = check_all_command_guards(command, "local")
    assert result["approved"] is False
    assert "never fixes" in result["description"]


@pytest.mark.parametrize("command", [
    "X=echo; $X hello",
    "X=echo && $X hello",
    "X=echo; if true; then $X hi; fi",
    "X=echo; { $X hi; }",
    "X=echo; ($X hi)",
    "X=echo; $X a | $X b",
    "X=echo; for i in 1 2; do $X $i; done",
    "export X=echo; $X hi",
    "X=echo; case a in a) $X hi;; esac",
])
def test_dominating_bindings_still_auto_approve(single_query, command):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        assert check_all_command_guards(command, "local")["approved"] is True


# P1 #2: executable shell payloads are checked for unresolved leaders too.
@pytest.mark.parametrize("command", [
    "bash -c '$X'",
    "sh -c '${X}'",
    "bash -lc 'cd /tmp && $X'",
    "bash <<'EOF'\n$X\nEOF",
    "bash <<EOF\n$X\nEOF",
    "cat <<'EOF' | bash\n$X\nEOF",
    "ssh host <<'EOF'\n$X\nEOF",
    "bash -c 'bash -c \"$X\"'",
])
def test_unresolved_leader_inside_shell_payload_refused(single_query, command):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_INCOMPLETE):
        result = check_all_command_guards(command, "local")
    assert result["approved"] is False
    assert "never fixes" in result["description"]


@pytest.mark.parametrize("command", [
    "bash -c 'X=echo; $X hi'",
    "bash -c 'echo $X'",
    "bash <<'EOF'\nX=ls\n$X -la\nEOF",
    "cat > /tmp/x.php <<'PHP'\n<?php\n$config = 1;\nPHP\nphp /tmp/x.php",
    "python3 - <<'PY'\n$X\nPY",           # a non-shell consumer: the body is not shell
    "cat <<EOF > notes.txt\n$X\nEOF",
])
def test_shell_payload_benign_and_non_shell_heredocs_still_approve(single_query, command):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        assert check_all_command_guards(command, "local")["approved"] is True


# P2: bounds never drop a value silently.
def test_loop_values_past_the_variant_cap_are_all_examined():
    words = " ".join(chr(ord("a") + k) for k in range(8))
    command = f'for c in {words} "rm -rf /home"; do $c; done'
    assert detect_hardline_command(command) == (True, "recursive delete of system directory")
    many = " ".join(f"w{k}" for k in range(40))
    assert detect_hardline_command(f'for c in {many} "rm -rf /home"; do $c; done') == (
        True, "recursive delete of system directory")


def test_loop_past_the_word_limit_is_unknown_not_resolved(single_query):
    many = " ".join(f"w{k}" for k in range(70))
    command = f"for c in {many}; do $c; done"
    assert [w for *_, w in opaque_command_leaders(command)] == ["$c"]
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        assert check_all_command_guards(command, "local")["approved"] is False


def test_loop_within_the_limit_stays_resolved(single_query):
    command = "for c in " + " ".join(f"tool{k}" for k in range(20)) + "; do $c --version; done"
    assert opaque_command_leaders(command) == []


# ---- regressions, set 3: combined guard, approve mode, guard-only ------------------

# P1 #1: a value computed by a command substitution is not in the text, so it proves nothing.
@pytest.mark.parametrize("command", [
    "X=$(cat command.txt); $X",
    "X=`cat command.txt`; $X",
    'X="$(cat command.txt)"; $X --flag',
    "X=$(cat a) Y=ls; $X",
    "X=echo; X=$(cat f); $X",
    'X="$(cat f) -rf"; $X /tmp/x',
])
def test_command_output_assignment_leader_refused_in_approve_mode(single_query, command):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        result = check_all_command_guards(command, "local")
    assert result["approved"] is False
    assert "never fixes" in result["description"]


def test_command_output_value_still_feeds_detection():
    # The literal part of a substitution value is still examined (tiny literal substitutions fold).
    assert detect_hardline_command('X="$(printf rm) -rf /home"; $X') == (
        True, "recursive delete of system directory")


# P1 #2: builtin/command-wrapped readers clobber their names too.
@pytest.mark.parametrize("command", [
    "X=echo; builtin read X; $X",
    "X=echo; command read X; $X",
    "X=echo; command -p read -r X; $X",
    "X=echo; builtin unset X; $X",
    "X=echo; builtin eval \"$Y\"; $X",
    "X=echo; command . ./env.sh; $X",
    "X=echo; builtin printf -v X '%s' \"$Y\"; $X",
])
def test_wrapped_clobbering_builtins_invalidate_in_approve_mode(single_query, command):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        result = check_all_command_guards(command, "local")
    assert result["approved"] is False
    assert "never fixes" in result["description"]


# P1 #3: independent choices combine as a cross product, not zipped.
CROSS_PRODUCT_COMMANDS = [
    "for X in rm echo; do for Y in ./build /home; do $X -rf $Y; done; done",
    "for Y in ./build /home; do for X in echo rm; do $X -rf $Y; done; done",
    "for X in echo rm; do for F in -v -rf; do for Y in ./a /home; do $X $F $Y; done; done; done",
]


@pytest.mark.parametrize("command", CROSS_PRODUCT_COMMANDS)
def test_cross_product_of_loop_values_is_hardline(command):
    assert detect_hardline_command(command) == (True, "recursive delete of system directory")


@pytest.mark.parametrize("command", CROSS_PRODUCT_COMMANDS)
def test_cross_product_blocked_by_combined_guard_in_approve_mode(single_query, command):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_INCOMPLETE):
        result = check_all_command_guards(command, "local")
    assert result["approved"] is False
    assert "hardline" in result["message"].lower()


def test_cross_product_past_the_whole_command_cap_is_checked_per_simple_command():
    # 3 x 4 = 12 combinations > the 8 whole-command variants: the simple command still gets all 12.
    xs = "echo ls rm"
    ys = "./a ./b ./c /home"
    command = f"for X in {xs}; do for Y in {ys}; do $X -rf $Y; done; done"
    assert detect_hardline_command(command) == (True, "recursive delete of system directory")
    assert uninspectable_reasons(command) == []


def test_cross_product_past_every_bound_is_incomplete_and_refused(single_query):
    # 9 x 9 = 81 combinations in one simple command is past the per-command bound: incomplete.
    xs = " ".join(f"t{k}" for k in range(9))
    ys = " ".join(f"./d{k}" for k in range(9))
    command = f"for X in {xs}; do for Y in {ys}; do $X $Y; done; done"
    assert uninspectable_reasons(command)
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        result = check_all_command_guards(command, "local")
    assert result["approved"] is False
    assert "could not be fully inspected" in result["description"]


def test_small_cross_product_resolves_every_combination():
    variants = resolve_shell_assignment_variants("for X in a b; do for Y in 1 2; do $X$Y; done; done")
    assert {v.rsplit("do ", 1)[1].split(";")[0] for v in variants} == {"a1", "a2", "b1", "b2"}


# P2: the payload-depth bound fails closed.
def _nested_bash(depth, inner="$X"):
    import shlex
    for _ in range(depth):
        inner = "bash -c " + shlex.quote(inner)
    return inner


def test_payload_nesting_past_the_depth_bound_is_refused(single_query):
    command = _nested_bash(4)
    assert uninspectable_reasons(command)
    for tirith in (_ALLOW_TIRITH, _INCOMPLETE):
        with single_query("approve"), patch("tools.approval._tirith_scan", return_value=tirith):
            result = check_all_command_guards(command, "local")
        assert result["approved"] is False
        assert "could not be fully inspected" in result["description"]


def test_payload_nesting_past_the_depth_bound_refused_even_if_benign(single_query):
    command = _nested_bash(5, "echo hi")
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        assert check_all_command_guards(command, "local")["approved"] is False


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_payload_nesting_within_the_bound_is_inspected(single_query, depth):
    assert uninspectable_reasons(_nested_bash(depth, "echo hi")) == []
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        assert check_all_command_guards(_nested_bash(depth, "echo hi"), "local")["approved"] is True
        assert check_all_command_guards(_nested_bash(depth), "local")["approved"] is False


# eval / source run text the command does not show: an eval'd unresolved leader is refused.
@pytest.mark.parametrize("command", ['eval "$Y"', "X=$(cat f); eval \"$X\"", "eval '$X'"])
def test_eval_of_unfixed_text_refused_in_approve_mode(single_query, command):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        assert check_all_command_guards(command, "local")["approved"] is False


def test_eval_of_same_command_value_is_hardline():
    assert detect_hardline_command('X="rm -rf /home"; eval "$X"') == (True, "recursive delete of system directory")


@pytest.mark.parametrize("command", [
    "X=echo; $X hello",
    "X=/usr/bin/echo; $X hi",
    "X=echo; command -v $X",
    "eval 'echo hi'",
    "X=echo; builtin cd /tmp; $X hi",
    "X=$HOME/bin/tool; $X --version",        # like writing $HOME/bin/tool directly
])
def test_round3_benign_controls_still_approve(single_query, command):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        assert check_all_command_guards(command, "local")["approved"] is True


# A program NAME that embeds an unfixed expansion is as unreadable as a bare `$X`.
@pytest.mark.parametrize("command", [
    "/bin/$X -rf ~/.local", "./$X", "${X%/*} -rf ~", "bin/py$V -m x", "sudo /bin/$X /home",
    "X=$(cat f); /bin/$X",
])
def test_partial_variable_program_name_refused_in_approve_mode(single_query, command):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        result = check_all_command_guards(command, "local")
    assert result["approved"] is False
    assert "never fixes" in result["description"]


@pytest.mark.parametrize("command", [
    "$HOME/bin/tool --version",               # literal program name under an expanded directory
    '"$VIRTUAL_ENV/bin/python" -m pytest',
    "X=ls; /bin/$X -la",                      # resolved in the same command
    '"$(dirname "$0")/run.sh"',
])
def test_literal_or_resolved_program_name_still_approves(single_query, command):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        assert check_all_command_guards(command, "local")["approved"] is True


# A large product inside print-only output (not piped, not redirected) cannot run anything, so it
# is not "incomplete"; the same product piped to a shell or redirected still is.
_NINE = " ".join(f"v{k}" for k in range(9))
_NINE_B = " ".join(f"w{k}" for k in range(9))


@pytest.mark.parametrize("body", ['echo "$A $B"', "printf '%s %s\\n' \"$A\" \"$B\"", 'C="$A$B"'])
def test_print_only_cross_product_is_not_incomplete(body):
    command = f"for A in {_NINE}; do for B in {_NINE_B}; do {body}; done; done"
    assert uninspectable_reasons(command) == []


@pytest.mark.parametrize("body", ['echo "$A $B" | bash', 'echo "$A" > "$B"', "printf -v X '%s' \"$A$B\""])
def test_executable_or_redirected_cross_product_stays_incomplete(body):
    command = f"for A in {_NINE}; do for B in {_NINE_B}; do {body}; done; done"
    assert uninspectable_reasons(command)


def test_assignment_product_is_checked_where_it_is_used(single_query):
    command = f'for A in {_NINE}; do for B in {_NINE_B}; do C="$A $B"; $C; done; done'
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        assert check_all_command_guards(command, "local")["approved"] is False


def test_eval_text_inside_a_data_heredoc_is_not_executed():
    command = "cat >> notes.py <<'EOF'\nassert f('X=\"rm -rf /home\"; eval \"$X\"')\nEOF"
    assert detect_hardline_command(command) == (False, None)
    assert uninspectable_reasons(command) == []


def test_eval_inside_a_shell_heredoc_is_still_seen():
    command = "bash <<'EOF'\nX=\"rm -rf /home\"; eval \"$X\"\nEOF"
    assert detect_hardline_command(command)[0] is True


@pytest.mark.parametrize("command", ['sudo -n env "PATH=$PATH" /usr/bin/py-spy dump --pid 1',
                                     'env "LD_PATH=$HOME/lib" ls'])
def test_quoted_env_assignment_operand_is_not_a_program(command):
    assert opaque_command_leaders(command) == []


# ---- regressions, set 4: every spelling of a name write invalidates the proof -------
# All guard-only (check_all_command_guards / classifiers); nothing reaches an executor.

READER_CLOBBER_COMMANDS = [
    # the four review spellings
    "X=echo; IFS= read -r X; $X",           # prefix assignment before the reader
    "REPLY=echo; read; $REPLY",             # implicit REPLY target
    "X=echo; R=read; $R X; $X",             # reader reached through a same-command variable
    "X=echo; printf -vX %s \"$Y\"; $X",     # attached -v target
    # the rest of the class
    "X=echo; read -raX; $X",
    "X=echo; LC_ALL=C builtin read X; $X",
    "X=echo; time read X; $X",
    "X=echo; ! read X; $X",
    "X=echo; wait -p X; $X",
    "X=echo; mapfile; $MAPFILE",
    "X=echo; getopts ab X; $X",
    "X=echo; let X=1; $X",
    "X=echo; N=X; read \"$N\"; $X",         # target name not in the text
    "X=echo; N=X; printf -v \"$N\" %s y; $X",
    "X=echo; N=X; declare \"$N=v\"; $X",
    "X=echo; declare -n R=X; $X",           # nameref: X can change through R
    "declare -n R=X; X=ls; $R",
    "X=echo; trap 'X=v' DEBUG; $X",
    "trap 'X=v' DEBUG; X=echo; $X",         # a trap handler outlives later assignments
    "eval \"$Y\"; X=echo; $X",
    "X=echo; E=eval; $E \"$Y\"; $X",
    "X=echo; B=builtin; $B read X; $X",
    "X=echo; L=let; $L X=1; $X",
    "X=echo; S=source; $S ./f; $X",
    "X=echo; D=declare; $D X=v; $X",
    "X=echo; select X in a; do break; done; $X",
    "_=echo; true rm; $_ -rf /tmp/x",       # $_ is rewritten by every command
]


@pytest.mark.parametrize("scan", [_ALLOW_TIRITH, _INCOMPLETE], ids=["tirith-allow", "tirith-block"])
@pytest.mark.parametrize("command", READER_CLOBBER_COMMANDS)
def test_name_writes_invalidate_bindings_in_approve_mode(single_query, command, scan):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=scan):
        result = check_all_command_guards(command, "local")
    assert result["approved"] is False
    assert "never fixes" in result["description"]


@pytest.mark.parametrize("command", [
    'X=echo; X[0]="rm -rf /home"; $X',       # element 0 is what $X expands to
    'X=echo; : ${X:="rm -rf /home"}; $X',    # default-assign expansion
    'X=echo; D=declare; $D X="rm -rf /home"; $X',
])
def test_hidden_destructive_writes_reach_detection(single_query, command):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        assert check_all_command_guards(command, "local")["approved"] is False


@pytest.mark.parametrize("command", [
    'X=echo; X[0]="rm -rf /home"; $X',
    'X=echo; : ${X:="rm -rf /home"}; $X',
])
def test_element_and_default_assignment_values_are_hardline(command):
    assert detect_hardline_command(command) == (True, "recursive delete of system directory")


@pytest.mark.parametrize("command", [
    "X=echo; $X hello",
    "X=echo; read -r Y; $X \"$Y\"",           # read of ANOTHER name
    "X=echo; IFS= read -r Y; $X \"$Y\"",
    "X=echo; printf -v Y %s z; $X \"$Y\"",
    "X=echo; command -v read; $X hi",         # lookup only
    "X=echo; /usr/bin/read X; $X hi",         # a path is never the builtin
    "X=echo; $HOME/bin/read X; $X hi",
    "X=echo; \"$VIRTUAL_ENV/bin/python\" -V; $X hi",
    "X=echo; export X; $X hi",
    "X=echo; declare -r X; $X hi",
    "X=echo; wait -n; $X hi",
    "source ./venv/bin/activate; PY=python3; $PY -V",
    "while IFS= read -r line; do echo \"$line\"; done < f",
])
def test_writes_to_other_names_still_auto_approve(single_query, command):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        assert check_all_command_guards(command, "local")["approved"] is True


@pytest.mark.parametrize("words, expected", [
    (["IFS=", "read", "-r", "X"], {"X"}),
    (["read"], {"REPLY"}),
    (["read", "-r", "-p", "prompt"], {"REPLY"}),
    (["read", "-raX"], {"X"}),
    (["read", "-a", "ARR", "X"], {"ARR", "X"}),
    (["printf", "-vX", "%s", "y"], {"X"}),
    (["printf", "-v", "X", "%s"], {"X"}),
    (["printf", "%s", "-v"], set()),
    (["mapfile", "-t"], {"MAPFILE"}),
    (["mapfile", "-t", "LINES"], {"LINES"}),
    (["getopts", "ab", "OPT"], {"OPT", "OPTARG", "OPTIND"}),
    (["wait", "-p", "PID"], {"PID"}),
    (["let", "X+=1", "Y++"], {"X", "Y"}),
    (["command", "-v", "read"], set()),
    (["/usr/bin/read", "X"], set()),
])
def test_command_clobbers_names(words, expected):
    from tools.approval_detection_clobbers import command_clobbers
    assert command_clobbers(words) == frozenset(expected)


@pytest.mark.parametrize("words", [["read", '"$N"'], ["printf", "-v", '"$N"', "%s"], ["source", "f"],
                                   ["declare", '"$N=v"']])
def test_command_clobbers_unknown_target_is_every_name(words):
    from tools.approval_detection_clobbers import ALL, command_clobbers
    assert command_clobbers(words) is ALL


@pytest.mark.parametrize("words", [["trap", "X=v", "DEBUG"], ["eval", '"$Y"'], ["declare", "-n", "R=X"]])
def test_command_clobbers_late_writers(words):
    from tools.approval_detection_clobbers import LATE, command_clobbers
    assert command_clobbers(words) == LATE


def test_command_clobbers_dynamic_leader_is_deferred():
    from tools.approval_detection_clobbers import DYNAMIC, command_clobbers
    assert command_clobbers(["$R", "X"]) == (DYNAMIC, 0)


def test_many_dynamic_command_words_resolve_in_linear_passes():
    # Each `$P ...` word's values depend on what every earlier dynamic word writes. Resolving that
    # by recursion was exponential in the number of such lines (a replayed 20-line script hung);
    # the fixpoint is a handful of linear passes.
    import time
    command = "P=/usr/bin/python3\n" + "\n".join(f"$P tool{k}.py arg" for k in range(40))
    started = time.monotonic()
    assert opaque_command_leaders(command) == []
    assert uninspectable_reasons(command) == []
    assert time.monotonic() - started < 10


def test_reader_reached_through_a_loop_variable_is_still_seen():
    # `$R R` overwrites R itself, so on the second iteration the command word is unknown.
    command = "R=read; for i in 1 2; do $R R; done; $R X"
    assert [w for *_, w in opaque_command_leaders(command)][:1] == ["$R"]


@pytest.mark.parametrize("command", [
    'S=/opt/s; R="python3 $S/resolve.py"; $R a; $R b',     # expansion in the ARGUMENTS only
    'home=/tmp/h; base=(env HOME="$home" python3 -m tool); "${base[@]}" one; "${base[@]}" two',
])
def test_dynamic_word_with_expanded_arguments_is_not_a_clobber(single_query, command):
    # Replay false positive: a `$` in a resolved command word's arguments made it "unreadable",
    # so it was taken to overwrite every name and the NEXT use of the same word was refused.
    assert opaque_command_leaders(command) == []
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        assert check_all_command_guards(command, "local")["approved"] is True


def test_dynamic_word_whose_program_is_itself_unknown_still_clobbers(single_query):
    command = 'X=echo; R="$B read"; $R X; $X'
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        assert check_all_command_guards(command, "local")["approved"] is False


# ---- regressions, set 5: redirections, mapfile callbacks, deferred function writes ----
# All guard-only (check_all_command_guards / classifiers); nothing reaches an executor.

ROUND5_REFUSE_COMMANDS = [
    # the three review spellings
    "X=echo; < input.txt read X; $X",                            # leading redirection
    "X=echo; mapfile -C 'read X; :' -c 1 A < input.txt; $X",     # callback runs shell code
    "X=echo; f() { read X; }; X=echo; f; $X",                    # write runs at the call
    # redirections anywhere in the reader
    "X=echo; read X < f; $X",
    "X=echo; 2>/dev/null read -r X; $X",
    "X=echo; <<<\"$s\" read X; $X",
    "X=echo; < a IFS= < b read < c X; $X",
    "X=echo; 0<f builtin read X; $X",
    "X=echo; exec {X}<f; $X",                                    # {NAME}< stores an fd number
    # callbacks and deferred writers
    "X=echo; mapfile -C'read X' A; $X",
    "mapfile -C 'X=v' A; X=echo; $X",
    "X=echo; function f { read X; }; X=echo; f; $X",
    "X=echo; f() ( read X ); X=echo; f; $X",
    # builtins with no handler count as writing every name
    "X=echo; enable -f ./x.so foo; $X",
    "X=echo; fc -s; $X",
    "X=echo; bind -x \"\\C-a: read X\"; $X",
    "X=echo; coproc X { cat; }; $X",
    "X=echo; B=bind; $B -x z; $X",
    # an alias whose text writes a name rewrites later commands
    "X=echo; alias echo=read; X=echo; $X",
    "X=echo; alias ll='read X'; $X",
    "X=echo; alias e=eval; $X",
]


@pytest.mark.parametrize("scan", [_ALLOW_TIRITH, _INCOMPLETE], ids=["tirith-allow", "tirith-block"])
@pytest.mark.parametrize("command", ROUND5_REFUSE_COMMANDS)
def test_redirected_callback_and_deferred_writes_refused(single_query, command, scan):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=scan):
        assert check_all_command_guards(command, "local")["approved"] is False


@pytest.mark.parametrize("command", [
    'f() { X="rm -rf /home"; }; X=echo; f; $X',     # the function's value survives the later X=echo
    '2>/dev/null X="rm -rf /home"; $X',            # a redirection before the assignment
    'X="rm -rf /home" 2>/dev/null; $X',
    'X=echo; < f X="rm -rf /home"; $X',
])
def test_deferred_and_redirected_destructive_values_are_hardline(command):
    assert detect_hardline_command(command) == (True, "recursive delete of system directory")


@pytest.mark.parametrize("command", [
    "X=echo; $X hello",
    "X=echo; read -r Y < f; $X \"$Y\"",
    "X=echo; < f read -r Y; $X \"$Y\"",
    "X=echo; mapfile -t A < f; $X \"${A[@]}\"",
    "X=echo; cd /tmp; $X hi",
    "X=echo; f() { echo hi; }; f; $X hi",
    "X=echo; 2>/dev/null ls; $X hi",
    "PY=python3; $PY -V 2>&1",
    "c=(ls -la); \"${c[@]}\" /tmp",
    "X=echo; alias ll=\"ls -l\"; $X hi",
    "X=echo; alias; $X hi",
])
def test_round5_benign_controls_still_approve(single_query, command):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        assert check_all_command_guards(command, "local")["approved"] is True


@pytest.mark.parametrize("words, expected", [
    (["mapfile", "-t", "A"], {"A"}),
    (["mapfile", "-C", "cb", "-c", "1", "A"], "late"),
    (["mapfile", "-Ccb", "A"], "late"),
    (["alias", "ll=ls -l"], set()),
    (["alias", "echo=read"], "late"),
    (["enable", "-f", "x.so", "foo"], None),
    (["fc", "-s"], None),
    (["cd", "/tmp"], set()),
    (["ls", "-l"], set()),
])
def test_command_clobbers_callbacks_and_unmodeled_builtins(words, expected):
    from tools.approval_detection_clobbers import command_clobbers
    result = command_clobbers(words)
    assert result == (frozenset(expected) if isinstance(expected, set) else expected)


def test_every_bash_builtin_is_classified():
    # A builtin on neither list would silently count as writing nothing; the lists must cover
    # every bash builtin so a new handler-less one fails closed instead.
    from tools.approval_detection_clobbers import _BASH_BUILTINS, _HANDLERS, _NON_WRITING_BUILTINS
    assert set(_HANDLERS) <= _BASH_BUILTINS
    assert not (_NON_WRITING_BUILTINS & set(_HANDLERS))



# ---- regressions, set 6: compound aliases, resolved heredoc consumers, glob leaders ----
# All guard-only (check_all_command_guards / classifiers); nothing reaches an executor.

ROUND6_REFUSE_COMMANDS = [
    # alias replacement text is shell syntax, not one word list
    "shopt -s expand_aliases\nalias r=':; read X'\nX=echo\nr\n$X",
    "shopt -s expand_aliases\nalias r=':|read X'\nX=echo\nr\n$X",
    "shopt -s expand_aliases\nalias r='true && read X'\nX=echo\nr\n$X",
    "shopt -s expand_aliases\nalias r='X=rm'\nX=echo\nr\n$X -rf /tmp/a",
    "shopt -s expand_aliases\nalias r='{ read X; }'\nX=echo\nr\n$X",
    # a heredoc consumer spelled as a variable is judged by what the variable holds
    "S=bash; $S <<'EOF'\n$X\nEOF",
    "S=/bin/bash; \"$S\" <<'EOF'\n$X\nEOF",
    "S=\"sudo bash\"; $S <<'EOF'\n$X\nEOF",
    "S=\"env sh\"; $S <<'EOF'\n$X\nEOF",
    "$S <<'EOF'\n$X\nEOF",
    "S=$(which bash); $S <<'EOF'\n$X\nEOF",
    # an unquoted glob-bearing value in command position runs whatever file matches
    'X="r?"; $X -rf /home',
    "X='r*'; $X -rf /home",
    'X="[r]m"; $X -rf /home',
    'X="/bin/r?"; $X -rf /home',
    'X="+(rm)"; shopt -s extglob; $X -rf /home',
    "D=/usr; $D/bin/r? -rf /home",
]


@pytest.mark.parametrize("scan", [_ALLOW_TIRITH, _INCOMPLETE], ids=["tirith-allow", "tirith-block"])
@pytest.mark.parametrize("command", ROUND6_REFUSE_COMMANDS)
def test_alias_heredoc_consumer_and_glob_leaders_refused(single_query, command, scan):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=scan):
        assert check_all_command_guards(command, "local")["approved"] is False


@pytest.mark.parametrize("command, executed", [
    ("S=bash; $S <<'EOF'\n$X\nEOF", True),
    ("S=zsh; $S -s <<'EOF'\n$X\nEOF", True),
    ("$UNSET <<'EOF'\n$X\nEOF", True),
    ("PY=python3; $PY - <<'EOF'\n$X\nEOF", False),
    ("C=cat; $C > f <<'EOF'\n$X\nEOF", False),
])
def test_variable_heredoc_consumer_payload_inspected_unless_proven_data(command, executed):
    # The payload's `$X` leader shows up as opaque only when its body is treated as shell code.
    payload_leaders = [f for f in opaque_command_leaders(command) if f[0] != command]
    assert bool(payload_leaders) is executed


@pytest.mark.parametrize("command", [
    "X=echo; $X hello",
    "alias ll='ls -l'; X=echo; $X hi",
    "alias ll='ls -l; pwd'; X=echo; $X hi",
    "PY=python3; $PY - <<'EOF'\nimport os; x=$X\nEOF",
    "C=cat; $C > f <<'EOF'\n$X\nEOF",
    'X="r?"; "$X" -rf ./build',          # quoted: no filename expansion, the literal program "r?"
    'X="ls -l *.py"; $X',                # the glob is an argument, not the program
    "X=ls; $X *.py",
    "ls /usr/bin/py*",
])
def test_round6_benign_controls_still_approve(single_query, command):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        assert check_all_command_guards(command, "local")["approved"] is True


@pytest.mark.parametrize("words, expected", [
    (["alias", "r=:; read X"], "late"),
    (["alias", "r=X=v"], "late"),
    (["alias", "r=( read X )"], "late"),
    (["alias", "ll=ls -l; pwd"], set()),
    (["alias", "g=git status | cat"], set()),
])
def test_command_clobbers_compound_alias_values(words, expected):
    from tools.approval_detection_clobbers import command_clobbers
    result = command_clobbers(words)
    assert result == (frozenset(expected) if isinstance(expected, set) else expected)



# ---- regressions, set 7: pipe-to-shell heredocs, per-span quoting, IFS, aliases with operands ------

ROUND7_REFUSE_COMMANDS = [
    # a resolved non-shell consumer does not make a body data when its output is piped to a shell
    "C=cat; $C <<'EOF' | bash\n$X\nEOF",
    "C=cat; $C <<'EOF' | sh\n$X\nEOF",
    "C=cat; $C <<'EOF' | sudo -u root bash\n$X\nEOF",
    "C=cat; $C <<'EOF' | env -i sh -s\n$X\nEOF",
    "C=cat; $C <<'EOF' | tr a-z a-z | sh\n$X\nEOF",
    "C=cat; $C <<'EOF' | $S\n$X\nEOF",
    # quoting is per span: the unquoted expansion or glob part is still filename-expanded
    'X="r?"; ""$X -rf /home',
    'X="r?"; \'\'$X -rf /home',
    'D=/usr; "$D"/bin/r? -rf /home',
    'D=/usr; "$D"/bin/[r]m -rf /home',
    "X='*'; ${X} -rf /home",
    # an unquoted value is split on IFS: a same-command IFS the value is split on, or an unknown one
    "IFS=$(cat f); X=echo; $X hi",
    # an alias's replacement is composed with the words that follow it at the invocation
    "shopt -s expand_aliases\nalias a='builtin '\nX=echo\na read X\n$X",
    "shopt -s expand_aliases\nalias a='command '\nX=echo\na read X\n$X",
    "shopt -s expand_aliases\nalias p=printf\nX=echo\np -v X %s \"$Y\"\n$X",
    "shopt -s expand_aliases\nalias r=read\nX=echo\nr X\n$X",
    "shopt -s expand_aliases\nalias a='true;'\nX=echo\na read X\n$X",
]


@pytest.mark.parametrize("scan", [_ALLOW_TIRITH, _INCOMPLETE], ids=["tirith-allow", "tirith-block"])
@pytest.mark.parametrize("command", ROUND7_REFUSE_COMMANDS)
def test_piped_heredoc_mixed_quoting_and_alias_operands_refused(single_query, command, scan):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=scan):
        assert check_all_command_guards(command, "local")["approved"] is False


@pytest.mark.parametrize("command", [
    'X="rm+-rf+/home"; IFS=+; $X',
    'IFS=+; X="rm+-rf+/home"; $X',
    "IFS=,; X=rm,-rf,/home; $X",
])
def test_ifs_split_destructive_value_is_hardline(command):
    assert detect_hardline_command(command)[0] is True


@pytest.mark.parametrize("command, executed", [
    ("C=cat; $C <<'EOF' | bash\n$X\nEOF", True),
    ("C=cat; $C <<'EOF' | sudo bash\n$X\nEOF", True),
    ("C=cat; $C <<'EOF' | tr a-z A-Z\n$X\nEOF", False),
    ("C=cat; $C <<'EOF' || true\n$X\nEOF", False),
])
def test_piped_heredoc_payload_inspected_only_when_a_shell_reads_it(command, executed):
    payload_leaders = [f for f in opaque_command_leaders(command) if f[0] != command]
    assert bool(payload_leaders) is executed


@pytest.mark.parametrize("command", [
    "X=echo; $X hello",
    "C=cat; $C > notes.txt <<'EOF'\n$X\nEOF",
    "C=cat; $C <<'EOF' | tr a-z A-Z\nhello\nEOF",
    'X="r?"; "$X" --version',             # fully quoted: the literal program "r?"
    "X='r*'; \"$X\" -rf ./build",
    '"$HOME"/bin/tool --help',
    'D=/usr; "$D"/bin/ls -la',
    'X="a b"; "$X"/bin/ls',
    "X=echo; IFS=o; $X hello",             # IFS holds no separator X's value is split into a program by
    "IFS=:; D=/usr; $D/bin/ls",
    "shopt -s expand_aliases\nalias ll='ls -l'\nX=echo\nll /tmp\n$X hi",
    "shopt -s expand_aliases\nalias e=echo\nX=echo\ne hi\n$X hi",
])
def test_round7_benign_controls_still_approve(single_query, command):
    with single_query("approve"), patch("tools.approval._tirith_scan", return_value=_ALLOW_TIRITH):
        assert check_all_command_guards(command, "local")["approved"] is True


@pytest.mark.parametrize("words, expected", [
    (["alias", "a=builtin "], "late"),
    (["alias", "a=command "], "late"),
    (["alias", "p=printf"], "late"),
    (["alias", "r=read"], "late"),
    (["alias", "a=true;"], "late"),
    (["alias", "a="], "late"),
    (["alias", "e=echo"], set()),
    (["alias", "ll=ls -l"], set()),
    (["alias", "g=git status | cat"], set()),
])
def test_command_clobbers_alias_composed_with_invocation_operands(words, expected):
    from tools.approval_detection_clobbers import command_clobbers
    result = command_clobbers(words)
    assert result == (frozenset(expected) if isinstance(expected, set) else expected)
