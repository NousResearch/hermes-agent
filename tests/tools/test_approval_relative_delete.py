"""A delete needs the same approval however its path is spelled (#151542).

The rm rules used to key on the FIRST CHARACTER of the argument, so the gate depended on how a
command was written rather than on what it removes: `rm /srv/app/x/*.log.gz` prompted while
`cd /srv/app/x && rm logs/archive/*.log.gz` — the same files — ran with no prompt. These tests pin
the effect-based rules in BOTH directions, because the two failure modes are not symmetric: a false
negative deletes files without asking, and a false positive trains people to approve blindly.

The rules are matched by `_RmEffectMatcher`, which walks the existing shell lexer
(`_iter_shell_command_word_spans` + `_read_shell_word` + `_scan_shell`) rather than a regex over raw
text. Every case below is a spelling that a raw-text regex got wrong: one of the two directions was
demonstrated by adversarial review of a first attempt.
"""

import pytest

from tools import approval_detection
from tools.approval_detection import detect_dangerous_command


def _flags(command: str) -> bool:
    return detect_dangerous_command(command)[0]


def _key(command: str):
    return detect_dangerous_command(command)[1]


# ── the reported asymmetry ───────────────────────────────────────────────────────────

def test_the_two_spellings_of_one_deletion_agree():
    absolute = "rm /srv/app/checkout-api/logs/archive/*.log.gz"
    relative = "cd /srv/app/checkout-api && rm logs/archive/*.log.gz"
    assert _flags(absolute) is True
    assert _flags(relative) is True


@pytest.mark.parametrize("command", [
    "rm -f logs/archive/*",
    "rm logs/archive/*.log.gz",
    "rm -f *.log",
    "rm logs/*.gz",
    "rm ../../etc/passwd",
    "rm -rf ../../etc",
    "rm a.txt b.txt",
    "rm -rf foo bar baz",
])
def test_the_reported_rows_are_gated(command):
    assert _flags(command), f"{command!r} deletes without an approval prompt"


# ── quoting: the direction a raw-text regex gets backwards ───────────────────────────

@pytest.mark.parametrize("command", [
    'rm "a.txt" "b.txt"',        # quoted operands are still two targets
    'rm -- "a b.txt" "c.txt"',
    "rm '../secret'",            # quoting does not neutralise a traversal
    'rm "../secret"',
    'rm "$DIR"/*.log',           # the idiomatic spelling: quoted dir, unquoted glob
    'rm "$DIR"/*',
])
def test_quoting_does_not_hide_a_deletion(command):
    assert _flags(command), f"{command!r} deletes but reads as prose"


def test_a_quoted_glob_is_not_a_glob():
    """Quoting suppresses expansion, so this names one literal file, not a pattern."""
    assert not _flags('rm "*.log"')


# ── command position: where the deletion is invoked from ─────────────────────────────

@pytest.mark.parametrize("command", [
    "cd /srv/app && rm logs/*.gz",
    "$(rm a b)",
    "then rm a b",
    "F=1 rm *.log",
    "FOO=bar BAZ=1 rm a b",
    "sudo -u root rm *.log",
    "sudo -E -H -u root rm a b",
    "nice -n 10 rm *.log",
    "builtin rm a b",
    "command rm a b",
    "nohup rm a b",
    'sh -c "rm a b"',
])
def test_deletion_is_found_wherever_rm_is_the_command(command):
    assert _flags(command), f"{command!r} deletes without an approval prompt"


# ── flags, redirections, comments, literals: the false-positive direction ────────────

@pytest.mark.parametrize("command", [
    "rm a.txt -v",            # GNU rm permutes flags: one target, not two
    "rm a.txt -f",
    "rm a.txt -i",
    "rm a.txt --verbose",
    "rm a.txt 2>/dev/null",   # a redirection target is not an operand
    "rm notes.txt # keep b.txt around",
    "rm a.txt -- ",           # end-of-options with nothing after it
    "rm notes.txt",           # one plain relative file stays quiet (#18083)
    "rm -f build/output.o",
    "rm -- -weird.txt",
    "rm --help",              # rm prints usage and exits without deleting, whatever follows
    "rm --help a b",
    "rm {lock}",              # no comma or range: one literal name, no expansion
    "rm -f {tmpfile}",
    "echo rm is a command",   # the word is an argument, not a command
    "man rm",
    "rm --help",
    "rmdir emptydir",
    "chmod +x rm",
    "grep -r 'rm *.log' .",
    'echo "rm a b"',
    "git commit -m 'rm small big'",
    "rm foo && echo done",
    "rm foo; ls",
    "wmic shadowcopy list && rm x",
])
def test_quiet_cases_stay_quiet(command):
    assert not _flags(command), f"{command!r} would prompt for no deletion"


def test_read_only_wmic_before_a_two_target_rm_is_not_the_wmic_rule():
    """Moved here from test_approval_windows's lookalike list: this rm removes two paths
    (`x` and `delete`, after `--`), so it is gated — by the rm effect rule, not by the WMIC
    shadowcopy rule, which is what that list was about."""
    assert _key("wmic shadowcopy list && rm x -- delete") == approval_detection.RM_MULTI_KEY


# ── labels must describe the effect, and keys must stay distinct ─────────────────────

@pytest.mark.parametrize("command", [
    "rm {1..50}.log",      # range
    "rm {a,b}.txt",        # list
    "rm backup{,.bak}",    # prefixed list, empties
    "rm file{1,2}.txt",
    "rm pre{1..3}post.log",
    "rm ./x{a,b}y",
])
def test_brace_expansion_is_gated_and_labelled_as_multiple_targets(command):
    """A brace range/list deletes several files. The `..` inside `{1..50}` is not a path
    component, so it must not be labelled a parent-directory path."""
    assert _key(command) == approval_detection.RM_MULTI_KEY


@pytest.mark.parametrize("command", [
    'rm "{a,b}"',          # quoting suppresses expansion: one literal filename
    "rm '{1..3}'",
    'rm "{1..3}.log"',
    r"rm a\ b",            # escaping does too, and one escaped space is ONE operand
    r"rm a\ b\ c",
    r"rm a.txt\ b.txt",
    r"rm \*.log",          # a literal star, not a glob
    r"rm a\*",
])
def test_expansion_suppressed_by_quoting_or_escaping_stays_quiet(command):
    assert not _flags(command), f"{command!r} expands to nothing"


def test_each_rule_has_a_distinct_approval_key_and_alias():
    """Approvals are persisted per key: _approval_scope accepts ANY alias of a key, so a shared
    alias would let one stored approval satisfy all three rules at once."""
    keys = [approval_detection.RM_GLOB_KEY, approval_detection.RM_DOTDOT_KEY,
            approval_detection.RM_MULTI_KEY]
    alias_sets = [frozenset(approval_detection._approval_key_aliases(k)) for k in keys]
    assert len(set(alias_sets)) == 3, "the three rm effect rules share an approval alias"


def test_approving_one_rule_does_not_approve_another(monkeypatch):
    """Behavioural form of the check above: seed a grant under one rule's key and prove the
    approval gate still refuses the other two."""
    from tools import approval

    keys = [approval_detection.RM_GLOB_KEY, approval_detection.RM_DOTDOT_KEY,
            approval_detection.RM_MULTI_KEY]
    for granted, others in ((keys[0], keys[1:]), (keys[1], [keys[0], keys[2]]),
                            (keys[2], [keys[0], keys[1]])):
        aliases = set(approval_detection._approval_key_aliases(granted))
        monkeypatch.setattr(approval, "_permanent_set", lambda a=aliases: set(a))
        monkeypatch.setattr(approval, "_session_approved", {})
        assert approval._approval_scope("s", granted) == "permanent"
        for other in others:
            assert approval._approval_scope("s", other) is None, (
                f"an approval granted for {granted!r} also covers {other!r}")


@pytest.mark.parametrize("command", [
    "rm $((1+1)) *.log",       # arithmetic expansion is ONE word; it must not hide the targets
    "rm $((1+1)) secret.txt",
    "rm $((1+1)) ../secret",
    "r\\\nm a b",              # backslash-newline is a continuation: the shell runs `rm`
    "r\\\nm *.log",
    "rm {x{a}b,y}",            # nested brace list
    "rm {a,{b,c}}",
    "rm {1..3}{4,5}",
])
def test_obfuscated_or_nested_spellings_are_gated(command):
    assert _flags(command), f"{command!r} deletes without an approval prompt"


@pytest.mark.parametrize("command", [
    "rm a[b",             # an unmatched `[` is a literal character to the shell
    "rm file[1",
    "rm {,,}",            # expands to empty fields only; rm errors without deleting
    "rm {}",
    "rm {a..}",           # a sequence missing an endpoint is literal
    "rm {1..1}",          # a one-member sequence is a single word
])
def test_shapes_that_delete_nothing_stay_quiet(command):
    assert not _flags(command), f"{command!r} deletes nothing"


@pytest.mark.parametrize("command", [
    "cat <<<x\nrm *.log",          # a here-string is not a heredoc header
    "cat <<<x\nrm a b",
    'echo "see <<EOF"\nrm a b',    # `<<` inside quotes is text
    "# cat <<EOF\nrm a b",         # `<<` inside a comment is text
    "x=$((1<<2))\nrm a b",         # `<<` as an arithmetic shift
    "echo $(( $(rm a b) + 1 ))",   # a substitution nested in arithmetic still runs
])
def test_a_heredoc_lookalike_cannot_hide_a_later_command(command):
    """An earlier version blanked everything after any `<<` as a heredoc body. Each of these makes
    the real deletion disappear behind that blanking, so the body is not blanked at all."""
    assert _flags(command), f"{command!r} deletes without an approval prompt"


@pytest.mark.parametrize("command", [
    "cat <<A <<B\nrm a b\nA\nrm c d\nB",   # several heredocs, both bodies are data
    "cat <<\\EOF\nrm a b\nEOF",            # backslash-quoted delimiter
    "cat <<-EOF\nx\n  EOF\nrm a b\nEOF",   # `<<-` strips leading TABS only, not spaces
])
def test_heredoc_bodies_are_still_scanned(command):
    """Documented limitation, not a goal: this detector has no heredoc state, so a body is read as
    commands — exactly as the pre-existing rules already do for `cat <<EOF` carrying `rm -rf /`.
    The safe direction is an extra prompt; blanking the body instead swallowed real deletions."""
    assert _flags(command)


def test_writing_a_script_keeps_its_heredoc_key():
    """The body is not a command, and the command keeps the key it already had on main — a stored
    approval for 'shell execution via heredoc' must still apply."""
    assert _key("cat > clean.sh <<'EOF'\nrm a b\nEOF") == "shell execution via heredoc"


@pytest.mark.parametrize("command,expected", [
    ("python -c 'print(1)'; rm a b", "script execution via -e/-c flag"),
    ("sh -c 'echo hi'; rm a b", "shell command via -c/-lc flag"),
    ("bash -lc 'echo hi' && rm *.log", "shell command via -c/-lc flag"),
    ("python <<EOF\nprint(1)\nEOF\nrm a b", "script execution via heredoc"),
    ("rg --pre 'touch x' pat; rm a b", "arbitrary program execution via rg --pre"),
    ("(dd if=/dev/sda of=/tmp/disk.img; rm *.log)", "disk copy"),
])
def test_a_delete_does_not_steal_consent_from_what_else_the_command_does(command, expected):
    """The rm rules describe what the DELETE removes, so they must never become the first key when
    the command also does something that needs its own approval. Matching them inside the variant
    loop did exactly that: `python -c ...; rm a b` reported a delete instead of the interpreter,
    and a stored grant for the delete key then let the interpreter run unprompted."""
    assert _key(command) == expected


def test_shell_payload_keeps_its_existing_key():
    """`sh -c 'rm a b'` already prompted as a shell-flag execution on main. The rm keys are yielded
    only after the execution-flag sweep, so the stored approval for that key keeps working."""
    assert _key("sh -c 'rm a b'") == "shell command via -c/-lc flag"


def test_nested_command_substitutions_stay_within_the_budget():
    """An absolute bound, not a linearity proof: this only asserts that the three extra rules do
    not push the documented nested-substitution shape through the budget. Measured best-of-3 with
    and without the three rules, their marginal cost was +7% at k=200 (0.93s -> 0.99s) and +2% at
    k=400 (3.39s -> 3.45s) — a constant share on top of the pre-existing quadratic scanner, which
    is what the 2s perf gate in this suite was already measuring."""
    import time

    command = "rm $(" * 200 + "x" + ")" * 200
    start = time.perf_counter()
    detect_dangerous_command(command)
    assert time.perf_counter() - start < 3.5, "rm effect rules add superlinear cost"


def test_existing_rm_rules_keep_their_keys():
    """Stored approvals are keyed by these; the new rules must not steal them."""
    assert _key("rm /srv/app/x/*.log.gz") == "delete in root path"
    assert _key("cd /srv/app && rm -r checkout-api/logs/archive") == "recursive delete"
    assert _key("find logs -delete") == "find -delete"
    assert _key("ls logs/archive | xargs rm") == "xargs with rm"
