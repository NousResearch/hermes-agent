"""Tests for the pattern-key allowlist in unattended (-q / cron) contexts.

Background: ``command_allowlist`` holds two shapes of entry, matched by two
different mechanisms at two different points in ``check_all_command_guards``:

  * command TEXT / glob (``"rm -rf build/*"``, ``"podman *"``) -- matched by
    ``_command_matches_permanent_allowlist`` at an early floor, BEFORE the
    unattended-context resolution;
  * a dangerous-pattern KEY (``"recursive delete"``, ``"execute_code"``) --
    matched by ``is_approved``, and this is the shape the harness itself writes
    when a user answers ``[a]lways`` at an approval prompt (``_persist_choice``).

Only the TEXT form used to survive an unattended context. The pattern-KEY form
was consulted solely on the attended path, so answering "Always allow" persisted
an entry into config.yaml that was silently inert in every ``-q`` and cron
session -- the sessions where no human is there to re-approve, i.e. precisely
where a standing decision matters most. ``_unattended_deny`` now applies the same
per-key ``is_approved`` filter the attended path uses.

Both directions are pinned deliberately. A test that only asserts the unblocking
half would let a later change ungate unattended sessions wholesale; a test that
only asserts the blocking half would let the silent-inertness bug return.

Note ``check_execute_code_guard`` is intentionally NOT changed by this: it gates
on the *presence of a human*, not on the classification of a command, and an
``execute_code`` allowlist entry is honoured there only via the gateway/ask
prompt path. That asymmetry is pinned below so it stays a decision rather than
an accident.

Tirith: ``tests/conftest.py`` sets ``TIRITH_ENABLED=false`` globally (the real
scanner auto-installs from GitHub), so the live scanner returns ``allow`` here.
Every tirith assertion below therefore drives a STUB with an explicit action, and
the stub is asserted to be in effect -- otherwise a control asserting "the scan
still runs" would pass against a disabled scanner and prove nothing.
"""

from unittest.mock import patch as mock_patch

import pytest

import tools.approval as approval_module
from tools.approval import check_all_command_guards, detect_dangerous_command, load_permanent

# NB: a path under / (including /tmp/...) matches "delete in root path" FIRST --
# detect_dangerous_command returns the first match, so a relative path is required
# to exercise the "recursive delete" key. Verified by enumerating the whole table.
RECURSIVE_DELETE_CMD = "rm -rf ./build/artifacts"
RECURSIVE_DELETE_KEY = "recursive delete"

# Shape of a content-level finding: no static DANGEROUS_PATTERNS match at all, so
# only the tirith branch of _unattended_deny can block it.
TIRITH_ONLY_CMD = 'eval "$(cat /tmp/payload)"'
TIRITH_RULE_ID = "analysis_incomplete"
TIRITH_BLOCK_RESULT = {
    "action": "block",
    "summary": "nested executable body could not be resolved",
    "findings": [{"rule_id": TIRITH_RULE_ID, "severity": "HIGH",
                  "title": "Nested executable body could not be resolved",
                  "description": "The shell will execute a dynamically selected value."}],
}
TIRITH_ALLOW_RESULT = {"action": "allow", "findings": [], "summary": ""}


@pytest.fixture(autouse=True)
def _clear_approval_state():
    approval_module._permanent_approved.clear()
    approval_module.clear_session("default")
    yield
    approval_module._permanent_approved.clear()
    approval_module.clear_session("default")


@pytest.fixture
def quiet_tirith():
    """Stub the scanner to 'allow' so pattern-key behaviour is measured alone."""
    with mock_patch("tools.tirith_security.check_command_security",
                    return_value=TIRITH_ALLOW_RESULT):
        yield


@pytest.fixture
def blocking_tirith():
    """Stub the scanner to 'block', and PROVE the stub is in effect.

    Without this assertion the tirith controls below would pass against the
    conftest-disabled scanner, which returns 'allow' and makes them vacuous.
    """
    with mock_patch("tools.tirith_security.check_command_security",
                    return_value=TIRITH_BLOCK_RESULT):
        from tools.tirith_security import check_command_security
        assert check_command_security(TIRITH_ONLY_CMD)["action"] == "block", \
            "blocking-tirith stub is not in effect; these controls would be vacuous"
        yield


@pytest.fixture(params=[
    ("single_query", {"HERMES_SINGLE_QUERY_SESSION": "1", "HERMES_INTERACTIVE": "1"}),
    ("cron", {"HERMES_CRON_SESSION": "1"}),
], ids=["-q", "cron"])
def unattended(request, monkeypatch):
    """Put the process in one unattended context with its mode forced to deny."""
    name, env = request.param
    for var in ("HERMES_SINGLE_QUERY_SESSION", "HERMES_CRON_SESSION", "HERMES_INTERACTIVE",
                "HERMES_GATEWAY_SESSION", "HERMES_SESSION_PLATFORM", "HERMES_EXEC_ASK",
                "HERMES_YOLO_MODE"):
        monkeypatch.delenv(var, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with mock_patch(f"tools.approval_context._get_{name}_approval_mode", return_value="deny"), \
            mock_patch("tools.approval_context._get_approval_mode", return_value="smart"):
        # Guard the premise: if the context did not actually activate, every
        # "still denied" assertion below would pass for the wrong reason.
        assert [c.name for c in approval_module._unattended_contexts()] == [name]
        yield name


def test_fixture_command_maps_to_the_expected_key():
    """Pin the fixture itself. detect_dangerous_command returns the FIRST match, so
    a fixture can silently drift onto another rule and test nothing intended."""
    is_dangerous, key, _desc = detect_dangerous_command(RECURSIVE_DELETE_CMD)
    assert is_dangerous is True
    assert key == RECURSIVE_DELETE_KEY, (
        f"{RECURSIVE_DELETE_CMD!r} now classifies as {key!r}; pick a command that still "
        f"exercises {RECURSIVE_DELETE_KEY!r} or update the constant")
    assert detect_dangerous_command(TIRITH_ONLY_CMD)[0] is False, (
        f"{TIRITH_ONLY_CMD!r} now matches a static pattern, so it no longer isolates "
        "the tirith branch")


class TestPatternKeyAllowlistInUnattendedContexts:
    """A pattern-KEY allowlist entry must mean the same thing headlessly as it
    does interactively -- and must not mean anything more."""

    def test_allowlisted_pattern_key_allows(self, unattended, quiet_tirith):
        """THE FIX. The key the harness persists on '[a]lways' is honoured in -q/cron."""
        load_permanent({RECURSIVE_DELETE_KEY})
        result = check_all_command_guards(RECURSIVE_DELETE_CMD, "local")
        assert result["approved"] is True, (
            f"allowlisted pattern key {RECURSIVE_DELETE_KEY!r} was ignored in the "
            f"{unattended} context; an 'Always allow' answer is silently inert again")

    def test_same_command_still_denied_without_the_entry(self, unattended, quiet_tirith):
        """THE OTHER DIRECTION. Nothing was ungated wholesale: remove the entry and
        the identical command is still blocked."""
        result = check_all_command_guards(RECURSIVE_DELETE_CMD, "local")
        assert result["approved"] is False
        assert "BLOCKED" in result["message"]

    def test_unrelated_allowlist_entry_does_not_allow(self, unattended, quiet_tirith):
        """An allowlist entry only covers ITS OWN key -- no blanket effect."""
        load_permanent({"git force push (rewrites remote history)"})
        result = check_all_command_guards(RECURSIVE_DELETE_CMD, "local")
        assert result["approved"] is False, (
            "an unrelated allowlist entry allowed a recursive delete")

    def test_legacy_key_alias_also_allows(self, unattended, quiet_tirith):
        """Stored entries may use the legacy regex-derived key; aliases must still
        match, or the fix silently fails for allowlists written by older versions."""
        aliases = set(approval_module._approval_key_aliases(RECURSIVE_DELETE_KEY))
        legacy = aliases - {RECURSIVE_DELETE_KEY}
        assert legacy, f"no legacy alias for {RECURSIVE_DELETE_KEY!r}; nothing to test"
        load_permanent(legacy)
        assert check_all_command_guards(RECURSIVE_DELETE_CMD, "local")["approved"] is True


class TestTirithStillRunsForAllowlistedPatternKeys:
    """The load-bearing half of the design: the fix filters findings PER KEY rather
    than short-circuiting, so a pattern-key entry cannot buy a pass on an unrelated
    content-level finding in the same command."""

    def test_pattern_key_entry_does_not_suppress_a_tirith_finding(
            self, unattended, blocking_tirith):
        """Allowlisting 'recursive delete' must NOT also allow a command whose only
        problem is a tirith content finding."""
        load_permanent({RECURSIVE_DELETE_KEY})
        result = check_all_command_guards(TIRITH_ONLY_CMD, "local")
        assert result["approved"] is False, (
            "a pattern-key allowlist entry suppressed an unrelated tirith finding -- "
            "the fix short-circuited the scan instead of filtering per key")
        assert "BLOCKED" in result["message"]

    def test_tirith_scan_still_runs_for_an_allowlisted_dangerous_command(
            self, unattended, blocking_tirith):
        """Stronger form: a command that is BOTH pattern-dangerous (allowlisted) and
        tirith-flagged must still block on the tirith finding alone."""
        load_permanent({RECURSIVE_DELETE_KEY})
        result = check_all_command_guards(RECURSIVE_DELETE_CMD, "local")
        assert result["approved"] is False, (
            "allowlisting the pattern key skipped the tirith scan entirely")

    def test_tirith_key_entry_allows_only_its_own_finding(self, unattended, blocking_tirith):
        """A tirith:<rule_id> entry is honoured for that rule -- symmetry with the
        attended path, which already filtered it in check_all_command_guards."""
        load_permanent({f"tirith:{TIRITH_RULE_ID}"})
        assert check_all_command_guards(TIRITH_ONLY_CMD, "local")["approved"] is True

    def test_wrong_tirith_rule_id_does_not_allow(self, unattended, blocking_tirith):
        """The tirith key is per-rule, not a blanket 'trust tirith' switch."""
        load_permanent({"tirith:some_other_rule"})
        assert check_all_command_guards(TIRITH_ONLY_CMD, "local")["approved"] is False


class TestFloorsStillBeatTheAllowlist:
    """Negative controls: the allowlist must not reach past the unconditional floors.
    Those run BEFORE the allowlist by design, so an entry cannot buy them off."""

    @pytest.mark.parametrize("command,key", [
        ("dd if=/dev/zero of=/dev/sdb bs=1M", "dd to raw block device"),
        ("mkfs.ext4 /dev/sdc1", "format filesystem (mkfs)"),
    ])
    def test_hardline_still_blocks_with_entry_allowlisted(
            self, unattended, quiet_tirith, command, key):
        load_permanent({key, command})
        result = check_all_command_guards(command, "local")
        assert result["approved"] is False
        assert result.get("hardline") is True, (
            f"allowlisting bought off the hardline floor for {command!r}")

    def test_user_deny_rule_still_blocks_with_entry_allowlisted(self, unattended, quiet_tirith):
        load_permanent({RECURSIVE_DELETE_KEY, RECURSIVE_DELETE_CMD})
        with mock_patch("tools.approval_context._get_approval_config",
                        return_value={"deny": ["rm -rf *"]}):
            result = check_all_command_guards(RECURSIVE_DELETE_CMD, "local")
        assert result["approved"] is False
        assert result.get("user_deny") is True, (
            "allowlisting bought off an approvals.deny rule")


class TestExecuteCodeGuardIsPresenceBased:
    """``check_execute_code_guard`` is deliberately NOT allowlist-gated in an
    unattended context: it asks 'is a human present to vet arbitrary Python?',
    which an allowlist entry cannot answer. Pinned so the asymmetry with
    ``check_all_command_guards`` stays deliberate rather than accidental."""

    def test_execute_code_still_denied_in_unattended_even_when_allowlisted(self, unattended):
        load_permanent({"execute_code"})
        result = approval_module.check_execute_code_guard("print('hi')", "local")
        assert result["approved"] is False, (
            "execute_code became allowlist-gated in an unattended context; if that is "
            "intended, the config loader should also stop silently accepting the entry")
        assert "BLOCKED" in result["message"]

    def test_execute_code_denied_without_the_entry_too(self, unattended):
        """Control: the denial above is not caused by the allowlist entry."""
        result = approval_module.check_execute_code_guard("print('hi')", "local")
        assert result["approved"] is False

    def test_execute_code_approve_mode_allows(self, monkeypatch):
        """The supported way to enable it headlessly is the mode knob, not the allowlist."""
        for var in ("HERMES_CRON_SESSION", "HERMES_GATEWAY_SESSION",
                    "HERMES_SESSION_PLATFORM", "HERMES_EXEC_ASK", "HERMES_YOLO_MODE"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        with mock_patch("tools.approval_context._get_single_query_approval_mode",
                        return_value="approve"), \
                mock_patch("tools.approval_context._get_approval_mode", return_value="smart"):
            result = approval_module.check_execute_code_guard("print('hi')", "local")
        assert result["approved"] is True


class TestSafeCommandsUnaffected:
    """Routine work must not start blocking -- an over-tightened gate teaches the
    operator to approve without reading, which is how a guard stops working."""

    @pytest.mark.parametrize("command", [
        "ls -la /tmp", "git status", "df -h", "ip link show", "systemctl status nginx",
    ])
    def test_safe_commands_allowed(self, unattended, quiet_tirith, command):
        assert check_all_command_guards(command, "local")["approved"] is True
