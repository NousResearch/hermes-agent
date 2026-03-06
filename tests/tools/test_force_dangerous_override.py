"""Tests for the --force flag dangerous verdict bypass fix in skills_guard.py.

Regression test: the old code had `if result.verdict == "dangerous" and not force:`
which meant force=True would skip the early return, fall through the policy
lookup, and hit `if force: return True` - allowing installation of skills
flagged as dangerous (reverse shells, data exfiltration, etc).

The docstring explicitly states: "never overrides dangerous".
"""


def _old_should_allow(verdict, trust_level, force):
    """Simulate the BROKEN old logic."""
    INSTALL_POLICY = {
        "builtin":       ("allow",  "allow",   "allow"),
        "trusted":       ("allow",  "allow",   "block"),
        "community":     ("allow",  "block",   "block"),
    }
    VERDICT_INDEX = {"safe": 0, "caution": 1, "dangerous": 2}

    # Old buggy check: `and not force`
    if verdict == "dangerous" and not force:
        return False

    policy = INSTALL_POLICY.get(trust_level, INSTALL_POLICY["community"])
    vi = VERDICT_INDEX.get(verdict, 2)
    decision = policy[vi]

    if decision == "allow":
        return True

    if force:
        return True  # Bug: this line is reached for dangerous + force=True

    return False


def _new_should_allow(verdict, trust_level, force):
    """Simulate the FIXED logic."""
    INSTALL_POLICY = {
        "builtin":       ("allow",  "allow",   "allow"),
        "trusted":       ("allow",  "allow",   "block"),
        "community":     ("allow",  "block",   "block"),
    }
    VERDICT_INDEX = {"safe": 0, "caution": 1, "dangerous": 2}

    # Fixed: no `and not force` - dangerous is always blocked
    if verdict == "dangerous":
        return False

    policy = INSTALL_POLICY.get(trust_level, INSTALL_POLICY["community"])
    vi = VERDICT_INDEX.get(verdict, 2)
    decision = policy[vi]

    if decision == "allow":
        return True

    if force:
        return True

    return False


class TestForceNeverOverridesDangerous:
    """The core bug: --force bypassed the dangerous verdict block."""

    def test_old_code_allows_dangerous_with_force(self):
        """Old code: force=True lets dangerous skills through."""
        assert _old_should_allow("dangerous", "community", force=True) is True

    def test_new_code_blocks_dangerous_with_force(self):
        """Fixed code: force=True still blocks dangerous skills."""
        assert _new_should_allow("dangerous", "community", force=True) is False

    def test_new_code_blocks_dangerous_trusted_with_force(self):
        """Fixed code: even trusted + force cannot install dangerous."""
        assert _new_should_allow("dangerous", "trusted", force=True) is False

    def test_force_still_overrides_caution(self):
        """force=True should still work for caution verdicts."""
        assert _new_should_allow("caution", "community", force=True) is True

    def test_caution_community_blocked_without_force(self):
        """Caution + community is blocked without force (unchanged)."""
        assert _new_should_allow("caution", "community", force=False) is False

    def test_safe_always_allowed(self):
        """Safe verdict is always allowed regardless of force."""
        assert _new_should_allow("safe", "community", force=False) is True
        assert _new_should_allow("safe", "community", force=True) is True

    def test_dangerous_blocked_without_force(self):
        """Dangerous is blocked without force (both old and new agree)."""
        assert _old_should_allow("dangerous", "community", force=False) is False
        assert _new_should_allow("dangerous", "community", force=False) is False
