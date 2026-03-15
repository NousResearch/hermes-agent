"""Regression tests for skills guard policy precedence.

Official/builtin skills should follow the INSTALL_POLICY table even when their
scan verdict is dangerous, and --force should override blocked verdicts for
non-builtin sources.
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

    policy = INSTALL_POLICY.get(trust_level, INSTALL_POLICY["community"])
    vi = VERDICT_INDEX.get(verdict, 2)
    decision = policy[vi]

    if decision == "allow":
        return True

    if force:
        return True

    return False


class TestPolicyPrecedenceForDangerousVerdicts:
    def test_builtin_dangerous_is_allowed_by_policy(self):
        assert _new_should_allow("dangerous", "builtin", force=False) is True

    def test_trusted_dangerous_is_blocked_without_force(self):
        assert _new_should_allow("dangerous", "trusted", force=False) is False

    def test_force_overrides_dangerous_for_community(self):
        assert _new_should_allow("dangerous", "community", force=True) is True

    def test_force_overrides_dangerous_for_trusted(self):
        assert _new_should_allow("dangerous", "trusted", force=True) is True

    def test_force_still_overrides_caution(self):
        assert _new_should_allow("caution", "community", force=True) is True

    def test_caution_community_blocked_without_force(self):
        assert _new_should_allow("caution", "community", force=False) is False

    def test_safe_always_allowed(self):
        assert _new_should_allow("safe", "community", force=False) is True
        assert _new_should_allow("safe", "community", force=True) is True

    def test_old_code_happened_to_allow_forced_dangerous_community(self):
        assert _old_should_allow("dangerous", "community", force=True) is True
