"""Tests for GNU long-option abbreviation bypass in DANGEROUS_PATTERNS.

GNU tools accept unique long-option prefix abbreviations at runtime
(e.g. ``chown --recur`` resolves to ``chown --recursive``). Two approval
patterns matched only the full flag name and could be evaded by passing a
valid abbreviation:

  chown    --recursive  →  --recur[a-z]*
  git push --force      →  --forc[a-z]*

The other long-flag patterns (rm/chmod/sed) were already covered on every
abbreviation by sibling short-flag / target patterns, so this file only
asserts the two gaps that were genuinely open plus the relevant regression
guards.
"""

import pytest

from tools.approval import detect_dangerous_command


class TestChownRecursiveLongOptionAbbreviation:
    """chown --recur* abbreviations targeting root must be caught.

    On main the bare ``--recur root`` form is caught only by an accidental
    case-insensitive ``r`` → ``R`` overlap with the short-flag pattern; longer
    abbreviations like ``--recurs``/``--recursi`` break that overlap and
    slipped through before the prefix change.
    """

    def test_chown_recursive_full_still_detected(self):
        dangerous, _, desc = detect_dangerous_command("chown --recursive root /etc")
        assert dangerous is True
        assert "chown" in desc.lower() or "root" in desc.lower()

    def test_chown_recur_root_detected(self):
        dangerous, _, _ = detect_dangerous_command("chown --recur root /etc")
        assert dangerous is True

    def test_chown_recurs_root_detected(self):
        dangerous, _, _ = detect_dangerous_command("chown --recurs root:root /var")
        assert dangerous is True, "chown --recurs is a valid abbreviation of --recursive"

    def test_chown_recursi_root_detected(self):
        dangerous, _, _ = detect_dangerous_command("chown --recursi root /etc")
        assert dangerous is True

    def test_chown_recur_non_root_not_flagged(self):
        """--recur* chown to a non-root user must not be flagged."""
        dangerous, _, _ = detect_dangerous_command("chown --recur nobody /opt/app")
        assert dangerous is False


class TestGitPushForceLongOptionAbbreviation:
    """git push --forc* abbreviations must be caught.

    The short ``-f`` pattern does not catch ``--forc`` (the ``\\b`` after the
    ``f`` does not match mid-word), so abbreviated long forms slipped through.
    """

    def test_git_push_force_full_still_detected(self):
        dangerous, _, desc = detect_dangerous_command("git push --force origin main")
        assert dangerous is True
        assert "force" in desc.lower()

    def test_git_push_forc_abbreviation_detected(self):
        dangerous, _, _ = detect_dangerous_command("git push --forc origin main")
        assert dangerous is True, "git push --forc is a valid abbreviation of --force"

    def test_git_push_forced_variant_detected(self):
        dangerous, _, _ = detect_dangerous_command("git push --forced origin main")
        assert dangerous is True

    def test_git_push_force_with_lease_detected(self):
        dangerous, _, _ = detect_dangerous_command(
            "git push --force-with-lease origin main"
        )
        assert dangerous is True

    def test_git_push_short_f_still_detected(self):
        """Existing -f pattern must not regress."""
        dangerous, _, _ = detect_dangerous_command("git push -f origin main")
        assert dangerous is True

    def test_git_push_no_force_not_flagged(self):
        dangerous, _, _ = detect_dangerous_command("git push origin main")
        assert dangerous is False

    def test_git_push_set_upstream_not_flagged(self):
        dangerous, _, _ = detect_dangerous_command(
            "git push --set-upstream origin feature"
        )
        assert dangerous is False


class TestFullFormRegressions:
    """The two changed long-flag patterns must still detect their full form."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "chown --recursive root /etc",
            "git push --force origin main",
        ],
    )
    def test_full_form_still_detected(self, cmd):
        dangerous, key, _ = detect_dangerous_command(cmd)
        assert dangerous is True, f"Full-form long flag not detected in: {cmd!r}"
        assert key is not None
