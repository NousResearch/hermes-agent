"""Tests for the global-flag bypass of ``<binary> <subcommand>``-anchored DANGEROUS_PATTERNS.

Several rules were anchored so the subcommand had to follow the command word immediately
(``\\bgit\\s+push\\b``), or admitted a run of dash-tokens only (``(-[^\\s]+\\s+)*``). Both spellings
are defeated by a global option that takes a SEPARATE value word, because the value is a bare
word the anchor cannot cross:

  git -C /tmp push --force      →  allowed, no prompt
  git -c k=v reset --hard       →  allowed, no prompt
  git --work-tree /tmp clean -fd → allowed, no prompt
  systemctl -M host stop nginx  →  allowed, no prompt
  killall -u root -9 foo        →  allowed, no prompt
  pkill -u root -9 foo          →  allowed, no prompt
  hermes -p ade update          →  allowed, no prompt
  docker --log-level x context use prod → allowed, no prompt

The fix threads ``_GLOBAL_FLAG_RUN`` between the command word and the anchored token. The run
consumes option TOKENS only (``-x``/``--xy``, optionally plus a glued ``=value`` or ONE following
value word), never a bare word on its own, so a compound command such as
``git status && echo push --force`` cannot be pulled into a match. Its separators are horizontal
whitespace, which keeps the run inside a single command segment.
"""

import pytest

from tools.approval import detect_dangerous_command


GIT_GLOBAL_FLAG_BYPASSES = [
    # (command, substring expected in the rule description)
    ("git -C /tmp push --force origin main", "force"),
    ("git -C /tmp push -f origin main", "force"),
    ("git -c user.name=x push --force origin main", "force"),
    ("git --git-dir=/opt/x push --force origin main", "force"),
    ("git --exec-path=/opt/x push --force origin main", "force"),
    ("git --namespace ns push --force origin main", "force"),
    ("git -C /tmp clean -fd", "clean"),
    ("git --work-tree /tmp -C /tmp clean -fd", "clean"),
    ("git -C /tmp reset --hard HEAD", "reset"),
    ("git -c core.pager=less reset --hard HEAD", "reset"),
    ("git -C /tmp branch -D feat", "branch"),
    ("git -C /tmp branch -d -f feat", "branch"),
    ("git -C /tmp branch --force --delete feat", "branch"),
]


class TestGitGlobalFlagBypass:
    """A git global option must not defeat the destructive-git rules.

    ``git`` accepts ``-C <path>``, ``-c <k>=<v>``, ``--git-dir``, ``--work-tree``,
    ``--exec-path`` and ``--namespace`` before the subcommand. Every one of them separates
    ``git`` from ``push``/``clean``/``reset``/``branch`` and so slipped every anchored rule.
    """

    @pytest.mark.parametrize("cmd,expected_word", GIT_GLOBAL_FLAG_BYPASSES)
    def test_git_global_flag_forms_detected(self, cmd, expected_word):
        dangerous, key, desc = detect_dangerous_command(cmd)
        assert dangerous is True, f"global-flag form not detected: {cmd!r}"
        assert key is not None
        assert expected_word in desc.lower()


class TestGitPlainFormRegressions:
    """The flag-free spellings the rules already caught must keep firing."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "git push --force origin main",
            "git push -f origin main",
            "git push --forc origin main",
            "git clean -fd",
            "git reset --hard HEAD",
            "git reset --h HEAD",
            "git branch -D feat",
            "git branch -d -f feat",
        ],
    )
    def test_plain_form_still_detected(self, cmd):
        dangerous, key, _ = detect_dangerous_command(cmd)
        assert dangerous is True, f"plain form regressed: {cmd!r}"
        assert key is not None


class TestGitNoFalsePositives:
    """The flag run must not turn benign git usage into a prompt.

    The run cannot consume a bare word, so it can never bridge from a harmless subcommand to a
    dangerous keyword appearing later in a compound command or inside a commit message.
    """

    @pytest.mark.parametrize(
        "cmd",
        [
            # The dangerous keyword belongs to a LATER command, not to git.
            "git status && echo push --force",
            "git log --oneline && echo clean -fd",
            "git diff && echo branch -D",
            "git -C /tmp status && echo push --force",
            # Dangerous keyword as prose inside a commit message.
            "git commit -m 'do not push --force here'",
            "git -c user.name=x commit -m 'push --force in the message'",
            # Benign git, with and without global flags.
            "git push --set-upstream origin feature",
            "git status",
            "git -C /tmp status",
            "git -C /tmp log --oneline",
            "git -C /tmp branch --list",
            # A path segment that merely spells a subcommand.
            "git -C /tmp/push status",
        ],
    )
    def test_benign_git_not_flagged(self, cmd):
        dangerous, _, desc = detect_dangerous_command(cmd)
        assert dangerous is False, f"false positive on: {cmd!r} -> {desc}"


class TestNonGitGlobalFlagBypass:
    """The same defect in the systemctl / killall / pkill / hermes / docker rules.

    ``systemctl`` and ``killall`` admitted ``(-[^\\s]+\\s+)*``, which accepts dash-tokens but not
    the value word that ``-M <host>`` / ``-u <user>`` supplies. ``pkill``, ``hermes update`` and
    ``docker context use`` admitted no flag run at all, even though sibling rules for the same
    binaries already did.
    """

    @pytest.mark.parametrize(
        "cmd,expected_word",
        [
            ("systemctl -M remote stop nginx", "service"),
            ("systemctl --host remote restart nginx", "service"),
            ("killall -u root -9 foo", "kill"),
            ("killall -u root -s KILL foo", "kill"),
            ("killall -u root -r 'fire.*'", "kill"),
            ("pkill -u root -9 foo", "kill"),
            ("hermes -p ade update", "update"),
            ("docker --log-level debug context use prod", "context"),
        ],
    )
    def test_global_flag_forms_detected(self, cmd, expected_word):
        dangerous, key, desc = detect_dangerous_command(cmd)
        assert dangerous is True, f"global-flag form not detected: {cmd!r}"
        assert key is not None
        assert expected_word in desc.lower()

    @pytest.mark.parametrize(
        "cmd",
        [
            "systemctl stop nginx",
            "systemctl restart nginx",
            "killall -9 foo",
            "killall -s KILL foo",
            "killall -r 'fire.*'",
            "pkill -9 foo",
            "hermes update",
            "hermes gateway stop",
            "docker context use prod",
        ],
    )
    def test_plain_form_still_detected(self, cmd):
        dangerous, key, _ = detect_dangerous_command(cmd)
        assert dangerous is True, f"plain form regressed: {cmd!r}"
        assert key is not None

    @pytest.mark.parametrize(
        "cmd",
        [
            "systemctl status nginx",
            "systemctl --type service list-units",
            "docker ps",
            "docker --log-level debug ps",
            "killall foo",
            "pkill foo",
            "hermes --version",
            "hermes -p ade run hello",
            # Horizontal-only separators keep the run inside one command segment: the `update`
            # here belongs to `npm`, not to `hermes`.
            "hermes --version\nnpm update",
        ],
    )
    def test_benign_not_flagged(self, cmd):
        dangerous, _, desc = detect_dangerous_command(cmd)
        assert dangerous is False, f"false positive on: {cmd!r} -> {desc}"


class TestLegacyApprovalKeysPreserved:
    """Widening a regex renames its derived approval key; the alias table must keep the old one.

    ``_PATTERN_KEY_ALIASES`` derives a legacy key from the regex TEXT, so an allowlist or session
    entry stored against the pre-fix spelling would silently stop matching without these aliases.
    """

    @pytest.mark.parametrize(
        "description,legacy_key",
        [
            ("git force push (rewrites remote history)", r"git\s+push"),
            ("git force push short flag (rewrites remote history)", r"git\s+push"),
            ("git clean with force (deletes untracked files)", r"git\s+clean\s+-[^\s]*f"),
            ("git branch force delete", r"git\s+branch\s+-D"),
            ("git branch force delete (long flags)", r"git\s+branch"),
            ("git reset --hard (destroys uncommitted changes)",
             r"git\s+reset\s+--h(?:a(?:r(?:d)?)?)?"),
            ("stop/restart system service",
             r"systemctl\s+(-[^\s]+\s+)*(stop|restart|disable|mask)"),
            ("force kill processes", r"pkill\s+-9"),
            ("force kill processes (killall -KILL)",
             r"killall\s+(-[^\s]*\s+)*-(9|KILL|SIGKILL)"),
            ("kill processes by regex (killall -r)", r"killall\s+(-[^\s]*\s+)*-r"),
            ("hermes update (restarts gateway, kills running agents)", r"hermes\s+update"),
            ("docker context use (switches default daemon for future commands)",
             r"docker\s+context\s+use"),
        ],
    )
    def test_legacy_key_still_aliases(self, description, legacy_key):
        from tools.approval_detection import _approval_key_aliases

        assert legacy_key in _approval_key_aliases(description)
        assert description in _approval_key_aliases(legacy_key)
