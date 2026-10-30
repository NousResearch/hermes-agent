"""Tests for git push ref-destructive flags and same-command flag scoping.

Closes the false negatives behind the incident PrimeIntellect-ai/prime-agent#2446
(62 remote branches dropped): `git push --mirror origin` deletes/force-updates every
remote ref that is absent locally, but the shipped DANGEROUS_PATTERNS had no
`--mirror` rule, so the guard ALLOWED it — and every global-option spelling
(`git -C DIR push …`, `git -c k=v push …`, `git --git-dir=… push …`) bypassed even
the existing `--force` rules because `\\bgit\\s+push\\b` required adjacency.

Also locks the DOTALL scoping issue: `\\bgit\\s+push\\b.*-f\\b` matched across
`;`/`&&`/`|` segment boundaries, so a harmless
`git push --dry-run origin main | sed 's/[0-9a-f]…/…/'` was blocked only because an
unrelated `-f` token appeared in a LATER command segment. The force/mirror rules are
now bounded to one command segment (same idiom as the git-branch long-flag rules).

And the `--force-with-lease` misattribution: the legacy `-f\\b` pattern matched the
`-f` inside `--force-with-lease` (hyphen is a word boundary). It now gets its own
detection as a gated pattern instead of the misleading "rewrites remote history"
verdict — never ALLOWED, so no coverage is lowered.
"""

import pytest

from tools.approval import detect_dangerous_command


class TestGitPushMirrorFalseNegative:
    """A mirror push must be dangerous: it force-updates and deletes remote refs."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "git push --mirror origin",
            "git push --mirror",
            "git -C /srv/app push --mirror origin",
            "git -C /srv/app push --force-with-lease origin",   # global-opts + lease
            "git -c protocol.version=2 push --mirror origin",
            "git --git-dir=/srv/bare.git push --mirror origin",
            "git -C /srv/app push -f origin main",              # -f behind global opts
            "git push origin --mirror",          # flag after remote
            "git push --mirr origin",            # git unambiguous abbreviation
            "git push --mirror --force-with-lease origin",
            "git push --prune origin",           # deletes remote refs absent locally
            "git -C /repo push --prune origin",
        ],
    )
    def test_mirror_and_scoping_push_detected(self, cmd):
        dangerous, _, desc = detect_dangerous_command(cmd)
        assert dangerous is True, f"mirror/force-class push must block: {cmd!r}"
        assert ("mirror" in desc.lower() or "force" in desc.lower()
                or "lease" in desc.lower() or "prune" in desc.lower())

    @pytest.mark.parametrize(
        "cmd",
        [
            "git push origin main",
            "git push origin mirror-sync",           # branch NAMED mirror-*
            "git push --dry-run origin main",
            "git commit -m 'do not use --mirror on shared remotes'",  # quoted prose
            "git push --no-mirror-check origin main",                # not a --mirror flag
            "git remote add mirror git@host:repo.git",
            "git fetch --all --prune",   # prune on FETCH only moves tracking refs — harmless
            "git push --all origin",     # publish-only semantics, deliberately NOT blocked
        ],
    )
    def test_benign_push_not_flagged_as_mirror(self, cmd):
        dangerous, _, desc = detect_dangerous_command(cmd)
        assert dangerous is False or "mirror" not in (desc or "").lower(), cmd


class TestGitPushFlagScoping:
    """The -f/--force rules must stay inside ONE command segment.

    Under the module-wide DOTALL flags, `.*` crossed `;`, `&&` and `|`,
    so a harmless push was blocked because an unrelated `-f` token lived
    in a later segment (measured on main: sed '[0-9a-f]\\{40\\}' and
    `git config -f` both tripped a --dry-run push).
    """

    @pytest.mark.parametrize(
        "cmd",
        [
            "git push --dry-run origin main | sed 's/[0-9a-f]\\{40\\}/…/'",
            "git push --dry-run origin main && git config -f .gitattributes foo",
            "git push --dry-run origin main; git branch -f tmp HEAD",
        ],
    )
    def test_unrelated_f_in_later_segment_not_blocked(self, cmd):
        dangerous, _, desc = detect_dangerous_command(cmd)
        assert dangerous is False, f"{cmd!r} blocked by cross-segment match: {desc}"

    @pytest.mark.parametrize(
        "cmd",
        [
            "git push -f origin main",
            "git push --force origin main",
            "git push -f origin main && echo done",   # -f IS in the push segment
            "git push origin main --force",
        ],
    )
    def test_real_force_push_still_blocked(self, cmd):
        dangerous, _, _ = detect_dangerous_command(cmd)
        assert dangerous is True, f"regression: force push must stay blocked: {cmd!r}"


class TestForceWithLeaseGatedNotHistoryWipe:
    """--force-with-lease is not a history wipe, but it is not allowed either.

    Coverage is never LOWERED: the safe spelling stays gated, but under its own
    description so the approval reason is honest (refuses stale overwrites; can
    still delete remote-only refs).
    """

    def test_force_with_lease_detected_with_own_reason(self):
        dangerous, _, desc = detect_dangerous_command(
            "git push --force-with-lease origin feature/eval-scale-1"
        )
        assert dangerous is True
        assert "lease" in desc.lower()

    def test_force_without_lease_keeps_history_reason(self):
        dangerous, _, desc = detect_dangerous_command("git push --force origin main")
        assert dangerous is True
        assert "history" in desc.lower()


class TestGitPushDeleteAndRefspec:
    """Ref deletion and force-via-refspec spellings must gate; plain refspecs must not."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "git push origin --delete feature/eval-scale-1",
            "git push origin -d feature/eval-scale-1",
            "git push origin :feature/eval-scale-1",
            "git push origin :refs/heads/feature/x",
        ],
    )
    def test_push_delete_detected(self, cmd):
        dangerous, _, desc = detect_dangerous_command(cmd)
        assert dangerous is True, cmd
        assert "delete" in desc.lower(), (cmd, desc)

    @pytest.mark.parametrize(
        "cmd",
        [
            "git push origin +main:main",
            "git push origin +main",
            "git -C /srv/app push origin +main:main",
        ],
    )
    def test_force_push_via_leading_plus_refspec_detected(self, cmd):
        """A leading ``+`` in a refspec is force-push semantics
        (``git push origin +main:main`` == ``--force``). The ``+`` is a refspec PREFIX
        (right after whitespace), not a flag or a name character."""
        dangerous, _, desc = detect_dangerous_command(cmd)
        assert dangerous is True, cmd
        assert "force" in desc.lower() or "+" in desc, (cmd, desc)

    @pytest.mark.parametrize(
        "cmd",
        [
            "git push origin main:main",
            "git push origin HEAD:main",
            "git push origin main+",
            "git push origin v2.0+build",
            "git push origin main && echo +done",
            "git push origin --all",
        ],
    )
    def test_benign_push_refspecs_not_flagged(self, cmd):
        """Non-force refspecs and ``+`` appearing NOT as a refspec prefix must stay
        clean (git forbids ``+``-prefixed tag names, so the FP risk surface is
        name-internal ``+``)."""
        dangerous, _, _ = detect_dangerous_command(cmd)
        assert dangerous is False, cmd
