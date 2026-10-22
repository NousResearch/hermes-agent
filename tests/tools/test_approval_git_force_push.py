"""The approval gate catches every git force-push spelling, and only the push's own.

Regression for #141085. The two force-push patterns in ``tools/approval_detection.py`` used a
``.*`` that crossed shell operators, so a LATER command's ``-f`` (``grep -f``, ``pkill -f``)
tripped the force-push prompt on a plain ``git push``; and they missed three real force pushes:
git's combined short flags (``-qf``), the ``+refspec`` spelling, and a push behind git's own
global options (``git -C <dir> push -f``).

The patterns are also the approval KEY source: a stored allowlist entry must keep resolving, so
the pre-change regex-derived key stays an alias of both force-push classes.
"""

import pytest

from tools.approval import detect_dangerous_command


class TestGitForcePushDetected:
    """Every spelling that rewrites remote history asks for approval."""

    @pytest.mark.parametrize("command", [
        "git push --force origin main",
        "git push --force-with-lease origin main",
        "git push --force-with-lease=refs/heads/main:refs/heads/main origin main",
        "git push -f origin main",
        # git's combined short flags: `-qf` is `-q -f`, so it forces.
        "git push -qf origin main",
        "git push -fq origin main",
        # the `+refspec` spelling forces without any flag.
        "git push origin +main",
        "git push +HEAD:refs/heads/main origin",
        # git's global options may sit between `git` and the subcommand.
        "git -C /repo push -f origin main",
        "git -c core.hooksPath=/dev/null push -f origin main",
        "git --no-pager push --force origin main",
        # a force push in a LATER segment is still a force push.
        "cd /r && git push -f origin main",
        "git push -q && git push -f",
        "git push origin main; git push -f",
    ])
    def test_force_push_forms_are_dangerous(self, command):
        dangerous, _key, description = detect_dangerous_command(command)
        assert dangerous is True, command
        assert "force" in description.lower(), command

    @pytest.mark.parametrize("command", [
        "git push origin main",
        "git push --dry-run origin main",
        "git push -u origin main",
        "git push --set-upstream origin main",
        "git push origin feature/foo-bar",
        "git push origin feature+f",
        "git push --follow-tags origin main",
        "git push --receive-pack=/usr/bin/git-receive-pack origin main",
    ])
    def test_an_ordinary_push_stays_unprompted(self, command):
        assert detect_dangerous_command(command) == (False, None, None)


class TestGitForcePushScope:
    """A LATER command's flag belongs to that command, not to the push."""

    @pytest.mark.parametrize("command", [
        "git push -q origin HEAD:wo/x && grep -f patterns.txt log",
        "git push origin main; pkill -f stale-watcher",
        "git push origin main | grep -f patterns.txt",
        "git push origin main && grep -f patterns.txt log",
        "git push --quiet origin main && git log -f",
    ])
    def test_a_later_command_flag_is_not_a_force_push(self, command):
        dangerous, _key, description = detect_dangerous_command(command)
        assert dangerous is False, f"{command!r} -> {description!r}"


def test_the_pre_change_approval_key_still_resolves():
    """An allowlist entry stored under the old regex-derived key still covers a force push."""
    from tools.approval_detection import _approval_key_aliases

    legacy_key = r"git\s+push"
    for description in ("git force push (rewrites remote history)",
                        "git force push short flag (rewrites remote history)"):
        assert legacy_key in _approval_key_aliases(description), description
        assert description in _approval_key_aliases(legacy_key), description
