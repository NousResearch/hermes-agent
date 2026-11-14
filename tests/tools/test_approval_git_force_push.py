"""The approval gate catches every git force-push spelling, and only the push's own.

Regression for #141085. The two force-push patterns in ``tools/approval_detection.py`` used a
``.*`` that crossed shell operators, so a LATER command's ``-f`` (``grep -f``, ``pkill -f``)
tripped the force-push prompt on a plain ``git push``; and they missed three real force pushes:
git's combined short flags (``-qf``), the ``+refspec`` spelling, and a push behind git's own
global options (``git -C <dir> push -f``).

The pre-change approval keys of all git rules are pinned in ``test_approval_git_global_options.py``.
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
        # the shell strips quotes, so a quoted flag or refspec still forces.
        "git push \"-f\" origin main",
        "git push origin '+main'",
        # a redirection's `&`/`|` is not a command separator.
        "git push origin main 2>&1 --force",
        "git push origin main &>/dev/null -f",
        # path-valued global options, attached or quoted values.
        "git --git-dir /x/.git push -f",
        "git -C/repo push -f",
        "git -C \"/my repo\" push -f",
        "git --exec-path /usr/lib/git-core push -f",
        "git --attr-source HEAD push -f",
        # a quoted `;`/`&`/`|` is part of an argument, not a command separator.
        "git push -o 'ci;skip' -f origin main",
        "git push -o \"a&b\" -f origin main",
        "git push --push-option='x|y' --force origin main",
        "git push --receive-pack='a|b' -f origin main",
        "git push --exec=\"x;y\" -f origin",
        "git push origin main '&' -f",
        "git push \"--force\" origin main",
        "git push -fo ci.skip origin main",
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
        # `-o` takes the rest of the cluster as its value: push option "foo", not `-f`.
        "git push -ofoo origin main",
        # a flag inside a quoted push option is the option's text.
        "git push -o 'note -f' origin main",
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
        "git push -q 2>&1 | grep -f patterns.txt",
        "git push origin main >| log; pkill -f stale-watcher",
        "git push -o 'a;b' origin main && grep -f patterns.txt",
    ])
    def test_a_later_command_flag_is_not_a_force_push(self, command):
        dangerous, _key, description = detect_dangerous_command(command)
        assert dangerous is False, f"{command!r} -> {description!r}"


@pytest.mark.parametrize("unit", [">&", "&>", ">|", "'a' ", '"a" ', "'", '"', "\\", "-c "])
def test_a_run_inside_the_push_segment_does_not_backtrack_exponentially(unit):
    """Overlapping push-argument alternatives used to re-split a `>&>&...` run 2**n ways."""
    import time

    start = time.monotonic()
    assert detect_dangerous_command("git push origin main " + unit * 200) == (False, None, None)
    assert time.monotonic() - start < 2
