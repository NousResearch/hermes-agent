"""Heredoc bodies that only feed a data consumer must not trip the hardline floor (#149601).

``cat`` and ``tee`` copy their stdin to a file or stdout and never execute it, and the non-shell
interpreters already allowlisted by ``tools.shell_heredoc`` read their stdin as source for a
*non-shell* language. A body line that begins with ``reboot``/``shutdown`` is therefore data: an
agent writing ``cat > notes.md <<'EOF'`` with a maintenance note is not rebooting anything, and the
hardline tier offers no approval that could let it through.

The masking stays conservative in the other direction — a body that a shell or an interpreter
executes, an unquoted body carrying an executable substitution, a body under a list operator, and
anything after the terminator keep their full text, so the floor still sees what will run.
"""

import pytest

from tools.approval_detection import detect_hardline_command


# Data-only delivery: the body is text, so none of these may hit the floor.
_INERT_BODIES = [
    # the reported shape: a quoted-delimiter note that mentions a reboot
    "cat > notes.md <<'EOF'\nreboot the board after the flash\nEOF\n",
    "cat > notes.md <<'EOF'\nsudo reboot\nEOF\n",
    # an unquoted delimiter still expands, but a plain body line executes nothing
    "cat > notes.md <<EOF\nreboot the board after the flash\nEOF\n",
    "tee notes.md <<'EOF'\nshutdown window is 02:00\nEOF\n",
    "tee -a notes.md <<EOF\nreboot the board after the flash\nEOF\n",
    # a `;`/`&&` inside a data body is text, not a separator the shell acts on
    "cat > notes.md <<'EOF'\nremember; reboot the board\nEOF\n",
    "cat > notes.md <<'EOF'\nstep 1 && sudo reboot\nEOF\n",
    # the non-shell interpreter path the shared helper already allowlists
    "python3 - <<'EOF'\nreboot the board after the flash\nEOF\n",
]

# Everything the conservative masking must leave visible to the floor.
_VISIBLE_BODIES = [
    # a shell executes a heredoc body
    "bash <<'EOF'\nreboot\nEOF\n",
    "sh -s <<'EOF'\nreboot\nEOF\n",
    "sudo bash <<'EOF'\nreboot\nEOF\n",
    # the body reaches a shell through a pipeline
    "cat <<'EOF' | bash\nreboot\nEOF\n",
    # an unquoted body really does execute a substitution
    "cat > notes.md <<EOF\n$(reboot)\nEOF\n",
    "cat > notes.md <<EOF\n`shutdown -h now`\nEOF\n",
    # commands after the terminator are not part of the body
    "cat > notes.md <<'EOF'\nnotes\nEOF\nreboot\n",
    "cat > notes.md <<'EOF'\nnotes\nEOF\n&& shutdown -h now",
    # an unterminated heredoc cannot be proven data
    "cat > notes.md <<'EOF'\nreboot\n",
    # the delivery command itself is not a data consumer
    "sudo reboot <<'EOF'\nignored\nEOF\n",
    # masking the body must not hide a redirect to a raw device
    "cat > /dev/sda <<'EOF'\ndata\nEOF\n",
    # a disk-wipe body under a shell keeps its hardline verdict
    "bash <<'EOF'\nrm -rf /\nEOF\n",
]


@pytest.mark.parametrize("command", _INERT_BODIES)
def test_data_heredoc_body_is_not_a_command(command):
    is_hardline, description = detect_hardline_command(command)
    assert not is_hardline, (
        f"a data-only heredoc body tripped the hardline floor: {command!r} "
        f"(got: {description})"
    )
    assert description is None


@pytest.mark.parametrize("command", _VISIBLE_BODIES)
def test_executable_heredoc_body_still_hits_the_floor(command):
    is_hardline, description = detect_hardline_command(command)
    assert is_hardline, f"hardline threat leaked through the heredoc masking: {command!r}"
    assert description


@pytest.mark.parametrize(
    "command",
    [
        # an unterminated quoted heredoc, a body that only LOOKS inert: the helper leaves the whole
        # command untouched whenever it cannot prove the body is data.
        "cat > notes.md <<'EOF'\nreboot\nEOF\ncat > other.md <<'UNCLOSED'\nreboot\n",
    ],
)
def test_unprovable_second_heredoc_keeps_first_body_visible(command):
    """A second, unterminated heredoc makes the whole command unprovable, so the real command shows."""
    is_hardline, description = detect_hardline_command(command)
    assert is_hardline, f"a threat after an unprovable heredoc was masked away: {command!r}"
    assert description
