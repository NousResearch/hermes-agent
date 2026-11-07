"""Direct tests for the masker the hardline floor relies on (see `tools.shell_heredoc`).

The floor asks "what executes", so it passes ``require_quoted_delimiter=False``: plain expansion
inside an unquoted body cannot run anything. The shared default stays strict for the guards that
must follow a body's expanded text (pinned by
``tests/tools/test_terminal_heredoc_background_guard.py::TestUnsafeHeredocPayloadRemainsVisible``
and ``tests/cron/test_lifecycle_guard_heredoc_walk.py``).
"""

from tools.shell_heredoc import strip_inert_heredoc_bodies


def test_tee_is_a_data_only_consumer():
    """``tee`` copies stdin to a file and never executes it, like ``cat``."""
    command = "tee notes.md <<'EOF'\nreboot\nEOF\n"
    assert strip_inert_heredoc_bodies(command) != command


def test_unquoted_body_with_a_substitution_stays_visible_under_the_floor_flag():
    """``$(...)``/backtick run in the shell before the consumer sees the text, flag or not."""
    for command in (
        "cat > notes.md <<EOF\n$(reboot)\nEOF\n",
        "cat > notes.md <<EOF\n`reboot`\nEOF\n",
    ):
        assert strip_inert_heredoc_bodies(command, require_quoted_delimiter=False) == command


def test_default_keeps_an_unquoted_body_visible():
    """Variable/arithmetic expansion may act on the body later, so the default does not mask it."""
    command = "cat > notes.md <<EOF\n$MAINTENANCE_NOTE\nEOF\n"
    assert strip_inert_heredoc_bodies(command) == command
