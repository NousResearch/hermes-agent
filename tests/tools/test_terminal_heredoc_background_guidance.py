"""Foreground guidance distinguishes heredoc data from shell backgrounding."""

import pytest

from tools.terminal_tool import _foreground_background_guidance


@pytest.mark.parametrize("delimiter", ["PY", "'PY'", '"PY"'])
def test_python_heredoc_bitwise_ampersand_is_allowed(delimiter):
    command = f"python3 - <<{delimiter}\nprint(oct(st.st_mode & 0o7777))\nPY\n"
    assert _foreground_background_guidance(command) is None


@pytest.mark.parametrize("command", [
    "cmd1 & cmd2",
    # Arithmetic left shifts are not heredoc openers.
    "echo $((1 <<SHIFT ))\ncmd1 & cmd2\nSHIFT\n",
    "((value <<SHIFT ))\ncmd1 & cmd2\nSHIFT\n",
    "python3 - <<'PY' & cmd2\nprint(1)\nPY\n",
    "python3 - <<'PY'\nprint(1)\nPY\ncmd1 & cmd2",
    "cmd1 & python3 - <<PY\nprint(1)\nPY\n",
    "python3 - <<'PY'\nprint(1)\nPY\ncmd1 &",
    # A quoted/commented opener is not a heredoc; don't hide following commands.
    "echo \"<<'PY'\"\ncmd1 & cmd2\nPY\n",
    "echo ok # <<PY\ncmd1 & cmd2\nPY\n",
    "echo 'multiline\n<<PY'\ncmd1 & cmd2\nPY\n",
    # Here-strings and unterminated heredocs must not consume later commands.
    "cat <<<PY\ncmd1 & cmd2\nPY\n",
    "python3 - <<PY\ncmd1 & cmd2\n",
    # Unquoted heredocs perform command substitution in the *outer shell*.
    "cat <<PY\n$(cmd1 & cmd2)\nPY\n",
    "cat <<PY\n`cmd1 & cmd2`\nPY\ncmd3 & cmd4",
])
def test_real_shell_backgrounding_remains_flagged(command):
    assert _foreground_background_guidance(command) is not None


def test_tab_stripped_heredoc_bitwise_ampersand_is_allowed():
    command = "python3 - <<-'PY'\n\tprint(oct(st.st_mode & 0o7777))\n\tPY\n"
    assert _foreground_background_guidance(command) is None


def test_multiple_heredocs_on_same_line_are_data():
    command = "cat <<ONE <<'TWO'\na & b\nONE\nc & d\nTWO\n"
    assert _foreground_background_guidance(command) is None


def test_heredoc_terminator_requires_exact_line():
    command = "python3 - <<PY\nPY_suffix\nprint(oct(st.st_mode & 0o7777))\nPY\n"
    assert _foreground_background_guidance(command) is None
