"""Regression tests for foreground/background guidance guardrails."""

from tools.terminal_tool import _foreground_background_guidance, _strip_quotes


def test_heredoc_python_bitwise_ampersand_does_not_look_like_backgrounding():
    command = """python3 - <<'PY'
from pathlib import Path
st = Path('/tmp').stat()
print(oct(st.st_mode & 0o777))
PY
"""

    assert _foreground_background_guidance(command) is None


def test_heredoc_body_keywords_do_not_trigger_shell_wrapper_guidance():
    command = """python3 - <<'PY'
print('nohup setsid disown & still plain Python data')
PY
"""

    assert _foreground_background_guidance(command) is None


def test_trailing_shell_ampersand_still_gets_background_guidance():
    guidance = _foreground_background_guidance("python3 -m http.server 8000 &")

    assert guidance is not None
    assert "background=true" in guidance


def test_inline_shell_ampersand_still_gets_background_guidance():
    guidance = _foreground_background_guidance("python3 server.py & curl localhost:8000")

    assert guidance is not None
    assert "background=true" in guidance


def test_strip_quotes_removes_heredoc_bodies_but_preserves_command_line():
    command = """python3 - <<'PY'
print('body & should disappear')
PY
printf done
"""

    stripped = _strip_quotes(command)

    assert "python3 - <<''" in stripped
    assert "printf done" in stripped
    assert "body & should disappear" not in stripped
