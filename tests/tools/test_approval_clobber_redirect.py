"""`>|` (bash's noclobber-overriding redirect) is a write like `>`: every redirect rule must see it."""

import pytest

from tools.approval import detect_dangerous_command, detect_hardline_command


@pytest.mark.parametrize("command", [
    "echo k >| ~/.ssh/authorized_keys",
    "echo x >| /etc/passwd",
    "cat a >|~/.bashrc",
    "date >|'/etc/hosts'",
])
def test_clobber_redirect_to_sensitive_target_needs_approval(command):
    assert detect_dangerous_command(command)[0] is True


def test_clobber_redirect_to_block_device_hits_hardline_floor():
    assert detect_hardline_command("cat f >| /dev/sda")[0] is True
    assert detect_dangerous_command("echo hi >| out.txt")[0] is False
    assert detect_dangerous_command("ls >| /dev/null")[0] is False
