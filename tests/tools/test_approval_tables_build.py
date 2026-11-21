"""Every pattern of the command-approval tables builds on RE2 (the build is lazy; this forces it)."""
import hermes_re
from tools.approval_detection import detect_dangerous_command, detect_hardline_command


def test_every_approval_pattern_builds():
    detect_dangerous_command("rm -rf /tmp/é")
    detect_hardline_command("ls")
    assert hermes_re.build_all(wide=True) == []
