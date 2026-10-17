"""find -exec/-execdir actions are judged by the program they actually launch."""

from tools.approval import detect_dangerous_command


def test_wrapper_between_exec_and_rm_still_needs_approval():
    """A wrapper word after -exec (sudo/env/nice/timeout/doas/...) used to hide the rm, so a mass
    delete ran unapproved; it must share the bare spelling's approval key. Running each matched
    file as a program asks too, while an argument merely named rm stays data."""
    rm_key = detect_dangerous_command("find . -exec rm {} \\;")[1]
    for cmd in ("find . -exec sudo rm '{}' \\;", "find . -exec nice rm {} \\;",
                "find . -exec timeout 5 rm {} +", "find . -execdir env FOO=1 /bin/rm {} +",
                "find . -exec doas rm {} \\;", "find . -exec sudo -u root rm {} +"):
        assert detect_dangerous_command(cmd)[:2] == (True, rm_key), cmd
    for cmd in ("find . -exec '{}' \\;", "find . -exec sh '{}' \\;"):
        assert detect_dangerous_command(cmd)[0] is True, cmd
    for cmd in ("find . -exec grep rm {} +", "find . -exec sudo cat {} +", "find . -exec chmod +x {} +"):
        assert detect_dangerous_command(cmd)[0] is False, cmd
