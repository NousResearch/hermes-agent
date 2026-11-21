"""`sh -c` keeps parsing options after `-c`; the script is the first operand (`bash -c -e -- 'rm -rf /'`)."""

import pytest

from tools.approval import detect_hardline_command


@pytest.mark.parametrize("command", [
    "bash -c -- 'rm -rf /'",
    "bash -c - 'rm -rf /'",
    "bash -c -e 'rm -rf /'",
    "bash -c -o errexit 'rm -rf /'",
    "sudo sh -c -x -- 'mkfs.ext4 /dev/sda'",
])
def test_options_after_dash_c_do_not_hide_the_script_from_the_hardline_floor(command):
    assert detect_hardline_command(command)[0] is True


@pytest.mark.parametrize("command", ["bash -c -- 'echo hi'", "bash -- script.sh -c 'rm -rf /'"])
def test_harmless_or_non_dash_c_scripts_stay_off_the_floor(command):
    assert detect_hardline_command(command)[0] is False
