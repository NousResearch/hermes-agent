"""Profile homes (~/.hermes/profiles/<name>/) carry their own config.yaml/.env and get the same
terminal-side write coverage as the default root (#79030)."""

import os

import pytest

from tools.approval_detection import detect_dangerous_command


@pytest.mark.parametrize(
    "command",
    [
        "sed -i s/manual/off/ ~/.hermes/profiles/work/config.yaml",
        "sed --in-place s/x/y/ $HOME/.hermes/profiles/work/config.yaml",
        "perl -pi -e s/x/y/ ~/.hermes/profiles/work/.env",
        "echo 'approvals: {mode: off}' | tee ~/.hermes/profiles/work/config.yaml",
        "echo 'approvals: {mode: off}' > ~/.hermes/profiles/work/config.yaml",
        "cp /tmp/evil.yaml ~/.hermes/profiles/work/config.yaml",
        "echo KEY=1 >> ~/.hermes/profiles/work/.env",
    ],
)
def test_profile_config_and_env_writes_need_approval(command):
    dangerous, _key, _desc = detect_dangerous_command(command)
    assert dangerous, command


def test_absolute_profile_path_needs_approval():
    home = os.path.expanduser("~")
    dangerous, _key, _desc = detect_dangerous_command(f"sed -i s/x/y/ {home}/.hermes/profiles/work/config.yaml")
    assert dangerous


@pytest.mark.parametrize(
    "command",
    [
        "cat ~/.hermes/profiles/work/config.yaml",
        "sed -i s/x/y/ ~/project/profiles/work/settings.yaml",
    ],
)
def test_reads_and_unrelated_files_stay_unflagged(command):
    dangerous, _key, _desc = detect_dangerous_command(command)
    assert not dangerous, command
