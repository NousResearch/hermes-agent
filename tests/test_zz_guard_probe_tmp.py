"""Ad-hoc verification: gateway-kill-shaped commands must still be blocked."""
import subprocess

import pytest


@pytest.mark.parametrize(
    "cmd",
    [
        ["pkill", "-f", "hermes-gateway"],
        ["killall", "hermes-gateway"],
        ["bash", "-c", "pkill -f hermes-gateway"],
        ["env", "X=1", "pkill", "-f", "hermes-gateway"],
        ["sudo", "pkill", "-f", "hermes-gateway"],
        ["nohup", "pkill", "-f", "hermes-gateway"],
        "pkill -f hermes-gateway",
    ],
)
def test_gateway_kill_still_blocked(cmd):
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(cmd)


def test_innocent_argv_not_blocked(tmp_path):
    p = tmp_path / "skill"
    p.write_text("hello")
    r = subprocess.run(["cat", str(p)], capture_output=True, text=True)
    assert r.stdout == "hello"


@pytest.mark.parametrize("cmd", ["pkill -f hermes-gateway", "bash -c \"pkill -f hermes-gateway\""])
def test_gateway_kill_shell_true_blocked(cmd):
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(cmd, shell=True)
