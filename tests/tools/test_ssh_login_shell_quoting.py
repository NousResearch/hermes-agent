"""The SSH transport must survive a non-POSIX *login* shell.

``ssh`` joins argv into one string and hands it to the remote user's login shell. ``shlex.quote``
produces POSIX single-quote escaping (``'\\''``), and fish decodes ``\\`` and ``\'`` inside single
quotes differently, so a payload carrying any quote leaks out of the argv:

    fish: Unsupported use of '='. In fish, please use 'set __hermes_ec $?'

Every file tool quotes its paths, so over SSH against such a host ``read_file``/``search_files``/
``patch`` failed 100% of the time. The payload now travels base64-encoded and is executed from a
temp file by ``bash``: no quotes, no ``$``, no backslashes at the login-shell layer.

These are behaviour contracts: the carrier must round-trip arbitrary shell text and must not hand
a single quote to the login shell.
"""
from __future__ import annotations

import base64
import re

import pytest

from tools.environments.ssh import _remote_script

# Text that breaks a POSIX-quoted payload under fish: quotes, `=`, `$`, backslashes, newlines.
HOSTILE = (
    "if [ -f '/etc/hostname' ]; then\n"
    "    echo \"a=b $HOME\"\n"
    "    printf '%s\\n' 'it'\\''s here'\n"
    "fi\n"
)


def _embedded_payload(script: str) -> str:
    """The base64 word the carrier asks the remote host to decode."""
    match = re.search(r"printf %s ([A-Za-z0-9+/=]+)", script)
    assert match, f"carrier does not embed a base64 payload: {script!r}"
    return base64.b64decode(match.group(1)).decode("utf-8")


@pytest.mark.parametrize("keep_stdin", [True, False])
def test_payload_round_trips_verbatim(keep_stdin: bool) -> None:
    script = _remote_script(HOSTILE, keep_stdin=keep_stdin)
    assert _embedded_payload(script) == HOSTILE


@pytest.mark.parametrize("keep_stdin", [True, False])
@pytest.mark.parametrize("login", [True, False])
def test_login_shell_never_sees_a_single_quote(login: bool, keep_stdin: bool) -> None:
    script = _remote_script(HOSTILE, login=login, keep_stdin=keep_stdin)
    assert "'" not in script, "a single quote reaches the login shell and fish will re-parse it"


def test_keeps_stdin_free_when_requested() -> None:
    # A temp-file carrier leaves the command's stdin for stdin_data.
    with_stdin = _remote_script("cat", keep_stdin=True)
    without_stdin = _remote_script("true", keep_stdin=False)
    assert "mktemp" in with_stdin
    assert "mktemp" not in without_stdin
    assert "bash" in without_stdin
