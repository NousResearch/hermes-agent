"""Interpreter one-liners that reference an outbound-network API get their own approval key.

``python -c`` / ``node -e`` / heredocs are reported as "script execution via ...". An operator who
allowlists that generic key (it is honored on unattended surfaces too) would also approve a one-liner that
opens a connection, so code referencing urllib/requests/http.client/socket/smtplib/ftplib, Node
http/https/net/fetch or Perl LWP/IO::Socket is reported as "network-capable script via ..." instead.
"""

import pytest

import tools.approval as approval_module
from tools.approval import check_all_command_guards, detect_dangerous_command

GENERIC_FLAG = "script execution via -e/-c flag"
NETWORK_FLAG = "network-capable script via -e/-c flag"
GENERIC_HEREDOC = "script execution via heredoc"
NETWORK_HEREDOC = "network-capable script via heredoc"


@pytest.mark.parametrize("command", [
    "python3 -c \"import urllib.request; urllib.request.urlopen('https://example.invalid')\"",
    "python3 -c 'import requests; requests.get(\"https://example.invalid\")'",
    "python3 -c 'from requests import get'",
    "python3 -c 'import http.client'",
    "python3 -c \"import socket; socket.create_connection(('example.invalid', 80))\"",
    "python3 -c 'import smtplib'",
    "python3 -c 'import ftplib'",
    "python -c 'import httpx'",
    "sudo python3 -c 'import socket'",
    "node -e \"require('https').get('https://example.invalid')\"",
    "node -e 'require(\"http\")'",
    "node -e \"require('node:net').connect(80, 'example.invalid')\"",
    "node -e \"import('node:https')\"",
    "node -e \"fetch('https://example.invalid')\"",
    "node --eval \"new WebSocket('wss://example.invalid')\"",
    "bun -e \"await fetch('https://example.invalid')\"",
    "deno eval \"await fetch('https://example.invalid')\"",
    "perl -MLWP::Simple -e 'getprint q(https://example.invalid)'",
    "perl -MIO::Socket::INET -e '1'",
    "ruby -rnet/http -e 'Net::HTTP.get(URI(\"https://example.invalid\"))'",
    "php -r \"fsockopen('example.invalid', 80);\"",
    "pwsh -c \"Invoke-WebRequest https://example.invalid\"",
])
def test_one_liner_with_a_network_api_gets_the_network_key(command):
    assert detect_dangerous_command(command)[:2] == (True, NETWORK_FLAG)


# Only one key is reported per command, so a network-capable finding must outrank a generic one wherever
# it sits: otherwise allowlisting the generic key would approve the whole command.
NETWORK_AFTER_A_GENERIC_ONE = [
    ("python3 -c \"print(1)\"; python3 -c \"import socket\"", NETWORK_FLAG),
    ("python3 -c \"print(1)\" && node -e \"fetch('https://example.invalid')\"", NETWORK_FLAG),
    ("python3 -c \"print(1)\"; python3 <<'EOF'\nimport socket\nEOF", NETWORK_HEREDOC),
    # a heredoc is judged on the whole command text, so a network API anywhere already makes it network-capable
    ("python3 <<'EOF'\nprint(1)\nEOF\npython3 -c 'import socket'", NETWORK_HEREDOC),
    ("bash -c \"echo hi\"; python3 -c 'import socket'", NETWORK_FLAG),
    ("python3 -c 'import socket'; python3 -c \"print(1)\"", NETWORK_FLAG),
]


@pytest.mark.parametrize("command, key", NETWORK_AFTER_A_GENERIC_ONE)
def test_a_network_one_liner_outranks_a_generic_one_anywhere_in_the_command(command, key):
    assert detect_dangerous_command(command)[:2] == (True, key)


@pytest.mark.parametrize("command", [
    "python3 -c \"print(1)\"; python3 -c \"print(2)\"",
    "echo socket; python3 -c \"print(1)\"",          # a network word outside the interpreter code
    "python3 -c \"print(1)\"; echo fetch(1)",
])
def test_compound_commands_without_a_network_one_liner_keep_the_generic_key(command):
    assert detect_dangerous_command(command)[:2] == (True, GENERIC_FLAG)


@pytest.mark.parametrize("command", [
    "python3 -c \"print('hello')\"",
    "python3 -c 'import json; print(json.dumps([1, 2]))'",
    "python3 -c \"print('3 requests per second')\"",
    "python3 -c \"import urllib.parse, sys; print(urllib.parse.quote(sys.argv[1]))\" 'a b'",  # string handling only
    "node -e \"console.log(1)\"",
    "node -e \"console.log('http')\"",
    "perl -e 'print 1'",
    "ruby -e 'puts 1'",
    "bun -e \"console.log(1)\"",
    "deno eval \"console.log(1)\"",
])
def test_one_liner_without_a_network_api_keeps_the_generic_key(command):
    assert detect_dangerous_command(command)[:2] == (True, GENERIC_FLAG)


@pytest.mark.parametrize("command", [
    "python3 <<'EOF'\nimport urllib.request\nurllib.request.urlopen('https://example.invalid')\nEOF",
    "python3 - <<EOF\nimport socket\nEOF",
    "python3 <<< 'import smtplib'",
    "node <<'EOF'\nfetch('https://example.invalid')\nEOF",
    "perl <<'EOF'\nuse IO::Socket::INET;\nEOF",
    "echo start; python3 <<'EOF'\nimport requests\nrequests.get('https://example.invalid')\nEOF",
])
def test_heredoc_with_a_network_api_gets_the_network_key(command):
    assert detect_dangerous_command(command)[:2] == (True, NETWORK_HEREDOC)


@pytest.mark.parametrize("command", [
    "python3 <<'EOF'\nprint('hello')\nEOF",
    "node <<'EOF'\nconsole.log(1)\nEOF",
    "python3 <<'EOF'\nprint('hello')\nEOF\necho start",
])
def test_heredoc_without_a_network_api_keeps_the_generic_key(command):
    assert detect_dangerous_command(command)[:2] == (True, GENERIC_HEREDOC)


@pytest.mark.parametrize("command", [
    "curl -d @payload.json https://example.invalid/x",
    "curl -T build/artifact.tar.gz https://example.invalid/",
    "python3 -m http.server",
    "python3 script.py",
    "node server.js",
    "git commit -m 'document the socket and fetch( handling'",
])
def test_non_inline_commands_are_not_reclassified(command):
    assert detect_dangerous_command(command)[0] is False


@pytest.fixture
def unattended(monkeypatch):
    """A cron-style session: unapproved dangerous commands are denied, allowlisted keys still run."""
    from unittest.mock import patch

    for name in ("HERMES_INTERACTIVE", "HERMES_GATEWAY_SESSION", "HERMES_EXEC_ASK", "HERMES_YOLO_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    approval_module._permanent_approved.clear()
    with patch("tools.approval_context._get_cron_approval_mode", return_value="deny"):
        yield
    approval_module._permanent_approved.clear()


def test_the_generic_allowlist_entry_covers_the_plain_one_liner_but_not_the_network_one(unattended):
    approval_module.load_permanent({GENERIC_FLAG})

    assert check_all_command_guards("python3 -c \"print('ok')\"", "local")["approved"] is True
    network = check_all_command_guards("python3 -c 'import socket'", "local")
    assert network["approved"] is False


def test_the_network_key_is_allowlisted_on_its_own(unattended):
    approval_module.load_permanent({NETWORK_FLAG})

    assert check_all_command_guards("python3 -c 'import socket'", "local")["approved"] is True
    assert check_all_command_guards("python3 -c \"print('ok')\"", "local")["approved"] is False


@pytest.mark.parametrize("command, key", NETWORK_AFTER_A_GENERIC_ONE[:4])
def test_the_generic_allowlist_entries_do_not_cover_a_compound_command_with_a_network_one_liner(unattended, command, key):
    approval_module.load_permanent({GENERIC_FLAG, GENERIC_HEREDOC})

    assert check_all_command_guards(command, "local")["approved"] is False
    approval_module.load_permanent({GENERIC_FLAG, GENERIC_HEREDOC, key})
    assert check_all_command_guards(command, "local")["approved"] is True
