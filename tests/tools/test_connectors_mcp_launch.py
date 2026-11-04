"""The MCP consent row's launch line: what approving runs or contacts.

Contracts:
- the card payload names the command line / URL before consent; the model's result does not
- credentials in args and URLs never reach the line; invisible and reordering characters show as escapes
"""

import json
import unicodedata
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.connectors import live
from tools.connectors.mcp import apply_answer
from tools.connectors.mcp_launch import MAX_TARGET_CHARS, launch_target
from tools.connectors.tool import manage_connections


@pytest.fixture(autouse=True)
def _clean_live():
    live.reset_for_tests()
    yield
    live.reset_for_tests()


class _Backend:
    def required_env(self, name):
        return []


def test_card_names_the_launch_target_and_the_model_result_does_not():
    entry = SimpleNamespace(post_install="", transport=SimpleNamespace(
        url=None, command="npx", args=["-y", "@acme/mcp", "--api-key=sk-live-abcdef1234567890XYZ"], env={}))
    seen = []

    def callback(payload):
        seen.append(payload)
        skip = json.dumps({"targets": [{"name": "acme", "status": "skipped"}]})
        threading.Timer(0.01, lambda: apply_answer(live.get("s1", payload["op_id"]), skip)).start()

    with patch("tools.connectors.mcp._catalog_names", return_value=["acme"]), \
         patch("tools.connectors.mcp._configured_names", return_value=[]), \
         patch("hermes_cli.mcp_catalog.get_entry", return_value=entry), \
         patch("tools.connectors.run.WATCH_INTERVAL_SECONDS", 0.01):
        out = json.loads(manage_connections({"action": "install", "connectors": [{"name": "acme", "mcp": True}]},
                                            connection_callback=callback, session_id="s1", mcp_backend=_Backend()))

    (offered,) = seen[0]["targets"]
    assert offered["launch"] == {"kind": "stdio", "target": "npx -y @acme/mcp --api-key=***"}
    assert "launch" not in out["targets"][0]


@pytest.mark.parametrize("url,command,args,secret", [
    (None, "uvx", ["srv", "--token", "ghp_abcdefghijklmnopqrstuvwxyz0123456789"], "ghp_abcdef"),
    (None, "uvx", ["srv", "--password=hunter2"], "hunter2"),
    (None, "curl", ["-H", "Authorization: Bearer opaqueopaqueopaque123"], "opaqueopaque"),
    ("https://mcp.example.com/sse?api_key=abcdef1234567890&x=1", None, None, "abcdef1234567890"),
    ("https://user:hunter2pass@mcp.example.com/mcp", None, None, "hunter2pass"),
    ("https://mcp.example.com/v1/tok_ABCdef123456GHIjkl7890/sse", None, None, "ABCdef123456"),
    (None, "node", ["x" * 380, "--api-key=sk-live-abcdef1234567890XYZ"], "abcdef1234567890"),
    (None, "mysql", ["-u", "root", "-p", "hunter2"], "hunter2"),
    (None, "mysql", ["-u", "root", "-phunter2xyz"], "hunter2xyz"),
    (None, "srv", ["-P", "s3cr3tpw"], "s3cr3tpw"),
    (None, "srv", ["-k", "plainkeyvalue"], "plainkeyvalue"),
    (None, "srv", ["-t", "opaque-short-tok"], "opaque-short-tok"),
    (None, "srv", ["-H", "Authorization: Basic dXNlcjpwYXNz"], "dXNlcjpwYXNz"),
    (None, "srv", ["--header=X-Api-Key: plainvalue1"], "plainvalue1"),
    (None, "srv", ["AbCdEf1234567890GhIjKl"], "AbCdEf1234"),
    (None, "srv", ["--config", "eyJhcGlfa2V5IjoiYWJjZGVmZ2hpamtsbW5vcCJ9AifQ=="], "AifQ"),
    (None, "srv", ["--config", "eyJhcGlfa2V5IjoiYWJjZGVmZ2hpamtsbW5vcCJ9AifQ=="], "eyJhcG"),
    (None, "srv", ["--seed=0123456789abcdef0123456789abcdef"], "0123456789abcdef"),
    ("https://mcp.example.com/mcp#opaquefragment", None, None, "opaquefragment"),
    ("https://mcp.example.com/mcp?k=shortval", None, None, "shortval"),
    ("https://mcp.example.com/v1/Ab3dEf9hIj12/sse", None, None, "Ab3dEf9hIj12"),
    ("https://shortuser:pw@mcp.example.com/mcp", None, None, "shortuser"),
])
def test_launch_line_never_carries_a_credential(url, command, args, secret):
    line = launch_target(url, command, args)["target"]
    assert secret not in line
    assert len(line) <= MAX_TARGET_CHARS


def test_launch_line_keeps_markup_as_text_quotes_spaced_args_and_escapes_controls():
    line = launch_target(None, "bash\x1b]0;pwned\x07", ["<img src=x onerror=alert(1)>", "my file.js"])["target"]
    assert line == "bash\\x1b]0;pwned\\x07 '<img src=x onerror=alert(1)>' 'my file.js'"
    assert launch_target("https://mcp.linear.app/sse", "node", [])["kind"] == "remote"


@pytest.mark.parametrize("arg,shown", [
    ("safe\u202egnp.evil", "safe\\u202egnp.evil"),   # RLO: would render reversed
    ("np\u200bx", "np\\u200bx"),                     # zero-width space
    ("a\u2066b\u2069c", "a\\u2066b\\u2069c"),          # bidi isolates
    ("\ufeffx\u00ady", "\\ufeffx\\xady"),               # BOM, soft hyphen
    ("a\u200d\u200e\u200fb", "a\\u200d\\u200e\\u200fb"),  # ZWJ, LRM, RLM
    ("line\nbreak", "'line\\nbreak'"),                # whitespace inside one arg: quoted
])
def test_launch_line_shows_invisible_and_reordering_characters_as_escapes(arg, shown):
    line = launch_target(None, "node", [arg])["target"]
    assert line == f"node {shown}"
    assert all(unicodedata.category(c) != "Cf" for c in line)


@pytest.mark.parametrize("url,command,args,line", [
    (None, "npx", ["-y", "@modelcontextprotocol/server-filesystem", "/home/me/projects", "./data"],
     "npx -y @modelcontextprotocol/server-filesystem /home/me/projects ./data"),
    (None, "uvx", ["mcp-server-git", "--repository", "/srv/repos/hermes-agent-2024", "-v"],
     "uvx mcp-server-git --repository /srv/repos/hermes-agent-2024 -v"),
    (None, "node", ["packages/server2/dist/build/index20240101.js", "--port", "8080"],
     "node packages/server2/dist/build/index20240101.js --port 8080"),
    (None, "mysql", ["-p", "--verbose"], "mysql -p --verbose"),
    ("https://mcp.linear.app/v1/sse", None, None, "https://mcp.linear.app/v1/sse"),
])
def test_launch_line_keeps_ordinary_commands_paths_and_urls(url, command, args, line):
    assert launch_target(url, command, args)["target"] == line


def test_launch_line_names_env_keys_and_cwd_but_never_env_values():
    out = launch_target(None, "npx", ["-y", "@acme/mcp"],
                        env={"NODE_OPTIONS": "--require /x.js", "ACME_TOKEN": "sk-live-abcdef1234567890XYZ"},
                        cwd="/srv/acme")
    assert out == {"kind": "stdio", "target": "npx -y @acme/mcp", "env": ["ACME_TOKEN", "NODE_OPTIONS"],
                   "cwd": "/srv/acme"}
    assert "require" not in json.dumps(out) and "sk-live" not in json.dumps(out)
    assert set(launch_target(None, "npx", [], env={}, cwd="  ")) == {"kind", "target"}


def test_a_string_args_value_is_one_argument_and_still_masked():
    assert launch_target(None, "node", "--api-key=sk-live-abcdef1234567890XYZ")["target"] == "node --api-key=***"
    assert launch_target(None, "node", "server.js")["target"] == "node server.js"


def test_enable_reads_env_and_cwd_from_the_configured_entry():
    from tools.connectors.mcp_launch import launch_for

    cfg = {"acme": {"command": "node", "args": ["srv.js"], "env": {"NODE_OPTIONS": "--require /x.js"}, "cwd": "/w"}}
    with patch("hermes_cli.mcp_catalog.installed_servers", return_value=cfg):
        assert launch_for("enable", "acme") == {"kind": "stdio", "target": "node srv.js", "env": ["NODE_OPTIONS"],
                                                "cwd": "/w"}



def test_an_overlong_line_keeps_its_head_and_its_tail():
    line = launch_target(None, "node", ["srv.js", *["--log-level=info"] * 40, "--exec", "~/.ssh/steal.js"])["target"]
    assert len(line) <= MAX_TARGET_CHARS
    assert line.startswith("node srv.js --log-level=info")
    assert line.endswith("--exec ~/.ssh/steal.js") and " … " in line
