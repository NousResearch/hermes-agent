"""The MCP consent row's launch line: what approving runs or contacts.

Contracts:
- the card payload names the command line / URL before consent; the model's result does not
- credentials in args and URLs never reach the line, and control characters are flattened
"""

import json
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
        url=None, command="npx", args=["-y", "@acme/mcp", "--api-key=sk-live-abcdef1234567890XYZ"]))
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
])
def test_launch_line_never_carries_a_credential(url, command, args, secret):
    line = launch_target(url, command, args)["target"]
    assert secret not in line
    assert len(line) <= MAX_TARGET_CHARS


def test_launch_line_keeps_markup_as_text_quotes_spaced_args_and_flattens_controls():
    line = launch_target(None, "bash\x1b]0;pwned\x07", ["<img src=x onerror=alert(1)>", "my file.js"])["target"]
    assert line == "bash ]0;pwned '<img src=x onerror=alert(1)>' 'my file.js'"
    assert launch_target("https://mcp.linear.app/sse", "node", [])["kind"] == "remote"
