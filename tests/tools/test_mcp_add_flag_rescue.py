"""Tests for cmd_mcp_add's REMAINDER --env/--connect-timeout rescue.

``mcp add --args`` is argparse.REMAINDER: tokens after the child command
argv land in cmd_args — including Hermes' own flags appended after it.
Before the rescue, ``--env KEY=VALUE`` and ``--connect-timeout N`` given
after the child argv were written verbatim into config args: the MCP
server ignored them (env never set!) and a literal secret leaked into
config.yaml. These tests pin the slot-rescue contract.
"""
import sys
import types

from hermes_cli import mcp_config as mc


class FakeArgs:
    def __init__(self, cmd_args, env=None, connect_timeout=None):
        self.name = "x"
        self.url = None
        self.mcp_command = "node"
        self.args = list(cmd_args)
        self.auth = None
        self.preset = None
        self.env = env or []
        self.connect_timeout = connect_timeout


@pytest.fixture
def stub_helpers(monkeypatch):
    """Stub everything cmd_mcp_add does after building server_config."""
    captured = {}

    def fake_validate(name, cfg):
        captured["cfg"] = cfg
        return ["STOP"]  # force early return right after validation

    monkeypatch.setattr(mc, "_warning", lambda *a, **k: None)
    monkeypatch.setattr(mc, "_error", lambda *a, **k: (_ for _ in ()).throw(AssertionError(str(a))))
    monkeypatch.setattr(mc, "_parse_env_assignments", lambda raw: dict(kv.split("=", 1) for kv in (raw or [])))
    monkeypatch.setattr(
        mc, "_apply_mcp_preset",
        lambda *a, **k: (k.get("url"), k.get("command"), list(k.get("cmd_args") or []), None),
    )
    monkeypatch.setattr(mc, "validate_mcp_server_entry", fake_validate)
    monkeypatch.setattr(mc, "_get_mcp_servers", lambda: {})
    return captured


def test_env_after_child_argv_is_rescued(stub_helpers):
    mc.cmd_mcp_add(FakeArgs(["-y", "pkg@1", "--env", "K=secret123"]))
    assert stub_helpers["cfg"]["args"] == ["-y", "pkg@1"]
    assert stub_helpers["cfg"]["env"] == {"K": "secret123"}


def test_connect_timeout_after_child_argv_is_rescued(stub_helpers):
    mc.cmd_mcp_add(FakeArgs(["-y", "pkg", "--connect-timeout", "90"]))
    assert stub_helpers["cfg"]["args"] == ["-y", "pkg"]
    assert stub_helpers["cfg"]["connect_timeout"] == 90.0


def test_env_equals_form_is_rescued(stub_helpers):
    mc.cmd_mcp_add(FakeArgs(["pkg", "--env=K=v"]))
    assert stub_helpers["cfg"]["env"] == {"K": "v"}


def test_child_flags_are_not_stolen(stub_helpers):
    """A --connect-timeout whose value is not numeric belongs to the child."""
    mc.cmd_mcp_add(FakeArgs(["pkg", "--connect-timeout", "wide"]))
    assert stub_helpers["cfg"]["args"] == ["pkg", "--connect-timeout", "wide"]
    assert "connect_timeout" not in stub_helpers["cfg"]


def test_no_trailing_flags_no_op(stub_helpers):
    mc.cmd_mcp_add(FakeArgs(["-y", "pkg"]))
    assert stub_helpers["cfg"]["args"] == ["-y", "pkg"]
    assert "env" not in stub_helpers["cfg"]
