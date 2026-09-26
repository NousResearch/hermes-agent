"""The tools[] pin's code identity comes from the live identity reader, not the retired shim.

``tools/mcp_tool_agent.py`` keyed the session pin by checkout/build SHA via
``hermes_cli.build_info.get_code_identity`` — a shim that since 16e99423d9 always answers
``{"sha": None, ..., "source": "unknown"}``. The pin therefore degraded to the release
``__version__`` (``"0.0.0"`` on an unstamped source checkout) for every process, so after a
``hermes update`` the pin's version STILL compared equal and the restore path replayed the
pre-update tools[] schema bytes — the exact staleness d956f0ae57 set out to fix.
"""

import types

from tools import mcp_tool_agent


def _pinned_def(name, description):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {}}}


def _agent(tools):
    return types.SimpleNamespace(
        tools=list(tools),
        valid_tool_names={t["function"]["name"] for t in tools},
        enabled_toolsets=None,
        disabled_toolsets=None,
    )


def test_tool_pin_version_uses_the_live_code_identity(monkeypatch):
    from hermes_cli import version_info

    monkeypatch.setattr(
        version_info, "get_code_identity",
        lambda refresh=False: {"sha": "f" * 40, "short_sha": "f" * 8, "version": "1.2.3", "source": "git"},
    )
    assert mcp_tool_agent.tool_pin_version() == "f" * 40


def test_pin_keyed_by_sha_replays_pinned_bytes_and_a_new_sha_does_not(monkeypatch):
    """End to end through the real seam: a pin keyed by THIS code's sha hands back the session's
    exact bytes; one keyed by another sha (the first process after ``hermes update``) takes the
    current definitions instead."""
    from hermes_cli import version_info

    monkeypatch.setattr(
        version_info, "get_code_identity",
        lambda refresh=False: {"sha": "b" * 40, "short_sha": "b" * 8, "version": "1.2.3", "source": "git"},
    )
    pin = {"version": mcp_tool_agent.tool_pin_version(), "tools": [_pinned_def("read_file", "OLD bytes")]}
    assert pin["version"] == "b" * 40

    same_code = _agent([_pinned_def("read_file", "NEW bytes")])
    mcp_tool_agent.restore_agent_tool_prefix(same_code, pin)
    assert same_code.tools[0]["function"]["description"] == "OLD bytes"  # same sha: pinned bytes stand

    post_update = _agent([_pinned_def("read_file", "NEW bytes")])  # a fresh process on new code
    mcp_tool_agent.restore_agent_tool_prefix(post_update, dict(pin, version="c" * 40))
    assert post_update.tools[0]["function"]["description"] == "NEW bytes"  # other code: current bytes


if __name__ == "__main__":
    raise SystemExit("run via pytest")
