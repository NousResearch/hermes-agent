"""GUI acknowledgement for /learn, /plan, /init (#52085).

The classic CLI and the messaging gateway both print an acknowledgement line
before submitting the builder prompt as a normal turn; the TUI/desktop went
through ``command.dispatch`` and got only ``{"type": "send", "message": ...}``
— no ``notice`` (so no ack line) and no ``display`` (so the clients echoed the
~5 KB model-facing prompt as the user's own bubble). These are the payload
contracts the GUI clients render:

* ``notice``  → system line (ui-tui createSlashHandler.ts, desktop slash.ts)
* ``display`` → the chat bubble text instead of ``message``

Wording is copied verbatim from the gateway acks (gateway/run_inbound.py).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


@pytest.fixture()
def server(hermes_home, monkeypatch):
    # Mocks scoped to the initial import only (see test_protocol.py rationale).
    with patch.dict("sys.modules", {
        "hermes_cli.env_loader": MagicMock(),
        "hermes_cli.banner": MagicMock(),
    }):
        mod = importlib.import_module("tui_gateway.server")
    # Pin config resolution to the isolated home (sibling test files import
    # tui_gateway.server at collection time and freeze _hermes_home to the
    # real home — see test_goal_command.py).
    monkeypatch.setattr(mod, "_hermes_home", hermes_home)
    monkeypatch.setattr(mod, "_cfg_cache", None)
    monkeypatch.setattr(mod, "_cfg_mtime", None)
    monkeypatch.setattr(mod, "_cfg_path", None)
    sid = "sid-prompt-builtin"
    mod._sessions[sid] = {"session_key": sid}
    yield mod, sid
    mod._sessions.clear()
    __import__("tui_gateway.server_requests", fromlist=["x"]).reset_for_tests()


def _dispatch(server, name: str, arg: str | None = "") -> dict:
    mod, sid = server
    resp = mod.handle_request({
        "id": "r1",
        "method": "command.dispatch",
        "params": {"name": name, "arg": arg, "session_id": sid},
    })
    assert "error" not in resp, resp
    return resp["result"]


# /learn ---------------------------------------------------------------------


def test_learn_with_argument_acks_and_labels_the_turn(server):
    result = _dispatch(server, "learn", "the deploy workflow")
    assert result["type"] == "send"
    assert result["notice"] == "Learning a skill from what you described…"
    assert result["display"] == "/learn the deploy workflow"
    # message stays the model-facing builder prompt, NOT what the UI shows.
    assert "[/learn]" in result["message"]
    assert result["message"] != result["notice"]
    assert result["message"] != result["display"]


def test_learn_bare_acks_this_conversation(server):
    result = _dispatch(server, "learn")
    assert result["type"] == "send"
    assert result["notice"] == "Learning a skill from this conversation…"
    assert result["display"] == "/learn"


# /plan ----------------------------------------------------------------------


def test_plan_with_task_acks_and_labels_the_turn(server):
    result = _dispatch(server, "plan", "refactor the parser")
    assert result["type"] == "send"
    assert result["notice"] == "Planning: refactor the parser"
    assert result["display"] == "/plan refactor the parser"
    assert "[/plan" in result["message"]
    assert result["message"] != result["notice"]


def test_plan_bare_acks_conversation_context(server):
    result = _dispatch(server, "plan")
    assert result["notice"] == "Planning from this conversation's context…"
    assert result["display"] == "/plan"


def test_plan_notice_truncates_long_task(server):
    task = "x" * 100
    result = _dispatch(server, "plan", task)
    assert result["notice"] == f"Planning: {'x' * 80}…"


def test_plan_notice_collapses_a_multiline_task(server):
    # The clients render ``notice`` as ONE system line, so a newline in the task
    # must not reach it (nor should runs of spaces/tabs).
    result = _dispatch(server, "plan", "fix the\nparser")
    assert result["notice"] == "Planning: fix the parser"
    assert "\n" not in result["notice"]


def test_plan_truncation_applies_to_the_collapsed_task(server):
    # 98 chars with double-space separators, 74 collapsed: over 80 raw, under 80
    # collapsed — the ellipsis decision and the slice both use the collapsed form.
    raw = "  ".join(["xy"] * 25)
    collapsed = " ".join(raw.split())
    assert len(raw) > 80 >= len(collapsed)
    result = _dispatch(server, "plan", raw)
    assert result["notice"] == f"Planning: {collapsed}"
    assert "…" not in result["notice"]


def test_plan_truncates_the_collapsed_form_when_still_too_long(server):
    # Collapsed length still > 80, so the ellipsis stays, and the cut lands on
    # the collapsed string — not on whitespace-padded raw text.
    raw = "  ".join(["abcdefghij"] * 9)
    collapsed = " ".join(raw.split())
    assert len(collapsed) > 80
    result = _dispatch(server, "plan", raw)
    assert result["notice"] == f"Planning: {collapsed[:80]}…"


# null / whitespace-only arg --------------------------------------------------
# ``CommandDispatchParams.arg`` is ``str | None`` (contracts/tools_commands.py) and
# ``validate_params`` defers type/required checks to handlers (contracts/registry.py), so an
# explicit ``"arg": null`` reaches here. ``params.get("arg", "")`` does NOT default a
# present-but-null key, so the handlers must normalise before touching the string.


def test_learn_with_null_arg_behaves_as_bare(server):
    result = _dispatch(server, "learn", None)
    assert result["type"] == "send"
    assert result["notice"] == "Learning a skill from this conversation…"
    assert result["display"] == "/learn"
    assert "[/learn]" in result["message"]


def test_plan_with_null_arg_behaves_as_bare(server):
    result = _dispatch(server, "plan", None)
    assert result["type"] == "send"
    assert result["notice"] == "Planning from this conversation's context…"
    assert result["display"] == "/plan"


def test_init_with_null_arg_behaves_as_bare(server, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _dispatch(server, "init", None)
    assert result["type"] == "send"
    assert result["notice"] == "Generating AGENTS.md from a project scan…"
    assert result["display"] == "/init"
    assert "[/init]" in result["message"]


def test_whitespace_only_arg_normalizes_to_the_bare_form(server):
    result = _dispatch(server, "learn", "   ")
    assert result["notice"] == "Learning a skill from this conversation…"
    assert result["display"] == "/learn"


def test_surrounding_whitespace_is_stripped_but_inner_text_kept(server):
    multi = "\n  refactor   the parser  \n"
    result = _dispatch(server, "plan", multi)
    assert result["display"] == "/plan refactor   the parser"
    # internal whitespace/newlines reach the builder verbatim, only edges are trimmed
    assert "refactor   the parser" in result["message"]
    assert not result["message"].startswith("\n")


# /init ----------------------------------------------------------------------


def test_init_notice_generates_when_no_agents_md(server, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # build_init_prompt_for_cwd resolves os.getcwd()
    result = _dispatch(server, "init")
    assert result["type"] == "send"
    assert result["notice"] == "Generating AGENTS.md from a project scan…"
    assert result["display"] == "/init"
    assert "[/init]" in result["message"]


def test_init_notice_updates_when_agents_md_exists(server, tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("# Existing\n\nRun `make lint`.\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = _dispatch(server, "init", "keep it terse")
    assert result["notice"] == "Updating AGENTS.md from a project scan…"
    assert result["display"] == "/init keep it terse"
