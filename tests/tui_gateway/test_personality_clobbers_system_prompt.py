"""Reproduction: /personality (desktop config.set) clobbers agent.system_prompt.

The desktop/TUI backend exposes a `config.set key=personality` RPC
(tui_gateway/server.py, the `@method("config.set")` handler). When the user
selects a personality (the desktop "assistant style" picker routes here), the
handler does:

    _write_config_key("display.personality", pname)
    _write_config_key("agent.system_prompt", new_prompt)   # <-- clobbers manual text

`agent.system_prompt` is the user's GLOBAL manual system prompt (read by
`/prompt` and the CLI's ephemeral-prompt path). Overwriting it with a
personality's text destroys the manual prompt. The sibling in-chat slash path
(server.py `/personality` handler) does NOT touch `agent.system_prompt` -- it
only sets the in-session `agent.ephemeral_system_prompt` -- so the two paths
disagree.

Contract tested here:
1. A pre-existing manual `agent.system_prompt` must survive selecting a
   personality via `config.set key=personality` (the global manual prompt is
   a SEPARATE concept from the per-session personality overlay).
2. Selecting a different personality must not leave a stale personality's
   text in `agent.system_prompt`.

This test drives the REAL `_write_config_key` / `_save_cfg` against a temp
HERMES_HOME config.yaml (no mocks of the persistence layer), so the captured
config file is genuine proof of the write.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import tui_gateway.server as server
import yaml


MANUAL_PROMPT = "manual_prompt_1"
PERSONALITY_1 = "personality_1"
PERSONALITY_2 = "personality_2"


def _seed_config(home: str) -> None:
    cfg = {
        "agent": {
            "system_prompt": MANUAL_PROMPT,
            "personalities": {
                "personality_1": PERSONALITY_1,
                "personality_2": PERSONALITY_2,
            },
        },
        "display": {"personality": "none"},
    }
    with open(os.path.join(home, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)


def _read_saved_system_prompt(home: str) -> str:
    with open(os.path.join(home, "config.yaml"), encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return (data.get("agent", {}) or {}).get("system_prompt", "")


def _set(params: dict) -> dict:
    return server._methods["config.set"]("rid-1", params)


def _make_session(session_id: str):
    """Mirror a live desktop session: real agent namespace + history buffer."""
    agent = MagicMock()
    agent.ephemeral_system_prompt = None
    return {
        "session_key": session_id,
        "agent": agent,
        "history": [],
        "history_version": 0,
        "history_lock": MagicMock(__enter__=lambda self: None, __exit__=lambda self, *a: None),
    }


def _run(home: str, session_id: str = ""):
    """Select `personality_1` via the desktop personality RPC.

    We stub the live-session pivot (_apply_personality_to_session) because the
    bug under test is the *config write* at server.py:11328-11329, not the
    in-session marker injection. Keeping it real would require a fully-built
    agent; the write happens before it regardless.
    """
    sid = session_id or "s1"
    session = _make_session(sid)
    with (
        patch.dict(
            server._sessions, {sid: session}, clear=False
        ),
        patch.object(server, "_apply_personality_to_session", return_value=(False, None)),
        # The handler calls these for the live-session pivot; not under test.
        patch.object(server, "_persist_live_session_runtime"),
        patch.object(server, "_emit"),
    ):
        resp = _set({"key": "personality", "value": "personality_1", "session_id": sid})
    return resp


def test_personality_selection_preserves_manual_system_prompt(tmp_path, monkeypatch):
    """Selecting a personality must NOT overwrite the manual agent.system_prompt."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Force the server module to pick up the temp home for config reads/writes.
    monkeypatch.setattr(server, "_hermes_home", Path(tmp_path))
    monkeypatch.setattr(server, "_cfg_path", None)
    monkeypatch.setattr(server, "_cfg_cache", None)
    _seed_config(str(tmp_path))

    resp = _run(str(tmp_path))

    assert resp["result"]["value"] == "personality_1", resp
    # THE BUG: the manual system_prompt was overwritten by the personality.
    saved = _read_saved_system_prompt(str(tmp_path))
    assert saved == MANUAL_PROMPT, (
        "agent.system_prompt was clobbered by the personality selection.\n"
        f"  expected (manual): {MANUAL_PROMPT!r}\n"
        f"  got (personality): {saved!r}"
    )


def test_switching_personality_leaves_no_stale_text(tmp_path, monkeypatch):
    """After choosing `personality_2`, agent.system_prompt must retain the manual prompt.

    The bug: selecting any personality overwrites agent.system_prompt with that
    personality's text. So after `personality_1` then `personality_2`, the field
    holds `personality_2`'s text -- still NOT the user's manual prompt. The field
    should have been left untouched (the personality overlay belongs in the
    in-session ephemeral prompt, not the durable global system prompt).
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_hermes_home", Path(tmp_path))
    monkeypatch.setattr(server, "_cfg_path", None)
    monkeypatch.setattr(server, "_cfg_cache", None)
    _seed_config(str(tmp_path))

    _run(str(tmp_path), session_id="a")
    # Now switch to a different personality.
    session = _make_session("a")
    with (
        patch.dict(
            server._sessions, {"a": session}, clear=False
        ),
        patch.object(server, "_apply_personality_to_session", return_value=(False, None)),
        patch.object(server, "_persist_live_session_runtime"),
        patch.object(server, "_emit"),
    ):
        resp = _set({"key": "personality", "value": "personality_2", "session_id": "a"})

    assert resp["result"]["value"] == "personality_2", resp
    saved = _read_saved_system_prompt(str(tmp_path))
    assert saved == MANUAL_PROMPT, (
        "agent.system_prompt still holds a personality's text after switching "
        f"personalities; the manual prompt was lost.\n"
        f"  expected (manual): {MANUAL_PROMPT!r}\n"
        f"  got: {saved!r}"
    )
