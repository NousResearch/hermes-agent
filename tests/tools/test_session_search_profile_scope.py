"""Cross-profile ``session_search`` reads under one process serving several profiles (#87779).

A secondary profile's turn must not read another profile's transcripts by naming it; the default
(root) profile keeps cross-profile reads so its ``@session:<profile>/<id>`` links resolve.
"""

import json
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_state import SessionDB
from tools.session_search_tool import session_search


def _seed(home: Path, sid: str, text: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    db = SessionDB(home / "state.db")
    try:
        db.create_session(sid, source="telegram")
        db.append_message(sid, role="user", content=text)
    finally:
        db.close()


@pytest.fixture
def two_profiles(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))  # the multiplexer's launch home
    eva = root / "profiles" / "eva"
    _seed(root, "s_owner", "owner secret marker")
    _seed(eva, "s_eva", "eva own marker")
    from agent import secret_scope
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    return root, eva


def _as(home: Path, **kwargs) -> dict:
    token = set_hermes_home_override(home)
    try:
        return json.loads(session_search(**kwargs))
    finally:
        reset_hermes_home_override(token)


def test_secondary_profile_cannot_read_another_profile(two_profiles):
    _root, eva = two_profiles
    for kwargs in ({"profile": "default"}, {"query": "secret", "profile": "default"},
                   {"session_id": "s_owner", "profile": "default"}, {"session_id": "default/s_owner"}):
        result = _as(eva, **kwargs)
        assert result["success"] is False, kwargs
        assert "owner secret marker" not in json.dumps(result) and "default profile" in result["error"]
    # A miss no longer nudges the model towards profile=.
    miss = _as(eva, session_id="s_owner")
    assert miss["success"] is False and "profile=" not in miss["error"]


def test_secondary_profile_still_reads_itself(two_profiles):
    _root, eva = two_profiles
    for kwargs in ({"session_id": "s_eva"}, {"session_id": "s_eva", "profile": "eva"},
                   {"session_id": "eva/s_eva"}):
        result = _as(eva, **kwargs)
        assert result["success"] is True, (kwargs, result)
        assert result["messages"][0]["content"] == "eva own marker"


def test_default_profile_keeps_cross_profile_reads(two_profiles):
    root, _eva = two_profiles
    for kwargs in ({"session_id": "s_eva", "profile": "eva"}, {"session_id": "eva/s_eva"}):
        result = _as(root, **kwargs)
        assert result["success"] is True, (kwargs, result)
        assert result["messages"][0]["content"] == "eva own marker"
    # A bare id that misses stays a miss (#106761), with the hint for the sanctioned form.
    miss = _as(root, session_id="s_eva")
    assert miss["success"] is False and "profile=" in miss["error"]


def test_standalone_secondary_profile_is_unchanged(two_profiles, monkeypatch):
    # A process launched for one profile only (no multiplexing, no routed scope) has no tenant
    # boundary to enforce: the explicit read behaves as before.
    _root, eva = two_profiles
    from agent import secret_scope
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setenv("HERMES_HOME", str(eva))
    result = json.loads(session_search(session_id="s_owner", profile="default"))
    assert result["success"] is True and result["messages"][0]["content"] == "owner secret marker"
