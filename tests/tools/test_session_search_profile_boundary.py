"""Real profile-home boundary checks for session_search."""

import json

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_state import SessionDB
from tools.registry import registry
from tools.session_search_tool import session_search


@pytest.fixture
def homes(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    alpha = root / "profiles" / "alpha"
    beta = root / "profiles" / "beta"
    for home in (root, alpha, beta):
        home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    databases = {}
    for name, home in (("alpha", alpha), ("beta", beta)):
        db = SessionDB(home / "state.db")
        db.create_session(f"{name}-session", source="cli")
        db.append_message(f"{name}-session", role="user", content=f"{name} private marker")
        db._conn.commit()
        databases[name] = db
    yield alpha, beta, databases
    for db in databases.values():
        db.close()


def _call(home, db, **args):
    token = set_hermes_home_override(home)
    try:
        return json.loads(registry.dispatch("session_search", args, db=db))
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize("args", [
    {"query": "private", "profile": "beta"},
    {"profile": "beta"},
    {"session_id": "beta-session", "profile": "beta"},
    {"session_id": "beta-session", "around_message_id": 1, "profile": "beta"},
    {"session_id": "beta/beta-session"},
    {"session_id": "@session:beta/beta-session"},
])
def test_target_opt_out_blocks_every_foreign_shape_before_open(homes, monkeypatch, args):
    alpha, beta, dbs = homes
    (beta / "config.yaml").write_text("session_search:\n  allow_cross_profile: false\n", encoding="utf-8")
    monkeypatch.setattr("tools.session_search_tool._resolve_profile_db",
                        lambda _profile: pytest.fail("foreign database must not open"))
    result = _call(alpha, dbs["alpha"], **args)
    assert result["success"] is False
    assert "Cross-profile session search is disabled" in result["error"]


def test_multiplex_a_to_b_to_a_and_self_search(homes):
    alpha, beta, dbs = homes
    # The historical default still permits an explicit foreign read.
    assert _call(alpha, dbs["alpha"], query="private", profile="beta")["success"]
    (beta / "config.yaml").write_text("session_search:\n  allow_cross_profile: false\n", encoding="utf-8")
    assert not _call(alpha, dbs["alpha"], query="private", profile="beta")["success"]
    own_beta = _call(beta, dbs["beta"], query="private")
    assert own_beta["success"] and [r["session_id"] for r in own_beta["results"]] == ["beta-session"]
    assert not _call(beta, dbs["beta"], query="private", profile="alpha")["success"]
    own_alpha = _call(alpha, dbs["alpha"], query="private")
    assert own_alpha["success"] and [r["session_id"] for r in own_alpha["results"]] == ["alpha-session"]


def test_lazy_database_path_uses_current_profile(homes, monkeypatch):
    import hermes_state

    alpha, beta, _ = homes
    # The hermetic suite pins DEFAULT_DB_PATH to its global fixture home.
    # Restore the production dynamic-home path for this multiplex test.
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    (beta / "config.yaml").write_text("session_search:\n  allow_cross_profile: false\n", encoding="utf-8")
    token = set_hermes_home_override(alpha)
    try:
        own = json.loads(session_search(query="private"))
        denied = json.loads(session_search(query="private", profile="beta"))
    finally:
        reset_hermes_home_override(token)
    assert own["success"] and [r["session_id"] for r in own["results"]] == ["alpha-session"]
    assert denied["success"] is False


def test_managed_policy_wins_and_malformed_managed_file_fails_closed(homes, monkeypatch, tmp_path):
    alpha, beta, dbs = homes
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    (alpha / "config.yaml").write_text("session_search:\n  allow_cross_profile: true\n", encoding="utf-8")
    (beta / "config.yaml").write_text("session_search:\n  allow_cross_profile: true\n", encoding="utf-8")
    policy = managed / "config.yaml"
    policy.write_text("session_search:\n  allow_cross_profile: false\n", encoding="utf-8")
    assert not _call(alpha, dbs["alpha"], profile="beta")["success"]
    policy.write_text("session_search: [invalid\n", encoding="utf-8")
    assert not _call(alpha, dbs["alpha"], profile="beta")["success"]
    policy.write_text("session_search:\n  allow_cross_profile: null\n", encoding="utf-8")
    assert not _call(alpha, dbs["alpha"], profile="beta")["success"]


def test_denied_read_neither_expands_target_secrets_nor_changes_its_files(homes):
    alpha, beta, dbs = homes
    (alpha / ".env").write_text("TOKEN=alpha-secret\n", encoding="utf-8")
    (beta / ".env").write_text("TOKEN=beta-secret\n", encoding="utf-8")
    (beta / "config.yaml").write_text(
        "providers:\n  example:\n    api_key: ${TOKEN}\n"
        "session_search:\n  allow_cross_profile: false\n",
        encoding="utf-8",
    )
    before = {path.relative_to(beta) for path in beta.rglob("*")}
    assert not _call(alpha, dbs["alpha"], profile="beta")["success"]
    assert {path.relative_to(beta) for path in beta.rglob("*")} == before


def test_explicit_null_policy_denies_cross_profile_read(homes):
    alpha, beta, dbs = homes
    (beta / "config.yaml").write_text("session_search:\n  allow_cross_profile: null\n", encoding="utf-8")
    assert not _call(alpha, dbs["alpha"], profile="beta")["success"]


def test_caller_opt_out_and_unreadable_policy_fail_closed(homes):
    alpha, beta, dbs = homes
    (alpha / "config.yaml").write_text("session_search:\n  allow_cross_profile: false\n", encoding="utf-8")
    assert not _call(alpha, dbs["alpha"], profile="beta")["success"]
    assert not _call(beta, dbs["beta"], profile="alpha")["success"]
    (alpha / "config.yaml").write_text("session_search: [invalid\n", encoding="utf-8")
    assert not _call(beta, dbs["beta"], profile="alpha")["success"]
    assert _call(alpha, dbs["alpha"], query="private")["success"]


def test_function_and_registry_paths_share_boundary(homes):
    alpha, beta, dbs = homes
    (beta / "config.yaml").write_text("session_search:\n  allow_cross_profile: false\n", encoding="utf-8")
    token = set_hermes_home_override(alpha)
    try:
        result = json.loads(session_search(db=dbs["alpha"], profile="beta"))
    finally:
        reset_hermes_home_override(token)
    assert result["success"] is False
