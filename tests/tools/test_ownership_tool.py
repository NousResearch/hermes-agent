"""ownership tool: permission matrix, target resolution, key validation.

The tool exists so non-admins can administer ownership without a shell, so the
load-bearing assertions are (a) the acting identity can only come from session
contextvars and (b) mutations still require owner-or-admin.
"""
import asyncio
import json

import pytest

import agent.automation_ownership as ao
import tools.ownership_tool as ot
from agent.automation_ownership import Identity

ALICE = Identity("slack", "U_ALICE", "Alice")
BOB = Identity("slack", "U_BOB", "Bob")
ADMIN = Identity("slack", "U_ADMIN", "Admin")

_DIRECTORY = {"U_ALICE": "Alice", "U_BOB": "Bob", "U_ADMIN": "Admin"}

# Captured before the autouse fixture stubs them, so the tests that exercise
# the real admin resolution / directory lookup can opt back in.
_real_is_admin = ot._is_admin
_real_user_directory = ot._user_directory


def _call(**args):
    return json.loads(asyncio.run(ot._ownership_handler(args)))


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    """Default wiring: Alice acting, a known directory, nobody an admin, no DMs."""
    monkeypatch.setattr(ot, "_user_directory", lambda platform: dict(_DIRECTORY))
    monkeypatch.setattr(ot, "_is_admin", lambda identity: False)
    monkeypatch.setattr(ao, "current_identity", lambda: ALICE)
    monkeypatch.setattr(ao, "_send_dm", lambda *a, **k: True)


def _owned_by(owner=ALICE, key="cron:j1"):
    ao._put_record(key, {
        "kind": "cron",
        "owner": {"platform": owner.platform, "user_id": owner.user_id,
                  "display_name": owner.display_name},
        "collaborators": [],
        "source": "creator",
    })
    return key


# --------------------------------------------------------------------------- #
# Identity is not forgeable
# --------------------------------------------------------------------------- #
def test_no_identity_refuses_every_action(monkeypatch):
    monkeypatch.setattr(ao, "current_identity", lambda: None)
    _owned_by()
    for action in ("list", "show", "claim", "transfer", "collab_add", "collab_remove"):
        out = _call(action=action, key="cron:j1", user="U_BOB", to_user="U_BOB")
        assert "identity" in out["error"], action


def test_claim_records_the_session_user_not_an_argument():
    # There is no argument that can name a different owner; extras are ignored.
    out = _call(action="claim", key="cron:new", user="U_BOB", to_user="U_BOB")
    assert out["ok"] is True
    assert ao.get_record("cron:new")["owner"]["user_id"] == "U_ALICE"


def test_disabled_ownership_short_circuits(monkeypatch):
    monkeypatch.setattr(ao, "is_enabled", lambda: False)
    assert "disabled" in _call(action="list")["error"]


# --------------------------------------------------------------------------- #
# claim / show / list
# --------------------------------------------------------------------------- #
def test_claim_unowned_succeeds_and_audits(monkeypatch):
    seen = []
    monkeypatch.setattr(ao, "record_ownership_change",
                        lambda action, key, actor, **kw: seen.append((action, key)))
    out = _call(action="claim", key="skill:weekly-report")
    assert out["ok"] is True and out["owner"] == "Alice"
    assert seen == [("automation_claim", "skill:weekly-report")]


def test_claim_already_owned_names_the_owner():
    _owned_by(BOB)
    out = _call(action="claim", key="cron:j1")
    assert "Bob" in out["error"]
    assert ao.get_record("cron:j1")["owner"]["user_id"] == "U_BOB"


def test_show_unowned_and_owned():
    assert _call(action="show", key="cron:ghost")["owned"] is False
    _owned_by(BOB)
    out = _call(action="show", key="cron:j1")
    assert out["owned"] is True and out["owner"] == "Bob"


def test_list_is_self_scoped():
    _owned_by(ALICE, "cron:mine")
    _owned_by(BOB, "cron:theirs")
    out = _call(action="list")
    assert out["owned"] == ["cron:mine"]


def test_list_for_another_user_is_admin_only(monkeypatch):
    _owned_by(BOB, "cron:theirs")
    assert "admin" in _call(action="list", user="U_BOB")["error"]

    monkeypatch.setattr(ot, "_is_admin", lambda identity: True)
    assert _call(action="list", user="U_BOB")["owned"] == ["cron:theirs"]


# --------------------------------------------------------------------------- #
# transfer / collaborators — owner-or-admin
# --------------------------------------------------------------------------- #
def test_transfer_by_owner_notifies_both_parties(monkeypatch):
    _owned_by(ALICE)
    dms = []
    monkeypatch.setattr(ao, "_send_dm",
                        lambda platform, user_id, msg: dms.append(user_id) or True)
    monkeypatch.setattr(ao, "notify_enabled", lambda: True)

    out = _call(action="transfer", key="cron:j1", to_user="Bob")
    assert out["ok"] is True
    assert ao.get_record("cron:j1")["owner"]["user_id"] == "U_BOB"
    # Alice acted, so only Bob is notified.
    assert dms == ["U_BOB"]


def test_transfer_by_non_owner_refused():
    _owned_by(BOB)
    assert "owner" in _call(action="transfer", key="cron:j1", to_user="U_ALICE")["error"].lower()
    assert ao.get_record("cron:j1")["owner"]["user_id"] == "U_BOB"


def test_transfer_by_admin_allowed(monkeypatch):
    monkeypatch.setattr(ot, "_is_admin", lambda identity: True)
    monkeypatch.setattr(ao, "current_identity", lambda: ADMIN)
    _owned_by(BOB)
    assert _call(action="transfer", key="cron:j1", to_user="U_ALICE")["ok"] is True


def test_collab_add_and_remove_by_owner():
    _owned_by(ALICE)
    assert _call(action="collab_add", key="cron:j1", user="Bob")["ok"] is True
    assert ao.get_record("cron:j1")["collaborators"][0]["user_id"] == "U_BOB"
    assert _call(action="collab_remove", key="cron:j1", user="U_BOB")["ok"] is True
    assert ao.get_record("cron:j1")["collaborators"] == []


def test_collab_add_by_non_owner_refused(monkeypatch):
    """The hole the tool must not inherit from the CLI: Bob adding himself."""
    _owned_by(ALICE)
    monkeypatch.setattr(ao, "current_identity", lambda: BOB)
    out = _call(action="collab_add", key="cron:j1", user="U_BOB")
    assert "owner" in out["error"].lower()
    assert ao.get_record("cron:j1")["collaborators"] == []


def test_mutating_an_unowned_key_is_refused():
    for action, kwargs in (("transfer", {"to_user": "U_BOB"}), ("collab_add", {"user": "U_BOB"})):
        out = _call(action=action, key="cron:ghost", **kwargs)
        assert "no recorded owner" in out["error"]


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ["", "cron", "nonsense:x", "cron:", ":j1"])
def test_bad_keys_rejected(key):
    assert "error" in _call(action="show", key=key)


def test_unknown_action_rejected():
    assert "Unknown action" in _call(action="wat")["error"]


def test_unknown_target_is_an_error_not_a_write():
    _owned_by(ALICE)
    assert "No user matching" in _call(action="transfer", key="cron:j1", to_user="U_GHOST")["error"]
    assert ao.get_record("cron:j1")["owner"]["user_id"] == "U_ALICE"


def test_ambiguous_name_is_an_error(monkeypatch):
    monkeypatch.setattr(ot, "_user_directory",
                        lambda platform: {"U1": "Sam", "U2": "sam"})
    _owned_by(ALICE)
    assert "more than one" in _call(action="transfer", key="cron:j1", to_user="Sam")["error"]


@pytest.mark.parametrize("raw", ["U_BOB", "Bob", "bob", "@Bob", "<@U_BOB>", "<@U_BOB|bob>"])
def test_target_resolution_accepts_ids_names_and_mentions(raw):
    ident, err = ot._resolve_target(raw, "slack")
    assert err == "" and ident.user_id == "U_BOB"


def test_target_resolution_without_a_directory_accepts_raw_id(monkeypatch):
    monkeypatch.setattr(ot, "_user_directory", lambda platform: {})
    ident, err = ot._resolve_target("U_WHOEVER", "slack")
    assert err == "" and ident.user_id == "U_WHOEVER"


def test_user_directory_reads_real_config(tmp_path, monkeypatch):
    """Exercises the unstubbed directory lookup end to end.

    Every other test here stubs ``_user_directory``, which hid a real bug: the
    gateway config loader did not bridge ``user_names`` into the platform
    ``extra``, so the lookup found no directory and fell through to accepting
    any string as a user id.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "slack:\n"
        "  user_roles:\n"
        "    U_ALICE: operator\n"
        "    U_NONAME: readonly\n"
        "  user_names:\n"
        "    U_ALICE: Alice\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("gateway.tool_access._config_cache", {})
    monkeypatch.setattr(ot, "_user_directory", _real_user_directory)  # undo the autouse stub

    assert ot._user_directory("slack") == {"U_ALICE": "Alice", "U_NONAME": ""}
    # A named user resolves by name; a role-only user still resolves by id.
    assert ot._resolve_target("Alice", "slack")[0].user_id == "U_ALICE"
    assert ot._resolve_target("U_NONAME", "slack")[0].user_id == "U_NONAME"
    # And an unknown id is now rejected rather than accepted verbatim.
    assert ot._resolve_target("U_GHOST", "slack")[0] is None


# --------------------------------------------------------------------------- #
# Admin resolution fails closed
# --------------------------------------------------------------------------- #
def test_is_admin_false_when_rbac_disabled(monkeypatch):
    class _Policy:
        enabled = False

        def grant_for(self, *a):
            return frozenset({"*"})

    monkeypatch.setattr("gateway.tool_access.policy_for_platform", lambda p: _Policy())
    assert _real_is_admin(ALICE) is False


def test_is_admin_true_for_star_grant(monkeypatch):
    class _Policy:
        enabled = True

        def grant_for(self, *a):
            return frozenset({"*"})

    monkeypatch.setattr("gateway.tool_access.policy_for_platform", lambda p: _Policy())
    assert _real_is_admin(ALICE) is True


def test_is_admin_false_when_resolution_raises(monkeypatch):
    def boom(_p):
        raise RuntimeError("config gone")

    monkeypatch.setattr("gateway.tool_access.policy_for_platform", boom)
    assert _real_is_admin(ALICE) is False
