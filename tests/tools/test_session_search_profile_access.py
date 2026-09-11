import json
import pytest
from tools.session_search_tool import check_profile_session_access, session_search
from hermes_cli import profiles as profiles_mod


def test_check_profile_session_access_self():
    allowed, reason = check_profile_session_access(caller_profile="work", target_profile="work")
    assert allowed is True
    assert "Self" in reason


def test_check_profile_session_access_unconfigured():
    config = {}
    allowed, reason = check_profile_session_access(caller_profile="work", target_profile="health", config=config)
    assert allowed is True
    assert "No session_access" in reason


def test_check_profile_session_access_explicit_deny():
    config = {
        "session_access": {
            "health": {
                "deny": ["default", "work"]
            }
        }
    }
    # Denied caller
    allowed, reason = check_profile_session_access(caller_profile="work", target_profile="health", config=config)
    assert allowed is False
    assert "explicitly denies access" in reason

    # Allowed caller
    allowed, reason = check_profile_session_access(caller_profile="personal", target_profile="health", config=config)
    assert allowed is True


def test_check_profile_session_access_explicit_allow():
    config = {
        "session_access": {
            "health": {
                "allow": ["personal"]
            }
        }
    }
    # Caller in allowlist
    allowed, reason = check_profile_session_access(caller_profile="personal", target_profile="health", config=config)
    assert allowed is True

    # Caller not in allowlist
    allowed, reason = check_profile_session_access(caller_profile="work", target_profile="health", config=config)
    assert allowed is False
    assert "allows access only to specified profiles" in reason


def test_check_profile_session_access_global_deny_policy():
    config = {
        "session_access": {
            "default_policy": "deny",
            "health": {
                "allow": ["personal"]
            }
        }
    }
    # Explicitly allowed
    allowed, reason = check_profile_session_access(caller_profile="personal", target_profile="health", config=config)
    assert allowed is True

    # Denied by global policy
    allowed, reason = check_profile_session_access(caller_profile="work", target_profile="finance", config=config)
    assert allowed is False
    assert "Global session_access default_policy is deny" in reason


def test_check_profile_session_access_blocked_by_default():
    config = {
        "session_access": {
            "health": {
                "session_read_blocked_by_default": True
            }
        }
    }
    allowed, reason = check_profile_session_access(caller_profile="work", target_profile="health", config=config)
    assert allowed is False
    assert "denies cross-profile access by default" in reason


def test_session_search_denies_unauthorized_profile_access(monkeypatch):
    config = {
        "session_access": {
            "health": {
                "deny": ["work"]
            }
        }
    }
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)
    monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: "work")

    # Calling session_search into blocked profile 'health'
    res_str = session_search(query="medical data", profile="health")
    res = json.loads(res_str)
    assert res["success"] is False
    assert "Access denied" in res["error"]
    assert "work" in res["error"]
    assert "health" in res["error"]
