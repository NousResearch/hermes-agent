"""``onboarding.state`` / ``onboarding.start`` and the desktop's welcome trial: a new run on a first
install claims it with the install already marked due, a rerun never does, a failed claim leaves no guest
and no chat and the next launch still due, a blocked claim with nothing to run on opens no chat, and only
the desktop's local backend opens onboarding by itself."""

from __future__ import annotations

import threading
from pathlib import Path

import httpx
import pytest

from agent.onboarding_flow import Facts
from hermes_cli import anon_auth, install_state
from tests.hermes_cli.anon_portal import as_local_desktop_backend, install_portal
from tui_gateway import server


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    home = tmp_path / "hermes-root"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_hermes_home", str(home))
    monkeypatch.setattr(server, "_db", None)
    monkeypatch.setattr("agent.onboarding_facts.read_facts", lambda **kw: Facts(os="macos", arch="arm64"))
    return home


@pytest.fixture
def portal(root, tmp_path, monkeypatch):
    fake = install_portal(monkeypatch, tmp_path)
    monkeypatch.setitem(server._methods, "session.resume",
                        lambda rid, params: {"result": {"session_id": f"rt-{params['session_id']}"}})
    monkeypatch.setitem(server._methods, "prompt.submit", lambda rid, params: {"result": {"status": "streaming"}})
    return fake


def _call(method: str, params: dict | None = None) -> dict:
    return server.handle_request({"jsonrpc": "2.0", "id": 9, "method": method, "params": params or {}})


def test_a_new_run_on_a_first_install_claims_the_trial_once(portal, monkeypatch):
    as_local_desktop_backend(monkeypatch)

    assert _call("onboarding.state")["result"]["run"] is True
    assert _call("onboarding.state")["result"]["eligible"] is True
    _call("onboarding.start")
    _call("onboarding.start")                    # a relaunch reopens the run: no second claim
    _call("onboarding.start", {"rerun": True})   # Settings: runs on the model it has

    assert portal.creates() == 1
    assert anon_auth.free_inference_active()


def test_a_rerun_on_a_first_install_never_claims(portal, monkeypatch):
    as_local_desktop_backend(monkeypatch)

    _call("onboarding.start", {"rerun": True})

    assert portal.creates() == 0


def test_a_failed_claim_opens_nothing_and_the_next_launch_tries_again(portal, monkeypatch):
    from hermes_cli import free_tier_bootstrap

    as_local_desktop_backend(monkeypatch)
    free_tier_bootstrap.run_bootstrap(announce=False)  # the first boot latches the install as due
    portal.create_response = httpx.Response(503, json={"error": "unavailable"})

    failed = _call("onboarding.start", {"fresh_launch": True})

    assert failed["error"]["code"] == 5079
    assert _call("onboarding.state")["result"]["session_id"] is None
    portal.create_response = None
    anon_auth.reset_mint_memo_for_tests()
    install_state.probe.cache_clear()  # the next launch asks first, then starts
    state = _call("onboarding.state")["result"]
    assert (state["run"], state["eligible"]) == (True, True)
    assert "result" in _call("onboarding.start", {"fresh_launch": True})
    assert anon_auth.free_inference_active()


def test_a_trial_another_install_took_first_opens_no_chat_it_cannot_run(portal, root, monkeypatch):
    """Two installs answered "due" before either claimed. The first takes the trial; the second start
    opens no chat with no model behind it, so the desktop lands on provider setup and sign-in. The
    install stays due: once a provider is set, the next start runs onboarding on it."""
    from hermes_cli import free_tier_bootstrap

    as_local_desktop_backend(monkeypatch)
    free_tier_bootstrap.run_bootstrap(announce=False)
    assert _call("onboarding.state")["result"]["eligible"] is True
    install_state.mark_machine_used()  # the other install's claim consumed the reservation

    refused = _call("onboarding.start", {"fresh_launch": True})

    assert refused["error"]["code"] == 5080
    state = _call("onboarding.state")["result"]
    assert (state["run"], state["eligible"], state["session_id"]) == (True, False, None)
    assert portal.creates() == 0
    (root / ".env").write_text("OPENROUTER_API_KEY=sk-or-test\n", encoding="utf-8")
    assert "result" in _call("onboarding.start", {"fresh_launch": True})


def test_a_crash_right_after_the_claim_keeps_onboarding_due(portal, monkeypatch):
    as_local_desktop_backend(monkeypatch)

    mark = install_state.mark_machine_used

    def die(*_args, **_kwargs):
        raise RuntimeError("the process died here")
    monkeypatch.setattr(install_state, "mark_machine_used", die)  # the guest is minted, then the crash

    assert _call("onboarding.start")["error"]["code"] == 5078
    monkeypatch.setattr(install_state, "mark_machine_used", mark)
    install_state.probe.cache_clear()  # the next launch

    state = _call("onboarding.state")["result"]
    assert state["run"] is True and state["eligible"] is True
    assert portal.creates() == 1


def test_only_the_local_desktop_backend_opens_onboarding_by_itself(portal, root, monkeypatch):
    assert _call("onboarding.state")["result"]["eligible"] is False   # CLI / SSH / URL backend

    as_local_desktop_backend(monkeypatch)
    install_state.mark_machine_used()
    assert _call("onboarding.state")["result"] == {"run": False, "eligible": False, "session_id": None, "run_id": None}

    (root / ".env").write_text("OPENROUTER_API_KEY=sk-or-test\n", encoding="utf-8")  # a key saved later
    assert _call("onboarding.state")["result"]["eligible"] is True


def test_eligible_follows_the_default_profile_not_the_launch_profile(portal, root, monkeypatch):
    from hermes_cli import profiles
    work = profiles.create_profile("work", no_alias=True)
    monkeypatch.setenv("HERMES_HOME", str(work))
    monkeypatch.setattr(server, "_hermes_home", str(work))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (root / ".env").write_text("OPENROUTER_API_KEY=sk-or-test\n", encoding="utf-8")
    as_local_desktop_backend(monkeypatch)
    install_state.mark_machine_used()

    assert _call("onboarding.state")["result"]["eligible"] is True


def test_a_claim_refreshes_the_setup_record(portal, monkeypatch):
    from hermes_cli import free_tier_bootstrap
    as_local_desktop_backend(monkeypatch)
    free_tier_bootstrap.run_bootstrap(announce=False)

    assert "result" in _call("onboarding.start")

    record = free_tier_bootstrap.current_record()
    assert record is not None and record.free_tier_account is True


def test_the_dev_fresh_install_switch_treats_a_used_machine_as_new(portal, monkeypatch):
    as_local_desktop_backend(monkeypatch)
    install_state.mark_machine_used()

    assert _call("onboarding.state", {"assume_fresh": True})["result"]["run"] is True
    assert "result" in _call("onboarding.start", {"assume_fresh": True})
    assert portal.creates() == 1


def test_the_due_answer_is_worked_out_at_boot_and_served_from_there(portal, monkeypatch):
    from hermes_cli import auth, free_tier_bootstrap

    as_local_desktop_backend(monkeypatch)
    free_tier_bootstrap.run_bootstrap(announce=False)
    real, callers = auth.resolve_provider, []
    monkeypatch.setattr(auth, "resolve_provider",
                        lambda *a, **kw: (callers.append(threading.current_thread().name), real(*a, **kw))[1])
    server.start_due_boot()

    answers = [_call("onboarding.state")["result"] for _ in range(3)]

    assert answers[0]["run"] is True and answers[0]["eligible"] is True and answers == [answers[0]] * 3
    assert callers and set(callers) == {"onboarding-due"}  # the requests themselves never walk the ladder


def _connector_guest_from_another_profile(_managed: Path) -> None:
    from hermes_cli.auth_nous import _write_shared_nous_state
    from tests.hermes_cli.anon_portal import PORTAL

    _write_shared_nous_state({"auth_method": anon_auth.ANON_AUTH_METHOD, "account_tier": anon_auth.ANON_ACCOUNT_TIER,
                              "anon_token": "anon_other", "client_id": anon_auth.ANON_CLIENT_ID,
                              "portal_base_url": PORTAL, anon_auth.WELCOME_CLAIM_FLAG: False}, renewal=False)


ELSEWHERE = {
    "shared_connector_guest": (_connector_guest_from_another_profile, "eligible"),
    "managed_onboarding_off": (lambda managed: (managed / "config.yaml").write_text("onboarding:\n  run: false\n",
                                                                                     encoding="utf-8"), "run"),
}


@pytest.mark.parametrize("change", list(ELSEWHERE))
def test_the_cached_due_answer_follows_shared_auth_and_the_managed_layer(portal, monkeypatch, tmp_path, change):
    from hermes_cli import free_tier_bootstrap

    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    as_local_desktop_backend(monkeypatch)
    free_tier_bootstrap.run_bootstrap(announce=False)
    cached = _call("onboarding.state")["result"]
    assert (cached["run"], cached["eligible"]) == (True, True)
    write, moved = ELSEWHERE[change]

    write(managed)

    assert _call("onboarding.state")["result"][moved] is False
