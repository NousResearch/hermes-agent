from types import SimpleNamespace

import httpx
import pytest

from hermes_cli.auth import _save_auth_store, get_provider_auth_state
from hermes_cli.config import save_config
from hermes_wisdom.client import WisdomAuthError
from hermes_wisdom.mediation_store import MediationStore
from hermes_wisdom.service import WisdomService
from hermes_wisdom.store import WisdomStore


@pytest.fixture
def account(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared"))
    save_config({"wisdom": {"enabled": True, "disclosure_acknowledged_at": "fixture"}})
    store = WisdomStore()
    store.activate_installation_identity("installation", "org")
    queue = MediationStore(store)
    identity = queue.enqueue("org", "feed:event", {"kind": "notice"})
    return store, queue, identity


@pytest.mark.parametrize("terminal", [True, False])
def test_cached_client_cannot_keep_revoked_account_advice_alive(account, terminal):
    store, queue, identity = account
    service = WisdomService(store=store, client=SimpleNamespace(display_org_id="org"))
    _save_auth_store({
        "providers": {
            "nous": {
                "last_auth_error": {
                    "code": "invalid_grant" if terminal else "server_error",
                    "relogin_required": terminal,
                },
            }
        }
    })

    if terminal:
        with pytest.raises(WisdomAuthError, match="sign in"):
            service.require_setup()
        assert store.active_org_id() is None
        store.verify_installation_identity("org")
        assert queue.enqueue("org", "feed:event", {"kind": "notice"}) == identity
        assert queue.assessments("org")[0]["state"] == "retired"
    else:
        service.require_setup()
        assert store.active_org_id() == "org"
        assert queue.assessments("org")[0]["state"] == "pending"


@pytest.mark.parametrize("status,code", [(401, "invalid_grant"), (503, "server_error")])
def test_runtime_refresh_only_retires_advice_on_persisted_terminal_failure(
    account, monkeypatch, status, code
):
    store, queue, _ = account
    _save_auth_store({
        "providers": {
            "nous": {
                "access_token": "expired-fixture",
                "refresh_token": "fixture-refresh",
                "scope": "inference:invoke",
                "expires_at": "2000-01-01T00:00:00+00:00",
            }
        }
    })
    calls = []

    def respond(request):
        calls.append(request.url.path)
        return httpx.Response(status, json={"error": code})

    monkeypatch.setattr(
        "hermes_cli.auth_nous._nous_http_client",
        lambda *args: httpx.Client(transport=httpx.MockTransport(respond)),
    )
    service = WisdomService(store=store)
    if status == 401:
        with pytest.raises(WisdomAuthError, match="sign in"):
            service.require_setup()
        assert get_provider_auth_state("nous")["last_auth_error"]["relogin_required"]
        assert store.active_org_id() is None
        store.verify_installation_identity("org")
        assert queue.assessments("org")[0]["state"] == "retired"
    else:
        service.require_setup()
        assert get_provider_auth_state("nous")["refresh_token"] == "fixture-refresh"
        assert queue.assessments("org")[0]["state"] == "pending"
    assert calls == ["/api/oauth/token"]
