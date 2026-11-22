"""Regression for #132728: registered Portal RPCs bind the selected profile.

Exercise real RPC registration, worker dispatch, OAuth refresh, and endpoint
resolution against two synthetic homes. Only external HTTP is substituted.
"""

import base64
import io
import json
import logging
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest


READS = ("billing.state", "subscription.state", "usage.bars")


def _jwt(label):
    def part(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")
    return ".".join((part({"alg": "none"}), part({
        "sub": label, "scope": "inference:invoke", "exp": int(time.time()) + 3600,
    }), "fixture"))


def _home(home, label, *, expired):
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("model:\n  provider: nous\n  default: fixture-model\n")
    (home / ".env").write_text(
        f"HERMES_PORTAL_BASE_URL=https://{label}-portal.invalid\n"
        f"NOUS_INFERENCE_BASE_URL=https://{label}-inference.invalid/v1\n"
    )
    (home / "auth.json").write_text(json.dumps({"version": 1, "providers": {"nous": {
        "access_token": _jwt(label), "refresh_token": f"{label}-refresh",
        "expires_at": "2000-01-01T00:00:00+00:00" if expired else "2999-01-01T00:00:00+00:00",
        "scope": "inference:invoke", "client_id": "fixture-client",
        "portal_base_url": "https://portal.nousresearch.com",
        "inference_base_url": "https://inference-api.nousresearch.com/v1",
    }}}))


@pytest.fixture(params=[False, True], ids=["valid-token", "expired-token"])
def runtime(request, tmp_path, monkeypatch):
    launch = tmp_path / "launch-home"
    secondary = launch / "profiles" / "secondary"
    _home(launch, "launch", expired=request.param)
    _home(secondary, "secondary", expired=request.param)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-auth"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from agent import billing_usage, billing_view, secret_scope, subscription_view
    from hermes_cli import auth, nous_account, nous_billing
    from tui_gateway import launch_profile_policy, server

    for key in ("HERMES_PORTAL_BASE_URL", "NOUS_PORTAL_BASE_URL", "NOUS_INFERENCE_BASE_URL",
                "HERMES_DEV_BILLING_FIXTURE", "HERMES_DEV_SUBSCRIPTION_FIXTURE", "HERMES_DEV_CREDITS_FIXTURE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    monkeypatch.setattr(server, "_hermes_home", str(launch))
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(auth, "_RESOLVE_TOKEN_CACHE", {})
    monkeypatch.setattr(nous_billing, "_token_cache", {})
    nous_account.reset_nous_portal_account_info_cache()
    previous = secret_scope.is_multiplex_active()
    launch_profile_policy.activate_multi_profile_hosting()
    calls = []

    def http_edge(request):
        calls.append((request.method, str(request.url), request.headers.get("authorization")))
        label = "secondary" if request.url.host == "secondary-portal.invalid" else "launch"
        if request.url.path.endswith("/token"):
            return httpx.Response(200, json={
                # Nous deliberately mirrors refreshed OAuth tokens across profiles.
                "access_token": _jwt("shared-account"), "refresh_token": "shared-rotated",
                "expires_in": 3600, "scope": "inference:invoke",
                "inference_base_url": f"https://{label}-inference.invalid/v1",
            })
        return httpx.Response(200, json={"paid_service_access": True})

    real_client = httpx.Client
    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(http_edge)
        return real_client(*args, **kwargs)
    monkeypatch.setattr(httpx, "Client", client)

    def urlopen(request, *args, **kwargs):
        calls.append((request.get_method(), request.full_url, request.get_header("Authorization")))
        response = io.BytesIO(json.dumps({
            "ok": True, "balanceUsd": "0", "role": "MEMBER", "effect": "no_op",
            "chargeId": "fixture-charge", "status": "settled", "amountUsd": "1",
        }).encode())
        response.status = 200
        return response
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    try:
        yield SimpleNamespace(server=server, homes={"launch": launch, "secondary": secondary},
                              calls=calls, expired=request.param)
    finally:
        secret_scope.set_multiplex_active(previous)


class Wire:
    def __init__(self):
        self.event = threading.Event()
        self.response = None

    def write(self, response):
        self.response = response
        self.event.set()
        return True


def _call(server, method, params):
    wire = Wire()
    response = server.dispatch({"jsonrpc": "2.0", "id": "probe", "method": method, "params": params}, wire)
    if response is None:
        assert wire.event.wait(10), "registered RPC did not return"
        response = wire.response
    assert "result" in response, response
    return response["result"]


@pytest.mark.parametrize("method", READS)
@pytest.mark.parametrize("profile", ["launch", "secondary"])
def test_reads_keep_scope_and_route_every_request(runtime, caplog, monkeypatch, method, profile):
    params = {} if profile == "launch" else {"profile": "secondary"}
    # A sibling's later ambient edit must not override the frozen launch scope or the selected profile.
    monkeypatch.setenv("HERMES_PORTAL_BASE_URL", "https://ambient-other.invalid")
    monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", "https://ambient-other-inference.invalid/v1")
    ambient = dict(os.environ)
    with caplog.at_level(logging.WARNING):
        result = _call(runtime.server, method, params)
    assert result.get("logged_in", result.get("available")) is True, result
    warnings = [r.getMessage() for r in caplog.records if "no profile secret scope" in r.getMessage()]
    assert not warnings, warnings
    assert runtime.calls
    assert all(url.startswith(f"https://{profile}-portal.invalid/") for _, url, _ in runtime.calls), runtime.calls
    assert any(verb == "POST" for verb, _, _ in runtime.calls) is runtime.expired, runtime.calls
    assert dict(os.environ) == ambient


@pytest.mark.parametrize("method", READS)
def test_reads_keep_profile_routing_when_switching_homes(runtime, caplog, method):
    with caplog.at_level(logging.WARNING):
        for profile in ("launch", "secondary", "launch"):
            runtime.calls.clear()
            result = _call(runtime.server, method, {} if profile == "launch" else {"profile": profile})
            assert result.get("logged_in", result.get("available")) is True, result
            # The third call can be answered from the same-profile account cache.
            for verb, url, bearer in runtime.calls:
                assert url.startswith(f"https://{profile}-portal.invalid/"), runtime.calls
                if verb == "GET":
                    assert bearer and bearer.startswith("Bearer "), runtime.calls
                    payload = bearer.removeprefix("Bearer ").split(".")[1]
                    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
                    assert claims["sub"] == ("shared-account" if runtime.expired else profile), runtime.calls
    assert not [r for r in caplog.records if "no profile secret scope" in r.getMessage()]


@pytest.mark.parametrize("method,params", [
    ("subscription.preview", {"subscription_type_id": "fixture-tier"}),
    ("subscription.change", {"cancel": True}),
    ("subscription.resume", {}),
    ("subscription.upgrade", {"subscription_type_id": "fixture-tier"}),
    ("billing.charge", {"amount_usd": "1"}),
    ("billing.charge_status", {"charge_id": "fixture-charge"}),
    ("billing.auto_reload", {"threshold": "1", "top_up_amount": "1"}),
])
@pytest.mark.parametrize("profile", ["launch", "secondary"])
def test_sibling_routes_use_the_same_profile_scope(runtime, caplog, method, params, profile):
    # These are HTTP-intercepted synthetic calls, including the money routes.
    with caplog.at_level(logging.WARNING):
        result = _call(runtime.server, method, params | ({} if profile == "launch" else {"profile": profile}))
    assert result["ok"] is True, result
    assert runtime.calls
    assert all(url.startswith(f"https://{profile}-portal.invalid/") for _, url, _ in runtime.calls), runtime.calls
    assert not [r for r in caplog.records if "no profile secret scope" in r.getMessage()]


@pytest.mark.parametrize("profile", ["launch", "secondary"])
def test_step_up_binds_the_profile_before_starting_the_external_flow(runtime, monkeypatch, profile):
    from agent.secret_scope import get_secret
    from hermes_cli import auth
    from hermes_constants import get_hermes_home

    seen = []
    def external_device_flow(**kwargs):
        seen.append((get_hermes_home(), get_secret("HERMES_PORTAL_BASE_URL")))
        return True
    monkeypatch.setattr(auth, "step_up_nous_billing_scope", external_device_flow)
    result = _call(runtime.server, "billing.step_up", {} if profile == "launch" else {"profile": profile})
    assert result == {"ok": True, "granted": True}, result
    assert seen == [(runtime.homes[profile], f"https://{profile}-portal.invalid")]


@pytest.mark.parametrize("method", READS)
def test_missing_profile_fails_before_any_portal_request(runtime, method):
    wire = Wire()
    response = runtime.server.dispatch({"jsonrpc": "2.0", "id": "probe", "method": method,
                                       "params": {"profile": "missing"}}, wire)
    if response is None:
        assert wire.event.wait(10)
        response = wire.response
    assert response["error"]["code"] == 4064, response
    assert runtime.calls == []
