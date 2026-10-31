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


@pytest.fixture
def sources():
    """A clean secret-source registry and per-home cache, restored afterwards."""
    from agent.secret_sources import registry
    from hermes_cli import env_loader
    registry._reset_registry_for_tests()
    env_loader.reset_secret_source_cache()
    yield registry
    registry._reset_registry_for_tests()
    env_loader.reset_secret_source_cache()


def _vault_sources(runtime, sources, runs, **fetchers):
    """Enable one fixture ``op``/``bws``-style source per ``fetchers`` item on the secondary profile, whose
    ``.env`` no longer carries a Portal override; each fetch appends its name to ``runs``."""
    from agent.secret_sources.base import SecretSource
    secondary = runtime.homes["secondary"]
    (secondary / ".env").write_text("NOUS_INFERENCE_BASE_URL=https://secondary-inference.invalid/v1\n")
    with (secondary / "config.yaml").open("a") as config:
        config.write("secrets:\n" + "".join(f"  {name}:\n    enabled: true\n" for name in fetchers))

    class _Vault(SecretSource):
        label, shape = "Fixture vault", "bulk"

        def __init__(self, name, result):
            self.name, self._result = name, result

        def fetch(self, cfg, home_path):
            runs.append(self.name)
            return self._result()
    for name, result in fetchers.items():
        sources.register_source(_Vault(name, result), replace=True)


def _portal():
    from agent.secret_sources.base import FetchResult
    return FetchResult(secrets={"HERMES_PORTAL_BASE_URL": "https://secondary-portal.invalid"})


def _down():
    from agent.secret_sources.base import ErrorKind, FetchResult
    return FetchResult().fail("helper exited 1", ErrorKind.AUTH_FAILED)


@pytest.mark.parametrize("method", READS)
def test_polled_reads_load_an_external_portal_override_once_not_every_poll(runtime, sources, method):
    """Desktop polls these views every 30 s. Their Portal override may come only from an external
    source (``op``/``bws``), and a failing source is never cached, so loading per call reran the slow
    CLI on every poll. A cold profile loads its sources once; later polls reuse that snapshot."""
    runs = []
    _vault_sources(runtime, sources, runs, vaultportal=_portal, vaultdown=_down)

    for _ in range(3):
        result = _call(runtime.server, method, {"profile": "secondary"})
        assert result.get("logged_in", result.get("available")) is True, result

    assert runtime.calls
    assert all(url.startswith("https://secondary-portal.invalid/") for _, url, _ in runtime.calls), runtime.calls
    assert sorted(runs) == ["vaultdown", "vaultportal"], runs


def test_a_polled_read_retries_a_failed_source_once_the_retry_interval_passes(runtime, sources, monkeypatch):
    """A source that failed (a vault still locked at serve start) is not rerun by every poll, but is retried
    after ``SECRET_SOURCE_RETRY_INTERVAL_S``, so its recovered Portal override is picked up."""
    from hermes_cli import env_loader
    clock, runs, attempts = [1000.0], [], iter([_down])
    monkeypatch.setattr(env_loader, "_monotonic", lambda: clock[0])
    _vault_sources(runtime, sources, runs, vaultportal=lambda: next(attempts, _portal)())

    def poll():
        runtime.calls.clear()
        _call(runtime.server, "billing.state", {"profile": "secondary"})
        return {url.split("/")[2] for _, url, _ in runtime.calls}

    assert "secondary-portal.invalid" not in poll() and runs == ["vaultportal"]
    clock[0] += env_loader.SECRET_SOURCE_RETRY_INTERVAL_S - 1
    assert "secondary-portal.invalid" not in poll() and runs == ["vaultportal"]
    clock[0] += 2
    assert "secondary-portal.invalid" in poll() and runs == ["vaultportal"] * 2


def test_overlapping_polled_reads_of_a_cold_profile_share_one_source_attempt(runtime, sources, monkeypatch):
    """Desktop's settings page mounts ``billing.state`` and ``subscription.state`` together: both reach the
    pool before the cold profile's first load finishes, and must not each run the failing source."""
    from hermes_cli import env_loader
    runs, attempts = [], []
    entered, release, second_waiting = threading.Event(), threading.Event(), threading.Event()

    def held_down():
        entered.set()
        assert release.wait(10)
        return _down()
    _vault_sources(runtime, sources, runs, vaultdown=held_down)
    real_load, real_lock = env_loader._hydrate_profile_secret_sources, env_loader._SECRET_SOURCE_CACHE_LOCK

    def counted_load(*args):  # one load attempt of the profile's sources
        attempts.append(threading.get_ident())
        return real_load(*args)

    class SignallingLock:  # tells the test when a second thread starts waiting for the loader lock
        def __enter__(self):
            if attempts and threading.get_ident() != attempts[0]:
                second_waiting.set()
            return real_lock.__enter__()

        def __exit__(self, *exc):
            return real_lock.__exit__(*exc)
    monkeypatch.setattr(env_loader, "_hydrate_profile_secret_sources", counted_load)
    monkeypatch.setattr(env_loader, "_SECRET_SOURCE_CACHE_LOCK", SignallingLock())

    wires = [Wire(), Wire()]
    assert runtime.server.dispatch({"jsonrpc": "2.0", "id": "a", "method": "billing.state",
                                    "params": {"profile": "secondary"}}, wires[0]) is None
    assert entered.wait(10)
    assert runtime.server.dispatch({"jsonrpc": "2.0", "id": "b", "method": "subscription.state",
                                    "params": {"profile": "secondary"}}, wires[1]) is None
    assert second_waiting.wait(10)  # the second read found the profile untried and waits for the first load
    release.set()
    assert all(wire.event.wait(10) for wire in wires)
    assert len(attempts) == 1, attempts
    assert runs == ["vaultdown"], runs


def test_a_polled_read_of_a_tried_profile_answers_while_another_load_holds_the_loader_lock(runtime, sources):
    """A poll whose profile was tried recently reuses its snapshot without the loader lock, so it does not
    wait behind another profile's slow fetch."""
    from hermes_cli import env_loader
    runs = []
    _vault_sources(runtime, sources, runs, vaultportal=_portal)
    _call(runtime.server, "billing.state", {"profile": "secondary"})  # loads; the profile now counts as tried
    held, release = threading.Event(), threading.Event()

    def hold_loader_lock():  # stands in for another profile's slow fetch
        with env_loader._SECRET_SOURCE_CACHE_LOCK:
            held.set()
            release.wait(10)
    holder = threading.Thread(target=hold_loader_lock, daemon=True)
    holder.start()
    try:
        assert held.wait(10)
        runtime.calls.clear()
        wire = Wire()
        assert runtime.server.dispatch({"jsonrpc": "2.0", "id": "poll", "method": "billing.state",
                                        "params": {"profile": "secondary"}}, wire) is None
        assert wire.event.wait(5), "the poll waited for the loader lock"
        assert "result" in wire.response, wire.response
        assert runtime.calls
    finally:
        release.set()
        holder.join(10)
    assert all(url.startswith("https://secondary-portal.invalid/") for _, url, _ in runtime.calls), runtime.calls
    assert runs == ["vaultportal"], runs


def test_profile_scoped_rejects_an_unknown_hydrate_secrets_option(runtime):
    with pytest.raises(ValueError, match="hydrate_secrets"):
        runtime.server._profile_scoped(lambda rid, params: None, hydrate_secrets="cold")


@pytest.mark.parametrize("method", ("display.status", "display.thumbnail"))
def test_polled_display_views_never_run_the_profiles_external_secret_source(runtime, monkeypatch, method):
    """Desktop polls the display thumbnail every 4 s; neither view reads a secret."""
    from hermes_cli import env_loader
    runs = []
    monkeypatch.setattr(env_loader, "hydrate_profile_secret_sources", lambda home: runs.append(home) or {})
    wire = Wire()
    req = {"jsonrpc": "2.0", "id": "probe", "method": method, "params": {"profile": "secondary"}}

    response = runtime.server.dispatch(req, wire)
    if response is None:
        assert wire.event.wait(10), "registered RPC did not return"
        response = wire.response

    assert response.get("id") == "probe", response
    assert runs == []


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
