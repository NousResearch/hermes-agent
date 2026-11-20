"""Routed Vercel sandboxes run on the serving profile's own token, project and team.

Without ``token``/``project_id``/``team_id``, ``Sandbox.create`` resolves credentials from
``os.environ``: on a multiplexed gateway that is the launch profile's ``.env``, so every routed
profile's sandbox was created, billed and run on the launch profile's Vercel account. These tests
drive the real, locked SDK and record each request where it leaves the HTTP client.
"""

import json
from types import SimpleNamespace

import httpx
import pytest

sandbox_sdk = pytest.importorskip("vercel.sandbox")

from tools.environments import vercel_sandbox

_REFUSAL = "VERCEL_TOKEN, VERCEL_PROJECT_ID and VERCEL_TEAM_ID are not all set for this profile"


def _triple(label):
    return {"VERCEL_TOKEN": f"vtok-{label}", "VERCEL_PROJECT_ID": f"prj_{label}", "VERCEL_TEAM_ID": f"team_{label}"}


def _dotenv(values):
    return "".join(f"{name}={value}\n" for name, value in values.items())


@pytest.fixture
def wire(monkeypatch):
    """Every SDK request: method, path, bearer token, ``teamId``, and the create body's project and
    source type. A create answers a running sandbox, or the next status in ``fail_creates``; every
    other request answers 404."""
    record = SimpleNamespace(requests=[], fail_creates=[])

    def send(_client, request, *_args, **_kwargs):
        try:
            body = json.loads(request.content or b"{}")
        except ValueError:  # file uploads send a tarball
            body = {}
        record.requests.append(SimpleNamespace(
            method=request.method, path=request.url.path,
            token=request.headers.get("authorization", "").removeprefix("Bearer "),
            team=request.url.params.get("teamId"), project=body.get("projectId"),
            source=(body.get("source") or {}).get("type")))
        if (request.method, request.url.path) != ("POST", "/v1/sandboxes"):
            return httpx.Response(404, json={"error": {"message": "not found"}}, request=request)
        if record.fail_creates:
            return httpx.Response(record.fail_creates.pop(0), json={"error": {"message": "refused"}}, request=request)
        sandbox = {"id": f"sbx_{len(record.requests)}", "memory": 2048, "vcpus": 1, "region": "iad1",
                   "runtime": "node22", "timeout": 300000, "status": "running", "requestedAt": 0,
                   "createdAt": 0, "updatedAt": 0, "cwd": "/vercel/sandbox"}
        return httpx.Response(200, json={"sandbox": sandbox, "routes": []}, request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(vercel_sandbox, "FileSyncManager", lambda **_kw: SimpleNamespace(
        sync=lambda force=False: None, sync_back=lambda: None))
    monkeypatch.setattr(vercel_sandbox.VercelSandboxEnvironment, "init_session", lambda self: None)
    return record


def _creates(requests):
    return [r for r in requests if (r.method, r.path) == ("POST", "/v1/sandboxes")]


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    """Launch profile A, its triple exported into the process env as a gateway start does, and a
    routed profile B; multiplexing on. ``home(label, dotenv)`` writes the profile's ``.env``."""
    from agent import secret_scope

    root = tmp_path / ".hermes"
    homes = {"A": root, "B": root / "profiles" / "b"}
    for home in homes.values():
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("VERCEL_OIDC_TOKEN", raising=False)
    for name, value in _triple("A").items():
        monkeypatch.setenv(name, value)
    was_active = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)

    def home(label, dotenv=None):
        profile = homes[label.split("-")[0]]
        (profile / ".env").write_text(_dotenv(_triple(label)) if dotenv is None else dotenv)
        return profile

    try:
        yield home
    finally:
        secret_scope.set_multiplex_active(was_active)


def _scope(home):
    from agent.runtime_scope import bind_profile_scope

    return bind_profile_scope(home, hydrate_secrets=False)


def test_every_request_of_a_routed_sandbox_uses_its_profiles_credentials(wire, profiles):
    for label in ("B", "A", "B"):
        start = len(wire.requests)
        with _scope(profiles(label)):
            env = vercel_sandbox.VercelSandboxEnvironment(persistent_filesystem=False)
            env._ensure_sandbox_ready()  # the refresh fails (404), so the sandbox is recreated
            env.cleanup()
        turn = wire.requests[start:]
        assert {(r.token, r.team) for r in turn} == {(f"vtok-{label}", f"team_{label}")}
        assert [r.project for r in _creates(turn)] == [f"prj_{label}"] * 2
        assert {r.method for r in turn if r.path.endswith("/stop")} == {"POST"}  # cleanup reached the SDK


def test_a_recreated_sandbox_rereads_the_profiles_credentials(wire, profiles):
    with _scope(profiles("B")):
        env = vercel_sandbox.VercelSandboxEnvironment(persistent_filesystem=False)
    with _scope(profiles("B-rotated")):
        env._ensure_sandbox_ready()
    assert [(r.token, r.project) for r in _creates(wire.requests)] == [("vtok-B", "prj_B"),
                                                                        ("vtok-B-rotated", "prj_B-rotated")]
    with _scope(profiles("B", "VERCEL_TOKEN=vtok-B\n")), pytest.raises(ValueError, match=_REFUSAL):
        env._ensure_sandbox_ready()
    assert len(_creates(wire.requests)) == 2


@pytest.mark.parametrize("dotenv", [
    "", "VERCEL_TOKEN=vtok-B\n", "VERCEL_TOKEN=vtok-B\nVERCEL_PROJECT_ID=prj_B\n",
    "VERCEL_TOKEN=vtok-B\nVERCEL_PROJECT_ID=prj_B\nVERCEL_TEAM_ID=\n", "VERCEL_OIDC_TOKEN=oidc-B\n"],
    ids=["absent", "token-only", "no-team", "empty-team", "oidc-only"])
def test_a_routed_profile_without_its_full_triple_is_refused_before_any_request(wire, profiles, dotenv):
    with _scope(profiles("B", dotenv)), pytest.raises(ValueError, match=_REFUSAL):
        vercel_sandbox.VercelSandboxEnvironment(persistent_filesystem=False)
    assert wire.requests == []


def test_snapshot_restore_retries_and_fallback_keep_the_profiles_credentials(wire, profiles):
    home = profiles("B")
    (home / "vercel_sandbox_snapshots.json").write_text(json.dumps({"task-1": "snap_1"}))
    wire.fail_creates = [503, 503, 503]  # every restore attempt fails, then a fresh sandbox is created
    with _scope(home):
        env = vercel_sandbox.VercelSandboxEnvironment(persistent_filesystem=True, task_id="task-1")
        env.cleanup()  # snapshots, then stops
    assert [r.source for r in _creates(wire.requests)] == ["snapshot"] * 3 + [None]
    assert {(r.token, r.team) for r in wire.requests} == {("vtok-B", "team_B")}
    assert {r.project for r in _creates(wire.requests)} == {"prj_B"}
    assert any(r.path.endswith("/snapshot") for r in wire.requests)
    assert json.loads((home / "vercel_sandbox_snapshots.json").read_text()) == {}


@pytest.mark.parametrize("env", [_triple("shell"), {"VERCEL_OIDC_TOKEN": "oidc-shell", "VERCEL_PROJECT_ID": "prj_shell",
                                                     "VERCEL_TEAM_ID": "team_shell"}], ids=["token", "oidc"])
def test_a_single_profile_keeps_the_sdks_own_credentials(wire, tmp_path, monkeypatch, env):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()
    for name in ("VERCEL_TOKEN", "VERCEL_PROJECT_ID", "VERCEL_TEAM_ID", "VERCEL_OIDC_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    vercel_sandbox.VercelSandboxEnvironment(persistent_filesystem=False).cleanup()
    token = env.get("VERCEL_TOKEN") or env["VERCEL_OIDC_TOKEN"]
    assert {(r.token, r.team) for r in wire.requests} == {(token, "team_shell")}
    assert [r.project for r in _creates(wire.requests)] == ["prj_shell"]


@pytest.mark.parametrize(("dotenv", "usable"), [
    (None, True), ("VERCEL_OIDC_TOKEN=oidc-B\n", False), ("VERCEL_TOKEN=vtok-B\n", False)],
    ids=["full-triple", "oidc-only", "token-only"])
def test_backend_selection_agrees_with_the_constructor_for_a_routed_profile(wire, profiles, dotenv, usable):
    from tools.terminal_tool_backends import _check_vercel

    with _scope(profiles("B", dotenv)):
        assert _check_vercel({"vercel_runtime": "", "container_disk": 51200}) is usable
        if usable:
            vercel_sandbox.VercelSandboxEnvironment(persistent_filesystem=False)
        else:
            with pytest.raises(ValueError, match=_REFUSAL):
                vercel_sandbox.VercelSandboxEnvironment(persistent_filesystem=False)
    assert {r.token for r in wire.requests} == ({"vtok-B"} if usable else set())
