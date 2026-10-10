"""Advisory completion stays on its owner and never starts a sandbox (#131751).

Use real RPC dispatch, profile files, terminal resolution and shell listings. A
LocalEnvironment serves the test-owned backend tree; Docker's lost-container
response is simulated at BaseEnvironment.execute, without a Docker daemon.
"""

from __future__ import annotations

import contextvars
import os
from pathlib import Path

import pytest

from agent import runtime_cwd, secret_scope
from gateway import session_context
from hermes_constants import get_hermes_home, get_hermes_home_override
from tools import terminal_scope, terminal_tool as tt, terminal_tool_lifecycle as lifecycle
from tools.environments.base import BaseEnvironment
from tools.environments.docker import DockerEnvironment
from tools.environments.local import LocalEnvironment
from tui_gateway import launch_profile_policy as launch_policy, server

# The facade publishes these helpers dynamically at import.
runtime_scope = getattr(server, "_session_profile_runtime_scope")
request = getattr(server, "handle_request")


def _cache_key(record, home):
    def resolve():
        session_context.set_session_vars(session_key=record["session_key"], profile=home.name)
        with runtime_scope(record, hydrate_secrets=False):
            return tt._resolve_container_task_id(record["session_key"])
    return contextvars.copy_context().run(resolve)


@pytest.mark.parametrize("pooled_launch", [False, True], ids=["root-launch", "named-launch"])
@pytest.mark.parametrize("launch_backend", ["docker", "local"])
@pytest.mark.parametrize("state", ["live", "cold", "retired", "draft", "missing", "deleted"])
def test_completion_preserves_owner_without_starting_environments(
    tmp_path, monkeypatch, pooled_launch, launch_backend, state,
):
    root = tmp_path / ".hermes"
    a = root / "profiles" / "pool" if pooled_launch else root
    b = root / "profiles" / "worker"
    homes = {"a": a, "b": b}
    backends = {"a": launch_backend, "b": "local" if launch_backend == "docker" else "docker"}
    records, targets, environments, preparations, recreated = {}, {}, {}, [], []
    host = tmp_path / "host"
    (host / "ws").mkdir(parents=True)
    (host / "ws" / "host-decoy.md").write_text("host only", encoding="utf-8")
    monkeypatch.chdir(host)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(a))
    monkeypatch.setenv("TERMINAL_ENV", launch_backend)
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
    monkeypatch.setattr(server, "_hermes_home", a)
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(launch_policy, "_snapshot", None)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr(tt, "_active_environments", {})
    monkeypatch.setattr(tt, "_last_activity", {})
    monkeypatch.setattr(tt, "_terminal_config_bridge_attempted", False)
    monkeypatch.setattr(session_context, "_session_context_engaged", False)

    for name, home in homes.items():
        local_tree, backend_tree = home / "local", home / "backend"
        (backend_tree / "ws").mkdir(parents=True)
        local_tree.mkdir()
        (local_tree / f"local-{name}.md").write_text(name, encoding="utf-8")
        (backend_tree / "ws" / f"backend-{name}.md").write_text(name, encoding="utf-8")
        cwd = str(local_tree) if backends[name] == "local" else "ws"
        (home / "config.yaml").write_text(
            f"terminal:\n  backend: {backends[name]}\n  cwd: {cwd}\n"
            "  container_persistent: false\n  docker_persist_across_processes: true\n",
            encoding="utf-8",
        )
        (home / ".env").write_text(
            f"COMPLETION_SHARED_TOKEN={name}-token\n{name.upper()}_ONLY_TOKEN={name}-only\n",
            encoding="utf-8",
        )
        record = {"session_key": "same-key", "profile_home": None if name == "a" else str(home), "cwd": cwd}
        records[name], targets[name] = record, cwd
        server._sessions[f"sid-{name}"] = record
        if backends[name] == "docker" and state == "live":
            with runtime_scope(record, hydrate_secrets=False):
                env = LocalEnvironment(cwd=str(backend_tree), timeout=3, env={"HOME": str(home)})
            monkeypatch.setattr(env, "_before_execute", lambda n=name: preparations.append(n))
            environments[name] = env
        elif backends[name] == "docker" and state == "retired":
            env = object.__new__(DockerEnvironment)
            env.cwd, env.timeout, env._persist_across_processes = str(backend_tree), 3, True
            monkeypatch.setattr(env, "_recreate_container", lambda n=name: recreated.append(n) or True)
            environments[name] = env

    # A secondary makes ambient reads fail closed; a later ambient backend write
    # must not become either tenant's policy, including a named pooled launch.
    launch_policy.activate_multi_profile_hosting()
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    foreign_env = None
    if state == "cold" and backends["b"] == "docker":
        # A legacy unqualified slot belongs to the launch profile even when the
        # secondary uses the same durable key. It must not be a fallback for B.
        with runtime_scope(records["a"], hydrate_secrets=False):
            foreign_env = LocalEnvironment(cwd=str(a / "backend"), timeout=3, env={"HOME": str(a)})
        tt._active_environments["same-key"] = foreign_env
    before_env = dict(os.environ)
    before_scope = (get_hermes_home_override(), secret_scope.current_secret_scope(),
                    terminal_scope.get_terminal_scope(), runtime_cwd.scoped_session_cwd())
    created, observed = [], []

    def prohibit_creation(*args, **kwargs):
        created.append(str(get_hermes_home()))
        raise RuntimeError("an advisory read attempted to create an environment")

    monkeypatch.setattr(tt, "_create_configured_env", prohibit_creation)
    original_listing = getattr(server, "_dir_listing_items")

    def observe(*args, **kwargs):
        try:
            shared = secret_scope.get_secret("COMPLETION_SHARED_TOKEN")
            foreign = secret_scope.get_secret("B_ONLY_TOKEN" if get_hermes_home() == a else "A_ONLY_TOKEN")
        except secret_scope.UnscopedSecretError:
            shared, foreign = "unbound", "unbound"
        observed.append((str(get_hermes_home()), shared, foreign,
                         tt._get_env_config()["env_type"], terminal_scope.get_terminal_scope() is not None))
        return original_listing(*args, **kwargs)

    monkeypatch.setattr(server, "_dir_listing_items", observe)
    original_execute = BaseEnvironment.execute

    def lost_container(env, command, cwd="", **kwargs):
        if isinstance(env, DockerEnvironment):
            owner = next(name for name, item in environments.items() if item is env)
            return {"output": "recovered" if owner in recreated else "Error response: No such container",
                    "returncode": 0 if owner in recreated else 1}
        return original_execute(env, command, cwd, **kwargs)

    monkeypatch.setattr(BaseEnvironment, "execute", lost_container)
    original_probe = lifecycle.get_active_env

    def retire_after_probe(key):
        env = original_probe(key)
        if env is not None and state == "live":
            # The descriptor was obtained, then its cache slot was retired.
            # Reacquiring through terminal_tool would cold-start a replacement.
            tt._active_environments.pop(tt._resolve_container_task_id(key), None)
        return env

    monkeypatch.setattr(lifecycle, "get_active_env", retire_after_probe)
    try:
        for name in ("a", "b", "a"):
            home, record = homes[name], records[name]
            if name in environments:
                tt._active_environments[_cache_key(record, home)] = environments[name]
            params = {"word": "@file:", "cwd": targets[name],
                      "profile": "worker" if name == "a" else "default"}
            if state == "draft":
                params["profile"] = ("pool" if pooled_launch else "default") if name == "a" else "worker"
            else:
                params["session_id"] = "unknown" if state == "missing" else f"sid-{name}"
            if state == "deleted":
                record["profile_home"] = str(tmp_path / f"deleted-{name}")
            response = request({"id": name, "method": "complete.path", "params": params})
            assert "result" in response, response
            texts = [item["text"] for item in response["result"]["items"]]
            if state in {"missing", "deleted"} or (backends[name] == "docker" and state != "live"):
                assert texts == [], response
            else:
                kind = "local" if backends[name] == "local" else "backend"
                assert texts == [f"@file:{kind}-{name}.md"], response
            assert created == []
            assert preparations == []
            assert recreated == []
            assert dict(os.environ) == before_env
            assert (get_hermes_home_override(), secret_scope.current_secret_scope(),
                    terminal_scope.get_terminal_scope(), runtime_cwd.scoped_session_cwd()) == before_scope

        if state in {"missing", "deleted"}:
            assert observed == []
        else:
            assert observed == [(str(homes[n]), f"{n}-token", None, backends[n], True) for n in ("a", "b", "a")]
        for name, env in environments.items():
            # Advisory reads suppress startup/recovery; ordinary commands retain it.
            with runtime_scope(records[name], hydrate_secrets=False):
                assert env.execute("printf 'ordinary command'")["returncode"] == 0
            assert recreated == ([name] if state == "retired" else [])
            assert preparations == ([name] if state == "live" else [])

        # An explicitly deleted draft owner must not adopt the launch tenant.
        response = request({"id": "draft-gone", "method": "complete.path",
                                          "params": {"word": "@file:", "profile": "deleted"}})
        assert response.get("result", {}).get("items") == [], response
        assert created == []
    finally:
        if foreign_env is not None:
            foreign_env.cleanup()
        for env in environments.values():
            if isinstance(env, LocalEnvironment):
                env.cleanup()
