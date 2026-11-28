"""Contract: under ``proxy.docker_internal_network`` the sandbox joins an internal network
and iron-proxy binds its gateway, or the sandbox refuses to start (fail closed).

A stub ``docker`` CLI on PATH stands in for the daemon so the real subprocess path runs.
"""

from __future__ import annotations

import json
import stat
import sys

import pytest

from agent.proxy_sources import iron_proxy as ip
from tools.environments import docker as docker_env
from tools.environments import docker_egress

pytestmark = pytest.mark.platforms("linux")

_STUB = r'''#!{python}
import json, os, sys
state = {state!r}
args = sys.argv[1:]
with open(os.path.join(state, "calls.jsonl"), "a") as f:
    f.write(json.dumps(args) + "\n")
nets = os.path.join(state, "nets.json")
known = json.load(open(nets)) if os.path.exists(nets) else {{}}
if args[:2] == ["network", "inspect"]:
    if args[2] in known or "*" in known:
        print(json.dumps([known.get(args[2], known.get("*"))])); sys.exit(0)
    sys.stderr.write("no such network\n"); sys.exit(1)
if args[:2] == ["network", "create"]:
    known[args[-1]] = {{"Name": args[-1], "Internal": "--internal" in args, "EnableIPv6": False,
                       "IPAM": {{"Config": [{{"Subnet": "172.31.0.0/16", "Gateway": "172.31.0.1"}}]}}}}
    json.dump(known, open(nets, "w")); sys.exit(0)
sys.exit(0)
'''


@pytest.fixture
def egress_ready(tmp_path, monkeypatch):
    """Configured, 'running' iron-proxy bound to the stub internal network's gateway."""
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    state = tmp_path / "docker-state"
    state.mkdir()
    stub = tmp_path / "bin" / "docker"
    stub.parent.mkdir()
    stub.write_text(_STUB.format(python=sys.executable, state=str(state)), encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(docker_env, "_docker_executable", str(stub))
    monkeypatch.setattr(ip, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(ip, "_port_listening", lambda h, p: True)

    from hermes_cli.config import load_config, save_config

    proxy_dir = ip._proxy_state_dir()
    ca = proxy_dir / "ca.crt"
    ca.write_text("fake-ca", encoding="utf-8")
    (proxy_dir / "ca.key").write_text("fake-key", encoding="utf-8")
    mapping = ip.TokenMapping(proxy_token=ip.mint_proxy_token("test"), real_env_name="OPENROUTER_API_KEY",
                              upstream_hosts=("openrouter.ai",))
    ip.write_proxy_config(ip.build_proxy_config(
        mappings=[mapping], ca_cert=ca, ca_key=proxy_dir / "ca.key", tunnel_port=9090,
        http_listen=["172.31.0.1:9090"]))
    ip.write_mappings([mapping])
    (proxy_dir / "iron-proxy.pid").write_text("99999", encoding="utf-8")
    cfg = load_config()
    cfg["proxy"] = {**(cfg.get("proxy") or {}), "enabled": True, "enforce_on_docker": True,
                    "docker_internal_network": True}
    save_config(cfg)
    return state


def _sandbox_host_args():
    """The egress ``host_args`` DockerEnvironment would hand to ``docker run``."""
    _volumes, env, host_args = docker_egress._egress_proxy_args_for_docker()
    return env, docker_egress.internal_network_host_args(docker_env.find_docker() or "docker", host_args)


def _calls(state):
    path = state / "calls.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()] if path.exists() else []


def test_sandbox_joins_internal_network_with_proxy_as_only_exit(egress_ready):
    env, host_args = _sandbox_host_args()

    created = [c for c in _calls(egress_ready) if c[:2] == ["network", "create"]]
    assert created and "--internal" in created[0] and "--ipv6=false" in created[0]
    network = created[0][-1]
    assert host_args[host_args.index("--network") + 1] == network
    assert "host.docker.internal:172.31.0.1" in host_args
    assert env["HTTPS_PROXY"] == "http://host.docker.internal:9090"
    # The network and its gateway are part of the reuse fingerprint, so a container
    # created on the default bridge is never reused once the flag is on.
    assert docker_egress._egress_reuse_fingerprint([], env, host_args) != (
        docker_egress._egress_reuse_fingerprint([], env, ["--add-host", "host.docker.internal:host-gateway"]))


def _route_out_network(state, monkeypatch):
    """A pre-existing network under the profile's name that has a route out."""
    (state / "nets.json").write_text(json.dumps({"*": {
        "Internal": False, "EnableIPv6": False,
        "IPAM": {"Config": [{"Subnet": "172.31.0.0/16", "Gateway": "172.31.0.1"}]}}}), encoding="utf-8")


def _proxy_down_without_enforcement(state, monkeypatch):
    """iron-proxy stopped and enforce_on_docker off: the legacy fallback is the open bridge."""
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    cfg["proxy"]["enforce_on_docker"] = False
    save_config(cfg)
    monkeypatch.setattr(ip, "_pid_alive", lambda pid: False)


@pytest.mark.parametrize("break_it, error", [
    (_route_out_network, "not internal"),
    (_proxy_down_without_enforcement, "not running"),
])
def test_sandbox_refuses_to_start_off_the_internal_network(egress_ready, monkeypatch, break_it, error):
    break_it(egress_ready, monkeypatch)

    with pytest.raises(RuntimeError, match=error):
        _sandbox_host_args()
