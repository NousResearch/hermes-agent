"""terminal.egress: a configured outbound allowlist reaches the cloud provider's native network
policy, and a backend that cannot enforce it refuses to start instead of running open."""

from __future__ import annotations

import json

import pytest


@pytest.fixture()
def egress_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()

    def _write(egress):
        (home / "config.yaml").write_text(json.dumps({"terminal": {"egress": egress}}), encoding="utf-8")
        return home

    monkeypatch.setenv("HERMES_HOME", str(home))
    return _write


def _create(env_type, **cc):
    from tools.terminal_tool_backends import _create_environment

    return _create_environment(env_type=env_type, image="img", cwd="/root", timeout=60,
                               container_config={"container_cpu": 1, "container_memory": 2048,
                                                 "container_disk": 0, "container_persistent": False, **cc},
                               task_id="egress-test")


@pytest.mark.parametrize("env_type", ["local", "docker", "ssh"])
def test_backend_without_native_allowlist_refuses_to_start(egress_home, env_type, monkeypatch):
    egress_home({"domains": ["api.github.com"]})
    started = []
    import tools.terminal_tool_backends as backends
    monkeypatch.setitem(backends._ENV_BUILDERS, env_type, lambda **kw: started.append(kw))

    with pytest.raises(ValueError, match=r"terminal\.egress is set, but the .* backend has no provider-enforced"):
        _create(env_type)
    assert started == []


def test_direct_modal_gets_native_allowlists_and_daytona_policy(egress_home, monkeypatch):
    egress_home({"domains": ["API.GitHub.com.", "*.pypi.org"], "cidrs": ["203.0.113.7", "10.0.0.0/8"]})
    import tools.terminal_tool_backends as backends
    captured = {}

    class _Modal:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(backends, "_ModalEnvironment", _Modal)
    monkeypatch.setattr(backends, "_get_modal_backend_state", lambda mode: {"selected_backend": "direct"})
    _create("modal", modal_mode="direct")
    sandbox_kwargs = captured["modal_sandbox_kwargs"]
    assert sandbox_kwargs["outbound_domain_allowlist"] == ["*.pypi.org", "api.github.com"]
    assert sandbox_kwargs["outbound_cidr_allowlist"] == ["10.0.0.0/8", "203.0.113.7/32"]

    # An empty allowlist is "no network", never "open".
    egress_home({"domains": [], "cidrs": []})
    _create("modal", modal_mode="direct")
    assert captured["modal_sandbox_kwargs"]["block_network"] is True

    # Managed Modal's create API takes no network policy: refuse rather than run open.
    monkeypatch.setattr(backends, "_get_modal_backend_state", lambda mode: {"selected_backend": "managed"})
    with pytest.raises(ValueError, match="managed Modal"):
        _create("modal")

    # Daytona: one list per sandbox, so a mixed list it cannot enforce exactly is refused.
    from tools.environments.sandbox_egress import SandboxEgress, daytona_create_kwargs, parse_sandbox_egress
    assert daytona_create_kwargs(SandboxEgress(("a.example.com",), ())) == {"domain_allow_list": "a.example.com"}
    with pytest.raises(ValueError, match="either a domain list or a CIDR list"):
        daytona_create_kwargs(SandboxEgress(("a.example.com",), ("1.1.1.1/32",)))
    with pytest.raises(ValueError, match="not a hostname"):
        parse_sandbox_egress({"domains": ["https://api.github.com/x"]})
