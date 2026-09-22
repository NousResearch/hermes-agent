from __future__ import annotations

import io
import types
from pathlib import Path

import pytest

from agent.secret_scope import current_secret_scope
from hermes_cli import env_loader
from tui_gateway.compute_host import ComputeHost


@pytest.fixture(autouse=True)
def _reset_secret_sources():
    from agent.secret_sources import registry

    env_loader.reset_secret_source_cache()
    registry._reset_registry_for_tests()
    yield
    env_loader.reset_secret_source_cache()
    registry._reset_registry_for_tests()


def _server_observing_scope(captured: dict):
    sessions: dict[str, dict] = {}

    def make_agent(_sid, key, **_kwargs):
        captured["scope"] = dict(current_secret_scope() or {})
        return types.SimpleNamespace(session_id=key)

    def init_session(sid, key, agent, history, **_kwargs):
        sessions[sid] = {
            "agent": agent,
            "session_key": key,
            "history": list(history),
        }

    return types.SimpleNamespace(
        _sessions=sessions,
        _make_agent=make_agent,
        _transfer_db_to_agent=lambda _agent, _db: True,
        _init_session=init_session,
    )


def _build_routed_session(monkeypatch, profile_home: Path, captured: dict) -> None:
    import hermes_state_registry

    monkeypatch.setattr(hermes_state_registry, "acquire", lambda _path: object())
    server = _server_observing_scope(captured)
    host = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
    try:
        host._build_server_session(
            server,
            {
                "sid": "routed-session",
                "session_key": "routed-key",
                "history": [],
                "profile_home": str(profile_home),
                "source": "desktop",
            },
            "routed-session",
        )
    finally:
        host.close()


def test_routed_profile_hydrates_bitwarden_with_inherited_bootstrap_only(
    tmp_path, monkeypatch
):
    profile_home = tmp_path / "profiles" / "target"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        "secrets:\n"
        "  bitwarden:\n"
        "    enabled: true\n"
        "    project_id: test-project\n"
        "    access_token_env: BWS_ACCESS_TOKEN\n"
        "    override_existing: true\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("BWS_ACCESS_TOKEN", "host-bootstrap-token")
    monkeypatch.setenv("GLM_API_KEY", "host-glm-key")
    monkeypatch.setenv("DEEPSEEK_PROXY_API_KEY", "host-deepseek-key")

    import agent.secret_sources.bitwarden as bitwarden

    fetch_args: dict = {}
    monkeypatch.setattr(bitwarden, "find_bws", lambda **_kwargs: Path("/fake/bws"))

    def fake_fetch(**kwargs):
        fetch_args.update(kwargs)
        return {
            "GLM_API_KEY": "target-glm-key",
            "DEEPSEEK_PROXY_API_KEY": "target-deepseek-key",
        }, []

    monkeypatch.setattr(bitwarden, "fetch_bitwarden_secrets", fake_fetch)

    captured: dict = {}
    _build_routed_session(monkeypatch, profile_home, captured)

    assert fetch_args["access_token"] == "host-bootstrap-token"
    assert captured["scope"] == {
        "GLM_API_KEY": "target-glm-key",
        "DEEPSEEK_PROXY_API_KEY": "target-deepseek-key",
    }


def test_routed_profile_without_external_sources_inherits_no_host_credentials(
    tmp_path, monkeypatch
):
    profile_home = tmp_path / "profiles" / "target"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("BWS_ACCESS_TOKEN", "host-bootstrap-token")
    monkeypatch.setenv("GLM_API_KEY", "host-glm-key")
    monkeypatch.setenv("DEEPSEEK_PROXY_API_KEY", "host-deepseek-key")

    captured: dict = {}
    _build_routed_session(monkeypatch, profile_home, captured)

    assert captured["scope"] == {}
