"""One Desktop backend serving two profiles: a secondary session's key save and model belong to IT.

``model.save_key`` bound by ``session_id`` wrote the LAUNCH profile's ``.env`` and exported the key
process-wide; ``session.create {profile}`` reported (and first-persisted) the launch profile's model.
"""

from __future__ import annotations

import os
from unittest.mock import Mock, patch

import pytest
import hermes_yaml as yaml

import tui_gateway.server as server


@pytest.fixture
def homes(tmp_path, monkeypatch):
    launch, worker = tmp_path / "launch", tmp_path / "profiles" / "worker"
    for home, model in ((launch, "launch-model"), (worker, "worker-model")):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(yaml.safe_dump({"model": {"default": model}}), encoding="utf-8")
        (home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    for seed in ("HERMES_MODEL", "HERMES_INFERENCE_MODEL"):
        monkeypatch.delenv(seed, raising=False)
    monkeypatch.setenv("GLM_API_KEY", "launch-own-key")
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setattr(server, "_cfg_cache", None)
    monkeypatch.setattr(server, "_cfg_sig", None)
    monkeypatch.setattr(server, "_cfg_path", None)
    monkeypatch.setattr(server, "_profile_home", lambda name: worker if name == "worker" else None)
    return launch, worker


def test_session_bound_save_key_writes_only_the_session_profiles_env(homes, monkeypatch):
    launch, worker = homes
    monkeypatch.setattr("hermes_cli.inventory.build_models_payload", Mock(return_value={"providers": []}))
    key = "glm-" + "worker-canary"
    session = {"agent": None, "profile_home": str(worker), "session_key": "worker-session"}
    with patch.dict(server._sessions, {"s-worker": session}, clear=False):
        resp = server._methods["model.save_key"](
            "rid", {"session_id": "s-worker", "slug": "zai", "api_key": key})
    assert "result" in resp, resp
    assert key in (worker / ".env").read_text(encoding="utf-8")
    assert key not in (launch / ".env").read_text(encoding="utf-8")
    assert os.environ["GLM_API_KEY"] == "launch-own-key", "a secondary's key never reaches the process env"


def test_session_create_for_a_profile_reports_and_persists_its_own_model(homes, monkeypatch, tmp_path):
    for name in ("_schedule_agent_build", "_schedule_session_cap_enforcement", "_register_session_cwd"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_completion_cwd", lambda *a, **k: str(tmp_path))
    known = set(server._sessions)
    try:
        worker_resp = server.handle_request(
            {"id": "1", "method": "session.create", "params": {"profile": "worker", "source": "desktop"}})
        launch_resp = server.handle_request(
            {"id": "2", "method": "session.create", "params": {"source": "desktop"}})
        assert worker_resp["result"]["info"]["model"] == "worker-model"
        assert launch_resp["result"]["info"]["model"] == "launch-model"
        worker_session = server._sessions[worker_resp["result"]["session_id"]]
        assert server._workdir_row_model_config(worker_session)[0] == "worker-model"
    finally:
        with server._sessions_lock:
            for sid in [s for s in server._sessions if s not in known]:
                server._sessions.pop(sid, None)
