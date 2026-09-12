"""Multiplexed gateway: module-level caches in tools/ and agent/ must not hand profile A's
config/.env-derived value to profile B. Real temp homes, real config.yaml/.env, real modules;
only HTTP transports are stubbed.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
import yaml

from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _make_home(root: Path, cfg: dict, env: str = "") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (root / ".env").write_text(env, encoding="utf-8")
    (root / "cache").mkdir(exist_ok=True)
    return root


class _scoped:
    def __init__(self, home: Path):
        self.home = home

    def __enter__(self):
        self._t1 = set_hermes_home_override(str(self.home))
        self._t2 = set_secret_scope(build_profile_secret_scope(self.home))

    def __exit__(self, *_):
        reset_secret_scope(self._t2)
        reset_hermes_home_override(self._t1)


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    a = _make_home(tmp_path / "A", {}, "CAMOFOX_URL=http://camofox-a:9377\n")
    b = _make_home(tmp_path / "A" / "profiles" / "B", {}, "CAMOFOX_URL=http://camofox-b:9377\n")
    monkeypatch.setenv("HERMES_HOME", str(a))
    monkeypatch.delenv("CAMOFOX_URL", raising=False)
    return a, b


def test_camofox_vnc_memo_is_keyed_by_the_profiles_server_url(two_homes, monkeypatch):
    """The one-shot VNC probe must not answer B (own CAMOFOX_URL) with A's server address."""
    import tools.browser_camofox as cam

    class _Resp:
        status_code = 200

        def __init__(self, url):
            self._port = 6001 if "camofox-a" in url else 6002

        def json(self):
            return {"ok": True, "vncPort": self._port}

    monkeypatch.setattr(cam.requests, "get", lambda url, *a, **k: _Resp(url))
    a, b = two_homes
    with _scoped(a):
        assert cam.check_camofox_available() is True
        assert cam.get_vnc_url() == "http://camofox-a:6001"
    with _scoped(b):
        assert cam.get_vnc_url() == "http://camofox-b:6002"
    with _scoped(a):
        assert cam.get_vnc_url() == "http://camofox-a:6001"


def test_home_keyed_caches_serve_each_profile_its_own_config(tmp_path, monkeypatch):
    """One mechanism (dict keyed by home / override bypass) across the sites that read per-profile
    config or per-home files: aux-vision routing, tirith binary path, learned image cost table, aux
    semaphore, MCP lock."""
    bin_a, bin_b = tmp_path / "binA" / "tirith", tmp_path / "binB" / "tirith"
    for p in (bin_a, bin_b):
        p.parent.mkdir()
        p.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        p.chmod(0o755)
    main = {"model": {"provider": "openai", "model": "gpt-4o"}}
    a = _make_home(tmp_path / "A", {**main, "security": {"tirith_path": str(bin_a)},
                                    "auxiliary": {"vision": {"provider": "auto"}, "summary": {"max_concurrency": 2}}})
    b = _make_home(tmp_path / "A" / "profiles" / "B", {**main, "security": {"tirith_path": str(bin_b)},
                                                       "auxiliary": {"vision": {"provider": "openai", "model": "gpt-4o-mini"},
                                                                     "summary": {"max_concurrency": 7}}})
    monkeypatch.setenv("HERMES_HOME", str(a))
    (a / "cache" / "image_token_costs.json").write_text(json.dumps({"m@gw.example": 1000}), encoding="utf-8")
    (b / "cache" / "image_token_costs.json").write_text(json.dumps({"m@gw.example": 3000}), encoding="utf-8")

    import agent.auxiliary_client as ac
    import agent.image_token_cost as itc
    import tools.computer_use.tool as cu
    import tools.tirith_security as tir
    from tools import mcp_tool_loop

    monkeypatch.setattr(tir, "_resolved_path", None)
    monkeypatch.setattr(tir, "_resolved_path_by_home", {})
    ac._reset_aux_semaphores()
    cu._AUX_VISION_ROUTE_CACHE.clear()

    with _scoped(a):
        assert cu._should_route_through_aux_vision() is False  # no explicit aux vision: native path
        assert tir._resolve_tirith_path(tir._load_security_config()["tirith_path"]) == str(bin_a)
        assert itc.learned_image_token_cost("m", "http://gw.example/v1") == 1000
        sem_a = ac._acquire_sync_aux_semaphore("summary")
        sem_a.acquire()
        cookie = mcp_tool_loop._try_acquire_mcp_discovery_lock()
        lock_a = cookie._fh.name
        cookie.release()
    with _scoped(b):
        assert cu._should_route_through_aux_vision() is True  # B named a dedicated vision model
        assert tir._resolve_tirith_path(tir._load_security_config()["tirith_path"]) == str(bin_b)
        assert itc.learned_image_token_cost("m", "http://gw.example/v1") == 3000
        sem_b = ac._acquire_sync_aux_semaphore("summary")
        cookie = mcp_tool_loop._try_acquire_mcp_discovery_lock()
        lock_b = cookie._fh.name
        cookie.release()
    assert Path(lock_a).parent == a and Path(lock_b).parent == b
    assert sem_b is not sem_a
    with _scoped(a):
        # B's differently-sized lookup must not have rebuilt the semaphore A is holding.
        assert ac._acquire_sync_aux_semaphore("summary") is sem_a
    sem_a.release()


def test_debounced_sync_push_fires_in_the_scheduling_profiles_context(two_homes, monkeypatch):
    """Timer threads start with empty ContextVars: the push must run under the writing profile's
    home, and B's write must not cancel A's pending push."""
    import tools.skill_manager_tool as smt
    import tools.skill_usage as su
    import tools.skills_sync_client as ssc
    from hermes_constants import get_hermes_home

    a, b = two_homes
    fired: dict[str, str] = {}
    both = threading.Event()

    def fake_push(*, message=""):
        fired[message] = str(get_hermes_home())
        if len(fired) == 2:
            both.set()

    monkeypatch.setattr(su, "is_sync_enabled", lambda name: True)
    monkeypatch.setattr(ssc, "maybe_push_skills", fake_push)
    monkeypatch.setattr(smt, "_SYNC_PUSH_DEBOUNCE_S", 0.05)
    monkeypatch.setattr(smt, "_sync_push_timers", {})
    with _scoped(a):
        smt._maybe_debounced_sync_push("skill-a")
    with _scoped(b):
        smt._maybe_debounced_sync_push("skill-b")
    assert both.wait(5), fired
    assert fired == {"sync: skill-a": str(a), "sync: skill-b": str(b)}


def test_endpoint_model_catalog_memo_is_keyed_by_credential(two_homes, monkeypatch):
    """Two profiles, same base_url, different api_key: a per-key gateway's catalog fetched with A's
    key must not be served to B from the in-memory memo (the disk memo already lives per home)."""
    import agent.model_metadata as mm

    a, b = two_homes
    mm._endpoint_model_metadata_cache.clear()
    mm._endpoint_model_metadata_cache_time.clear()
    mm._ensure_requests()

    class _Resp:
        status_code, ok = 200, True

        def __init__(self, headers):
            self._who = headers.get("Authorization", "").rsplit("-", 1)[-1]

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"id": f"model-for-{self._who}", "context_length": 1}]}

        def close(self):
            pass

    monkeypatch.setattr(mm.requests, "get", lambda url, headers=None, **k: _Resp(headers or {}))
    with _scoped(a):
        assert set(mm.fetch_endpoint_model_metadata("http://gw.example/v1", api_key="key-A")) == {"model-for-A"}
    with _scoped(b):
        assert set(mm.fetch_endpoint_model_metadata("http://gw.example/v1", api_key="key-B")) == {"model-for-B"}
    with _scoped(a):
        assert set(mm.fetch_endpoint_model_metadata("http://gw.example/v1", api_key="key-A")) == {"model-for-A"}
