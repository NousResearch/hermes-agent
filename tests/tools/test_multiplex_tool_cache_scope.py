"""Multiplexed gateway: module-level caches in tools/ and agent/ must not hand profile A's
config/.env-derived value to profile B. Real temp homes, real config.yaml/.env, real modules;
only HTTP transports are stubbed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import hermes_yaml as yaml

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

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

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
    config or per-home files: aux-vision routing, learned image cost table, aux
    semaphore, MCP lock."""
    main = {"model": {"provider": "openai", "model": "gpt-4o"}}
    a = _make_home(tmp_path / "A", {**main,
                                    "auxiliary": {"vision": {"provider": "auto"}, "summary": {"max_concurrency": 2}}})
    b = _make_home(tmp_path / "A" / "profiles" / "B", {**main,
                                                       "auxiliary": {"vision": {"provider": "openai", "model": "gpt-4o-mini"},
                                                                     "summary": {"max_concurrency": 7}}})
    monkeypatch.setenv("HERMES_HOME", str(a))
    (a / "cache" / "image_token_costs.json").write_text(json.dumps({"m@gw.example": 1000}), encoding="utf-8")
    (b / "cache" / "image_token_costs.json").write_text(json.dumps({"m@gw.example": 3000}), encoding="utf-8")

    import agent.auxiliary_client as ac
    import agent.image_token_cost as itc
    import tools.computer_use.tool as cu
    from tools import mcp_tool_loop

    ac._reset_aux_semaphores()

    with _scoped(a):
        assert cu._should_route_through_aux_vision() is False  # no explicit aux vision: native path
        assert itc.learned_image_token_cost("m", "http://gw.example/v1") == 1000
        sem_a = ac._acquire_sync_aux_semaphore("summary")
        sem_a.acquire()
        cookie = mcp_tool_loop._try_acquire_mcp_discovery_lock()
        lock_a = cookie._fh.name
        cookie.release()
    with _scoped(b):
        assert cu._should_route_through_aux_vision() is True  # B named a dedicated vision model
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


def test_endpoint_model_catalog_memo_is_keyed_by_credential(two_homes, monkeypatch):
    """Two profiles, same base_url, different api_key: a per-key gateway's catalog fetched with A's
    key must not be served to B from the in-memory memo (the disk memo already lives per home)."""
    from contextlib import contextmanager
    import httpx
    import agent.model_metadata as mm

    a, b = two_homes
    mm._endpoint_model_metadata_cache.clear()
    mm._endpoint_model_metadata_cache_time.clear()

    @contextmanager
    def stream(url, headers=None, **kwargs):
        who = headers.get("Authorization", "").rsplit("-", 1)[-1]
        yield httpx.Response(200, request=httpx.Request("GET", url),
                             json={"data": [{"id": f"model-for-{who}", "context_length": 1}]})

    monkeypatch.setattr(mm.model_metadata_http, "stream", stream)
    with _scoped(a):
        assert set(mm.fetch_endpoint_model_metadata("http://gw.example/v1", api_key="key-A")) == {"model-for-A"}
    with _scoped(b):
        assert set(mm.fetch_endpoint_model_metadata("http://gw.example/v1", api_key="key-B")) == {"model-for-B"}
    with _scoped(a):
        assert set(mm.fetch_endpoint_model_metadata("http://gw.example/v1", api_key="key-A")) == {"model-for-A"}
