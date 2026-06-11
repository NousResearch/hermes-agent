"""End-to-end test for the multi-profile Hermes dashboard endpoints.

Verifies that the dashboard config / env / messaging-platform endpoints
target the profile named in the request body, not the profile the
dashboard was launched with.  Regression test for the silent cross-profile
write bug where configuring Feishu in the "writer" profile would write
to the launch profile's .env (issue: multi-profile API gap).

Run with:

    venv/bin/python3 -m pytest tests/test_web_server_multi_profile.py -q

or, if the isolation plugin is wired into pyproject.toml:

    venv/bin/python3 -m pytest tests/test_web_server_multi_profile.py -q --no-isolate

The test uses tmp_path for hermes home directories and a stub
``HERMES_HOME`` so the actual user config is not touched.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_profile_env(tmp_path, monkeypatch):
    """Set up a hermes layout with several profile directories.

    Returns a dict ``{"homes": {name: Path, ...}, "default_home": Path}``.
    The default (launch-profile) home is what the web server's process
    sees as ``get_hermes_home()``; the named profile homes are what
    the endpoints should write to when ``profile=<name>`` is passed.
    """
    root = tmp_path / "hermes_root"
    default_home = root / "default"
    default_home.mkdir(parents=True)
    profiles_root = root / "profiles"
    profiles_root.mkdir()

    homes: dict[str, Path] = {"default": default_home}
    for name in ("writer", "chief", "cto"):
        home = profiles_root / name
        home.mkdir(parents=True)
        homes[name] = home

    # The dashboard is launched with HERMES_HOME pointing at the default
    # (launch) profile, mimicking the real `hermes dashboard` flow.
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    # The profiles.py helper looks for ~/.hermes/profiles — point it at
    # our isolated root by symlinking, or override its root resolver.
    # The simplest approach: monkeypatch ``hermes_cli.profiles._get_profiles_root``.
    from hermes_cli import profiles as _profiles_mod
    monkeypatch.setattr(_profiles_mod, "_get_profiles_root", lambda: profiles_root)
    # Also patch the default-home resolver to return our default home, so
    # the "default" profile points to default_home rather than ~/.hermes.
    monkeypatch.setattr(_profiles_mod, "_get_default_hermes_home", lambda: default_home)

    # Reload hermes_constants so it picks up HERMES_HOME from env.
    import importlib
    from hermes_constants import (
        get_hermes_home, get_env_path, get_config_path,
    )
    # Force-resolve at this point so the cached env_path / config_path
    # inside the web_server's helpers (if any) reflect the test layout.
    assert str(get_hermes_home()) == str(default_home)

    yield {"homes": homes, "default_home": default_home, "profiles_root": profiles_root}


@pytest.fixture
def client(multi_profile_env):
    """A FastAPI TestClient pointed at the real web_server app.

    Skips when fastapi/httpx is not installed in the test environment.
    """
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")

    from fastapi.testclient import TestClient
    from hermes_cli.web_server import app, _SESSION_TOKEN

    with TestClient(app) as c:
        # Inject the session token on every request so we pass the
        # dashboard's auth middleware (set up for protecting
        # env/config writes against XSRF).  Tests exercise the
        # *write-path* logic, not the auth gate.
        c.headers.update({"X-Hermes-Session-Token": _SESSION_TOKEN})
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_env(home: Path, key: str) -> str | None:
    env_path = home / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1]
    return None


def _read_yaml(home: Path, dotted: str):
    import yaml
    p = home / "config.yaml"
    if not p.exists():
        return None
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    for part in dotted.split("."):
        if not isinstance(data, dict):
            return None
        data = data.get(part)
    return data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMultiProfileConfig:
    """The dashboard's write endpoints must honor the ``profile`` field."""

    def test_put_messaging_platform_writes_to_named_profile(
        self, client, multi_profile_env
    ):
        """PUT /api/messaging/platforms/feishu with profile=writer writes
        FEISHU_* to writer/.env, not the launch-profile (default) .env."""
        default_home = multi_profile_env["default_home"]
        writer_home = multi_profile_env["homes"]["writer"]
        chief_home = multi_profile_env["homes"]["chief"]

        resp = client.put(
            "/api/messaging/platforms/feishu",
            json={
                "profile": "writer",
                "env": {
                    "FEISHU_APP_ID": "cli_writer_test_123",
                    "FEISHU_APP_SECRET": "writer_secret_abc",
                },
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ok"] is True
        assert body["profile"] == "writer"

        # Writer got the credentials.
        assert _read_env(writer_home, "FEISHU_APP_ID") == "cli_writer_test_123"
        assert _read_env(writer_home, "FEISHU_APP_SECRET") == "writer_secret_abc"

        # The launch profile and other profiles were not touched.
        assert _read_env(default_home, "FEISHU_APP_ID") is None
        assert _read_env(chief_home, "FEISHU_APP_ID") is None

    def test_put_messaging_platform_unknown_profile_returns_404(
        self, client, multi_profile_env
    ):
        resp = client.put(
            "/api/messaging/platforms/feishu",
            json={
                "profile": "nonexistent_xyz",
                "env": {"FEISHU_APP_ID": "x"},
            },
        )
        assert resp.status_code == 404
        assert "Unknown profile" in resp.json()["detail"]

    def test_put_messaging_platform_omitted_profile_preserves_legacy(
        self, client, multi_profile_env
    ):
        """profile=None / omitted must continue to write to the launch
        profile's .env, matching the pre-fix behavior."""
        default_home = multi_profile_env["default_home"]

        resp = client.put(
            "/api/messaging/platforms/feishu",
            json={"env": {"FEISHU_APP_ID": "cli_legacy"}},
        )
        assert resp.status_code == 200, resp.text
        # Profile field is null in the response.
        assert resp.json()["profile"] is None
        # Wrote to the launch profile.
        assert _read_env(default_home, "FEISHU_APP_ID") == "cli_legacy"

    def test_put_env_writes_to_named_profile(self, client, multi_profile_env):
        """PUT /api/env with profile=writer writes to writer/.env."""
        writer_home = multi_profile_env["homes"]["writer"]
        default_home = multi_profile_env["default_home"]

        resp = client.put(
            "/api/env",
            json={"profile": "writer", "key": "OPENAI_API_KEY", "value": "sk-writer"},
        )
        assert resp.status_code == 200, resp.text
        assert _read_env(writer_home, "OPENAI_API_KEY") == "sk-writer"
        assert _read_env(default_home, "OPENAI_API_KEY") is None

    def test_delete_env_removes_from_named_profile(self, client, multi_profile_env):
        """DELETE /api/env with profile=writer removes from writer/.env
        and returns 404 if the key lives only in another profile."""
        writer_home = multi_profile_env["homes"]["writer"]

        # Seed: write a key to writer first.
        (writer_home / ".env").write_text("WRITER_KEY=writer_val\n", encoding="utf-8")

        # Delete from writer — should succeed.
        resp = client.request(
            "DELETE", "/api/env", json={"profile": "writer", "key": "WRITER_KEY"}
        )
        assert resp.status_code == 200, resp.text
        assert _read_env(writer_home, "WRITER_KEY") is None

        # Delete from writer again — should 404 (key no longer there).
        resp = client.request(
            "DELETE", "/api/env", json={"profile": "writer", "key": "WRITER_KEY"}
        )
        assert resp.status_code == 404

    def test_put_messaging_platform_enabled_flag_per_profile(
        self, client, multi_profile_env
    ):
        """The ``enabled`` flag is per-profile config.yaml; verify
        enabling Feishu in writer doesn't enable it in default."""
        writer_home = multi_profile_env["homes"]["writer"]
        default_home = multi_profile_env["default_home"]

        resp = client.put(
            "/api/messaging/platforms/feishu",
            json={"profile": "writer", "enabled": True},
        )
        assert resp.status_code == 200, resp.text

        # Writer's config.yaml has the flag set.
        assert _read_yaml(writer_home, "platforms.feishu.enabled") is True
        # Default home's config.yaml does not have it (or has it False).
        assert _read_yaml(default_home, "platforms.feishu.enabled") in (None, False)

    def test_pydantic_model_accepts_profile_field(self):
        """Sanity check: the new Pydantic models parse ``profile`` correctly
        and default to None — guarantees no breaking change for clients
        that don't send the field."""
        from hermes_cli.web_server import (
            ConfigUpdate, EnvVarUpdate, EnvVarDelete, EnvVarReveal,
            MessagingPlatformUpdate,
        )

        assert ConfigUpdate(config={}).profile is None
        assert ConfigUpdate(config={}, profile="writer").profile == "writer"

        assert EnvVarUpdate(key="X", value="y").profile is None
        assert EnvVarUpdate(key="X", value="y", profile="cto").profile == "cto"

        assert EnvVarDelete(key="X").profile is None
        assert EnvVarDelete(key="X", profile="writer").profile == "writer"

        assert EnvVarReveal(key="X").profile is None
        assert EnvVarReveal(key="X", profile="chief").profile == "chief"

        assert MessagingPlatformUpdate().profile is None
        assert MessagingPlatformUpdate(profile="writer").profile == "writer"
