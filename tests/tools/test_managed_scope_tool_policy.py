"""Administrator (managed-scope) policy must reach the tool-layer readers.

``/etc/hermes/config.yaml`` is the IT-pushed, user-immutable config layer
(``hermes_cli/managed_scope.py``). It is merged by ``load_config()`` and by
``load_user_config_effective()``, but NOT by ``read_raw_config()``, which parses the
user's own ``~/.hermes/config.yaml`` only. Three sandbox/network policy readers in
``tools/`` used the raw reader, so a value that exists ONLY in the managed scope was
silently ignored — the same class as the MCP discovery gate (#91073).
"""

from __future__ import annotations

import pytest

from hermes_cli import managed_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import credential_files, env_passthrough, url_safety

_MANAGED_CONFIG = """\
security:
  allow_private_urls: true
terminal:
  env_passthrough:
    - MANAGED_ONLY_VAR
  credential_files:
    - managed-creds.json
"""


@pytest.fixture
def managed_profile(tmp_path, monkeypatch):
    """Active profile whose own config.yaml carries none of the policy keys, plus a
    managed scope that carries all of them."""
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("display:\n  personality: default\n", encoding="utf-8")
    (home / "managed-creds.json").write_text("{}\n", encoding="utf-8")
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(_MANAGED_CONFIG, encoding="utf-8")

    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
    managed_scope.invalidate_managed_cache()
    url_safety._reset_allow_private_cache()
    env_passthrough._config_passthrough.clear()
    credential_files._config_files.clear()
    token = set_hermes_home_override(home)
    try:
        yield home
    finally:
        reset_hermes_home_override(token)
        managed_scope.invalidate_managed_cache()
        url_safety._reset_allow_private_cache()
        env_passthrough._config_passthrough.clear()
        credential_files._config_files.clear()


def test_managed_layer_is_the_only_source(managed_profile):
    """Guard for the fixture itself: the policy keys exist ONLY in the managed layer."""
    from hermes_cli.config import read_raw_config
    from hermes_cli.config_effective import load_user_config_effective

    raw = read_raw_config()
    assert raw.get("security") is None and raw.get("terminal") is None
    effective = load_user_config_effective()
    assert effective["security"]["allow_private_urls"] is True
    assert effective["terminal"]["env_passthrough"] == ["MANAGED_ONLY_VAR"]


def test_managed_allow_private_urls_is_honored(managed_profile):
    assert url_safety._resolve_allow_private_urls() is True


def test_managed_env_passthrough_is_honored(managed_profile):
    assert env_passthrough.is_env_passthrough("MANAGED_ONLY_VAR") is True


def test_managed_credential_files_are_mounted(managed_profile):
    mounts = credential_files.get_credential_file_mounts()
    assert [m["container_path"] for m in mounts] == ["/root/.hermes/managed-creds.json"]
    assert mounts[0]["host_path"] == str(managed_profile / "managed-creds.json")


def test_managed_credential_files_reject_master_store(managed_profile):
    """A managed mount list cannot bypass the guard for Hermes' own credentials."""
    (managed_profile / "auth.json").write_text("{}\n", encoding="utf-8")
    managed_dir = managed_scope.get_managed_dir()
    assert managed_dir is not None
    (managed_dir / "config.yaml").write_text(
        _MANAGED_CONFIG.replace(
            "    - managed-creds.json\n",
            "    - auth.json\n    - managed-creds.json\n",
        ),
        encoding="utf-8",
    )
    managed_scope.invalidate_managed_cache()
    credential_files._config_files.clear()

    # The skill-registration path already rejects this file. The managed
    # config path must do the same without dropping an ordinary credential.
    assert credential_files.register_credential_file("auth.json") is False
    assert [m["container_path"] for m in credential_files.get_credential_file_mounts()] == [
        "/root/.hermes/managed-creds.json",
    ]
