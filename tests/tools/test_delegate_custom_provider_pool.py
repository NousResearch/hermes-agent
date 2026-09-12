"""Delegated named-custom routes use their endpoint-scoped credential pool."""

import json
from types import SimpleNamespace

import yaml

from tools.delegate_tool_config import _resolve_child_credential_pool


CHILD_URL = "https://child.example/v1"
PARENT_URL = "https://parent.example/v1"


def _write_named_custom_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "custom_providers": [
                    {"name": "parent-custom", "base_url": PARENT_URL},
                    {"name": "child-custom", "base_url": CHILD_URL},
                ]
            }
        ),
        encoding="utf-8",
    )
    (home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {},
                "credential_pool": {
                    "custom:parent-custom": [
                        {
                            "id": "parent-key",
                            "auth_type": "api_key",
                            "source": "manual",
                            "access_token": "parent-secret",
                        }
                    ],
                    "custom:child-custom": [
                        {
                            "id": "child-key",
                            "auth_type": "api_key",
                            "source": "manual",
                            "access_token": "child-secret",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )


def test_bare_and_canonical_custom_names_resolve_the_same_pool(tmp_path, monkeypatch):
    _write_named_custom_home(tmp_path, monkeypatch)
    parent = SimpleNamespace(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        _credential_pool=None,
    )

    pools = [
        _resolve_child_credential_pool(name, parent, CHILD_URL)
        for name in ("child-custom", "custom:child-custom")
    ]

    assert all(pool is not None for pool in pools)
    assert {pool.provider for pool in pools} == {"custom:child-custom"}
    assert [{entry.id for entry in pool.entries()} for pool in pools] == [
        {"child-key"},
        {"child-key"},
    ]


def test_named_custom_child_does_not_inherit_another_endpoint_pool(
    tmp_path, monkeypatch
):
    _write_named_custom_home(tmp_path, monkeypatch)
    from agent.credential_pool import load_pool

    parent_pool = load_pool("custom:parent-custom")
    parent = SimpleNamespace(
        provider="parent-custom",
        base_url=PARENT_URL,
        _credential_pool=parent_pool,
    )

    child_pool = _resolve_child_credential_pool("child-custom", parent, CHILD_URL)

    assert child_pool is not None
    assert child_pool is not parent_pool
    assert child_pool.provider == "custom:child-custom"
    assert {entry.id for entry in child_pool.entries()} == {"child-key"}
