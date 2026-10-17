"""Cross-provider delegated children must share one pool so sibling leases balance (#140495).

A child whose provider differs from the parent's used to get a fresh ``load_pool()`` instance of
its own, whose lease counts are all zero, so ``acquire_lease()``'s ``(active_leases, priority)``
pick pinned every concurrent sibling to the priority-0 entry under every pool strategy. Sharing
one instance per provider per process restores the balancing same-provider children already get
from the parent's pool.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool_config import (
    _DELEGATED_POOLS,
    _loaded_pool,
    _resolve_child_credential_pool,
)

BASE = "https://openrouter.ai/api/v1"
ROWS = [
    {
        "id": f"k{i}",
        "label": f"key-{i}",
        "auth_type": "api_key",
        "access_token": f"synthetic-key-{i}",
        "base_url": BASE,
        "priority": i,
        "source": "manual",
    }
    for i in range(2)
]
ENTRY_IDS = {row["id"] for row in ROWS}


@pytest.fixture(autouse=True)
def _clean_pool_cache():
    """The shared-pool cache is process-global; no test may see another's instance."""
    _DELEGATED_POOLS.clear()
    yield
    _DELEGATED_POOLS.clear()


@pytest.fixture(autouse=True)
def _no_seeding_env(monkeypatch):
    for var in (
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)


def _spawn_children(parent_provider: str, n: int = 4) -> list[str]:
    """Lease *n* concurrent children (no release in between), as a delegate_task fan-out does."""
    from hermes_cli.auth import write_credential_pool
    from tools.delegate_tool_child_run import _lease_child_credential

    write_credential_pool("openrouter", ROWS)
    parent = SimpleNamespace(
        provider=parent_provider,
        _credential_pool=None,  # an anthropic parent owns no openrouter pool
        base_url=BASE,
        requested_provider=parent_provider,
    )
    leased = []
    for _ in range(n):
        child = SimpleNamespace(base_url=BASE, _swap_credential=lambda entry: None)
        child._credential_pool = _resolve_child_credential_pool(
            "openrouter", parent, BASE
        )
        _, lease_id = _lease_child_credential(child)
        leased.append(lease_id)
    return leased


def test_cross_provider_children_spread_across_pool_entries():
    leased = _spawn_children("anthropic")
    assert set(leased) == ENTRY_IDS


def test_cross_provider_children_share_one_pool_instance():
    from hermes_cli.auth import write_credential_pool

    write_credential_pool("openrouter", ROWS)
    parent = SimpleNamespace(
        provider="anthropic",
        _credential_pool=None,
        base_url=BASE,
        requested_provider="anthropic",
    )
    pools = [
        _resolve_child_credential_pool("openrouter", parent, BASE),
        _resolve_child_credential_pool("openrouter", parent, BASE),
    ]
    assert all(pool is not None for pool in pools)
    assert pools[0] is pools[1]


def test_released_lease_lets_next_child_reuse_priority_zero():
    """Teardown's release_lease() lands on the shared pool: after both keys were leased, releasing
    key-0 lets a later child lease it again instead of piling onto key-1 forever."""
    from hermes_cli.auth import write_credential_pool
    from tools.delegate_tool_child_run import _lease_child_credential

    write_credential_pool("openrouter", ROWS)
    parent = SimpleNamespace(
        provider="anthropic",
        _credential_pool=None,
        base_url=BASE,
        requested_provider="anthropic",
    )
    child = SimpleNamespace(base_url=BASE, _swap_credential=lambda entry: None)
    child._credential_pool = _resolve_child_credential_pool("openrouter", parent, BASE)
    pool, first_id = _lease_child_credential(child)
    _, second_id = _lease_child_credential(
        SimpleNamespace(
            base_url=BASE, _swap_credential=lambda entry: None, _credential_pool=pool
        )
    )
    assert first_id != second_id

    pool.release_lease(first_id)
    _, third_id = _lease_child_credential(
        SimpleNamespace(
            base_url=BASE, _swap_credential=lambda entry: None, _credential_pool=pool
        )
    )
    assert third_id == first_id


def test_pool_without_credentials_is_not_cached():
    """A provider with no usable credentials returns None every time without caching, so a later
    ``auth add`` is still visible to the next delegated child."""
    empty_pool = MagicMock()
    empty_pool.has_credentials.return_value = False
    with patch("agent.credential_pool.load_pool", return_value=empty_pool) as load_mock:
        assert _loaded_pool("openrouter") is None
        assert _loaded_pool("openrouter") is None
    assert load_mock.call_count == 2
    assert "openrouter" not in _DELEGATED_POOLS
