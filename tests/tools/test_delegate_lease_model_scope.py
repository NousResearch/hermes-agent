"""Subagent credential leases are scoped to the child's model (#147822).

An entitlement rejection benches only ``(credential, model)``, so a bench on one model must not
take the credential away from a child running another model. The endpoint fallback leases by id,
past the pool's availability filter, so it has to honour the same scope itself.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.credential_pool import AUTH_TYPE_OAUTH, CredentialPool, PooledCredential
from tools.delegate_tool_child_run import _lease_child_credential

PUBLIC = "https://public.invalid/v1"
PRIVATE = "https://private.invalid/v1"


def _entry(entry_id: str, priority: int, base_url: str = PUBLIC) -> PooledCredential:
    return PooledCredential(
        provider="openai-codex", id=entry_id, label=entry_id, auth_type=AUTH_TYPE_OAUTH, priority=priority,
        source="manual", access_token=f"token-{entry_id}", expires_at="2099-01-01T00:00:00Z", base_url=base_url,
    )


def _bench(pool: CredentialPool, entry_id: str, model: str) -> None:
    pool.mark_exhausted_and_rotate(
        status_code=400, credential_id=entry_id, failure_reason="model_entitlement", model=model,
        error_context={"message": f"The '{model}' model is not supported when using Codex with a ChatGPT account."},
    )


def _child(pool: CredentialPool, model, base_url: str = PUBLIC) -> SimpleNamespace:
    return SimpleNamespace(model=model, base_url=base_url, _credential_pool=pool, _swap_credential=MagicMock())


@pytest.fixture
def isolated_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


def test_a_bench_on_another_model_leaves_the_credential_leasable(isolated_auth):
    pool = CredentialPool("openai-codex", [_entry("a", 0), _entry("b", 1)])
    _bench(pool, "a", "model-a")

    _, lease_id = _lease_child_credential(_child(pool, "model-b"))
    assert lease_id == "a"
    pool.release_lease(lease_id)

    _, lease_id = _lease_child_credential(_child(pool, "model-a"))
    assert lease_id == "b"


def test_a_child_with_no_model_stays_conservative(isolated_auth):
    pool = CredentialPool("openai-codex", [_entry("a", 0), _entry("b", 1)])
    _bench(pool, "a", "model-a")

    _, lease_id = _lease_child_credential(_child(pool, None))
    assert lease_id == "b"


def test_endpoint_fallback_skips_an_entry_benched_for_the_childs_model(isolated_auth):
    pool = CredentialPool(
        "openai-codex",
        [_entry("pub", 0), _entry("priv-a", 1, PRIVATE), _entry("priv-b", 2, PRIVATE)],
    )
    _bench(pool, "priv-a", "model-a")
    pool.acquire_lease("priv-b")  # tilt the least-leased pick away from the child's endpoint
    pool.acquire_lease("priv-a")

    child = _child(pool, "model-a", PRIVATE)
    _, lease_id = _lease_child_credential(child)

    assert lease_id == "priv-b"
    assert child._swap_credential.call_args[0][0].id == "priv-b"
