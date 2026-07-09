"""Regression tests for #61195 — delegation.base_url must stay authoritative.

Bug shape: ``delegation.base_url: https://api.anthropic.com`` with a primary
``model.provider: openrouter``.  The child's ``provider``/``base_url`` resolve
correctly through ``_resolve_delegation_credentials() -> _build_child_agent()
-> init_agent()``, but the startup credential lease in ``_run_single_child()``
called ``_swap_credential()`` with whatever pool entry it found — and
``_swap_credential`` applies the entry's ``base_url`` unconditionally.  A pool
entry carrying the parent's endpoint retargets the child's outbound HTTP to
``https://openrouter.ai/api/v1`` while ``agent.provider`` still says
``anthropic`` — the exact mismatched pair from the issue's errors.log, failing
with 401 "Missing Authentication header".

The fix is structural, at the choke point every rotation flows through:

* ``_swap_credential`` refuses cross-provider entries outright
  (``_credential_entry_compatible``), mirroring the caller-side guards from
  #33088 / #56885 that previously had to be re-added one call site at a time.
* An explicit ``delegation.base_url`` is pinned (``_delegation_endpoint_pin``):
  same-provider rotation swaps keys but keeps the configured endpoint.
* ``_resolve_delegation_credentials`` prefers the user's own Anthropic
  credential for a direct api.anthropic.com endpoint instead of inheriting
  the parent's (foreign-provider) key.
"""

import threading
import unittest
from unittest.mock import MagicMock, patch

from agent.credential_pool import CredentialPool, PooledCredential
from tools.delegate_tool import (
    _build_child_agent,
    _resolve_delegation_credentials,
)

ANTHROPIC_URL = "https://api.anthropic.com"
OPENROUTER_URL = "https://openrouter.ai/api/v1"

DELEGATION_CFG = {"base_url": ANTHROPIC_URL, "model": "claude-sonnet-4-5"}


def _entry(provider, token, base_url):
    return PooledCredential(
        provider=provider,
        id=f"{provider}-{token[-6:]}",
        label=f"{provider} entry",
        auth_type="api_key",
        priority=0,
        source=f"env:{provider.upper()}_API_KEY",
        access_token=token,
        base_url=base_url,
    )


def _make_parent():
    parent = MagicMock()
    parent.base_url = OPENROUTER_URL
    parent.api_key = "sk-or-v1-parent-key"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "moonshotai/kimi-k2.7-code"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent.openrouter_min_coding_score = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent._credential_pool = CredentialPool(
        "openrouter", [_entry("openrouter", "sk-or-v1-pool-key", OPENROUTER_URL)]
    )
    parent._fallback_chain = None
    parent.reasoning_config = None
    parent.prefill_messages = None
    parent.max_tokens = None
    parent.acp_command = None
    parent.acp_args = []
    parent.session_id = "parent-session"
    parent._current_turn_id = "turn-1"
    return parent


def _build_delegated_child(child_pool):
    """Build a real child through the production path with a controlled pool."""
    parent = _make_parent()
    with patch(
        "agent.anthropic_adapter.resolve_anthropic_token",
        return_value="sk-ant-native-key",
    ):
        creds = _resolve_delegation_credentials(DELEGATION_CFG, parent)
        with patch(
            "tools.delegate_tool._resolve_child_credential_pool",
            return_value=child_pool,
        ):
            child = _build_child_agent(
                task_index=0,
                goal="regression",
                context=None,
                toolsets=None,
                model=creds["model"],
                max_iterations=5,
                task_count=1,
                parent_agent=parent,
                override_provider=creds["provider"],
                override_base_url=creds["base_url"],
                override_api_key=creds["api_key"],
                override_api_mode=creds["api_mode"],
                pin_base_url=bool(creds.get("base_url_pinned")),
                role="leaf",
            )
    return child


def _lease_like_run_single_child(child):
    """Verbatim startup-lease logic from delegate_tool._run_single_child."""
    child_pool = getattr(child, "_credential_pool", None)
    if child_pool is not None:
        leased = child_pool.acquire_lease()
        if leased is not None:
            entry = child_pool.current()
            if entry is not None and hasattr(child, "_swap_credential"):
                child._swap_credential(entry)


class TestDelegationCredentialResolution(unittest.TestCase):
    def test_direct_anthropic_endpoint_resolves_native_key(self):
        """delegation.base_url=api.anthropic.com must not inherit the parent's
        foreign-provider key when the user has an Anthropic credential."""
        parent = _make_parent()
        with patch(
            "agent.anthropic_adapter.resolve_anthropic_token",
            return_value="sk-ant-native-key",
        ):
            creds = _resolve_delegation_credentials(DELEGATION_CFG, parent)
        self.assertEqual(creds["provider"], "anthropic")
        self.assertEqual(creds["base_url"], ANTHROPIC_URL)
        self.assertEqual(creds["api_mode"], "anthropic_messages")
        self.assertEqual(creds["api_key"], "sk-ant-native-key")
        self.assertTrue(creds["base_url_pinned"])

    def test_no_native_key_falls_back_to_parent_inheritance(self):
        """Without an Anthropic credential the pre-existing behavior stands:
        api_key=None -> _build_child_agent inherits the parent's key."""
        parent = _make_parent()
        with patch(
            "agent.anthropic_adapter.resolve_anthropic_token", return_value=None
        ):
            creds = _resolve_delegation_credentials(DELEGATION_CFG, parent)
        self.assertIsNone(creds["api_key"])

    def test_explicit_delegation_api_key_still_wins(self):
        parent = _make_parent()
        cfg = dict(DELEGATION_CFG, api_key="sk-ant-explicit")
        with patch(
            "agent.anthropic_adapter.resolve_anthropic_token",
            return_value="sk-ant-native-key",
        ):
            creds = _resolve_delegation_credentials(cfg, parent)
        self.assertEqual(creds["api_key"], "sk-ant-explicit")


class TestStartupLeaseCannotRetargetChild(unittest.TestCase):
    """The #61195 shapes, end-to-end through real AIAgent construction."""

    def test_cross_provider_pool_entry_is_refused(self):
        """A pool holding the parent-provider's entries (the contamination
        shape behind the reported ``provider=anthropic
        base_url=https://openrouter.ai/api/v1`` log line) must not be able to
        retarget the child."""
        pool = CredentialPool(
            "openrouter", [_entry("openrouter", "sk-or-v1-pool-key", OPENROUTER_URL)]
        )
        child = _build_delegated_child(pool)
        _lease_like_run_single_child(child)

        self.assertEqual(child.provider, "anthropic")
        self.assertEqual(child.base_url, ANTHROPIC_URL)
        self.assertEqual(child._anthropic_base_url, ANTHROPIC_URL)
        # The foreign key must not have been applied either.
        self.assertNotEqual(child.api_key, "sk-or-v1-pool-key")

    def test_same_provider_entry_with_foreign_base_url_keeps_pinned_endpoint(self):
        """Same-provider entry whose base_url points elsewhere: rotate the
        key, keep the explicitly configured delegation endpoint."""
        pool = CredentialPool(
            "anthropic", [_entry("anthropic", "sk-ant-pool-key", OPENROUTER_URL)]
        )
        child = _build_delegated_child(pool)
        _lease_like_run_single_child(child)

        self.assertEqual(child.api_key, "sk-ant-pool-key")
        self.assertEqual(child.base_url, ANTHROPIC_URL)
        self.assertEqual(child._anthropic_base_url, ANTHROPIC_URL)

    def test_same_provider_rotation_still_works(self):
        """Pool rotation for delegated children must keep working — the fix
        must not pin children to a single key (pool sharing exists so
        subagents can rotate on rate limits)."""
        pool = CredentialPool(
            "anthropic", [_entry("anthropic", "sk-ant-pool-key", ANTHROPIC_URL)]
        )
        child = _build_delegated_child(pool)
        _lease_like_run_single_child(child)

        self.assertEqual(child.api_key, "sk-ant-pool-key")
        self.assertEqual(child.base_url, ANTHROPIC_URL)

    def test_unscoped_entry_keeps_legacy_swap_behavior(self):
        """Entries without a provider string are unscoped (legacy pools) and
        still swap — matching recover_with_credential_pool's semantics."""
        entry = _entry("", "sk-legacy-key", None)
        pool = CredentialPool("anthropic", [entry])
        child = _build_delegated_child(pool)
        _lease_like_run_single_child(child)

        self.assertEqual(child.api_key, "sk-legacy-key")
        self.assertEqual(child.base_url, ANTHROPIC_URL)


class TestPinScopedToProvider(unittest.TestCase):
    def test_pin_released_after_provider_switch(self):
        """The endpoint pin is scoped to the delegated provider: after a
        provider switch (fallback chain), rotation for the new provider must
        follow the new provider's entries, not the stale pin."""
        child = _build_delegated_child(None)
        # Simulate what the fallback path does on a provider switch.
        child.provider = "openrouter"
        child.api_mode = "chat_completions"
        child.base_url = OPENROUTER_URL
        child._client_kwargs = {"api_key": "sk-or-v1-old", "base_url": OPENROUTER_URL}
        with patch.object(child, "_replace_primary_openai_client", return_value=True):
            child._swap_credential(
                _entry("openrouter", "sk-or-v1-rotated", OPENROUTER_URL)
            )
        self.assertEqual(child.api_key, "sk-or-v1-rotated")
        self.assertEqual(child.base_url, OPENROUTER_URL)


class TestNonDelegationSwapUnchanged(unittest.TestCase):
    def test_same_provider_entry_endpoint_still_follows_entry(self):
        """Primary (non-delegated) agents keep the existing semantics: a
        same-provider entry that carries its own base_url (per-key proxy)
        retargets the client as before — no pin applies."""
        child = _build_delegated_child(None)
        del child._delegation_endpoint_pin
        with patch(
            "agent.anthropic_adapter.build_anthropic_client",
            return_value=MagicMock(),
        ):
            child._swap_credential(
                _entry("anthropic", "sk-ant-proxy-key", "https://my-proxy.example/v1")
            )
        self.assertEqual(child.api_key, "sk-ant-proxy-key")
        self.assertEqual(child.base_url, "https://my-proxy.example/v1")


if __name__ == "__main__":
    unittest.main()
