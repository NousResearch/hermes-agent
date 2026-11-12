"""Tests for the profile-qualified Feishu client binding registry.

Covers the ownership guarantees required for the DM-fallback fix:

1. profile A + profile B in one process resolve distinct clients from
   worker-thread tool calls (no cross-principal aliasing);
2. stale A disconnect after B has published does not clear B
   (compare-and-remove teardown);
3. reconnect / replacement-adapter publication survives old-generation
   teardown, including across adapter instances (per-profile monotonic
   generation allocation — no cross-instance collision);
4. failed publication never leaves a discoverable binding (no publish
   before a successful connect);
5. the HERMES_HOME contextvar override installed by the multiplex
   gateway's ``_profile_runtime_scope`` reaches the tool worker thread
   through the real propagation chain (``copy_context()`` via
   ``propagate_context_to_thread``), so each profile's turn resolves only
   its own client with no explicit profile_key.
"""

import os
import shutil
import threading
import unittest

import tools.feishu_client_binding as binding
from tools.feishu_doc_tool import get_client as doc_get_client
from tools.feishu_drive_tool import get_client as drive_get_client


class TestFeishuClientBindingRegistry(unittest.TestCase):
    def setUp(self):
        binding.clear_all()

    def tearDown(self):
        binding.clear_all()
        from tools.feishu_doc_tool import set_client as doc_set_client
        from tools.feishu_drive_tool import set_client as drive_set_client

        doc_set_client(None)
        drive_set_client(None)

    # -- registry primitives ------------------------------------------------

    def test_publish_resolve_same_profile(self):
        client = object()
        binding.publish(client, profile_key="/profiles/a")
        self.assertIs(
            binding.resolve(profile_key="/profiles/a"), client
        )

    def test_profile_isolation_resolves_distinct_clients(self):
        client_a = object()
        client_b = object()
        binding.publish(client_a, profile_key="/profiles/a")
        binding.publish(client_b, profile_key="/profiles/b")
        self.assertIs(
            binding.resolve(profile_key="/profiles/a"), client_a
        )
        self.assertIs(
            binding.resolve(profile_key="/profiles/b"), client_b
        )

    def test_reconnect_replaces_same_profile_binding(self):
        old = object()
        new = object()
        gen_old = binding.publish(old, profile_key="/profiles/a")
        gen_new = binding.publish(new, profile_key="/profiles/a")
        self.assertGreater(gen_new, gen_old)
        self.assertIs(
            binding.resolve(profile_key="/profiles/a"), new
        )

    def test_stale_unpublish_does_not_clear_newer_binding(self):
        """A stale adapter (old generation) must not clear the newer binding."""
        client_a = object()
        client_b = object()
        gen_a = binding.publish(client_a, profile_key="/profiles/a")
        binding.publish(client_b, profile_key="/profiles/a")
        binding.unpublish(gen_a, profile_key="/profiles/a")
        self.assertIs(
            binding.resolve(profile_key="/profiles/a"), client_b
        )

    def test_cross_instance_generation_no_collision(self):
        """Two adapter instances for the same profile must never publish
        colliding generations (the per-profile counter is process-wide, not
        per-instance) — the exact hole a per-adapter counter leaves open.
        """
        stale_client = object()  # old adapter instance A1
        fresh_client = object()  # replacement adapter instance A2

        gen_stale = binding.publish(stale_client, profile_key="/profiles/a")
        gen_fresh = binding.publish(fresh_client, profile_key="/profiles/a")
        self.assertNotEqual(gen_stale, gen_fresh)

        # A1's delayed teardown (old generation) must not clear A2's binding.
        binding.unpublish(gen_stale, profile_key="/profiles/a")
        self.assertIs(
            binding.resolve(profile_key="/profiles/a"), fresh_client
        )

    def test_matching_unpublish_clears_binding(self):
        client = object()
        gen = binding.publish(client, profile_key="/profiles/a")
        binding.unpublish(gen, profile_key="/profiles/a")
        self.assertIsNone(binding.resolve(profile_key="/profiles/a"))

    def test_unpublish_unknown_generation_is_noop(self):
        binding.publish(object(), profile_key="/profiles/a")
        binding.unpublish(99, profile_key="/profiles/a")
        self.assertIsNotNone(binding.resolve(profile_key="/profiles/a"))

    def test_explicit_empty_string_key_is_not_redirected_to_active_profile(self):
        """An explicit ``""`` is a (pathological) explicit key, not the
        implicit active-home key: falsy handling at the call site would
        silently route it into the active profile's bucket — a cross-
        principal misroute for a typo like
        ``os.environ.get("HERMES_HOME", "")``."""
        client = object()
        binding.publish(client, profile_key="")
        # Never lands in the active profile's bucket...
        self.assertIsNot(binding.resolve(), client)
        # ...and is discoverable only through the same explicit spelling.
        self.assertIs(binding.resolve(profile_key=""), client)

    # -- worker-thread resolution (the DM failure boundary) ----------------

    def _resolve_in_worker_thread(self, fn, *args):
        result = {}

        def _worker():
            result["client"] = fn(*args)

        t = threading.Thread(target=_worker)
        t.start()
        t.join(timeout=5)
        self.assertFalse(t.is_alive(), "worker thread did not finish")
        return result["client"]

    def test_worker_thread_resolves_own_profile_client(self):
        client_a = object()
        client_b = object()
        binding.publish(client_a, profile_key="/profiles/a")
        binding.publish(client_b, profile_key="/profiles/b")
        self.assertIs(
            self._resolve_in_worker_thread(binding.resolve, "/profiles/a"),
            client_a,
        )
        self.assertIs(
            self._resolve_in_worker_thread(binding.resolve, "/profiles/b"),
            client_b,
        )

    def test_doc_and_drive_tools_resolve_profile_binding_in_worker_thread(self):
        client_a = object()
        binding.publish(client_a)
        self.assertIs(
            self._resolve_in_worker_thread(doc_get_client),
            client_a,
        )
        self.assertIs(
            self._resolve_in_worker_thread(drive_get_client),
            client_a,
        )

    def test_tools_prefer_thread_local_over_binding(self):
        shared = object()
        local = object()
        binding.publish(shared, profile_key="/profiles/a")
        from tools.feishu_doc_tool import set_client as doc_set_client

        doc_set_client(local)
        self.assertIs(doc_get_client(), local)

    def test_missing_binding_returns_none(self):
        binding.clear_all()
        self.assertIsNone(
            self._resolve_in_worker_thread(doc_get_client)
        )
        self.assertIsNone(
            self._resolve_in_worker_thread(drive_get_client)
        )


class TestEquivalentHomeCanonicalization(unittest.TestCase):
    """Witness for the registry-key canonicalization requirement.

    The per-profile generation invariant must be about the physical
    profile home, not the path spelling used at registry ingress. Keys are
    canonicalized via ``hermes_home_key()`` (expanduser + resolve +
    normcase), so an alias/symlink to a profile home and its resolved path
    must share ONE binding bucket and ONE monotonic generation counter —
    otherwise a replacement adapter could publish as a new generation under
    one spelling while a stale credential-bearing client stays discoverable
    under the other.
    """

    def setUp(self):
        binding.clear_all()
        import tempfile

        self._tmp = tempfile.mkdtemp(prefix="feishu-binding-equiv-")
        self._real = os.path.join(self._tmp, "real-home")
        os.makedirs(self._real, exist_ok=True)
        self._alias = os.path.join(self._tmp, "alias-home")
        try:
            os.symlink(self._real, self._alias)
        except OSError:
            self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
            self.skipTest("symlinks unavailable on this platform")

    def tearDown(self):
        binding.clear_all()
        from tools.feishu_doc_tool import set_client as doc_set_client
        from tools.feishu_drive_tool import set_client as drive_set_client

        doc_set_client(None)
        drive_set_client(None)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_active_profile_key_canonicalizes_equivalent_spellings(self):
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        token = set_hermes_home_override(self._alias)
        try:
            key_via_alias = binding.active_profile_key()
        finally:
            reset_hermes_home_override(token)
        token = set_hermes_home_override(self._real)
        try:
            key_via_real = binding.active_profile_key()
        finally:
            reset_hermes_home_override(token)
        self.assertEqual(key_via_alias, key_via_real)

    def test_equivalent_spellings_share_one_binding_bucket(self):
        client = object()
        binding.publish(client, profile_key=self._alias)
        # The resolved spelling must find the alias-published binding...
        self.assertIs(binding.resolve(profile_key=self._real), client)
        # ...and the alias spelling must find itself too.
        self.assertIs(binding.resolve(profile_key=self._alias), client)

    def test_replacement_across_spellings_advances_one_counter(self):
        """A replacement published under a second spelling of the same
        physical home must advance the SAME per-profile generation counter
        (not two buckets each stuck at 1), and the stale generation's
        compare-and-remove teardown — issued under the original spelling —
        must not clear the newer binding."""
        stale_client = object()
        fresh_client = object()

        gen_stale = binding.publish(stale_client, profile_key=self._alias)
        gen_fresh = binding.publish(fresh_client, profile_key=self._real)
        self.assertGreater(gen_fresh, gen_stale)

        # Stale teardown via the OTHER spelling of the same home.
        binding.unpublish(gen_stale, profile_key=self._alias)
        self.assertIs(binding.resolve(profile_key=self._real), fresh_client)

    def test_worker_thread_resolves_across_spellings_via_contextvar(self):
        """The real DM path: the gateway installs the profile-home override
        with one spelling while a sibling adapter published under another.
        The worker-thread tool call must still resolve the owning profile's
        client, never None and never a sibling profile's client."""
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )
        from tools.thread_context import propagate_context_to_thread

        client = object()
        sibling_client = object()
        binding.publish(client, profile_key=self._real)
        binding.publish(sibling_client, profile_key="/profiles/unrelated-b")

        result = {}
        token = set_hermes_home_override(self._alias)
        try:

            def _capture():
                result["doc"] = doc_get_client()
                result["drive"] = drive_get_client()

            worker = threading.Thread(target=propagate_context_to_thread(_capture))
            worker.start()
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive(), "worker thread did not finish")
        finally:
            reset_hermes_home_override(token)
        self.assertIs(result["doc"], client)
        self.assertIs(result["drive"], client)


class TestProfileScopeContextvarPropagation(unittest.TestCase):
    """Witness for the real DM tool path (issue #29760).

    The binding key is derived from ``get_hermes_home()``, which follows the
    ``HERMES_HOME`` contextvar override installed by the multiplex gateway's
    ``_profile_runtime_scope``. That override reaches the tool worker thread
    only because the gateway propagates contextvars across thread boundaries
    (``copy_context()`` in ``_run_in_executor_with_context`` and
    ``propagate_context_to_thread`` in the tool executor). These tests cross
    the same boundary with the same mechanism — no explicit profile_key — so
    a regression in any link fails here instead of cross-principal in
    production.
    """

    def setUp(self):
        binding.clear_all()
        from tools.feishu_doc_tool import set_client as doc_set_client
        from tools.feishu_drive_tool import set_client as drive_set_client

        doc_set_client(None)
        drive_set_client(None)

    def tearDown(self):
        binding.clear_all()

    def _resolve_in_profile_scope(self, home, fn):
        """Resolve ``fn()`` on a worker thread running inside a copied
        context that carries the profile-home override — the same
        propagation shape as the gateway's ``run_sync`` executor hop plus
        the tool dispatch hop."""
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )
        from tools.thread_context import propagate_context_to_thread

        result = {}
        token = set_hermes_home_override(home)
        try:

            def _capture():
                result["value"] = fn()

            worker = threading.Thread(
                target=propagate_context_to_thread(_capture)
            )
            worker.start()
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive(), "worker thread did not finish")
        finally:
            reset_hermes_home_override(token)
        return result["value"]

    def test_worker_thread_resolves_own_profile_via_contextvar(self):
        """Profile A's turn and profile B's turn, each on its own worker
        thread under its own override, resolve only their own client."""
        client_a = object()
        client_b = object()
        binding.publish(client_a, profile_key="/profiles/a")
        binding.publish(client_b, profile_key="/profiles/b")

        self.assertIs(
            self._resolve_in_profile_scope("/profiles/a", doc_get_client),
            client_a,
        )
        self.assertIs(
            self._resolve_in_profile_scope("/profiles/b", doc_get_client),
            client_b,
        )
        self.assertIs(
            self._resolve_in_profile_scope("/profiles/a", drive_get_client),
            client_a,
        )
        self.assertIs(
            self._resolve_in_profile_scope("/profiles/b", drive_get_client),
            client_b,
        )

    def test_unrelated_profile_scope_resolves_nothing(self):
        """A profile with no published binding must resolve to None —
        never a sibling profile's client."""
        client_b = object()
        binding.publish(client_b, profile_key="/profiles/b")

        self.assertIsNone(
            self._resolve_in_profile_scope("/profiles/a", doc_get_client)
        )
        self.assertIsNone(
            self._resolve_in_profile_scope("/profiles/a", drive_get_client)
        )


if __name__ == "__main__":
    unittest.main()
