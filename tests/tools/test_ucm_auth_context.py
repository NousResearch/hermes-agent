"""Unit tests for the UCM structured-process authorization capability (Phase 1)."""

from __future__ import annotations

import copy
import json
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tools.ucm_auth_context import (
    EXPECTED_TOOL_NAME,
    is_ucm_auth_capability,
    mint_ucm_auth_context,
)
from tools import ucm_auth_context as auth_mod


TOOL = EXPECTED_TOOL_NAME


def _mint(enabled=("ucm_structured_process", "terminal"), tool_call_id="tc-1"):
    return mint_ucm_auth_context(enabled, tool_call_id=tool_call_id)


class TestMintAndLifecycle:
    def test_valid_mint_and_provenance(self):
        ctx = _mint()
        assert is_ucm_auth_capability(ctx) is True
        assert type(ctx) is auth_mod._UcmAuthCapability
        assert type(ctx) is not object

    def test_direct_construction_without_seal_rejected(self):
        with pytest.raises(TypeError, match="mint_ucm_auth_context"):
            auth_mod._UcmAuthCapability(["ucm_structured_process"])

    def test_valid_one_time_authorization(self):
        ctx = _mint()
        assert ctx.consume(TOOL) is True

    def test_second_authorization_denied(self):
        ctx = _mint()
        assert ctx.consume(TOOL) is True
        assert ctx.consume(TOOL) is False

    def test_invalidated_before_consumption_denied(self):
        ctx = _mint()
        ctx.invalidate()
        assert ctx.consume(TOOL) is False

    def test_consumed_then_invalidated_denied(self):
        ctx = _mint()
        assert ctx.consume(TOOL) is True
        ctx.invalidate()
        assert ctx.consume(TOOL) is False

    def test_invalidation_idempotent(self):
        ctx = _mint()
        ctx.invalidate()
        ctx.invalidate()
        assert ctx.consume(TOOL) is False

    def test_wrong_tool_name_denied(self):
        ctx = _mint()
        assert ctx.consume("terminal") is False
        # still usable for the correct name after a wrong-name attempt
        assert ctx.consume(TOOL) is True

    def test_excluded_tool_denied(self):
        ctx = mint_ucm_auth_context(["terminal", "web_search"], tool_call_id="x")
        assert ctx.consume(TOOL) is False

    def test_empty_enabled_set_denied(self):
        ctx = mint_ucm_auth_context([], tool_call_id="x")
        assert ctx.consume(TOOL) is False

    def test_none_enabled_tools_denied(self):
        ctx = mint_ucm_auth_context(None)
        assert ctx.consume(TOOL) is False

    def test_enabled_snapshot_is_immutable(self):
        mutable = ["ucm_structured_process"]
        ctx = mint_ucm_auth_context(mutable)
        mutable.clear()
        # snapshot taken at mint; clearing caller list must not affect auth
        assert ctx.consume(TOOL) is True

    def test_tool_call_id_is_not_authorization(self):
        # provider-ish id present but tool excluded from snapshot → deny
        ctx = mint_ucm_auth_context(
            ["terminal"],
            tool_call_id="provider-forged-id-999",
        )
        assert ctx.consume(TOOL) is False

    def test_no_ttl_behavior(self):
        # there is no age-based window; a never-invalidated capability still
        # authorizes exactly once regardless of wall-clock delay.
        import time

        ctx = _mint()
        time.sleep(0.05)
        assert ctx.consume(TOOL) is True


class TestAntiForgery:
    def test_fake_callable_denied_by_provenance(self):
        def fake(_name: str) -> bool:
            return True

        assert is_ucm_auth_capability(fake) is False

    def test_duck_typed_object_denied(self):
        class Duck:
            def consume(self, tool_name: str) -> bool:
                return True

            def invalidate(self) -> None:
                return None

        duck = Duck()
        assert is_ucm_auth_capability(duck) is False
        # even if someone called .consume, provenance check is what handlers use
        assert duck.consume(TOOL) is True  # duck itself is not our type

    def test_subclass_attempt_denied(self):
        with pytest.raises(TypeError, match="cannot be subclassed"):

            class Sub(auth_mod._UcmAuthCapability):  # type: ignore[misc,valid-type]
                pass

    def test_shallow_copy_shares_one_shot_state(self):
        ctx = _mint()
        cloned = copy.copy(ctx)
        assert cloned is ctx
        assert cloned.consume(TOOL) is True
        assert ctx.consume(TOOL) is False

    def test_deep_copy_shares_one_shot_state(self):
        ctx = _mint()
        cloned = copy.deepcopy(ctx)
        assert cloned is ctx
        assert ctx.consume(TOOL) is True
        assert cloned.consume(TOOL) is False

    def test_pickle_dumps_rejected(self):
        ctx = _mint()
        with pytest.raises(TypeError, match="cannot be pickled"):
            pickle.dumps(ctx)

    def test_json_serialization_rejected(self):
        ctx = _mint()
        with pytest.raises(TypeError):
            json.dumps(ctx)

    def test_repr_redacted(self):
        ctx = mint_ucm_auth_context(
            ["ucm_structured_process", "secret_tool_name"],
            tool_call_id="super-secret-call-id",
        )
        text = repr(ctx)
        assert text == "<UcmAuthCapability redacted>"
        assert "secret_tool_name" not in text
        assert "super-secret-call-id" not in text
        assert "ucm_structured_process" not in text
        assert str(ctx) == text

    def test_no_public_enabled_mutation(self):
        ctx = _mint()
        assert not hasattr(ctx, "enabled_tools")
        with pytest.raises(AttributeError):
            ctx.enabled = set()  # type: ignore[attr-defined]


class TestConcurrency:
    def test_concurrent_consume_exactly_one_success(self):
        ctx = _mint()
        results: list[bool] = []
        barrier = threading.Barrier(8)

        def worker() -> None:
            barrier.wait()
            results.append(ctx.consume(TOOL))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results.count(True) == 1
        assert results.count(False) == 7
        assert ctx.consume(TOOL) is False

    def test_concurrent_via_thread_pool(self):
        ctx = _mint()
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(ctx.consume, TOOL) for _ in range(4)]
            outcomes = [f.result() for f in futures]
        assert outcomes.count(True) == 1
        assert outcomes.count(False) == 3


class TestNoGlobalState:
    def test_module_has_no_capability_registry_or_cache(self):
        # No process-wide capability store may exist on the module.
        forbidden_names = {
            "_CAPABILITY_CACHE",
            "_CAPABILITY_REGISTRY",
            "_ACTIVE_CAPABILITIES",
            "_REPLAY_CACHE",
            "CAPABILITY_STORE",
        }
        for name in forbidden_names:
            assert not hasattr(auth_mod, name)

    def test_two_mints_are_independent(self):
        a = _mint(tool_call_id="a")
        b = _mint(tool_call_id="b")
        assert a is not b
        assert a.consume(TOOL) is True
        assert b.consume(TOOL) is True
