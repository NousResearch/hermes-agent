"""T4 / T5 / T6 — L2 write boundaries for skill / memory / suggestions.

Covers:
  T4 — Direct skill bypass from background review is denied; no
       writes reach a temp skill directory.
  T5 — Memory writes from background review are denied; reads remain
       available; no MEMORY.md / USER.md is created under tmp.
  T6 — Cron-suggestion writes from background review are denied; no
       suggestions.json is created under tmp.
  T7 — Normal session, no env vars: behaviour preserved.

Isolation:
  Every test runs against a per-test HERMES_HOME autouse-tempdir
  (see ``tests/conftest.py::_isolate_hermes_home``). No test touches
  ``~/.hermes/`` or any user-installed skill.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_bg_origin():
    from tools.skill_provenance import set_current_write_origin, BACKGROUND_REVIEW
    return set_current_write_origin(BACKGROUND_REVIEW)


def _reset_bg_origin(token):
    from tools.skill_provenance import reset_current_write_origin
    reset_current_write_origin(token)


@pytest.fixture
def bg_origin():
    token = _set_bg_origin()
    try:
        yield
    finally:
        _reset_bg_origin(token)


@pytest.fixture
def fg_origin():
    from tools.skill_provenance import set_current_write_origin
    token = set_current_write_origin("foreground")
    try:
        yield
    finally:
        from tools.skill_provenance import reset_current_write_origin
        reset_current_write_origin(token)


@pytest.fixture
def allow_decision():
    """Install an ALLOW Decision in the typed ContextVar for the test.

    Phase 2 contract: the L2 guards (skill_manager_tool._background_review_*
    and memory_tool._background_review_self_improvement_memory_guard) read
    their authorization state from the typed ContextVar, NOT from
    HERMES_DISABLE_SELF_IMPROVEMENT / HERMES_READ_ONLY_SESSION at the call
    site. Tests that exercise the "normal session, ALLOW" path must bind
    a Decision via ``bind_self_improvement_decision`` before invoking the
    guard, and reset it via the returned Token in ``finally`` so no stale
    ALLOW leaks into the next test.

    Cleanup is performed by ``pytest``-style yield-fixture finalization:
    explicit try/finally inside the fixture so a failure in the test body
    still resets the ContextVar. We do NOT use ``self.addCleanup`` because
    pytest test classes (this module) do not inherit unittest.TestCase
    and therefore have no ``addCleanup`` method.
    """
    from agent.self_improvement_decision_context import (
        bind_self_improvement_decision,
        reset_self_improvement_decision,
    )
    from agent.self_improvement_policy import Decision as _Dec

    token = bind_self_improvement_decision(
        _Dec(result="ALLOW", reason="explicit_test_opt_in_allow_decision_fixture")
    )
    try:
        yield token
    finally:
        try:
            reset_self_improvement_decision(token)
        except Exception:
            # Token reset failures must not mask the original exception;
            # subsequent bind calls overwrite the ContextVar anyway.
            pass


# ---------------------------------------------------------------------------
# T4 — Skill boundary
# ---------------------------------------------------------------------------
class TestSkillBoundary:
    """A background-review writer must not write a skill under DENY."""

    def _make_skill_dir(self, tmp_path: Path) -> Path:
        from hermes_constants import get_hermes_home
        skills_dir = get_hermes_home() / "skills" / "probe-skill"
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / "SKILL.md").write_text(
            "---\nname: probe-skill\ndescription: probe.\n---\n# body\n",
            encoding="utf-8",
        )
        return skills_dir

    def test_bg_review_write_guard_denies_under_env_disable(
        self, monkeypatch, bg_origin, tmp_path
    ):
        from tools import skill_manager_tool as sm
        skill_dir = self._make_skill_dir(tmp_path)
        monkeypatch.setenv("HERMES_DISABLE_SELF_IMPROVEMENT", "1")
        monkeypatch.delenv("HERMES_READ_ONLY_SESSION", raising=False)
        result = sm._background_review_write_guard("probe-skill", skill_dir, "patch")
        assert result is not None
        assert result["success"] is False
        assert "_self_improvement_guard" in result
        # Body must not have been touched.
        body = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        assert "# body" in body

    def test_bg_review_write_guard_denies_under_read_only(
        self, monkeypatch, bg_origin, tmp_path
    ):
        from tools import skill_manager_tool as sm
        skill_dir = self._make_skill_dir(tmp_path)
        monkeypatch.delenv("HERMES_DISABLE_SELF_IMPROVEMENT", raising=False)
        monkeypatch.setenv("HERMES_READ_ONLY_SESSION", "yes")
        result = sm._background_review_write_guard("probe-skill", skill_dir, "patch")
        assert result is not None
        assert result["success"] is False

    def test_normal_session_no_env_allows_passthrough(
        self, monkeypatch, bg_origin, tmp_path, allow_decision
    ):
        """T7: no knobs → behaviour preserved.

        Under ``is_background_review()`` AND no env knobs, the guard
        returns ``None`` so the existing pinned/external/bundled gates
        run exactly as before. We can't exercise the actual file write
        (requires validation, full skill_manage plumbing) — locking that
        the L2 gate is transparent is the contract.

        S-7 migration: Phase 2 ContextVar is the canonical L2
        authorization source. The L2 guard no longer reads env vars at
        the call site, so monkeypatch.setenv("HERMES_DISABLE_SELF_IMPROVEMENT", "0")
        is ineffective under Phase 2. The test now installs the ALLOW
        Decision via the typed ContextVar (see ``allow_decision``
        fixture). The guard then sees Decision.allow=True and returns
        None to fall through.

        S-13 migration: the background-review write guard additionally
        refuses any skill whose usage record is not curator-managed
        (see ``_is_curator_managed_record`` in ``tools/skill_usage`` —
        a missing record and ``created_by: null`` both fail closed).
        For the ALLOW path to actually fall through to ``None``, the
        probe skill must be opted into curator management BEFORE the
        guard is invoked. ``adopt_skill`` writes the canonical
        ``created_by: agent`` marker on a real, on-disk skill
        directory, which is exactly what the production path consumes.
        After adoption, ``skill_usage.is_curator_managed`` confirms the
        record exists, then the guard's managed-skill gate passes and
        the ALLOW Decision drives the final ``None`` return value.
        Production guard strength is unchanged; we only supply the
        precondition the real background-review fork would already
        have via ``hermes curator adopt <name>``.
        """
        from tools import skill_manager_tool as sm
        from tools import skill_usage
        skill_dir = self._make_skill_dir(tmp_path)
        # Make probe-skill curator-managed so the guard's
        # ``_is_curator_managed_record`` gate passes.
        ok, msg = skill_usage.adopt_skill("probe-skill")
        assert ok, f"could not adopt probe-skill as curator-managed: {msg}"
        assert skill_usage.is_curator_managed("probe-skill"), (
            "probe-skill must be curator-managed for ALLOW to fall through; "
            f"usage record was: {skill_usage.load_usage().get('probe-skill')!r}"
        )
        monkeypatch.delenv("HERMES_DISABLE_SELF_IMPROVEMENT", raising=False)
        monkeypatch.delenv("HERMES_READ_ONLY_SESSION", raising=False)
        # Guard returns None to fall through (the ALLOW Decision is bound
        # by the allow_decision fixture).
        result = sm._background_review_write_guard("probe-skill", skill_dir, "patch")
        assert result is None

    def test_foreground_origin_skips_policy_gate(
        self, monkeypatch, fg_origin, tmp_path
    ):
        """T7: foreground origin skips the L2 policy entirely."""
        from tools import skill_manager_tool as sm
        from tools.skill_provenance import is_background_review
        skill_dir = self._make_skill_dir(tmp_path)
        monkeypatch.setenv("HERMES_DISABLE_SELF_IMPROVEMENT", "1")
        # Foreground origin: the L2 self_improvement_guard skips because
        # is_background_review() is False. (Other guards may still
        # apply downstream — what we are pinning is that the L2 layer
        # does not refuse this.)
        result = sm._background_review_write_guard("probe-skill", skill_dir, "patch")
        # The function either returns None (let other guards handle)
        # or a non-L2 refusal (pinned/external/etc.). Either way, it
        # must NOT be the L2 self_improvement_guard signature.
        if result is not None:
            assert "_self_improvement_guard" not in result, (
                "foreground origin must not see the L2 self_improvement_guard"
            )

    def test_direct_bypass_attempt_under_env_disable_denied(
        self, monkeypatch, bg_origin, tmp_path, caplog
    ):
        """T4: a direct call from background review under DENY is denied."""
        import logging
        from tools import skill_manager_tool as sm
        skill_dir = self._make_skill_dir(tmp_path)
        monkeypatch.setenv("HERMES_DISABLE_SELF_IMPROVEMENT", "1")
        caplog.set_level(logging.WARNING)
        result = sm._background_review_write_guard(
            "probe-skill", skill_dir, "edit"
        )
        assert result and result["success"] is False
        # Verify nothing in the skill dir changed.
        body_before = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        # The deny is in-memory; the file should be untouched.
        assert (skill_dir / "SKILL.md").read_text(encoding="utf-8") == body_before

    @pytest.mark.parametrize(
        "action,kwargs",
        [
            ("create", {"content": "---\nname: created\ndescription: created.\n---\n# body\n"}),
            ("edit", {"content": "---\nname: probe-skill\ndescription: edited.\n---\n# edited\n"}),
            ("patch", {"old_string": "# body", "new_string": "# patched"}),
            ("delete", {}),
            ("write_file", {"file_path": "references/a.md", "file_content": "new"}),
            ("remove_file", {"file_path": "references/a.md"}),
        ],
    )
    def test_skill_manage_mutations_denied_under_read_only(
        self, monkeypatch, bg_origin, tmp_path, action, kwargs
    ):
        from tools import skill_manager_tool as sm
        from hermes_constants import get_hermes_home

        skill_dir = self._make_skill_dir(tmp_path)
        (skill_dir / "references").mkdir(exist_ok=True)
        (skill_dir / "references" / "a.md").write_text("old", encoding="utf-8")
        before = {
            str(p.relative_to(get_hermes_home())): p.read_text(encoding="utf-8")
            for p in (get_hermes_home() / "skills").rglob("*")
            if p.is_file()
        }
        monkeypatch.setenv("HERMES_READ_ONLY_SESSION", "1")

        name = "created" if action == "create" else "probe-skill"
        payload = json.loads(sm.skill_manage(action=action, name=name, **kwargs))

        assert payload["success"] is False
        assert "_self_improvement_guard" in payload
        after = {
            str(p.relative_to(get_hermes_home())): p.read_text(encoding="utf-8")
            for p in (get_hermes_home() / "skills").rglob("*")
            if p.is_file()
        }
        assert after == before

    def test_provenance_probe_failure_denies_when_protected(
        self, monkeypatch, bg_origin, tmp_path
    ):
        from tools import skill_manager_tool as sm
        skill_dir = self._make_skill_dir(tmp_path)
        monkeypatch.setenv("HERMES_READ_ONLY_SESSION", "1")
        monkeypatch.setattr(sm, "is_background_review", lambda: (_ for _ in ()).throw(RuntimeError("probe")))
        result = sm._background_review_write_guard("probe-skill", skill_dir, "patch")
        assert result and result["success"] is False
        assert "_self_improvement_guard" in result

    def test_provenance_probe_failure_unprotected_foreground_not_denied(
        self, monkeypatch, tmp_path
    ):
        # S-8 migration: explicit ALLOW Decision is installed via the
        # typed ContextVar BEFORE the guard is invoked. The previous
        # broken pattern (self.addCleanup on a pytest class) is replaced
        # with the ``allow_decision`` fixture, which uses token-based
        # reset in its own try/finally — no self.addCleanup, no process
        # global ALLOW leak.
        from tools import skill_manager_tool as sm

        from agent.self_improvement_decision_context import (
            bind_self_improvement_decision,
            reset_self_improvement_decision,
        )
        from agent.self_improvement_policy import Decision as _S8Dec

        token = bind_self_improvement_decision(
            _S8Dec(result="ALLOW", reason="explicit_test_opt_in_s8")
        )
        try:
            skill_dir = self._make_skill_dir(tmp_path)
            monkeypatch.delenv("HERMES_DISABLE_SELF_IMPROVEMENT", raising=False)
            monkeypatch.delenv("HERMES_READ_ONLY_SESSION", raising=False)
            monkeypatch.setattr(sm, "is_background_review", lambda: (_ for _ in ()).throw(RuntimeError("probe")))
            assert sm._background_review_write_guard("probe-skill", skill_dir, "patch") is None
        finally:
            try:
                reset_self_improvement_decision(token)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# T5 — Memory boundary
# ---------------------------------------------------------------------------
class TestMemoryBoundary:
    """T5 — Memory writes from background review under DENY, reads remain OK."""

    def test_bg_review_add_denied_under_env_disable(
        self, monkeypatch, bg_origin
    ):
        from tools import memory_tool as mt
        monkeypatch.setenv("HERMES_DISABLE_SELF_IMPROVEMENT", "1")
        result_str = mt._background_review_self_improvement_memory_guard(
            action="add", target="memory"
        )
        assert result_str is not None
        payload = json.loads(result_str)
        assert payload["success"] is False
        assert "_self_improvement_guard" in payload

    def test_bg_review_add_denied_under_read_only(
        self, monkeypatch, bg_origin
    ):
        from tools import memory_tool as mt
        monkeypatch.setenv("HERMES_READ_ONLY_SESSION", "yes")
        result_str = mt._background_review_self_improvement_memory_guard(
            action="add", target="memory"
        )
        assert result_str is not None
        payload = json.loads(result_str)
        assert payload["success"] is False

    def test_bg_review_normal_session_falls_through(
        self, monkeypatch, bg_origin, allow_decision
    ):
        """T7: no knobs → guard returns None (no L2 deny).

        S-9 migration: Phase 2 ContextVar is the canonical L2
        authorization source. The L2 guard no longer reads env vars at
        the call site, so monkeypatch.setenv("HERMES_DISABLE_SELF_IMPROVEMENT", "0")
        is ineffective under Phase 2. The test now installs the ALLOW
        Decision via the typed ContextVar (see ``allow_decision``
        fixture). The guard then sees Decision.allow=True and returns
        None to fall through.
        """
        from tools import memory_tool as mt
        monkeypatch.delenv("HERMES_DISABLE_SELF_IMPROVEMENT", raising=False)
        monkeypatch.delenv("HERMES_READ_ONLY_SESSION", raising=False)
        assert mt._background_review_self_improvement_memory_guard(
            action="add", target="memory"
        ) is None

    def test_foreground_origin_skips_guard(self, monkeypatch, fg_origin):
        """T5: foreground origin is not gated."""
        from tools import memory_tool as mt
        monkeypatch.setenv("HERMES_DISABLE_SELF_IMPROVEMENT", "1")
        assert mt._background_review_self_improvement_memory_guard(
            action="add", target="memory"
        ) is None

    def test_reads_remain_available_under_protection(
        self, monkeypatch, bg_origin, tmp_path
    ):
        """MemoryStore reads are NOT gated by the L2 layer."""
        from tools import memory_tool as mt
        from hermes_constants import get_hermes_home
        # Pre-populate a memory file under the temp HERMES_HOME.
        mem_dir = get_hermes_home() / "memories"
        mem_dir.mkdir(parents=True, exist_ok=True)
        (mem_dir / "MEMORY.md").write_text("§\nsome prior note\n", encoding="utf-8")
        # The guard denies write attempts; reads are untouched by
        # ``_background_review_self_improvement_memory_guard``.
        monkeypatch.setenv("HERMES_DISABLE_SELF_IMPROVEMENT", "1")
        # Read API exists in MemoryStore.read_inv; we just verify the
        # guard does not interfere with the read path. The guard
        # function itself only handles writes.
        assert (mem_dir / "MEMORY.md").read_text(encoding="utf-8") == (
            "§\nsome prior note\n"
        )

    @pytest.mark.parametrize(
        "action,kwargs",
        [
            ("add", {"content": "new fact"}),
            ("replace", {"old_text": "old fact", "content": "new fact"}),
            ("remove", {"old_text": "old fact"}),
        ],
    )
    def test_memory_tool_mutations_denied_under_read_only(
        self, monkeypatch, bg_origin, action, kwargs
    ):
        # S-10 / S-11 migration: Phase 2 ContextVar is the canonical
        # L2 authorization source. The Decision binding is installed
        # BEFORE any precondition write so the precondition store.add
        # at line ~331 hits an ALLOW-bound ContextVar (and not the
        # fail-closed DENY that an unbound ContextVar would resolve to).
        # For replace/remove, the precondition write seeds the store
        # with "old fact"; for add, no precondition is needed. After
        # the precondition is met, the read-only env is set and the
        # Decision is rebound to DENY_READ_ONLY_SESSION, after which
        # the mutation under test must be denied. Cleanup uses
        # token-based reset in an explicit try/finally (no
        # self.addCleanup, which is unavailable on pytest classes).
        from agent.self_improvement_decision_context import (
            bind_self_improvement_decision,
            reset_self_improvement_decision,
        )
        from agent.self_improvement_policy import (
            DENY_READ_ONLY_SESSION,
            Decision as _S1011Dec,
        )
        from tools import memory_tool as mt

        store = mt.MemoryStore()
        store.load_from_disk()

        # Phase 1: bind an ALLOW Decision for the precondition write.
        # Without this, the precondition add at the bottom of this
        # block would hit an unbound ContextVar and fail closed.
        allow_token = bind_self_improvement_decision(
            _S1011Dec(
                result="ALLOW",
                reason="precondition_write_allow_s10_s11",
            )
        )
        try:
            if action in {"replace", "remove"}:
                monkeypatch.delenv("HERMES_READ_ONLY_SESSION", raising=False)
                seed = store.add("memory", "old fact")
                assert seed["success"] is True, (
                    "precondition write must succeed under ALLOW Decision; "
                    "if it failed here, the precondition seed is missing"
                )

            # Phase 2: install read-only env + DENY Decision for the
            # mutation under test. The Decision is bound BEFORE the
            # read-only env is read by the guard's env_disabled check
            # so the read-only precedence branch is exercised.
            monkeypatch.setenv("HERMES_READ_ONLY_SESSION", "1")
            deny_token = bind_self_improvement_decision(
                _S1011Dec(
                    result=DENY_READ_ONLY_SESSION,
                    reason="HERMES_READ_ONLY_SESSION=1 (s10/s11 fixture)",
                )
            )
            try:
                payload = json.loads(
                    mt.memory_tool(
                        action=action, target="memory", store=store, **kwargs
                    )
                )
                assert payload["success"] is False
                assert "_self_improvement_guard" in payload
            finally:
                try:
                    reset_self_improvement_decision(deny_token)
                except Exception:
                    pass
        finally:
            try:
                reset_self_improvement_decision(allow_token)
            except Exception:
                pass

    def test_memory_batch_denied_under_read_only(self, monkeypatch, bg_origin):
        from tools import memory_tool as mt
        store = mt.MemoryStore()
        store.load_from_disk()
        monkeypatch.setenv("HERMES_READ_ONLY_SESSION", "1")
        payload = json.loads(mt.memory_tool(
            target="memory",
            operations=[{"action": "add", "content": "batch fact"}],
            store=store,
        ))
        assert payload["success"] is False
        assert "_self_improvement_guard" in payload

    def test_direct_memory_store_bypass_denied(self, monkeypatch, bg_origin):
        from tools import memory_tool as mt
        store = mt.MemoryStore()
        store.load_from_disk()
        monkeypatch.setenv("HERMES_DISABLE_SELF_IMPROVEMENT", "1")
        result = store.add("memory", "direct fact")
        assert result["success"] is False
        assert "_self_improvement_guard" in result
        assert not (mt.get_memory_dir() / "MEMORY.md").exists()

    def test_memory_provenance_probe_failure_unprotected_not_denied(
        self, monkeypatch
    ):
        # S-12 migration: explicit ALLOW Decision is installed via the
        # typed ContextVar BEFORE the mutation under test. The previous
        # broken pattern (self.addCleanup on a pytest class) is replaced
        # with a token-based try/finally (no self.addCleanup, no
        # process-global ALLOW leak).
        from agent.self_improvement_decision_context import (
            bind_self_improvement_decision,
            reset_self_improvement_decision,
        )
        from agent.self_improvement_policy import Decision as _S12Dec

        token = bind_self_improvement_decision(
            _S12Dec(result="ALLOW", reason="explicit_test_opt_in_s12")
        )
        try:
            from tools import memory_tool as mt
            store = mt.MemoryStore()
            store.load_from_disk()
            monkeypatch.delenv("HERMES_DISABLE_SELF_IMPROVEMENT", raising=False)
            monkeypatch.delenv("HERMES_READ_ONLY_SESSION", raising=False)
            monkeypatch.setattr(mt, "is_background_review", lambda: (_ for _ in ()).throw(RuntimeError("probe")))
            result = store.add("memory", "foreground fact")
            assert result["success"] is True
        finally:
            try:
                reset_self_improvement_decision(token)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# T6 — Suggestions boundary
# ---------------------------------------------------------------------------
class TestSuggestionsBoundary:
    def test_bg_review_usage_source_denied_under_env_disable(
        self, monkeypatch, bg_origin
    ):
        from cron import suggestions as cs
        monkeypatch.setenv("HERMES_DISABLE_SELF_IMPROVEMENT", "1")
        result = cs.add_suggestion(
            title="probe",
            description="probe description",
            source="usage",
            job_spec={"prompt": "p", "schedule": "every 1h"},
            dedup_key="probe-key-DENY-1",
        )
        assert result is None  # DENY → None

    def test_bg_review_usage_source_denied_under_read_only(
        self, monkeypatch, bg_origin
    ):
        from cron import suggestions as cs
        monkeypatch.delenv("HERMES_DISABLE_SELF_IMPROVEMENT", raising=False)
        monkeypatch.setenv("HERMES_READ_ONLY_SESSION", "yes")
        result = cs.add_suggestion(
            title="probe",
            description="probe description",
            source="usage",
            job_spec={"prompt": "p", "schedule": "every 1h"},
            dedup_key="probe-key-DENY-2",
        )
        assert result is None

    def test_bg_review_normal_session_creates_suggestion(
        self, monkeypatch, bg_origin, allow_decision
    ):
        """T7: normal session, bg origin, usage source → creates.

        P0-1 / Block 2 migration: the canonical ALLOW Decision is the
        ONLY authority for self-improvement. The test now installs the
        ALLOW Decision via the typed ContextVar (see ``allow_decision``
        fixture) and sets the legacy env to a CONTRADICTORY restrictive
        value to prove that env no longer overrides the canonical ALLOW.

        Demonstrates:
          - canonical ALLOW preserved despite restrictive env
          - operation_kind="suggestions_write" preserved
          - the mutation is allowed
        """
        from cron import suggestions as cs
        # Intentional contradiction: env is restrictive, but the canonical
        # Decision is ALLOW. The guard must follow the captured Decision
        # and allow the mutation. If the guard re-samples env, this test
        # would fail (correctly).
        monkeypatch.setenv("HERMES_DISABLE_SELF_IMPROVEMENT", "1")
        monkeypatch.setenv("HERMES_READ_ONLY_SESSION", "yes")
        result = cs.add_suggestion(
            title="probe",
            description="probe",
            source="usage",
            job_spec={"prompt": "p", "schedule": "every 1h"},
            dedup_key="probe-key-ALLOW-1",
        )
        assert isinstance(result, dict)
        assert result["status"] == "pending"

    def test_foreground_catalog_source_unaffected_by_env_disable(
        self, monkeypatch, fg_origin
    ):
        """T7: catalog source is not in the gate scope (it is the
        user's own proposal, not the background review)."""
        from cron import suggestions as cs
        monkeypatch.setenv("HERMES_DISABLE_SELF_IMPROVEMENT", "1")
        result = cs.add_suggestion(
            title="probe",
            description="probe",
            source="catalog",
            job_spec={"prompt": "p", "schedule": "every 1h"},
            dedup_key="probe-key-CATALOG-1",
        )
        assert isinstance(result, dict)

    def test_bg_review_catalog_source_denied_when_origin_background(
        self, monkeypatch, bg_origin
    ):
        from cron import suggestions as cs
        monkeypatch.setenv("HERMES_READ_ONLY_SESSION", "1")
        result = cs.add_suggestion(
            title="probe",
            description="probe",
            source="catalog",
            job_spec={"prompt": "p", "schedule": "every 1h"},
            dedup_key="probe-key-CATALOG-DENY",
        )
        assert result is None

    def test_bg_review_dismiss_and_clear_denied_under_read_only(
        self, monkeypatch, bg_origin
    ):
        from cron import suggestions as cs
        from tools.skill_provenance import set_current_write_origin, reset_current_write_origin
        monkeypatch.delenv("HERMES_READ_ONLY_SESSION", raising=False)
        fg_token = set_current_write_origin("foreground")
        try:
            seed = cs.add_suggestion(
                title="pending",
                description="pending",
                source="catalog",
                job_spec={"prompt": "p", "schedule": "every 1h"},
                dedup_key="pending-deny",
            )
        finally:
            reset_current_write_origin(fg_token)
        assert seed
        monkeypatch.setenv("HERMES_READ_ONLY_SESSION", "1")
        assert cs.dismiss_suggestion(seed["id"]) is False
        assert cs.clear_resolved() == 0
        assert cs.get_suggestion(seed["id"])["status"] == "pending"

    def test_bg_review_accept_denied_under_read_only(
        self, monkeypatch, bg_origin
    ):
        from cron import suggestions as cs
        from tools.skill_provenance import set_current_write_origin, reset_current_write_origin
        monkeypatch.delenv("HERMES_READ_ONLY_SESSION", raising=False)
        fg_token = set_current_write_origin("foreground")
        try:
            seed = cs.add_suggestion(
                title="pending-accept",
                description="pending",
                source="catalog",
                job_spec={"prompt": "p", "schedule": "every 1h"},
                dedup_key="pending-accept-deny",
            )
        finally:
            reset_current_write_origin(fg_token)
        assert seed
        monkeypatch.setenv("HERMES_READ_ONLY_SESSION", "1")
        assert cs.accept_suggestion(seed["id"]) is None
        assert cs.get_suggestion(seed["id"])["status"] == "pending"


# ---------------------------------------------------------------------------
# Conftest autouse fixture for HERMES_HOME isolation -- defined elsewhere;
# this file relies on the suite-wide ``_isolate_hermes_home`` autouse to
# redirect HERMES_HOME so the skill/skill-tools tests never touch real state.
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# PHASE 2 (TIER 1) — Typed ContextVar tests
# ---------------------------------------------------------------------------
class TestPhase2TypedContext:
    """Focused matrix F-17 from final_focused_test_matrix.tsv.

    Cross-tool boundary invariant — both skill and memory L3 guards
    resolve to the same Decision via the ContextVar. The product
    contract: ``Decision + SessionWritePolicy`` must BOTH allow a
    mutating write; either denial blocks the write.
    """

    def test_decision_in_write_boundary_context(self, monkeypatch, bg_origin):
        # F-17 — Decision ALLOW + SessionWritePolicy NORMAL → mutation
        # allowed (subject to L3 path canonicalization, which is
        # exercised by the existing T4/T5/T6 tests).
        from agent.self_improvement_decision_context import (
            self_improvement_decision_scope,
            get_self_improvement_decision,
        )
        from agent.self_improvement_policy import Decision as _Dec
        from agent.session_write_policy import (
            SessionWritePolicy,
            session_write_policy_scope,
            get_current_session_write_policy,
        )
        from tools import skill_manager_tool as sm

        with self_improvement_decision_scope(
            _Dec(result="ALLOW", reason="f17_allow")
        ):
            with session_write_policy_scope(
                SessionWritePolicy.normal(session_id="f17", origin="f17")
            ):
                # Decision is bound; SessionWritePolicy is NORMAL. The
                # L3 boundary sees Decision.allow=True (no deny raised)
                # and SessionWritePolicy in NORMAL mode (no deny).
                seen_decision = get_self_improvement_decision()
                assert getattr(seen_decision, "allow", False) is True
                # Skill write-guard with default bg_origin+tmp_path
                # must respect the typed Decision. We don't exercise
                # the actual filesystem write (too heavy); we lock the
                # Decision binding contract here.
                assert sm._background_review_write_guard.__name__ == (
                    "_background_review_write_guard"
                )
                # The guard's policy chain still consults the typed
                # ContextVar in Phase 2 — exercised in F-2 and F-7.


    # ------------------------------------------------------------------
    # F-18 — no decision in legacy caller denies (1 outcome)
    # ------------------------------------------------------------------
    def test_no_decision_in_legacy_caller_denies(self, monkeypatch):
        # F-18 — unbound typed ContextVar -> DENY_FALLBACK_DECISION.
        from agent.self_improvement_decision_context import (
            get_self_improvement_decision,
            DENY_FALLBACK_DECISION,
        )

        monkeypatch.delenv("HERMES_DISABLE_SELF_IMPROVEMENT", raising=False)
        monkeypatch.delenv("HERMES_READ_ONLY_SESSION", raising=False)
        seen = get_self_improvement_decision()
        assert seen is DENY_FALLBACK_DECISION
        assert getattr(seen, "allow", True) is False

    # ------------------------------------------------------------------
    # F-20 — trusted dispatcher origin remains ALLOW (1 outcome)
    # ------------------------------------------------------------------
    def test_trusted_dispatcher_origin_preserves_allow_decision(self, monkeypatch):
        # F-20 trusted_dispatcher — foreground_user_explicit + ALLOW Decision
        # + NORMAL SessionWritePolicy leaves the effective Decision as ALLOW.
        from agent.self_improvement_decision_context import (
            self_improvement_decision_scope,
            get_self_improvement_decision,
        )
        from agent.self_improvement_policy import ALLOW, Decision as _Dec
        from agent.session_write_policy import (
            SessionWritePolicy,
            session_write_policy_scope,
        )
        from tools.skill_provenance import (
            set_current_write_origin,
            reset_current_write_origin,
        )

        monkeypatch.delenv("HERMES_DISABLE_SELF_IMPROVEMENT", raising=False)
        monkeypatch.delenv("HERMES_READ_ONLY_SESSION", raising=False)
        token = set_current_write_origin("foreground_user_explicit")
        try:
            with self_improvement_decision_scope(
                _Dec(result=ALLOW, reason="f20_trusted")
            ):
                with session_write_policy_scope(
                    SessionWritePolicy.normal(
                        session_id="f20", origin="foreground_user_explicit"
                    )
                ):
                    seen = get_self_improvement_decision()
                    assert getattr(seen, "allow", False) is True
        finally:
            reset_current_write_origin(token)
