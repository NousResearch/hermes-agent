"""Explicit work references fence real delegation construction and result settlement."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from types import SimpleNamespace

import pytest

from hermes_cli.task_delivery.store import WorkLedger
from hermes_state import SessionDB
from tools import delegate_tool as delegate
from tools.delegate_tool_dispatch import _Batch, _release_unstarted_child


def _context(task="fix", session="owner", role="implement", **extra):
    payload = {"session_id": session, "task_id": task, "role": role, **extra}
    return "[HERMES_WORK_ITEM_V1]\n" + json.dumps(payload) + "\n\nKeep the existing behavior."


def _route(profile_home=None):
    creds = dict.fromkeys(("provider", "base_url", "api_key", "api_mode", "command", "args"))
    creds["model"] = "fixture"
    return SimpleNamespace(error=None, creds=creds, routing_cfg={}, reasoning_effort=None,
                           profile_home=profile_home, report=lambda: None)


def _build(parent, tasks, *, routes=None, rebuild=None):
    return delegate._build_children(
        tasks, [None] * len(tasks), routes or [_route() for _ in tasks], top_role="leaf",
        max_iterations=1, parent_agent=parent, live_deleg_id=None, live_writers=[], rebuild=rebuild,
    )


def _batch(parent, children):
    batch = _Batch(**{field.name: None for field in fields(_Batch)})
    batch.parent_agent, batch.children = parent, children
    return batch


@pytest.fixture
def world(tmp_path, monkeypatch):
    built, closed = [], []

    def builder(**kwargs):
        child = SimpleNamespace(_subagent_id=f"child-{len(built)}", supplied_context=kwargs["context"])
        built.append(child)
        return child

    monkeypatch.setattr(delegate, "_build_child_preserving_parent_tools", builder)
    monkeypatch.setattr(delegate, "pin_child_service_tier", lambda *args: None)
    monkeypatch.setattr(delegate, "_detach_child", lambda *args: None)
    monkeypatch.setattr(delegate, "_close_child", lambda child, *args: closed.append(child))
    with SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session("owner", source="gui")
        ledger = WorkLedger(db)
        for task in ("fix", "other"):
            ledger.register("owner", task, title="Repair behavior", acceptance="The actual request succeeds")
        parent = SimpleNamespace(session_id="owner", _session_db=db)
        yield parent, ledger, built, closed


def test_managed_claim_precedes_any_child_and_competing_dispatch_builds_only_once(world):
    parent, ledger, built, _ = world
    task = {"goal": "Repair the service", "context": _context()}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: _build(parent, [task]), range(2)))
    winners = [children for children, error in results if error is None]
    assert len(winners) == 1 and len(built) == 1
    claim = ledger.get("owner", "fix")["claim"]
    assert claim and claim["owner"] in built[0].supplied_context
    assert str(claim["generation"]) in built[0].supplied_context
    assert any("claimed" in str(error) for _, error in results if error)
    assert task["context"] == _context(), "caller context is immutable"


def test_all_protocols_and_profiles_are_validated_before_claim_or_build(world, tmp_path):
    parent, ledger, built, _ = world
    good = {"goal": "Repair first module", "context": _context()}
    invalid_contexts = [
        _context(task="other", session="another-session"),
        _context(task="other", role="admin"),
        _context(task="other", files="service.py"),
        _context(task="other", db_path=str(tmp_path / "other.db")),
        '[HERMES_WORK_ITEM_V1]\n{"session_id":"owner","session_id":"owner","task_id":"other","role":"implement"}\n\ntext',
        '[HERMES_WORK_ITEM_V1]\n{bad json}\n\ntext',
    ]
    for context in invalid_contexts:
        children, error = _build(parent, [good, {"goal": "Repair other module", "context": context}])
        assert error and not children and not built
        assert ledger.get("owner", "fix")["claim"] is None
    children, error = _build(parent, [good], routes=[_route(tmp_path / "another-profile")])
    assert error and "profile" in error and not children and not built
    children, error = _build(SimpleNamespace(session_id="owner", _session_db=None), [good])
    assert error and not children and not built


def test_batch_duplicate_and_later_builder_failure_release_only_new_claims(world, monkeypatch):
    parent, ledger, built, closed = world
    task = {"goal": "Repair the service", "context": _context()}
    children, error = _build(parent, [task, task])
    assert error and not children and not built
    assert ledger.get("owner", "fix")["claim"] is None
    old = ledger.claim("owner", "other", owner="existing-worker", role="implement", request_id="existing")
    children, error = _build(parent, [task, {"goal": "Other module", "context": _context(task="other")}])
    assert error and not children and not built
    assert ledger.get("owner", "fix")["claim"] is None
    assert ledger.get("owner", "other")["claim"] == old["claim"]
    ledger.release("owner", "other", owner="existing-worker", generation=old["claim"]["generation"], reason="fixture")
    real_builder = delegate._build_child_preserving_parent_tools

    def fail_second(**kwargs):
        if kwargs["task_index"] == 1:
            raise RuntimeError("fixture builder failure")
        return real_builder(**kwargs)

    monkeypatch.setattr(delegate, "_build_child_preserving_parent_tools", fail_second)
    children, error = _build(parent, [task, {"goal": "Other module", "context": _context(task="other")}])
    assert error and not children and len(closed) == 1
    assert all(ledger.get("owner", name)["claim"] is None for name in ("fix", "other"))


def test_completed_execution_does_not_claim_a_delivery_or_passing_review(world, monkeypatch):
    parent, ledger, _, _ = world
    task = {"goal": "Repair the service", "context": _context()}
    children, error = _build(parent, [task])
    assert error is None
    monkeypatch.setattr(delegate, "_run_single_child", lambda **kwargs: {
        "task_index": 0, "status": "completed", "summary": '{"passed":true}',
    })
    result = _batch(parent, children).run_child(*children[0])
    item = ledger.get("owner", "fix")
    assert result["status"] == "completed"
    assert item["implementation"] == "needs_artifact"
    assert item["claim"] is None and item["candidate"] is None
    assert item["review"] is None and item["acceptance"] is None


def test_explicit_delivery_survives_execution_settlement_and_blocks_a_new_builder(world, tmp_path, monkeypatch):
    parent, ledger, built, _ = world
    task = {"goal": "Repair the service", "context": _context()}
    children, error = _build(parent, [task])
    assert error is None
    artifact = tmp_path / "candidate.patch"
    artifact.write_text("synthetic patch")

    def execute(**kwargs):
        claim = ledger.get("owner", "fix")["claim"]
        ledger.deliver("owner", "fix", owner=claim["owner"], generation=claim["generation"],
                       path=artifact, summary="Concrete candidate")
        return {"task_index": 0, "status": "completed", "summary": "Delivered"}

    monkeypatch.setattr(delegate, "_run_single_child", execute)
    _batch(parent, children).run_child(*children[0])
    assert ledger.get("owner", "fix")["implementation"] == "delivered"
    refused, error = _build(parent, [task])
    assert not refused and "delivered" in error and len(built) == 1


def test_managed_continue_is_refused_and_ordinary_best_of_n_keeps_context(world):
    parent, ledger, built, _ = world
    managed = {"goal": "Repair the service", "context": _context()}
    children, error = _build(parent, [managed], rebuild=[{}])
    assert error and "continu" in error and not children and not built
    assert ledger.get("owner", "fix")["claim"] is None
    ordinary = {"goal": "Review this same implementation", "context": "Discuss [HERMES_WORK_ITEM_V1] as text."}
    children, error = _build(SimpleNamespace(), [ordinary, ordinary])
    assert error is None and len(children) == 2
    assert all(child.supplied_context == ordinary["context"] for _, _, child in children)


def test_unstarted_child_release_and_execution_exception_preserve_reconciliation(world, monkeypatch):
    parent, ledger, _, _ = world
    task = {"goal": "Repair the service", "context": _context()}
    children, error = _build(parent, [task])
    assert error is None
    monkeypatch.setattr("tools.delegate_tool_resumable.release_unstarted", lambda child: None)
    monkeypatch.setattr("tools.delegate_tool_dispatch._detach_child", lambda *args: None)
    monkeypatch.setattr("tools.delegate_tool_dispatch._close_child", lambda *args: None)
    _release_unstarted_child(_batch(parent, children), children[0][2])
    assert ledger.get("owner", "fix")["claim"] is None
    children, error = _build(parent, [task])
    assert error is None

    def raises(**kwargs):
        raise RuntimeError("fixture execution failed")

    monkeypatch.setattr(delegate, "_run_single_child", raises)
    with pytest.raises(RuntimeError, match="fixture"):
        _batch(parent, children).run_child(*children[0])
    item = ledger.get("owner", "fix")
    assert item["claim"] is None and item["implementation"] == "needs_reconciliation"


def test_background_runner_uses_the_same_work_execution_settlement(world, monkeypatch):
    from tools import delegate_tool_dispatch as dispatch
    parent, ledger, _, _ = world
    task = {"goal": "Repair the service", "context": _context()}
    children, error = _build(parent, [task])
    assert error is None
    batch = _batch(parent, children)
    batch.task_list, batch.live_writers, batch.live_paths, batch.refused = [task], [], [], []
    batch.creds, batch.overall_start = {"model": "fixture"}, time.monotonic()
    monkeypatch.setattr(delegate, "_run_single_child", lambda **kwargs: {
        "task_index": 0, "status": "completed", "summary": "No artifact yet",
    })
    monkeypatch.setattr(dispatch, "_finalize_child_results", lambda *args: None)
    monkeypatch.setattr(dispatch, "finish_delegation_unit", lambda *args, **kwargs: None)
    monkeypatch.setattr("tools.delegation_live_log.update_manifest_statuses", lambda *args, **kwargs: None)
    completed = []

    def accept(**kwargs):
        completed.append(kwargs["runner"]())
        return {"status": "dispatched", "delegation_id": "synthetic-unit"}

    monkeypatch.setattr("tools.async_delegation.dispatch_async_delegation_batch", accept)
    receipt = dispatch._dispatch_unit(batch, "synthetic-unit", None, {})
    assert receipt["status"] == "dispatched" and completed[0]["results"][0]["status"] == "completed"
    assert ledger.get("owner", "fix")["implementation"] == "needs_artifact"


def test_stale_claim_is_checked_before_execution_and_cannot_change_its_replacement(world, monkeypatch):
    parent, ledger, _, _ = world
    task = {"goal": "Repair the service", "context": _context()}
    children, error = _build(parent, [task])
    assert error is None
    old = ledger.get("owner", "fix")["claim"]
    ledger.release("owner", "fix", owner=old["owner"], generation=old["generation"], reason="Owner cancelled")
    replacement = ledger.claim("owner", "fix", owner="replacement", role="implement", request_id="new")
    ran = []
    monkeypatch.setattr(delegate, "_run_single_child", lambda **kwargs: ran.append(kwargs))
    monkeypatch.setattr("tools.delegate_tool_dispatch._detach_child", lambda *args: None)
    monkeypatch.setattr("tools.delegate_tool_dispatch._close_child", lambda *args: None)
    result = _batch(parent, children).run_child(*children[0])
    assert result["status"] == "error" and ran == []
    assert ledger.get("owner", "fix")["claim"] == replacement["claim"]


def test_dispatch_failure_before_execution_releases_work_without_touching_plain_tasks(world, monkeypatch):
    from tools import delegate_tool_dispatch as dispatch
    parent, ledger, _, _ = world
    task = {"goal": "Repair the service", "context": _context()}
    children, error = _build(parent, [task])
    assert error is None
    batch = _batch(parent, children)
    batch.task_list = [task]
    monkeypatch.setattr(dispatch, "begin_delegation_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(delegate, "session_joins_delegation", lambda parent: False)
    monkeypatch.setattr("tools.delegate_tool_resumable.release_unstarted", lambda child: None)
    monkeypatch.setattr(dispatch, "_detach_child", lambda *args: None)
    monkeypatch.setattr(dispatch, "_close_child", lambda *args: None)

    def rejected(*args):
        raise RuntimeError("fixture async admission failed")

    monkeypatch.setattr(dispatch, "_dispatch_background", rejected)
    with pytest.raises(RuntimeError, match="fixture"):
        dispatch._run_batch(batch, True)
    assert ledger.get("owner", "fix")["claim"] is None


def test_async_session_wrapper_uses_its_existing_profile_database(world, monkeypatch):
    from hermes_state import AsyncSessionDB
    parent, ledger, _, _ = world
    parent._session_db = AsyncSessionDB(parent._session_db)
    children, error = _build(parent, [{"goal": "Repair the service", "context": _context()}])
    assert error is None and ledger.get("owner", "fix")["claim"] is not None
    monkeypatch.setattr(delegate, "_run_single_child", lambda **kwargs: {
        "task_index": 0, "status": "interrupted", "summary": "Stopped",
    })
    _batch(parent, children).run_child(*children[0])
    assert ledger.get("owner", "fix")["implementation"] == "needs_reconciliation"


def test_review_execution_summary_is_not_a_structured_review_delivery(world, tmp_path, monkeypatch):
    parent, ledger, _, _ = world
    item = ledger.claim("owner", "fix", owner="implementer", role="implement", request_id="implementation")
    artifact = tmp_path / "candidate.patch"
    artifact.write_text("candidate")
    ledger.deliver("owner", "fix", owner="implementer", generation=item["claim"]["generation"],
                   path=artifact, summary="Implemented")
    children, error = _build(parent, [{"goal": "Review the service", "context": _context(role="review")}])
    assert error is None
    monkeypatch.setattr(delegate, "_run_single_child", lambda **kwargs: {
        "task_index": 0, "status": "completed", "summary": '{"passed":true,"findings":[]}',
    })
    _batch(parent, children).run_child(*children[0])
    item = ledger.get("owner", "fix")
    assert item["implementation"] == "delivered" and item["review"] is None
    assert item["claim"] is None and item["acceptance"] is None


@pytest.mark.parametrize("reference_session", ["owner", "compressed-tip"])
def test_verified_compression_continuation_keeps_the_original_work_scope(world, monkeypatch, reference_session):
    parent, ledger, built, _ = world
    db = parent._session_db
    db.end_session("owner", "compression")
    db.create_session("compressed-tip", source="gui", parent_session_id="owner")
    db.end_session("compressed-tip", "compression")
    db.create_session("current-tip", source="gui", parent_session_id="compressed-tip")
    parent.session_id = "current-tip"
    children, error = _build(parent, [{"goal": "Repair the service", "context": _context(session=reference_session)}])
    assert error is None and len(children) == 1
    item = ledger.get("current-tip", "fix")
    assert item["session_id"] == "owner" and item["claim"] == ledger.get("owner", "fix")["claim"]
    details = json.loads(built[0].supplied_context.split("[HERMES_WORK_CLAIM_V1]\n", 1)[1].split("\n", 1)[0])
    assert details["session_id"] == "owner"
    monkeypatch.setattr(delegate, "_run_single_child", lambda **kwargs: {
        "task_index": 0, "status": "completed", "summary": "Still needs an artifact",
    })
    _batch(parent, children).run_child(*children[0])
    assert ledger.get("owner", "fix")["implementation"] == "needs_artifact"


@pytest.mark.parametrize("marker", ["_branched_from", "_reset_from", "_delegate_from"])
def test_explicit_fork_never_inherits_the_compressed_parents_work_claim(world, marker):
    parent, ledger, built, _ = world
    db = parent._session_db
    db.end_session("owner", "compression")
    db.create_session("fork", source="gui", parent_session_id="owner", model_config={marker: "owner"})
    parent.session_id = "fork"
    children, error = _build(parent, [{"goal": "Repair the service", "context": _context()}])
    assert error and "session" in error and not children and not built
    assert ledger.get("owner", "fix")["claim"] is None


@pytest.mark.parametrize("role", ["implement", "review"])
def test_claim_argument_examples_execute_real_renewal_and_role_delivery(world, tmp_path, role):
    from hermes_cli.task_delivery.cli import configure, execute
    parent, ledger, built, _ = world
    artifact = tmp_path / "candidate.patch"
    artifact.write_text("candidate")
    if role == "review":
        item = ledger.claim("owner", "fix", owner="implementer", role="implement", request_id="first")
        ledger.deliver("owner", "fix", owner="implementer", generation=item["claim"]["generation"],
                       path=artifact, summary="Concrete candidate")
    children, error = _build(parent, [{"goal": "Complete managed work", "context": _context(role=role)}])
    assert error is None and children
    details = json.loads(built[0].supplied_context.split("[HERMES_WORK_CLAIM_V1]\n", 1)[1].split("\n", 1)[0])
    parser = configure(argparse.ArgumentParser())
    renewed = execute(parser.parse_args(details["commands"]["renew"][2:]), parent._session_db)
    assert renewed["claim"]["generation"] == details["generation"]
    assert renewed["claim"]["expires_at"] >= details["expires_at"]
    if role == "review":
        report = tmp_path / "review.json"
        report.write_text(json.dumps({"request_id": "review-1", "reviewer": "independent",
            "candidate_digest": renewed["candidate"]["digest"], "passed": True, "findings": []}))
        action, replacements = "review", {"<review-report.json>": str(report)}
    else:
        action, replacements = "deliver", {"<artifact-path>": str(artifact), "<delivery-summary>": "Implemented"}
    argv = [replacements.get(value, value) for value in details["commands"][action][2:]]
    completed = execute(parser.parse_args(argv), parent._session_db)
    assert completed["claim"] is None and completed["acceptance"] is None
    if role == "review":
        assert completed["review"]["status"] == "passed"
    else:
        assert completed["implementation"] == "delivered"
