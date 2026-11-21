"""Atomic batches and background-review read marks: a stale target aborts and rolls back the batch,
and a rollback puts back exactly the read marks the batch's own writes moved (a journal), leaving every
other mark (parallel workers, other skills, hook-made writes) as it was. Shared fixtures and helpers
live in test_skill_stale_read_guard."""
import json

import pytest

from tests.tools.test_skill_manager_tool import VALID_SKILL_CONTENT
import tests.tools.test_skill_stale_read_guard as _guard
from tests.tools.test_skill_stale_read_guard import REF, WRITES, _dispatch, _fork, _other_writer_appends, _view

skills = _guard.skills  # the shared fixture, used by name below


def test_a_stale_target_aborts_a_batch_and_rolls_back_its_earlier_ops(skills):
    skill_md, ref = skills / "bg-skill" / "SKILL.md", skills / "bg-skill" / REF
    with _fork():
        _view("SKILL.md")
        _view(REF)
        before = skill_md.read_bytes()
        _other_writer_appends(ref)
        theirs = ref.read_bytes()
        result = _dispatch({"operations": [
            {**WRITES["patch"][0], "name": "bg-skill"}, {**WRITES["patch_ref"][0], "name": "bg-skill"}]})
    assert result["success"] is False and result.get("_stale_read") is True, result
    assert result["failed_index"] == 1
    assert skill_md.read_bytes() == before and ref.read_bytes() == theirs


def test_a_rolled_back_batch_needs_one_reread_then_can_be_retried(skills):
    """A rollback drops the marks the batch's own writes left: the retry is refused as unread, and
    after one re-read it goes through."""
    with _fork():
        _view("SKILL.md")
        failed = _dispatch({"operations": [
            {**WRITES["patch"][0], "name": "bg-skill"},
            {"action": "patch", "name": "bg-skill", "old_string": "no such text", "new_string": "x"}]})
        assert failed["success"] is False and failed["failed_index"] == 1
        _retry_needs_one_reread("SKILL.md", {**WRITES["patch"][0], "name": "bg-skill"})


def test_a_rollback_keeps_a_mark_a_parallel_read_added_meanwhile(skills, monkeypatch):
    """Copied tool workers of one review share its marks; restoring the pre-batch marks must not
    erase a read another worker recorded while the batch ran."""
    import tools.skill_manager_batch as batch
    from tools.skill_manager_guards import mark_background_review_skill_read
    real_rollback = batch._rollback

    def rollback_while_another_worker_reads(*a, **kw):
        mark_background_review_skill_read(skills / "bg-skill" / REF)
        return real_rollback(*a, **kw)

    monkeypatch.setattr(batch, "_rollback", rollback_while_another_worker_reads)
    with _fork():
        _view("SKILL.md")
        failed = _dispatch({"operations": [
            {**WRITES["patch"][0], "name": "bg-skill"},
            {"action": "patch", "name": "bg-skill", "old_string": "no such text", "new_string": "x"}]})
        assert failed["success"] is False
        result = _dispatch({**WRITES["patch_ref"][0], "name": "bg-skill"})
    assert result["success"] is True, result


def _other_skill(skills, nested=False):
    """A second skill: a sibling of bg-skill, or (nested) a separate skill inside bg-skill's folder,
    which discovery finds as its own skill with its own lock."""
    content = VALID_SKILL_CONTENT.replace("name: test-skill", "name: other-skill")
    if nested:
        other = skills / "bg-skill" / "other-skill"
        other.mkdir()
        (other / "SKILL.md").write_text(content, encoding="utf-8")
    else:
        assert _dispatch({"action": "create", "name": "other-skill", "content": content})["success"]
        other = skills / "other-skill"
    (other / "references").mkdir()
    (other / REF).write_text("Other thing.\n", encoding="utf-8")
    return other


class _ParallelWorker:
    """Runs dispatches the way a parallel tool call does: in a context copied when it was submitted
    (tools.thread_context.propagate_context_to_thread), i.e. before the batch started, so it sees the
    review's shared marks but not anything the batch set in its own context."""

    def __init__(self):
        self.ctx = None

    def capture(self):
        import contextvars
        self.ctx = contextvars.copy_context()

    def dispatch(self, args):
        return self.ctx.run(_dispatch, args)


def _failing_batch_on_bg_skill():
    return {"operations": [
        {**WRITES["patch"][0], "name": "bg-skill"},
        {"action": "patch", "name": "bg-skill", "old_string": "no such text", "new_string": "x"}]}


@pytest.mark.parametrize("worker_op", ["patch", "remove_file"])
def test_a_rollback_leaves_marks_of_skills_outside_the_batch_alone(skills, monkeypatch, worker_op):
    """While a batch on bg-skill runs, a parallel worker of the same review writes other-skill (whose
    lock the batch never held). The batch's rollback must not put back other-skill's old marks: a
    stale digest would refuse the worker's next write, and a resurrected mark of a removed file would
    let a file put back at that path be overwritten unread."""
    import tools.skill_manager_batch as batch
    other = _other_skill(skills)
    real_rollback = batch._rollback
    worker = _ParallelWorker()

    def rollback_after_a_parallel_write(*a, **kw):
        if worker_op == "patch":
            assert worker.dispatch({"action": "patch", "name": "other-skill", "file_path": REF,
                                    "old_string": "Other thing.", "new_string": "Other thing, done."})["success"]
        else:
            assert worker.dispatch({"action": "remove_file", "name": "other-skill", "file_path": REF})["success"]
        return real_rollback(*a, **kw)

    monkeypatch.setattr(batch, "_rollback", rollback_after_a_parallel_write)
    with _fork():
        _view(REF)
        from tools.skills_tool import skill_view
        assert json.loads(skill_view("other-skill", REF))["success"] is True
        worker.capture()
        assert _dispatch(_failing_batch_on_bg_skill())["success"] is False
        if worker_op == "patch":
            result = _dispatch({"action": "patch", "name": "other-skill", "file_path": REF,
                                "old_string": "Other thing, done.", "new_string": "Other thing, twice."})
            assert result["success"] is True, result
        else:
            (other / "references").mkdir(exist_ok=True)
            (other / REF).write_text("Other thing.\n", encoding="utf-8")
            result = _dispatch({"action": "write_file", "name": "other-skill", "file_path": REF,
                                "file_content": "Overwritten.\n"})
            assert result.get("_read_before_write_required") is True, result


@pytest.mark.parametrize("nested", [False, True], ids=["sibling", "nested"])
def test_a_rollback_does_not_bless_a_revert_of_a_skill_outside_the_batch(skills, monkeypatch, nested):
    """A parallel worker writes other-skill, then someone else puts its old bytes back. The fork once
    read those old bytes, but the revert is another writer's change made after the fork's own write:
    the batch's rollback must not re-validate other-skill's mark against them. A skill nested inside
    the batch skill's folder is still another skill (own lock), not part of the batch."""
    import tools.skill_manager_batch as batch
    other = _other_skill(skills, nested)
    original = (other / REF).read_bytes()
    real_rollback = batch._rollback
    worker = _ParallelWorker()

    def worker_writes_then_someone_reverts(*a, **kw):
        assert worker.dispatch({"action": "patch", "name": "other-skill", "file_path": REF,
                                "old_string": "Other thing.", "new_string": "Other thing, done."})["success"]
        (other / REF).write_bytes(original)
        return real_rollback(*a, **kw)

    monkeypatch.setattr(batch, "_rollback", worker_writes_then_someone_reverts)
    with _fork():
        from tools.skills_tool import skill_view
        assert json.loads(skill_view("other-skill", REF))["success"] is True
        _view("SKILL.md")
        worker.capture()
        assert _dispatch(_failing_batch_on_bg_skill())["success"] is False
        result = _dispatch({"action": "write_file", "name": "other-skill", "file_path": REF,
                            "file_content": "Overwritten.\n"})
    assert result.get("_stale_read") is True, result


def test_a_rollback_keeps_a_nested_skills_removed_mark_removed(skills, monkeypatch):
    """A parallel worker removes a supporting file of a skill nested in bg-skill's folder (dropping
    its mark). bg-skill's rollback copies its whole folder back, which brings that file back; the
    nested skill's mark must stay dropped, so writing the file needs a fresh read."""
    import tools.skill_manager_batch as batch
    other = _other_skill(skills, nested=True)
    real_rollback = batch._rollback
    worker = _ParallelWorker()

    def worker_removes(*a, **kw):
        assert worker.dispatch({"action": "remove_file", "name": "other-skill", "file_path": REF})["success"]
        return real_rollback(*a, **kw)

    monkeypatch.setattr(batch, "_rollback", worker_removes)
    with _fork():
        from tools.skills_tool import skill_view
        assert json.loads(skill_view("other-skill", REF))["success"] is True
        _view("SKILL.md")
        worker.capture()
        assert _dispatch(_failing_batch_on_bg_skill())["success"] is False
        assert (other / REF).exists()
        result = _dispatch({"action": "write_file", "name": "other-skill", "file_path": REF,
                            "file_content": "Overwritten.\n"})
    assert result.get("_read_before_write_required") is True, result


def test_a_rollback_drops_a_mark_whose_file_now_holds_bytes_the_fork_never_read(skills):
    """A writer that takes no lock changes a file of the batch's skill after the fork read it and
    before the batch starts; the rollback restores those bytes. The fork never read them, so its
    mark must not be moved onto them: the next write to that file is refused."""
    ref = skills / "bg-skill" / REF
    with _fork():
        _view(REF)
        _view("SKILL.md")
        _other_writer_appends(ref)
        theirs = ref.read_bytes()
        assert _dispatch(_failing_batch_on_bg_skill())["success"] is False
        result = _dispatch({**WRITES["write_file"][0], "name": "bg-skill"})
    assert result.get("_stale_read") is True or result.get("_read_before_write_required") is True, result
    assert ref.read_bytes() == theirs


def _retry_needs_one_reread(rel, args):
    """Inside a fork, after a rolled-back batch: the write is refused as unread (never as allowed), and
    after one re-read it goes through."""
    refused = _dispatch(args)
    assert refused["success"] is False and refused.get("_read_before_write_required") is True, refused
    _view(rel)
    retry = _dispatch(args)
    assert retry["success"] is True, retry


def _failing_batch(*ops):
    return {"operations": [*ops, {"action": "patch", "name": "bg-skill", "old_string": "no such text",
                                  "new_string": "x"}]}


def test_a_rolled_back_write_beside_a_template_skill_md_needs_one_reread(skills):
    """A SKILL.md template kept in a support folder is not a skill boundary: the batch's own mark on
    the file beside it is dropped like any other."""
    (skills / "bg-skill" / "references" / "SKILL.md").write_text("Template.\n", encoding="utf-8")
    patch_ref = {**WRITES["patch_ref"][0], "name": "bg-skill"}
    with _fork():
        _view(REF)
        assert _dispatch(_failing_batch(patch_ref))["success"] is False
        _retry_needs_one_reread(REF, patch_ref)


def test_two_writes_to_one_file_in_a_rolled_back_batch_drop_the_mark(skills):
    patch_ref = {**WRITES["patch_ref"][0], "name": "bg-skill"}
    again = {"action": "patch", "name": "bg-skill", "file_path": REF, "old_string": "Do it.", "new_string": "Do it twice."}
    with _fork():
        _view(REF)
        assert _dispatch(_failing_batch(patch_ref, again))["success"] is False
        assert (skills / "bg-skill" / REF).read_bytes() == b"Do the thing.\n"
        _retry_needs_one_reread(REF, patch_ref)


def test_a_file_removed_by_a_rolled_back_batch_needs_one_reread(skills):
    with _fork():
        _view(REF)
        assert _dispatch(_failing_batch({"action": "remove_file", "name": "bg-skill", "file_path": REF}))[
            "success"] is False
        assert (skills / "bg-skill" / REF).read_bytes() == b"Do the thing.\n"
        _retry_needs_one_reread(REF, {**WRITES["patch_ref"][0], "name": "bg-skill"})


def test_a_change_landing_after_the_rollback_is_not_taken_as_read(skills, monkeypatch):
    """Between the rollback's restore and the mark fix-up, a writer that takes no lock changes the
    file. The fork never read those bytes: its next write is refused."""
    import tools.skill_manager_batch as batch
    ref = skills / "bg-skill" / REF
    real_rollback = batch._rollback

    def rollback_then_someone_writes(*a, **kw):
        out = real_rollback(*a, **kw)
        ref.write_bytes(b"Someone else's bytes.\n")
        return out

    monkeypatch.setattr(batch, "_rollback", rollback_then_someone_writes)
    with _fork():
        _view(REF)
        assert _dispatch(_failing_batch({**WRITES["patch_ref"][0], "name": "bg-skill"}))["success"] is False
        result = _dispatch({**WRITES["write_file"][0], "name": "bg-skill"})
    assert result.get("_stale_read") is True or result.get("_read_before_write_required") is True, result
    assert ref.read_bytes() == b"Someone else's bytes.\n"


def test_a_read_made_after_the_rollback_is_kept(skills, monkeypatch):
    """A parallel worker reads the file right after the rollback (new bytes someone wrote); its mark
    must survive the batch's fix-up, so the fork can write what it just read."""
    import tools.skill_manager_batch as batch
    ref = skills / "bg-skill" / REF
    real_rollback = batch._rollback

    def rollback_then_change_and_read(*a, **kw):
        out = real_rollback(*a, **kw)
        ref.write_bytes(b"Newer bytes.\n")
        _view(REF)
        return out

    monkeypatch.setattr(batch, "_rollback", rollback_then_change_and_read)
    with _fork():
        _view(REF)
        assert _dispatch(_failing_batch({**WRITES["patch_ref"][0], "name": "bg-skill"}))["success"] is False
        result = _dispatch({"action": "patch", "name": "bg-skill", "file_path": REF,
                            "old_string": "Newer bytes.", "new_string": "Newest."})
    assert result["success"] is True, result


def test_a_parallel_read_of_the_bytes_the_batch_wrote_is_kept(skills, monkeypatch):
    """The batch writes B; a parallel worker reads that same B; then the batch fails and restores A.
    The worker's mark records a read of B, not the batch's write: the rollback must leave it, so a
    write against the restored A is refused."""
    import tools.skill_manager_batch as batch
    real_rollback = batch._rollback
    worker = _ParallelWorker()

    def worker_reads_then_rollback(*a, **kw):
        worker.ctx.run(_view, REF)
        return real_rollback(*a, **kw)

    monkeypatch.setattr(batch, "_rollback", worker_reads_then_rollback)
    with _fork():
        _view(REF)
        worker.capture()
        assert _dispatch(_failing_batch({**WRITES["patch_ref"][0], "name": "bg-skill"}))["success"] is False
        assert (skills / "bg-skill" / REF).read_bytes() == b"Do the thing.\n"
        result = _dispatch({**WRITES["write_file"][0], "name": "bg-skill"})
    assert result.get("_stale_read") is True, result


def test_a_read_of_bytes_the_batch_once_wrote_after_the_rollback_is_kept(skills, monkeypatch):
    """After the rollback restores A, someone writes the batch's B again and a parallel worker reads
    it. That fresh read must survive the batch's mark fix-up, so the worker can write what it read."""
    import tools.skill_manager_batch as batch
    ref = skills / "bg-skill" / REF
    real_rollback = batch._rollback
    worker = _ParallelWorker()
    written = {}

    def rollback_then_b_again_and_read(*a, **kw):
        written["b"] = ref.read_bytes()
        out = real_rollback(*a, **kw)
        ref.write_bytes(written["b"])
        worker.ctx.run(_view, REF)
        return out

    monkeypatch.setattr(batch, "_rollback", rollback_then_b_again_and_read)
    with _fork():
        _view(REF)
        worker.capture()
        assert _dispatch(_failing_batch({**WRITES["patch_ref"][0], "name": "bg-skill"}))["success"] is False
        result = worker.dispatch({"action": "patch", "name": "bg-skill", "file_path": REF,
                                  "old_string": "Do it.", "new_string": "Do it now."})
    assert result["success"] is True, result


def test_a_write_made_by_a_hook_during_a_batch_keeps_its_mark(skills, monkeypatch):
    """A synchronous hook that runs after a batch op (via _record_success, e.g. on_skill_lifecycle)
    writes another skill. That write is not the batch's: the batch's rollback leaves its file and
    must leave its mark too, so the next write to it needs no re-read."""
    import tools.skill_manager_tool as smt
    other = _other_skill(skills)
    real_record = smt._record_success
    fired = {}

    def hook_writes_other_skill(action, name, *a, **kw):
        real_record(action, name, *a, **kw)
        if name == "bg-skill" and not fired:
            fired["done"] = True
            assert smt.skill_manage(action="patch", name="other-skill", file_path=REF,
                                    old_string="Other thing.", new_string="Other thing, hooked.")
    monkeypatch.setattr(smt, "_record_success", hook_writes_other_skill)
    with _fork():
        _view(REF)
        from tools.skills_tool import skill_view
        assert json.loads(skill_view("other-skill", REF))["success"] is True
        assert _dispatch(_failing_batch({**WRITES["patch_ref"][0], "name": "bg-skill"}))["success"] is False
        assert (other / REF).read_text(encoding="utf-8") == "Other thing, hooked.\n"
        result = _dispatch({"action": "patch", "name": "other-skill", "file_path": REF,
                            "old_string": "Other thing, hooked.", "new_string": "Other thing, again."})
    assert result["success"] is True, result


def test_a_mark_a_hook_dropped_during_a_batch_stays_dropped_after_the_rollback(skills, monkeypatch):
    """The batch removes REF (read as A). A hook running after that op (not the batch's) finds a file B
    at that path, reads it and removes it too. When the batch rolls back and restores A, the hook's
    later drop stands: two drops are two different updates, and writing REF needs a fresh read."""
    import tools.skill_manager_tool as smt
    ref = skills / "bg-skill" / REF
    real_record = smt._record_success
    fired = {}

    def hook_reads_and_removes_a_new_file(action, name, *a, **kw):
        real_record(action, name, *a, **kw)
        if action == "remove_file" and not fired:
            fired["done"] = True
            ref.parent.mkdir(exist_ok=True)
            ref.write_text("Hook's bytes.\n", encoding="utf-8")
            _view(REF)
            assert json.loads(smt.skill_manage(action="remove_file", name="bg-skill", file_path=REF))["success"]
    monkeypatch.setattr(smt, "_record_success", hook_reads_and_removes_a_new_file)
    with _fork():
        _view(REF)
        failing = _failing_batch({"action": "remove_file", "name": "bg-skill", "file_path": REF})
        assert _dispatch(failing)["success"] is False
        assert fired and ref.read_bytes() == b"Do the thing.\n"
        result = _dispatch({**WRITES["write_file"][0], "name": "bg-skill"})
    assert result.get("_read_before_write_required") is True, result


def test_a_parallel_read_during_the_batch_writes_scan_is_not_taken_as_the_batchs(skills, monkeypatch):
    """The batch's patch has published B and is being security-scanned when a parallel worker reads B.
    That read is the worker's: after the batch fails and restores A it must survive, so the next write
    is refused as stale (B was read, A is there), not treated as never read."""
    import tools.skill_manager_tool as smt
    real_scan = smt._security_scan_skill
    worker = _ParallelWorker()
    fired = {}

    def scan_while_a_worker_reads(*a, **kw):
        if not fired:
            fired["done"] = True
            worker.ctx.run(_view, REF)
        return real_scan(*a, **kw)
    monkeypatch.setattr(smt, "_security_scan_skill", scan_while_a_worker_reads)
    with _fork():
        _view(REF)
        worker.capture()
        assert _dispatch(_failing_batch({**WRITES["patch_ref"][0], "name": "bg-skill"}))["success"] is False
        assert fired and (skills / "bg-skill" / REF).read_bytes() == b"Do the thing.\n"
        result = _dispatch({**WRITES["write_file"][0], "name": "bg-skill"})
    assert result.get("_stale_read") is True, result


def test_a_parallel_read_between_two_batch_writes_is_not_overwritten_by_an_older_mark(skills, monkeypatch):
    """A -> the batch writes B -> a parallel worker reads B -> the batch writes C -> the batch fails and
    restores A. The fork must not be left holding a mark for A it would write against unread: the
    next write is refused, and the file stays A."""
    import tools.skill_manager_tool as smt
    ref = skills / "bg-skill" / REF
    real_record = smt._record_success
    worker = _ParallelWorker()
    fired = {}

    def worker_reads_after_the_first_write(action, name, *a, **kw):
        real_record(action, name, *a, **kw)
        if not fired:
            fired["done"] = True
            worker.ctx.run(_view, REF)
    monkeypatch.setattr(smt, "_record_success", worker_reads_after_the_first_write)
    again = {"action": "patch", "name": "bg-skill", "file_path": REF, "old_string": "Do it.", "new_string": "Do it twice."}
    with _fork():
        _view(REF)
        worker.capture()
        assert _dispatch(_failing_batch({**WRITES["patch_ref"][0], "name": "bg-skill"}, again))["success"] is False
        assert fired and ref.read_bytes() == b"Do the thing.\n"
        result = _dispatch({**WRITES["write_file"][0], "name": "bg-skill"})
    assert result["success"] is False, result
    assert result.get("_stale_read") is True or result.get("_read_before_write_required") is True, result
    assert ref.read_bytes() == b"Do the thing.\n"
