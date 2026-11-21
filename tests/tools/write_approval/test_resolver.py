"""The resolver contract for write approvals (tools/write_approval.py).

``preview_pending(subsystem, pending_id)`` returns the ``expect`` a GUI holds. ``resolve_pending(
subsystem, pending_id, decision, *, expect=None, memory_store=None)`` takes ``approve`` or
``reject`` and returns ``status``: ``applied``, ``rejected``, ``already_resolved`` (+ ``prior``),
``conflict`` (+ ``reason``) or ``needs_review``. An approval applies only if every target still
equals its staged base, checked under the applier's lock. An approve with an ``expect`` (GUI) also
pins the record bytes and the profile; without one (CLI) the staged base decides. A legacy record
has no base, so a GUI approve needs an ``expect`` and the CLI applies it unchecked (``legacy_unchecked``).

Tests patch module attributes looked up at call time (the claim, the applier locks, the appliers).
Crash tests run real child processes killed with SIGKILL: an in-process exception still runs cleanup.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

from tests.waiting import wait_until
from tools import write_approval as wa
from tools.write_approval import preview_pending, resolve_pending

_log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
WAIT = 30  # seconds: a deadline for a thread or child to finish, never a delay; a hang fails here
CHILD_TIMEOUT = 120


@pytest.fixture
def home(monkeypatch, tmp_path):
    h = tmp_path / ".hermes"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


# --- helpers: the API, and staging through the production entry points -------------------------
def _expect(subsystem, pending_id):
    return json.loads(json.dumps(preview_pending(subsystem, pending_id)["expect"]))


def _set_approval(subsystem, enabled):
    import hermes_cli.config as cfg
    c = cfg.load_config()
    c.setdefault(subsystem, {})["write_approval"] = enabled
    cfg.save_config(c)


def _load_store():
    from tools.memory_tool import load_on_disk_store
    return load_on_disk_store()


def _store_for(subsystem):
    return _load_store() if subsystem == wa.MEMORY else None


def _skill_md(name, body="# body"):
    return f"---\nname: {name}\ndescription: {name} skill\n---\n{body}\n"


def _create_skill(name, body="# body"):
    from tools.skill_manager_tool import skill_manage
    _set_approval("skills", False)  # create writes directly only when the gate is off
    out = json.loads(skill_manage(action="create", name=name, content=_skill_md(name, body)))
    assert out.get("success"), out


def _skill_dir(name):
    return wa._find_skill_path(name)


def _skill_md_path(name):
    return _skill_dir(name) / "SKILL.md"


def _stage_skill(**kwargs):
    """Stage through skill_manage with skills.write_approval on, as the agent does."""
    from tools.skill_manager_tool import skill_manage
    _set_approval("skills", True)
    out = json.loads(skill_manage(**kwargs))
    assert out.get("staged") is True, out
    return out["pending_id"]


def _stage_skill_patch():
    return _stage_skill(action="patch", name="probe", old_string="# body", new_string="# body\nADDED-LINE")


def _stage_alpha_beta_batch():
    _create_skill("alpha", "# alpha body")
    _create_skill("beta", "# beta body")
    return _stage_skill(action="patch", name="alpha", operations=[
        {"action": "patch", "name": "alpha", "old_string": "# alpha body", "new_string": "# alpha body v2"},
        {"action": "patch", "name": "beta", "old_string": "# beta body", "new_string": "# beta body v2"}])


def _stage_memory(**kwargs):
    """Stage through memory_tool with memory.write_approval on, as the agent does."""
    from tools.memory_tool import memory_tool
    _set_approval("memory", True)
    out = json.loads(memory_tool(store=_load_store(), **kwargs))
    assert out.get("staged") is True, out
    return out["pending_id"]


def _stage_memory_add(content):
    return _stage_memory(action="add", content=content)


def _seed_memory(*entries):
    """Entries that exist before anything is staged (a direct write, gate off)."""
    _set_approval("memory", False)
    store = _load_store()
    for entry in entries:
        assert store.add("memory", entry)["success"], entry


def _drift_memory(old, new):
    """An edit made after staging. The gate is off, so it is a direct write."""
    from tools.memory_tool import memory_tool
    _set_approval("memory", False)
    out = json.loads(memory_tool(action="patch", old_text=old, content=new, store=_load_store()))
    assert out.get("success"), out


def _stage_legacy_skill_patch():
    """A record as stage_write wrote it before base revisions: no base, no record binding."""
    _create_skill("probe")
    pid = uuid.uuid4().hex[:8]
    d = wa._pending_dir("skills")
    d.mkdir(parents=True, exist_ok=True)
    payload = {"action": "patch", "name": "probe", "old_string": "# body", "new_string": "# body\nADDED-LINE"}
    record = {"id": pid, "subsystem": "skills", "action": "patch", "summary": "patch 'probe'",
              "origin": "foreground", "created_at": time.time(), "payload": payload}
    (d / f"{pid}.json").write_text(json.dumps(record), encoding="utf-8")
    return pid


def _on_disk(subsystem, skill="probe"):
    """Bytes of everything an approval may write: MEMORY.md/USER.md, or one skill's files."""
    from tools import memory_tool, write_approval as wa
    if subsystem == wa.MEMORY:
        d = memory_tool.get_memory_dir()
        return {p.name: p.read_bytes() for p in sorted(d.glob("*.md"))} if d.exists() else {}
    skill_dir = _skill_dir(skill)
    if skill_dir is None:
        return {}
    return {str(p.relative_to(skill_dir)): p.read_bytes() for p in sorted(skill_dir.rglob("*")) if p.is_file()}


def _assert_refused(r, why=""):
    assert r["status"] == "conflict" and r.get("reason"), (why, r)


# --- helpers: threads and child processes, each bounded by WAIT / CHILD_TIMEOUT ---------------
class _Racer:
    """One resolver call on a named thread. join() returns its result or re-raises its error."""

    def __init__(self, name, call):
        self._call = call
        self.result = self.error = None
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.start = self._thread.start

    def _run(self):
        try:
            self.result = self._call()
        except Exception as exc:
            self.error = exc
            _log.exception("%s raised", self._thread.name)

    def wait_for(self, event, what):
        """Block until ``event`` is set. If this racer ends first, its own error is the failure."""
        wait_until(lambda: event.is_set() or not self._thread.is_alive(), timeout=WAIT, what=what)
        if not event.is_set():
            self.join()  # re-raises this racer's error, if it had one
            raise AssertionError(f"{self._thread.name} ended before {what}: {self.result!r}")

    def join(self):
        self._thread.join(WAIT)
        assert not self._thread.is_alive(), f"{self._thread.name} did not finish within {WAIT}s"
        if self.error is not None:
            raise self.error
        return self.result


def _lock_probe(real_lock):
    """Wrap an applier lock so the test learns when the resolver thread starts waiting on it."""
    waiting = threading.Event()

    def lock(path):
        if threading.current_thread().name == "resolver":
            waiting.set()
        return real_lock(path)
    return waiting, lock


def _child(script, spec, home):
    env = {**os.environ, "HERMES_HOME": str(home)}
    return subprocess.run([sys.executable, "-c", script, json.dumps(spec)], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, timeout=CHILD_TIMEOUT, check=False)


# Kill after a write lands: "applier" (the applier's write) or "first_op" (op 1 of a two-op batch).
_KILL_SCRIPT = r"""
import json, os, signal, sys
from tools import memory_tool, skill_manager_tool, write_approval as wa
spec = json.loads(sys.argv[1])

def sigkill_after_write(real):
    def applier(*args, **kwargs):
        out = real(*args, **kwargs)              # the write lands ...
        os.kill(os.getpid(), signal.SIGKILL)     # ... and the process dies with no cleanup
        return out
    return applier

if spec["kill"] == "first_op":
    real_op, done = skill_manager_tool._skill_manage_from, []

    def op_then_sigkill(payload, **kwargs):
        if payload.get("action") == "batch":     # the whole batch replays through here, then per op
            return real_op(payload, **kwargs)
        out = real_op(payload, **kwargs)         # one op of the batch: its write has landed
        done.append(payload.get("name"))
        if len(done) == 1:
            os.kill(os.getpid(), signal.SIGKILL)  # op 1 is durable; op 2 never starts
        return out

    skill_manager_tool._skill_manage_from = op_then_sigkill
elif spec["subsystem"] == wa.MEMORY:
    memory_tool.apply_memory_pending = sigkill_after_write(memory_tool.apply_memory_pending)
else:
    skill_manager_tool.apply_skill_pending = sigkill_after_write(skill_manager_tool.apply_skill_pending)
store = memory_tool.load_on_disk_store() if spec["subsystem"] == wa.MEMORY else None
wa.resolve_pending(spec["subsystem"], spec["pid"], "approve", expect=spec["expect"], memory_store=store)
print("RESULT {}")  # reached only if the kill never happened
"""

# A fresh process: its first call is a list or a resolve. Every applier call is logged to a file.
_FRESH_PROCESS = r"""
import json, sys
from tools import memory_tool, skill_manager_tool, write_approval as wa
spec = json.loads(sys.argv[1])
subsystem, pid = spec["subsystem"], spec["pid"]

def log_applies(module, attr):
    real = getattr(module, attr)
    def applier(*args, **kwargs):
        with open(spec["calls_file"], "a", encoding="utf-8") as f:
            f.write("applied\n")
        return real(*args, **kwargs)
    setattr(module, attr, applier)

for module, attr in ((memory_tool, "apply_memory_pending"), (skill_manager_tool, "apply_skill_pending"),
                     (skill_manager_tool, "_skill_manage_from")):
    log_applies(module, attr)

def resolve():
    store = memory_tool.load_on_disk_store() if subsystem == wa.MEMORY else None
    return wa.resolve_pending(subsystem, pid, "approve", expect=spec["expect"], memory_store=store)

def listed():
    return [r.get("status") for r in wa.list_pending(subsystem) if r["id"] == pid]

if spec["order"] == "list_first":
    seen = listed(); result = resolve()
else:
    result = resolve(); seen = listed()
print("RESULT " + json.dumps({"resolved": result, "listed": seen}, default=str))
"""


def _kill_mid_apply(home, kill, subsystem, pid, expect):
    """SIGKILL a child mid-apply (see _KILL_SCRIPT). A new reader must not need to wait for its claim."""
    proc = _child(_KILL_SCRIPT, {"kill": kill, "subsystem": subsystem, "pid": pid, "expect": expect}, home)
    assert proc.returncode == -signal.SIGKILL, (proc.returncode, proc.stdout, proc.stderr)
    assert (wa._pending_dir(subsystem) / f"{pid}.claimed").exists(), "the SIGKILLed approval left no claim"


def _fresh(home, subsystem, pid, expect, order, calls_file):
    """Run a fresh process that lists and resolves the record in the given order."""
    spec = {"subsystem": subsystem, "pid": pid, "expect": expect, "order": order, "calls_file": str(calls_file)}
    proc = _child(_FRESH_PROCESS, spec, home)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, (proc.returncode, proc.stdout, proc.stderr)
    return json.loads(lines[-1][len("RESULT "):])


def _assert_needs_review_never_requeued_or_reapplied(first, calls_file):
    assert first["resolved"]["status"] == "needs_review", first
    assert not [s for s in first["listed"] if s != "needs_review"], f"listed as pending again: {first}"
    assert not calls_file.exists(), "a fresh process ran the applier again"


# --- (a) approve applies once ----------------------------------------------------------------
@pytest.mark.parametrize("subsystem, expect_again", [("memory", False), ("skills", True)],
                         ids=["memory-add", "skill-patch"])
def test_a_repeat_approve_reports_the_prior_outcome_and_writes_nothing(home, subsystem, expect_again):
    if subsystem == wa.MEMORY:
        pid = _stage_memory_add("fact applied once")
    else:
        _create_skill("probe")
        pid = _stage_skill_patch()
    expect = _expect(subsystem, pid)
    first = resolve_pending(subsystem, pid, "approve", expect=expect, memory_store=_store_for(subsystem))
    assert first["status"] == "applied", first
    before = _on_disk(subsystem)

    # A repeat is already_resolved even when our own write has made the expect stale.
    again = resolve_pending(subsystem, pid, "approve", expect=expect if expect_again else None,
                            memory_store=_store_for(subsystem))
    assert again["status"] == "already_resolved" and again["prior"] == "applied", again
    assert _on_disk(subsystem) == before
    if subsystem == wa.MEMORY:
        assert _load_store().memory_entries == ["fact applied once"]
    else:
        assert before["SKILL.md"].count(b"ADDED-LINE") == 1


# --- (b) approve vs reject race on one id, forced through the claim --------------------------
@pytest.mark.parametrize("winner_decision", ["approve", "reject"])
def test_approve_and_reject_racing_on_one_id_apply_once_and_agree(home, monkeypatch, winner_decision):
    """Both decisions claim first, so a reject cannot discard a record an approve is applying. The
    winner holds the claim until the loser has contended for it; the loser reports the winner's
    durable outcome as its prior, and the effect happens exactly once."""
    from contextlib import contextmanager
    loser_decision = "reject" if winner_decision == "approve" else "approve"
    text = "race fact"
    pid = _stage_memory_add(text)
    expect = _expect(wa.MEMORY, pid)
    stores = {"winner": _load_store(), "loser": _load_store()}  # loaded before either racer starts
    real_claim, real_lock = wa._claim_pending, wa._try_claim_lock
    holding, contended, release = threading.Event(), threading.Event(), threading.Event()

    def claim(*args, **kwargs):
        got = real_claim(*args, **kwargs)
        if threading.current_thread().name == "winner" and got is not None:
            holding.set()
            release.wait(WAIT)  # keep the claim until the loser has contended for it
        return got

    @contextmanager
    def observe_lock(path):
        with real_lock(path) as owned:
            if not owned and threading.current_thread().name == "loser":
                contended.set()
            yield owned

    def racer(name, decision):
        return _Racer(name, lambda: resolve_pending(
            wa.MEMORY, pid, decision, expect=expect, memory_store=stores[name]))
    monkeypatch.setattr(wa, "_claim_pending", claim)
    monkeypatch.setattr(wa, "_try_claim_lock", observe_lock)
    winner, loser = racer("winner", winner_decision), racer("loser", loser_decision)
    winner.start()
    try:
        winner.wait_for(holding, "the winner's claim")
        loser.start()
        loser.wait_for(contended, "the loser's contention for the held claim")
    finally:
        release.set()
    won, lost = winner.join(), loser.join()

    assert won["status"] == {"approve": "applied", "reject": "rejected"}[winner_decision], won
    assert lost["status"] == "already_resolved" and lost.get("prior") == won["status"], (won, lost)
    assert _load_store().memory_entries.count(text) == (1 if winner_decision == "approve" else 0)
    assert wa.get_pending(wa.MEMORY, pid) is None
    stored = resolve_pending(wa.MEMORY, pid, loser_decision, memory_store=_load_store())
    assert stored["status"] == "already_resolved" and stored.get("prior") == won["status"], (won, lost, stored)


def test_a_slow_live_decision_is_not_recovered_or_closed(home, monkeypatch):
    from tools import memory_tool, write_approval as wa
    pid = _stage_memory_add("slow decision")
    expect = _expect(wa.MEMORY, pid)
    store = _load_store()
    entered, resume = threading.Event(), threading.Event()
    apply = memory_tool.apply_memory_pending

    def hold_apply(*args, **kwargs):
        entered.set()
        assert resume.wait(WAIT)
        return apply(*args, **kwargs)
    monkeypatch.setattr(memory_tool, "apply_memory_pending", hold_apply)
    decision = _Racer("slow decision", lambda: wa.resolve_pending(
        wa.MEMORY, pid, "approve", expect=expect, memory_store=store))
    decision.start()
    try:
        decision.wait_for(entered, "the live applier")
        claimed = wa._pending_path(wa.MEMORY, pid).with_suffix(".claimed")
        old = time.time() - 3600
        os.utime(claimed, (old, old))
        assert not wa.list_needs_review(wa.MEMORY)
        monkeypatch.setattr(wa, "_CLAIM_WAIT_SECONDS", 0.2)
        assert wa.resolve_pending(wa.MEMORY, pid, "reject")["status"] == "conflict"
        assert wa._read_outcome(wa.MEMORY, pid)["status"] == "applying"
    finally:
        resume.set()
        result = decision.join()
    assert result["status"] == "applied"
    assert _load_store().memory_entries == ["slow decision"]


# --- (c) target drift after the preview: refused as a whole, nothing partly written -----------
@pytest.mark.parametrize("batch", [False, True], ids=["single", "batch"])
def test_a_skill_drifted_after_its_preview_is_refused_and_writes_nothing(home, batch):
    if batch:
        pid, names = _stage_alpha_beta_batch(), ["alpha", "beta"]
    else:
        _create_skill("probe")
        pid, names = _stage_skill_patch(), ["probe"]
    expect = _expect(wa.SKILLS, pid)

    # The last target changes after the preview. Its anchor still matches, so only the base refuses.
    last = _skill_md_path(names[-1])
    drifted = last.read_text(encoding="utf-8") + "EDITED BY HAND\n"
    last.write_text(drifted, encoding="utf-8")
    before = {n: _on_disk(wa.SKILLS, n) for n in names}
    _assert_refused(resolve_pending(wa.SKILLS, pid, "approve", expect=expect))
    assert {n: _on_disk(wa.SKILLS, n) for n in names} == before, "an earlier op was partly written"
    assert last.read_text(encoding="utf-8") == drifted
    assert wa.get_pending(wa.SKILLS, pid) is not None


@pytest.mark.parametrize("batch", [False, True], ids=["single", "batch"])
def test_a_memory_entry_drifted_after_its_preview_is_refused_and_writes_nothing(home, batch):
    first = "Repo lives in ~/src/app; tests via make test"
    _seed_memory(first, "Staging DB: pg-staging-2 (old cluster, retiring)")
    if batch:
        pid = _stage_memory(operations=[
            {"action": "patch", "old_text": "~/src/app", "content": "~/src/app-v2"},
            {"action": "patch", "old_text": "pg-staging-2", "content": "pg-staging-3"}])
    else:
        pid = _stage_memory(action="patch", old_text="pg-staging-2", content="pg-staging-3")
    expect = _expect(wa.MEMORY, pid)

    _drift_memory("pg-staging-2", "pg-staging-9")  # the pinned entry changes after the preview
    drifted = [first, "Staging DB: pg-staging-9 (old cluster, retiring)"]
    assert _load_store().memory_entries == drifted
    before = _on_disk(wa.MEMORY)
    _assert_refused(resolve_pending(wa.MEMORY, pid, "approve", expect=expect, memory_store=_load_store()))
    assert _on_disk(wa.MEMORY) == before
    assert _load_store().memory_entries == drifted
    assert wa.get_pending(wa.MEMORY, pid) is not None


@pytest.mark.parametrize("restore_fails", [False, True], ids=["clean-rollback", "failed-rollback"])
def test_skill_batch_refusal_is_retryable_only_after_clean_rollback(home, monkeypatch, restore_fails):
    from types import SimpleNamespace
    from tools import skill_manager_batch as batch, write_approval as wa
    _create_skill("probe")
    path = _skill_md_path("probe")
    original = path.read_bytes()
    pid = _stage_skill(action="patch", name="probe", operations=[
        {"action": "patch", "name": "probe", "old_string": "# body", "new_string": "# changed"},
        {"action": "patch", "name": "probe", "old_string": "no such line", "new_string": "unused"},
    ])
    snapshots = home.parent / "batch-snapshots"
    snapshots.mkdir()
    monkeypatch.setattr(batch, "tempfile", SimpleNamespace(mkdtemp=lambda **kwargs: str(snapshots)))
    copytree = batch.shutil.copytree

    def copy_with_restore_fault(src, dst, *args, **kwargs):
        if restore_fails and Path(src) == snapshots / "probe":
            raise OSError("injected snapshot restore failure")
        return copytree(src, dst, *args, **kwargs)
    monkeypatch.setattr(batch.shutil, "copytree", copy_with_restore_fault)
    reply = wa.resolve_pending(wa.SKILLS, pid, "approve")
    if not restore_fails:
        assert reply["status"] == "conflict", reply
        assert path.read_bytes() == original
        assert [item["id"] for item in wa.list_pending(wa.SKILLS)] == [pid]
        assert not wa.list_needs_review(wa.SKILLS)
        assert not snapshots.exists()
        return

    assert reply["status"] == "needs_review", reply
    landed = path.read_bytes()
    assert b"# changed" in landed
    assert (snapshots / "probe" / "SKILL.md").read_bytes() == original
    assert not wa.list_pending(wa.SKILLS)
    [review] = wa.list_needs_review(wa.SKILLS)
    assert review["id"] == pid and "rollback" in review["reason"].lower()
    assert wa.resolve_pending(wa.SKILLS, pid, "approve")["status"] == "needs_review"
    assert path.read_bytes() == landed


def test_failed_cleanup_of_batch_created_skill_requires_review(home, monkeypatch):
    # A batch can adopt an empty directory: failing to remove its new SKILL.md is a failed rollback.
    adopted = home / "skills" / "probe"
    adopted.mkdir(parents=True)
    pid = _stage_skill(action="create", name="probe", operations=[
        {"action": "create", "name": "probe", "content": _skill_md("probe")},
        {"action": "patch", "name": "probe", "old_string": "no such line", "new_string": "unused"},
    ])
    path = adopted / "SKILL.md"
    unlink = Path.unlink

    def unlink_with_cleanup_fault(target, *args, **kwargs):
        if target == path:
            raise PermissionError("injected rollback cleanup failure")
        return unlink(target, *args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(Path, "unlink", unlink_with_cleanup_fault)
        reply = wa.resolve_pending(wa.SKILLS, pid, "approve")
    assert reply["status"] == "needs_review", reply
    assert path.read_text(encoding="utf-8") == _skill_md("probe")
    assert not wa.list_pending(wa.SKILLS)
    assert [item["id"] for item in wa.list_needs_review(wa.SKILLS)] == [pid]


@pytest.mark.platforms("posix")  # the applier locks are flock-based; the probe needs them to block
@pytest.mark.parametrize("subsystem", ["memory", "skills"])
def test_drift_landing_while_the_resolver_waits_for_the_applier_lock_is_refused(home, monkeypatch, subsystem):
    from tools import skill_manager_tool, skill_usage
    from tools.memory_tool import ENTRY_DELIMITER, MemoryStore
    if subsystem == wa.MEMORY:
        _seed_memory("Staging DB: pg-staging-2 (old cluster)")
        pid = _stage_memory(action="patch", old_text="pg-staging-2", content="pg-staging-3")
        expect = _expect(wa.MEMORY, pid)
        path = MemoryStore._path_for("memory")
        waiting, probe = _lock_probe(MemoryStore._file_lock)
        monkeypatch.setattr(MemoryStore, "_file_lock", staticmethod(probe))
        store = _load_store()
        drifted_entries = ["Staging DB: pg-staging-9 (old cluster)"]
        resolver = _Racer("resolver", lambda: resolve_pending(
            wa.MEMORY, pid, "approve", expect=expect, memory_store=store))
        with MemoryStore._file_lock(path):  # a concurrent writer holds the applier lock
            resolver.start()
            resolver.wait_for(waiting, "the resolver's arrival at the applier lock")
            path.write_text(ENTRY_DELIMITER.join(drifted_entries), encoding="utf-8")  # the writer's edit
        _assert_refused(resolver.join(), "memory")
        assert _load_store().memory_entries == drifted_entries, "the resolver wrote over the drift"
    else:
        _create_skill("probe")
        pid = _stage_skill_patch()
        expect = _expect(wa.SKILLS, pid)
        drifted = _skill_md("probe") + "EDITED BY HAND\n"
        waiting, probe = _lock_probe(skill_usage.skill_file_lock)
        monkeypatch.setattr(skill_usage, "skill_file_lock", probe)
        resolver = _Racer("resolver", lambda: resolve_pending(wa.SKILLS, pid, "approve", expect=expect))
        with skill_manager_tool._skill_mutation_lock("probe"):  # a concurrent writer holds the applier lock
            resolver.start()
            resolver.wait_for(waiting, "the resolver's arrival at the applier lock")
            _skill_md_path("probe").write_text(drifted, encoding="utf-8")  # the writer's edit
        _assert_refused(resolver.join(), "skills")
        assert _skill_md_path("probe").read_text(encoding="utf-8") == drifted, "the resolver wrote over the drift"
    assert wa.get_pending(subsystem, pid) is not None


# --- (d) staged-base corner cases: what counts as a base, per op -----------------------------
@pytest.mark.parametrize("batch", [False, True], ids=["single", "batch"])
def test_skill_edit_during_preview_cannot_authorize_unseen_bytes(home, monkeypatch, batch):
    _create_skill("probe", "# body\nreviewed context")
    path = _skill_md_path("probe")
    if batch:
        # The token binds raw bytes; the renderer sees decoded text and folds earlier ops.
        path.write_bytes(path.read_text(encoding="utf-8").replace("\n", "\r\n").encode("utf-8-sig"))
        pid = _stage_skill(action="patch", name="probe", operations=[
            {"action": "patch", "name": "probe", "old_string": "# body", "new_string": "# first edit"},
            {"action": "patch", "name": "probe", "old_string": "# first edit", "new_string": "# second edit"}])
    else:
        pid = _stage_skill_patch()
    original = path.read_bytes()
    edited = original.replace(b"reviewed context", b"other context")
    render = wa.difflib.unified_diff

    def edit_during_render(*args, **kwargs):
        path.write_bytes(edited)  # an ordinary editor need not take Hermes' mutation lock
        return render(*args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(wa.difflib, "unified_diff", edit_during_render)
        shown = wa.preview_pending(wa.SKILLS, pid)
    assert shown["status"] == "ready", shown
    for op in shown["ops"]:
        assert "reviewed context" in op["diff"] and "other context" not in op["diff"]
    if batch:
        assert "-# first edit" in shown["ops"][1]["diff"] and "+# second edit" in shown["ops"][1]["diff"]
    assert shown["expect"]["targets"]["probe"]["SKILL.md"] == "sha256:" + hashlib.sha256(original).hexdigest()
    assert not shown["changed_since_staged"]
    decision = wa.resolve_pending(wa.SKILLS, pid, "approve", expect=shown["expect"])
    assert decision["status"] == "conflict"
    assert path.read_bytes() == edited and wa.get_pending(wa.SKILLS, pid) is not None


def test_a_preview_taken_after_drift_names_it_and_approves_against_the_current_file(home):
    _create_skill("probe")
    pid = _stage_skill_patch()
    _skill_md_path("probe").write_text(_skill_md("probe") + "EDITED BY HAND\n", encoding="utf-8")
    shown = preview_pending(wa.SKILLS, pid)  # the patch still applies: ready, with the drift named
    assert shown["status"] == "ready" and shown["changed_since_staged"] == ["probe/SKILL.md"], shown
    assert shown["notice"] and "current file" in shown["notice"], shown
    assert "EDITED BY HAND" in shown["ops"][0]["diff"], "the diff is not against the current file"
    r = resolve_pending(wa.SKILLS, pid, "approve", expect=json.loads(json.dumps(shown["expect"])))
    assert r["status"] == "applied", r
    content = _skill_md_path("probe").read_text(encoding="utf-8")
    assert "ADDED-LINE" in content and "EDITED BY HAND" in content


def test_memory_preview_checks_each_pin_after_preceding_batch_operations(home):
    pid = _stage_memory(operations=[
        {"action": "add", "content": "temporary sample entry"},
        {"action": "patch", "old_text": "temporary", "content": "final"},
    ])
    shown = preview_pending(wa.MEMORY, pid)
    assert shown["status"] == "ready", shown
    assert shown["ops"][1]["matched_entry"] == "temporary sample entry"
    assert _load_store().memory_entries == []
    result = resolve_pending(wa.MEMORY, pid, "approve", expect=shown["expect"], memory_store=_load_store())
    assert result["status"] == "applied", result
    assert _load_store().memory_entries == ["final sample entry"]


def test_a_memory_add_is_not_refused_by_drift_in_other_entries(home):
    pid = _stage_memory_add("new fact")
    expect = _expect(wa.MEMORY, pid)
    _seed_memory("unrelated existing entry")  # an add has no target: unrelated drift must not block it
    r = resolve_pending(wa.MEMORY, pid, "approve", expect=expect, memory_store=_load_store())
    assert r["status"] == "applied", r
    assert _load_store().memory_entries == ["unrelated existing entry", "new fact"]


@pytest.mark.parametrize("case", ["create", "write_file", "delete"])
def test_a_staged_op_is_refused_when_its_target_appeared_since_staging(home, case):
    if case == "create":
        name = "crashy"
        pid = _stage_skill(action="create", name=name, content=_skill_md(name))  # the base is absent
    else:
        name = "probe"
        _create_skill(name)
        if case == "write_file":
            pid = _stage_skill(action="write_file", name=name, file_path="references/extra.md",
                               file_content="staged\n")
        else:
            pid = _stage_skill(action="delete", name=name)
    expect = _expect(wa.SKILLS, pid)

    if case == "create":
        _create_skill(name, body="# hand made")  # the base was absent, and now it is not
        shown = preview_pending(wa.SKILLS, pid)
        assert shown["status"] == "conflict" and "already exists" in shown["reason"], shown
    elif case == "write_file":
        extra = _skill_dir(name) / "references" / "extra.md"  # the path the op creates is now taken
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_text("written by hand\n", encoding="utf-8")
    else:
        late = _skill_dir(name) / "references" / "late.md"  # a file the delete would remove
        late.parent.mkdir(parents=True, exist_ok=True)
        late.write_text("added after staging\n", encoding="utf-8")
    before = _on_disk(wa.SKILLS, name)

    _assert_refused(resolve_pending(wa.SKILLS, pid, "approve", expect=expect))
    assert _on_disk(wa.SKILLS, name) == before, "a staged op wrote over a change made since staging"
    assert wa.get_pending(wa.SKILLS, pid) is not None


def test_a_batch_that_creates_then_patches_a_skill_applies_when_nothing_else_changed(home):
    pid = _stage_skill(action="patch", name="gamma", operations=[
        {"action": "create", "name": "gamma", "content": _skill_md("gamma")},
        {"action": "patch", "name": "gamma", "old_string": "# body", "new_string": "# body\nADDED-LINE"}])
    # The patch's own base is the file the create made, not disk. Nothing else changed, so this applies.
    r = resolve_pending(wa.SKILLS, pid, "approve", expect=_expect(wa.SKILLS, pid))
    assert r["status"] == "applied", r
    assert b"ADDED-LINE" in _on_disk(wa.SKILLS, "gamma")["SKILL.md"]


@pytest.mark.parametrize("action,batch", [("delete", False), ("remove_file", False), ("remove_file", True)])
def test_fresh_preview_does_not_override_destructive_staged_base(home, action, batch):
    _create_skill("probe")
    ref = "references/kept.md"
    target = _skill_dir("probe") / ref
    target.parent.mkdir(exist_ok=True)
    target.write_text("staged base\n", encoding="utf-8")
    removal = {"action": action, "name": "probe"}
    if action == "remove_file":
        removal["file_path"] = ref
    if batch:
        _create_skill("other")
        pid = _stage_skill(action="patch", name="probe", operations=[
            {"action": "patch", "name": "other", "old_string": "# body", "new_string": "# new"}, removal])
    else:
        pid = _stage_skill(**removal)
    target.write_text("newer notes\n", encoding="utf-8")
    other_before = _on_disk(wa.SKILLS, "other") if batch else None

    shown = preview_pending(wa.SKILLS, pid)
    assert shown["status"] == "stale", shown
    assert shown["changed_since_staged"] == ([f"probe/{ref}"] if action == "remove_file" else ["probe"])
    for expect in (None, shown["expect"]):
        _assert_refused(resolve_pending(wa.SKILLS, pid, "approve", expect=expect))
    assert target.read_text(encoding="utf-8") == "newer notes\n"
    if batch:
        assert _on_disk(wa.SKILLS, "other") == other_before


def test_a_whole_skill_delete_preview_lists_empty_directories_and_detects_new_ones(home):
    _create_skill("probe")
    (_skill_dir("probe") / "empty").mkdir()
    pid = _stage_skill(action="delete", name="probe")
    shown = preview_pending(wa.SKILLS, pid)
    assert shown["status"] == "ready", shown
    assert "empty/" in shown["ops"][0]["diff"] and "SKILL.md" in shown["ops"][0]["diff"]
    later = _skill_dir("probe") / "later"
    later.mkdir()
    _assert_refused(resolve_pending(wa.SKILLS, pid, "approve", expect=shown["expect"]))
    assert later.is_dir()


@pytest.mark.platforms("posix")
def test_staged_skill_changes_work_through_a_symlinked_profile_home(home, tmp_path, monkeypatch):
    _create_skill("probe")
    alias = tmp_path / "home-alias"
    alias.symlink_to(home, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(alias))
    for action, kwargs in [("patch", {"old_string": "# body", "new_string": "# updated"}), ("delete", {})]:
        pid = _stage_skill(action=action, name="probe", **kwargs)
        shown = preview_pending(wa.SKILLS, pid)
        assert shown["status"] == "ready", shown
        result = resolve_pending(wa.SKILLS, pid, "approve", expect=shown["expect"])
        assert result["status"] == "applied", result


@pytest.mark.platforms("posix")
def test_whole_skill_preview_rejects_a_special_file_without_opening_it(home, monkeypatch):
    _create_skill("probe")
    skill_dir = _skill_dir("probe")
    assert skill_dir is not None
    pipe = skill_dir / "extra.pipe"
    os.mkfifo(pipe)
    open_file = os.open

    def require_regular(path, *args, **kwargs):
        assert path != pipe, "a reviewer attempted to open a special file"
        return open_file(path, *args, **kwargs)
    monkeypatch.setattr(os, "open", require_regular)
    pid = _stage_skill(action="delete", name="probe")
    shown = preview_pending(wa.SKILLS, pid)
    assert shown["status"] == "conflict", shown
    _assert_refused(resolve_pending(wa.SKILLS, pid, "approve", expect=shown["expect"]))
    assert pipe.exists()


def test_skill_preview_orders_patch_chain_across_equivalent_path_spellings(home):
    _create_skill("probe")
    root = _skill_dir("probe")
    assert root is not None
    file = root / "references" / "notes.md"
    file.parent.mkdir()
    file.write_text("first draft\n", encoding="utf-8")
    pid = _stage_skill(action="patch", name="probe", operations=[
        {"action": "patch", "file_path": "references/notes.md", "old_string": "first", "new_string": "second"},
        {"action": "patch", "file_path": "references/./notes.md", "old_string": "second", "new_string": "final"},
    ])
    result = preview_pending(wa.SKILLS, pid)
    assert result["status"] == "ready", result
    assert "-second draft" in result["ops"][1]["diff"] and "+final draft" in result["ops"][1]["diff"]
    assert resolve_pending(wa.SKILLS, pid, "approve", expect=result["expect"])["status"] == "applied"
    assert file.read_text(encoding="utf-8") == "final draft\n"


@pytest.mark.parametrize("ops", [
    [{"action": "patch", "name": "foo", "old_string": "# body", "new_string": "# body\nfirst edit"},
     {"action": "patch", "name": "cat/foo", "old_string": "first edit", "new_string": "second edit"}],
    [{"action": "create", "name": "bar", "category": "cat",
      "content": "---\nname: bar\ndescription: A new skill.\n---\n# bar\n"},
     {"action": "write_file", "name": "cat/bar", "file_path": "references/notes.md", "file_content": "Notes for bar.\n"}],
], ids=["existing-skill", "skill-created-in-the-batch"])
def test_a_batch_naming_one_skill_two_ways_is_refused(home, ops):
    """Each spelling would be checked and previewed on its own while replay edits one folder."""
    skill = home / "skills" / "cat" / "foo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(_skill_md("foo"), encoding="utf-8")

    def category_files():
        return {p.relative_to(home): p.read_bytes() for p in skill.parent.rglob("*") if p.is_file()}

    before = category_files()
    pid = wa.stage_write(wa.SKILLS, {"action": "batch", "operations": ops}, summary="aliased", origin="foreground")["id"]

    shown = preview_pending(wa.SKILLS, pid)
    assert shown["status"] == "conflict" and "consistent spelling" in shown["reason"], shown
    for expect in (shown["expect"], None):
        _assert_refused(resolve_pending(wa.SKILLS, pid, "approve", expect=expect))
    assert category_files() == before


@pytest.mark.parametrize("empty_file", [False, True])
def test_preview_does_not_offer_approval_for_a_patch_with_no_match(home, empty_file):
    _create_skill("probe")
    root = _skill_dir("probe")
    assert root is not None
    target = root / "references" / "notes.md"
    if empty_file:
        target.parent.mkdir()
        target.write_text("", encoding="utf-8")
    pid = _stage_skill(action="patch", name="probe", file_path="references/notes.md",
                       old_string="anchor", new_string="new")
    shown = preview_pending(wa.SKILLS, pid)
    assert shown["status"] == "conflict", shown
    _assert_refused(resolve_pending(wa.SKILLS, pid, "approve", expect=shown["expect"]))
    assert target.exists() is empty_file


# --- (e) record bytes and profile binding of expect ------------------------------------------
def test_staging_skips_ids_in_pending_claimed_and_resolved_records(home, monkeypatch):
    from types import SimpleNamespace
    active, interrupted, done = (_stage_memory_add(t) for t in ["pending", "claimed", "resolved"])
    pending = wa._pending_path(wa.MEMORY, active)
    claimed = wa._pending_path(wa.MEMORY, interrupted)
    claimed = claimed.replace(claimed.with_suffix(".claimed"))
    assert wa.resolve_pending(wa.MEMORY, done, "reject")["status"] == "rejected"
    before = {p: p.read_bytes() for p in [pending, claimed, wa._outcome_path(wa.MEMORY, done)]}
    fresh = "feedface"
    assert not wa._pending_path(wa.MEMORY, fresh).exists()
    numbers = iter([active, interrupted, done, fresh])
    monkeypatch.setattr(wa, "uuid", SimpleNamespace(uuid4=lambda: SimpleNamespace(hex=next(numbers))))

    assert _stage_memory_add("different proposal") == fresh
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_changed_record_bytes_refuse_approve_and_reject_with_a_reason(home, decision):
    pid = _stage_memory_add("record fact")
    expect = _expect(wa.MEMORY, pid)
    record = wa._pending_dir(wa.MEMORY) / f"{pid}.json"
    record.write_bytes(record.read_bytes() + b"\n")  # the same JSON value, one byte different
    record_bytes, before = record.read_bytes(), _on_disk(wa.MEMORY)

    r = resolve_pending(wa.MEMORY, pid, decision, expect=expect, memory_store=_load_store())
    _assert_refused(r, decision)
    assert _on_disk(wa.MEMORY) == before, "a refused decision wrote something"
    assert record.read_bytes() == record_bytes, "the record was touched by a refused decision"
    assert wa.get_pending(wa.MEMORY, pid) is not None, "a refused decision still discarded the record"


def test_an_expect_from_one_profile_is_refused_by_another_holding_the_same_id(home, monkeypatch, tmp_path):
    pid = _stage_memory_add("profile fact")
    expect = _expect(wa.MEMORY, pid)
    record_bytes = (wa._pending_dir(wa.MEMORY) / f"{pid}.json").read_bytes()
    # Profile B holds a record with the same id and the same bytes: only the profile differs.
    home_b = tmp_path / "profile-b" / ".hermes"
    (home_b / "pending" / "memory").mkdir(parents=True)
    (home_b / "pending" / "memory" / f"{pid}.json").write_bytes(record_bytes)
    monkeypatch.setenv("HERMES_HOME", str(home_b))
    assert wa.get_pending(wa.MEMORY, pid) is not None and _load_store().memory_entries == []
    r = resolve_pending(wa.MEMORY, pid, "approve", expect=expect, memory_store=_load_store())
    _assert_refused(r, "profile A's expect against profile B")
    assert _load_store().memory_entries == [], "profile B's memory was written from profile A's expect"
    assert wa.get_pending(wa.MEMORY, pid) is not None
    monkeypatch.setenv("HERMES_HOME", str(home))  # deciding in A never touches B
    applied = resolve_pending(wa.MEMORY, pid, "approve", expect=expect, memory_store=_load_store())
    assert applied["status"] == "applied", applied
    assert _load_store().memory_entries == ["profile fact"]
    monkeypatch.setenv("HERMES_HOME", str(home_b))
    assert _load_store().memory_entries == []
    assert wa.get_pending(wa.MEMORY, pid) is not None, "deciding in profile A discarded profile B's record"


@pytest.mark.parametrize("decision, final_status", [("approve", "applied"), ("reject", "rejected")])
def test_the_outcome_hashes_the_exact_record_bytes_including_a_bom(home, decision, final_status):
    pid = _stage_memory_add("bom record")
    pending = wa._pending_path(wa.MEMORY, pid)
    raw = b"\xef\xbb\xbf" + pending.read_bytes()
    pending.write_bytes(raw)
    expect = _expect(wa.MEMORY, pid)
    result = resolve_pending(wa.MEMORY, pid, decision, expect=expect, memory_store=_load_store())
    assert result["status"] == final_status, result
    assert wa._read_outcome(wa.MEMORY, pid)["record_sha256"] == hashlib.sha256(raw).hexdigest()
    assert _load_store().memory_entries == (["bom record"] if decision == "approve" else [])


# --- (f) crash after the write: needs_review on first access from a fresh process ------------
@pytest.mark.platforms("posix")
@pytest.mark.parametrize("order", ["list_first", "resolve_first"])
@pytest.mark.parametrize("kind", ["memory_add", "skill_create"])
def test_sigkill_after_the_write_leaves_needs_review_on_first_access_from_a_fresh_process(home, kind, order):
    if kind == "memory_add":
        subsystem, pid = wa.MEMORY, _stage_memory_add("crash fact")
    else:
        subsystem, pid = wa.SKILLS, _stage_skill(action="create", name="crashy", content=_skill_md("crashy"))

    def landed():
        if subsystem == wa.MEMORY:
            return _load_store().memory_entries.count("crash fact") == 1
        return _on_disk(wa.SKILLS, "crashy").get("SKILL.md", b"").count(b"name: crashy") == 1
    expect = _expect(subsystem, pid)
    _kill_mid_apply(home, "applier", subsystem, pid, expect)
    assert landed(), "the write did not land before the kill"
    calls = home / "applier-calls.log"
    _assert_needs_review_never_requeued_or_reapplied(_fresh(home, subsystem, pid, expect, order, calls), calls)
    again = _fresh(home, subsystem, pid, expect, "resolve_first", calls)  # a later process: still durable
    assert again["resolved"]["status"] == "needs_review", again
    assert landed() and not calls.exists(), "a later process re-ran the applier"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("order", ["list_first", "resolve_first"])
def test_sigkill_after_the_first_op_of_a_batch_is_durable_needs_review_with_no_replay(home, order):
    pid = _stage_alpha_beta_batch()
    expect = _expect(wa.SKILLS, pid)
    _kill_mid_apply(home, "first_op", wa.SKILLS, pid, expect)
    assert _on_disk(wa.SKILLS, "alpha")["SKILL.md"].count(b"# alpha body v2") == 1, "op 1's write did not land"
    assert b"beta body v2" not in _on_disk(wa.SKILLS, "beta")["SKILL.md"]
    calls = home / "applier-calls.log"
    _assert_needs_review_never_requeued_or_reapplied(_fresh(home, wa.SKILLS, pid, expect, order, calls), calls)
    again = _fresh(home, wa.SKILLS, pid, expect, "resolve_first", calls)
    assert again["resolved"]["status"] == "needs_review", again
    alpha = _on_disk(wa.SKILLS, "alpha")["SKILL.md"]
    assert alpha.count(b"# alpha body v2") == 1 and b"v2 v2" not in alpha, "op 1 was replayed"
    assert b"beta body v2" not in _on_disk(wa.SKILLS, "beta")["SKILL.md"], "op 2 ran after the kill"


# --- (g) a refused write stays pending with a reason, or needs review if its mark sticks -----
@pytest.mark.parametrize("mark_cannot_clear", [False, True], ids=["store-refuses", "mark-cannot-clear"])
def test_a_refused_write_stays_pending_unless_its_mark_cannot_be_cleared(home, monkeypatch, mark_cannot_clear):
    from tools.memory_tool import MemoryStore
    pid = _stage_memory_add("x" * 120 + " too long for the limit")
    expect = _expect(wa.MEMORY, pid)
    small = MemoryStore(memory_char_limit=60)  # the same write fails the char limit at approve time
    small.load_from_disk()
    before = _on_disk(wa.MEMORY)
    outcome = wa._outcome_path(wa.MEMORY, pid)
    real_unlink = Path.unlink

    def unlink(self, *args, **kwargs):
        if self == outcome:
            raise PermissionError("cannot clear the mark")
        return real_unlink(self, *args, **kwargs)
    if mark_cannot_clear:
        monkeypatch.setattr(Path, "unlink", unlink)
        r = resolve_pending(wa.MEMORY, pid, "approve", memory_store=small)
        monkeypatch.setattr(Path, "unlink", real_unlink)
        assert r["status"] == "needs_review" and r.get("reason"), r
        assert [item["id"] for item in wa.list_needs_review(wa.MEMORY)] == [pid]
    else:
        _assert_refused(resolve_pending(wa.MEMORY, pid, "approve", expect=expect, memory_store=small))
        assert wa.get_pending(wa.MEMORY, pid) is not None  # not discarded
    assert _on_disk(wa.MEMORY) == before


# --- (h) CLI: the same contract without an expect ---------------------------------------------
@pytest.mark.parametrize("case", ["memory_add", "skill_create", "skill_patch"])
def test_cli_approve_keeps_its_output_and_leaves_the_same_disk_result_as_resolve_pending(home, case):
    import shutil
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    subsystem = wa.MEMORY if case == "memory_add" else wa.SKILLS
    skill = "parity" if case == "skill_create" else "probe"

    def stage():
        if case == "memory_add":
            return _stage_memory_add("cli fact")
        if case == "skill_create":
            return _stage_skill(action="create", name="parity", content=_skill_md("parity"))
        _create_skill("probe")
        return _stage_skill_patch()

    pid = stage()
    out = handle_pending_subcommand(subsystem, ["approve", pid], memory_store=_store_for(subsystem))
    assert out == f"Approved 1 {subsystem} write(s)."
    assert wa.get_pending(subsystem, pid) is None
    cli_disk = _on_disk(subsystem, skill)
    assert cli_disk, "the CLI approve wrote nothing"
    if case == "memory_add":
        assert _load_store().memory_entries == ["cli fact"]
    elif case == "skill_patch":
        assert b"ADDED-LINE" in cli_disk["SKILL.md"]

    for sub in ("memories", "skills", "pending"):
        shutil.rmtree(home / sub, ignore_errors=True)
    pid = stage()
    r = resolve_pending(subsystem, pid, "approve", expect=_expect(subsystem, pid), memory_store=_store_for(subsystem))
    assert r["status"] == "applied", r
    assert _on_disk(subsystem, skill) == cli_disk
    assert wa.get_pending(subsystem, pid) is None


@pytest.mark.parametrize("surface", ["cli", "resolve_pending_without_expect"])
def test_tokenless_approve_refuses_a_drifted_skill_patch_via_its_staged_base(home, surface):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    _create_skill("probe")
    pid = _stage_skill_patch()  # the staged base is recorded here; no preview is ever taken
    drifted = _skill_md("probe") + "EDITED BY HAND\n"
    _skill_md_path("probe").write_text(drifted, encoding="utf-8")
    if surface == "cli":
        out = handle_pending_subcommand(wa.SKILLS, ["approve", pid])
        assert "Approved 1" not in out, out
    else:
        _assert_refused(resolve_pending(wa.SKILLS, pid, "approve"))  # no expect: only the staged base decides
    assert _skill_md_path("probe").read_text(encoding="utf-8") == drifted, \
        "approve applied a patch to a file that changed since it was staged"
    assert wa.get_pending(wa.SKILLS, pid) is not None


# --- (i) legacy records: no staged base -----------------------------------------------------
@pytest.mark.parametrize("surface, expect_taken, drifted, applies", [
    ("gui", False, False, False),  # no expect: a legacy record has no base to check
    ("gui", True, False, True),
    ("gui", True, True, False),    # drift after the preview
    ("cli", False, True, True),    # legacy_unchecked: the CLI has nothing to compare with
], ids=["gui-no-expect", "gui-expect", "gui-expect-after-drift", "cli-unchecked"])
def test_a_legacy_record_without_a_base_needs_an_expect_in_the_gui_and_applies_unchecked_in_the_cli(
        home, surface, expect_taken, drifted, applies):
    """A documented exception for records staged before base revisions, not a safety property."""
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    pid = _stage_legacy_skill_patch()
    expect = _expect(wa.SKILLS, pid) if expect_taken else None
    if drifted:
        _skill_md_path("probe").write_text(_skill_md("probe") + "EDITED BY HAND\n", encoding="utf-8")
    before = _on_disk(wa.SKILLS)

    if surface == "cli":
        out = handle_pending_subcommand(wa.SKILLS, ["approve", pid])
        assert "Approved 1" in out, out
        content = _skill_md_path("probe").read_bytes()
        assert b"ADDED-LINE" in content and b"EDITED BY HAND" in content
        return
    r = resolve_pending(wa.SKILLS, pid, "approve", expect=expect)
    if applies:
        assert r["status"] == "applied", r
        assert _on_disk(wa.SKILLS)["SKILL.md"].count(b"ADDED-LINE") == 1
    else:
        _assert_refused(r)
        assert _on_disk(wa.SKILLS) == before
        assert wa.get_pending(wa.SKILLS, pid) is not None


# --- (j) stacked skill proposals: approving A changes B's file --------------------------------
_STACKED_BODY = "# body\n\n## tail"


def _stage_stacked():
    """A and B both patch probe/SKILL.md, staged while it is unchanged; A is approved first."""
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    _create_skill("probe", _STACKED_BODY)
    a = _stage_skill(action="patch", name="probe", old_string="# body", new_string="# body\nA-LINE")
    b = _stage_skill(action="patch", name="probe", old_string="## tail", new_string="## tail\nB-LINE")
    assert handle_pending_subcommand(wa.SKILLS, ["approve", a]) == "Approved 1 skills write(s)."
    return b


@pytest.mark.parametrize("surface", ["fresh_preview", "edited_after_preview", "cli"])
def test_stacked_b_is_checked_against_the_file_a_left_behind(home, surface):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    b = _stage_stacked()
    if surface == "cli":
        before = _on_disk(wa.SKILLS)
        out = handle_pending_subcommand(wa.SKILLS, ["approve", b])
        assert "Approved 0" in out and "changed since it was staged: probe/SKILL.md" in out, out
        assert _on_disk(wa.SKILLS) == before
        assert wa.get_pending(wa.SKILLS, b) is not None
    elif surface == "fresh_preview":
        shown = preview_pending(wa.SKILLS, b)  # the reviewer sees A's result
        assert shown["status"] == "ready" and shown["changed_since_staged"] == ["probe/SKILL.md"], shown
        assert shown["notice"], shown
        assert "A-LINE" in shown["ops"][0]["diff"] and "+B-LINE" in shown["ops"][0]["diff"], shown["ops"]
        r = resolve_pending(wa.SKILLS, b, "approve", expect=_expect(wa.SKILLS, b))
        assert r["status"] == "applied", r
        content = _skill_md_path("probe").read_text(encoding="utf-8")
        assert content.count("A-LINE") == 1 and content.count("B-LINE") == 1, content
        assert wa.get_pending(wa.SKILLS, b) is None
    else:
        expect = _expect(wa.SKILLS, b)  # the reviewer saw A's result
        edited = _skill_md_path("probe").read_text(encoding="utf-8") + "EDITED AFTER THE PREVIEW\n"
        _skill_md_path("probe").write_text(edited, encoding="utf-8")
        r = resolve_pending(wa.SKILLS, b, "approve", expect=expect)
        _assert_refused(r)
        assert "previewed" in r["reason"], r
        assert _skill_md_path("probe").read_text(encoding="utf-8") == edited
        assert wa.get_pending(wa.SKILLS, b) is not None


def test_stacked_b_whose_anchor_a_removed_previews_stale_and_is_refused(home):
    _create_skill("probe", _STACKED_BODY)
    a = _stage_skill(action="patch", name="probe", old_string="## tail", new_string="## end")
    b = _stage_skill(action="patch", name="probe", old_string="## tail", new_string="## tail\nB-LINE")
    assert resolve_pending(wa.SKILLS, a, "approve")["status"] == "applied"
    before = _on_disk(wa.SKILLS)

    shown = preview_pending(wa.SKILLS, b)
    assert shown["status"] == "stale" and "probe/SKILL.md" in shown["reason"], shown
    assert shown["notice"] is None, shown
    r = resolve_pending(wa.SKILLS, b, "approve", expect=json.loads(json.dumps(shown["expect"])))
    _assert_refused(r)
    assert _on_disk(wa.SKILLS) == before
    assert wa.get_pending(wa.SKILLS, b) is not None


# --- (k) needs_review: listed, and closed only by a reject that names its id ---------------
def _sigkilled_mid_apply(home, subsystem):
    """A staged write whose approval was SIGKILLed after its write landed (now needs_review)."""
    if subsystem == wa.MEMORY:
        pid = _stage_memory_add("crash fact")
    else:
        pid = _stage_skill(action="create", name="crashy", content=_skill_md("crashy"))
    _kill_mid_apply(home, "applier", subsystem, pid, _expect(subsystem, pid))
    return pid


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("subsystem", ["memory", "skills"])
def test_a_sigkilled_approval_needs_review_until_an_explicit_reject_closes_it(home, subsystem):
    pid = _sigkilled_mid_apply(home, subsystem)
    landed = _on_disk(subsystem, "crashy")
    [item] = wa.list_needs_review(subsystem)  # the first access settles the stale claim
    assert set(item) == {"id", "summary", "action", "origin", "staged_at", "decided_at", "reason"}, item
    assert item["id"] == pid and item["summary"], item
    # Neither memory text nor the proposed skill description belongs in queue metadata.
    assert ("crash fact" if subsystem == wa.MEMORY else "crashy skill") not in json.dumps(item)
    assert item["action"] == ("add" if subsystem == "memory" else "create"), item
    assert item["origin"] == "foreground" and item["staged_at"] and item["decided_at"] and item["reason"], item
    assert wa.list_pending(subsystem) == [] and wa.pending_count(subsystem) == 0

    store = _store_for(subsystem)
    assert resolve_pending(subsystem, pid, "approve", memory_store=store)["status"] == "needs_review"
    # A write that needs review has no preview, so any expect is stale; closing it takes a plain reject.
    stale_expect = {"profile": wa._profile_identity(), "record_sha256": "0" * 64, "targets": {}}
    stale = resolve_pending(subsystem, pid, "reject", expect=stale_expect, close_interrupted=True)
    assert stale["status"] == "needs_review", stale
    assert resolve_pending(subsystem, pid, "reject")["status"] == "needs_review"  # a bulk reject can't close it
    r = resolve_pending(subsystem, pid, "reject", close_interrupted=True)
    assert r["status"] == "rejected" and "nothing was rolled back" in r["warning"], r
    assert wa.list_needs_review(subsystem) == []
    assert not (wa._pending_dir(subsystem) / f"{pid}.claimed").exists()
    outcome = wa._read_outcome(subsystem, pid)
    assert outcome["status"] == "rejected" and outcome["interrupted"], outcome
    for decision in ("approve", "reject"):
        later = resolve_pending(subsystem, pid, decision, memory_store=store)
        assert later["status"] == "already_resolved" and later["prior"] == "rejected", later
    shown = preview_pending(subsystem, pid)
    assert shown["status"] == "already_resolved" and shown["prior"] == "rejected", shown
    assert _on_disk(subsystem, "crashy") == landed, "closing the write (or deciding after it) changed a file"


def _interrupted_memory_claim():
    pid = _stage_memory_add("interrupted write")
    pending = wa._pending_path(wa.MEMORY, pid)
    claimed = pending.with_suffix(".claimed")
    pending.replace(claimed)
    wa._write_outcome(wa.MEMORY, pid, "applying", {})
    return pid, claimed


def test_reject_all_never_closes_a_write_interrupted_after_it_listed_the_queue(home, monkeypatch):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    pid = _stage_memory_add("listed, then interrupted")
    listed = wa.list_pending(wa.MEMORY)
    pending = wa._pending_path(wa.MEMORY, pid)
    pending.replace(pending.with_suffix(".claimed"))  # another client claimed it and died mid-write
    wa._write_outcome(wa.MEMORY, pid, "applying", {})

    with monkeypatch.context() as patch:
        patch.setattr(wa, "list_pending", lambda subsystem: listed)
        assert handle_pending_subcommand(wa.MEMORY, ["reject", "all"]) == "Rejected 0 pending memory write(s)."
    assert resolve_pending(wa.MEMORY, pid, "reject")["status"] == "needs_review"
    assert [item["id"] for item in wa.list_needs_review(wa.MEMORY)] == [pid]


def test_concurrent_rejects_of_an_interrupted_write_close_it_once(home):
    pid, _ = _interrupted_memory_claim()
    start = threading.Barrier(2, timeout=WAIT)

    def reject():
        start.wait()
        return resolve_pending(wa.MEMORY, pid, "reject", close_interrupted=True)
    first, second = _Racer("first reject", reject), _Racer("second reject", reject)
    first.start()
    second.start()
    results = sorted((first.join(), second.join()), key=lambda r: r["status"])
    assert [r["status"] for r in results] == ["already_resolved", "rejected"], results
    assert results[0]["prior"] == "rejected"


def test_closing_an_interrupted_write_survives_claim_cleanup_failure(home, monkeypatch):
    pid, claimed = _interrupted_memory_claim()
    unlink = Path.unlink

    def fail_claim_cleanup(path, *args, **kwargs):
        if path == claimed:
            raise PermissionError("claim cleanup unavailable")
        return unlink(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_claim_cleanup)
        assert resolve_pending(wa.MEMORY, pid, "reject", close_interrupted=True)["status"] == "rejected"
        assert claimed.exists()
        assert wa.list_needs_review(wa.MEMORY) == []
    assert wa.list_needs_review(wa.MEMORY) == []
    assert not claimed.exists()
    shown = preview_pending(wa.MEMORY, pid)
    assert (shown["status"], shown["prior"]) == ("already_resolved", "rejected")


@pytest.mark.platforms("posix")
def test_cli_lists_needs_review_and_only_an_explicit_reject_clears_it(home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    assert handle_pending_subcommand(wa.SKILLS, ["pending"]) == "No pending skills writes."
    pid = _sigkilled_mid_apply(home, wa.SKILLS)
    out = handle_pending_subcommand(wa.SKILLS, ["pending"])
    assert out.startswith("No pending skills writes.\n\nNeeds review (1)"), out
    assert pid in out and "/skills reject <id>" in out and "check the result by hand" in out, out
    assert handle_pending_subcommand(wa.SKILLS, ["reject", "all"]) == "Rejected 0 pending skills write(s)."
    assert [item["id"] for item in wa.list_needs_review(wa.SKILLS)] == [pid]
    refused = handle_pending_subcommand(wa.SKILLS, ["approve", pid])
    assert "needs review" in refused and f"/skills reject {pid}" in refused, refused
    done = handle_pending_subcommand(wa.SKILLS, ["reject", pid])
    assert done.startswith(f"Rejected skills write '{pid}'. Warning:") and "partly changed" in done, done
    assert handle_pending_subcommand(wa.SKILLS, ["pending"]) == "No pending skills writes."


# --- (l) failures around the decision's own bookkeeping ---------------------------------------
@pytest.mark.parametrize("broken", ["_find_skill", "_skill_mutation_locks"])
def test_a_lookup_or_lock_failure_before_the_write_is_a_conflict_and_stays_pending(home, monkeypatch, broken):
    from tools import skill_manager_tool
    _create_skill("probe")
    pid = _stage_skill(action="patch", name="probe", old_string="# body", new_string="# body\nADDED")
    before = _on_disk(wa.SKILLS)

    def fail(*_args, **_kwargs):
        raise OSError(f"{broken} failed")

    with monkeypatch.context() as patch:
        patch.setattr(skill_manager_tool, broken, fail)
        r = resolve_pending(wa.SKILLS, pid, "approve")
    _assert_refused(r)
    assert f"{broken} failed" in r["reason"], r
    assert _on_disk(wa.SKILLS) == before
    assert wa.get_pending(wa.SKILLS, pid) is not None
    assert not os.path.lexists(wa._outcome_path(wa.SKILLS, pid))


def test_staging_returns_an_error_record_when_the_skill_lookup_fails(home, monkeypatch):
    """stage_write is best-effort: capturing the base revisions must not raise out of it."""
    from tools import skill_manager_tool, write_approval as wa
    _create_skill("probe")

    def lookup_fails(*_args, **_kwargs):
        raise OSError("skill lookup failed")
    monkeypatch.setattr(skill_manager_tool, "_find_skill", lookup_fails)
    record = wa.stage_write(wa.SKILLS, {"action": "patch", "name": "probe", "old_string": "# body",
                                        "new_string": "# body\nADDED"}, summary="probe patch", origin="foreground")
    assert record.get("error"), record
    assert wa.list_pending(wa.SKILLS) == []


def test_recovery_leaves_a_final_outcome_it_cannot_read_untouched(home, monkeypatch):
    """A claim left behind after an applied outcome, plus a transient read error on that outcome:
    recovery must not overwrite the final outcome with needs_review."""
    pid = _stage_memory_add("kept final")
    real_finish = wa._finish_claim
    monkeypatch.setattr(wa, "_finish_claim", lambda subsystem, pending_id, *, applied: (
        None if applied else real_finish(subsystem, pending_id, applied=applied)))
    assert resolve_pending(wa.MEMORY, pid, "approve", memory_store=_load_store())["status"] == "applied"
    monkeypatch.setattr(wa, "_finish_claim", real_finish)
    outcome = wa._outcome_path(wa.MEMORY, pid)
    assert wa._pending_path(wa.MEMORY, pid).with_suffix(".claimed").exists()

    real_read = wa._record_bytes

    def outcome_unreadable(path):
        if Path(path) == outcome:
            raise PermissionError("transient")
        return real_read(path)
    monkeypatch.setattr(wa, "_record_bytes", outcome_unreadable)
    wa.pending_count(wa.MEMORY)  # runs stale-claim recovery
    monkeypatch.setattr(wa, "_record_bytes", real_read)
    assert json.loads(outcome.read_text(encoding="utf-8"))["status"] == "applied"


def test_a_plain_file_where_the_outcome_store_belongs_is_refused(home):
    store = home / "pending" / "resolved"
    store.mkdir(parents=True)
    (store / wa.MEMORY).write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError):
        wa._outcome_path(wa.MEMORY, "deadbeef")
