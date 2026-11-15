"""Profiles sharing a skills root exclude each other when they write one skill. The per-skill
mutation lock used to live only under the ACTING profile's own skills dir, so two profiles patching one
shared skill held different lock files and could interleave their read-modify-write cycles (a lost
update). Each write now also takes the lock of the skill folder it changes, in the root holding it."""
import json
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SKILL = "---\nname: shared-skill\ndescription: A shared skill.\n---\n\n# Shared\n\nStep 1: Do the thing.\n"


def _profile(home: Path, shared: Path) -> Path:
    (home / "skills").mkdir(parents=True)
    (home / "config.yaml").write_text(
        "skills:\n  write_approval: false\n  external_dirs:\n    - " + str(shared) + "\n", encoding="utf-8")
    return home


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    """Two profiles (a, b) that both list one shared external skills root holding ``shared-skill``;
    this process runs as profile b."""
    shared = tmp_path / "shared"
    (shared / "shared-skill").mkdir(parents=True)
    (shared / "shared-skill" / "SKILL.md").write_text(SKILL, encoding="utf-8")
    a, b = _profile(tmp_path / "a", shared), _profile(tmp_path / "b", shared)
    monkeypatch.setenv("HERMES_HOME", str(b))
    from agent import skill_utils
    monkeypatch.setattr(skill_utils, "_EXTERNAL_DIRS_CACHE", {}, raising=False)
    return {"shared": shared, "a": a, "b": b}


def _lock(root: Path, name: str) -> Path:
    """The per-skill lock file a writer of *name* takes under *root* (digest of the basename)."""
    import hashlib
    return root / ".locks" / (hashlib.sha256(name.encode("utf-8")).hexdigest() + ".lock")


def _patch(old, new):
    from tools.skill_manager_tool import skill_manage
    return json.loads(skill_manage(action="patch", name="shared-skill", old_string=old, new_string=new))


HOLD = textwrap.dedent("""
    import sys, time, pathlib
    from tools.skill_manager_tool import _skill_mutation_lock
    ready, release = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    with _skill_mutation_lock(sys.argv[3] if len(sys.argv) > 3 else "shared-skill"):
        ready.write_text("held")
        deadline = time.monotonic() + 30
        while not release.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
""")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX flock semantics; msvcrt path is equivalent")
def test_a_write_from_another_profile_waits_for_the_shared_skills_lock(fleet, tmp_path):
    """Profile a holds the skill's mutation lock in another process; profile b's patch of the same
    shared skill must wait for it instead of running alongside."""
    ready, release = tmp_path / "ready", tmp_path / "release"
    env = {**os.environ, "HERMES_HOME": str(fleet["a"]), "PYTHONPATH": str(REPO)}
    holder = subprocess.Popen([sys.executable, "-c", HOLD, str(ready), str(release)], cwd=REPO, env=env)
    try:
        for _ in range(1500):
            if ready.exists() or holder.poll() is not None:
                break
            threading.Event().wait(0.02)
        assert ready.exists(), "lock holder did not start"
        box = {}
        writer = threading.Thread(target=lambda: box.update(result=_patch("Step 1: Do the thing.",
                                                                            "Step 1: Do it well.")))
        writer.start()
        writer.join(timeout=3)
        blocked = writer.is_alive()
        unchanged = "Do it well." not in (fleet["shared"] / "shared-skill" / "SKILL.md").read_text()
    finally:
        release.write_text("go")
        holder.wait(timeout=30)
    writer.join(timeout=30)
    assert blocked and unchanged, "profile b wrote the shared skill while profile a held its lock"
    assert box["result"]["success"] is True, box


def _hold_and_patch(fleet, tmp_path, name, old, new, writer_name=None):
    """Profile a holds the mutation lock for *name* in another process; profile b patches *writer_name*
    (default: *name*).
    Returns (b was blocked while a held the lock, b's result after release)."""
    ready, release = tmp_path / "ready", tmp_path / "release"
    env = {**os.environ, "HERMES_HOME": str(fleet["a"]), "PYTHONPATH": str(REPO)}
    holder = subprocess.Popen([sys.executable, "-c", HOLD, str(ready), str(release), name], cwd=REPO, env=env)
    box = {}
    try:
        for _ in range(1500):
            if ready.exists() or holder.poll() is not None:
                break
            threading.Event().wait(0.02)
        assert ready.exists(), "lock holder did not start"
        from tools.skill_manager_tool import skill_manage
        writer = threading.Thread(target=lambda: box.update(result=json.loads(skill_manage(
            action="patch", name=writer_name or name, old_string=old, new_string=new))))
        writer.start()
        writer.join(timeout=3)
        blocked = writer.is_alive()
    finally:
        release.write_text("go")
        holder.wait(timeout=30)
    writer.join(timeout=30)
    return blocked, box.get("result")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX flock semantics; msvcrt path is equivalent")
def test_a_display_name_with_a_slash_locks_the_skill_it_names(fleet, tmp_path):
    """The shared skill's folder is actual-skill; its display (frontmatter) name is team/foo. Profile a
    also has an unrelated local skill folder named foo. Both profiles' writes named team/foo change
    the shared skill, so they must exclude each other."""
    actual = fleet["shared"] / "actual-skill"
    actual.mkdir()
    (actual / "SKILL.md").write_text(SKILL.replace("name: shared-skill", "name: team/foo"), encoding="utf-8")
    local_foo = fleet["a"] / "skills" / "foo"
    local_foo.mkdir()
    (local_foo / "SKILL.md").write_text(SKILL.replace("shared-skill", "foo"), encoding="utf-8")
    blocked, result = _hold_and_patch(fleet, tmp_path, "team/foo", "Step 1: Do the thing.", "Step 1: Team.")
    assert blocked, "profile b wrote team/foo while profile a held its lock"
    assert result["success"] is True, result
    assert "Step 1: Team." in (actual / "SKILL.md").read_text(encoding="utf-8")


def test_a_shared_skills_lock_lives_in_its_own_root(fleet):
    assert _patch("Step 1: Do the thing.", "Step 1: Do it well.")["success"] is True
    assert _lock(fleet["shared"], "shared-skill").exists(), "no lock file in the shared root"
    assert _lock(fleet["b"] / "skills", "shared-skill").exists(), "the profile's own lock is still taken"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX flock semantics; msvcrt path is equivalent")
def test_every_name_that_resolves_to_a_shared_skill_takes_its_root_lock(fleet, tmp_path):
    """Profile a holds the lock for a categorized shared skill by its folder name; profile b writes it
    by its display name (with a slash). Both resolve to one folder, so b must wait for a."""
    cat = fleet["shared"] / "mlops" / "cat-skill"
    cat.mkdir(parents=True)
    (cat / "SKILL.md").write_text(SKILL.replace("name: shared-skill", "name: team/cat"), encoding="utf-8")
    blocked, result = _hold_and_patch(fleet, tmp_path, "cat-skill", "Step 1: Do the thing.", "Step 1: Cat.",
                                      writer_name="team/cat")
    assert blocked, "profile b wrote team/cat while profile a held cat-skill's lock"
    assert result["success"] is True, result
    assert "Step 1: Cat." in (cat / "SKILL.md").read_text(encoding="utf-8")


def _wait_for(ready, holder):
    for _ in range(1500):
        if ready.exists() or holder.poll() is not None:
            break
        threading.Event().wait(0.02)
    assert ready.exists(), "lock holder did not start"


def _held_while(fleet, tmp_path, holder_home, hold_name, write, during=lambda: True):
    """Another process (profile *holder_home*) holds the locks for *hold_name*; this process runs
    *write*. Returns (write was blocked and *during()* held true while the lock was held, write's
    result after release)."""
    ready, release = tmp_path / "ready", tmp_path / "release"
    env = {**os.environ, "HERMES_HOME": str(holder_home), "PYTHONPATH": str(REPO)}
    holder = subprocess.Popen([sys.executable, "-c", HOLD, str(ready), str(release), hold_name], cwd=REPO, env=env)
    box = {}
    try:
        _wait_for(ready, holder)
        writer = threading.Thread(target=lambda: box.update(result=json.loads(write())))
        writer.start()
        writer.join(timeout=3)
        blocked = writer.is_alive() and during()
    finally:
        release.write_text("go")
        holder.wait(timeout=30)
    writer.join(timeout=30)
    return blocked, box.get("result")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX flock semantics; msvcrt path is equivalent")
def test_the_owning_profile_and_a_profile_using_its_root_as_external_exclude_each_other(fleet, tmp_path, monkeypatch):
    """Profile a keeps actual-skill (display name team/foo) in its OWN skills dir; profile b lists a's
    skills dir in external_dirs. Both writes of team/foo change that one folder, so b must wait while
    a holds its locks, although a sees the skill as local."""
    actual = fleet["a"] / "skills" / "actual-skill"
    actual.mkdir()
    (actual / "SKILL.md").write_text(SKILL.replace("name: shared-skill", "name: team/foo"), encoding="utf-8")
    (fleet["b"] / "config.yaml").write_text(
        "skills:\n  write_approval: false\n  external_dirs:\n    - " + str(fleet["a"] / "skills") + "\n", encoding="utf-8")
    from agent import skill_utils
    monkeypatch.setattr(skill_utils, "_EXTERNAL_DIRS_CACHE", {}, raising=False)
    from tools.skill_manager_tool import skill_manage
    blocked, result = _held_while(fleet, tmp_path, fleet["a"], "team/foo", lambda: skill_manage(
        action="patch", name="team/foo", old_string="Step 1: Do the thing.", new_string="Step 1: Owned."),
        during=lambda: "Owned" not in (actual / "SKILL.md").read_text(encoding="utf-8"))
    assert blocked, "profile b wrote a's skill while a held its locks"
    assert result["success"] is True, result
    assert "Step 1: Owned." in (actual / "SKILL.md").read_text(encoding="utf-8")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX flock semantics; msvcrt path is equivalent")
def test_a_create_into_a_shared_create_dir_waits_for_another_profiles_lock(fleet, tmp_path, monkeypatch):
    """Profiles a and b share one create_dir. While a holds the locks for brain-skill, b's create of
    brain-skill must wait (not just leave a lock file behind)."""
    brain = tmp_path / "brain"
    for prof in ("a", "b"):
        (fleet[prof] / "config.yaml").write_text(
            "skills:\n  write_approval: false\n  create_dir: " + str(brain) + "\n", encoding="utf-8")
    from agent import skill_utils
    monkeypatch.setattr(skill_utils, "_EXTERNAL_DIRS_CACHE", {}, raising=False)
    from tools.skill_manager_tool import skill_manage
    blocked, result = _held_while(fleet, tmp_path, fleet["a"], "brain-skill", lambda: skill_manage(
        action="create", name="brain-skill", content=SKILL.replace("shared-skill", "brain-skill")),
        during=lambda: not (brain / "brain-skill").exists())
    assert blocked, "profile b created while a held the lock"
    assert result["success"] is True, result
    assert (brain / "brain-skill" / "SKILL.md").exists()


def _folder_lock_module():
    """The folder-lock helpers. Without them (main, the earlier design) a write takes one lock only,
    so a lock file reached by two spellings cannot arise and these hazards do not exist."""
    locks = pytest.importorskip("tools.skill_manager_locks")
    if not hasattr(locks, "_target_root_lock"):
        pytest.skip("no folder lock in this code: the two-spellings hazard does not exist")
    return locks


def _spelled_twice(tmp_path, fleet, monkeypatch, locks, of):
    """Make the folder lock of every name be the profile's own lock of *of*, reached through a
    symlinked spelling of the profile's skills dir."""
    link = tmp_path / "skills-link"
    link.symlink_to(fleet["b"] / "skills", target_is_directory=True)
    monkeypatch.setattr(locks, "_target_root_lock", lambda name: link / ".locks" / locks._skill_lock_path(of).name)


def _finishes(fn, timeout=10):
    box = {}
    worker = threading.Thread(target=lambda: box.update(result=fn()), daemon=True)
    worker.start()
    worker.join(timeout=timeout)
    assert not worker.is_alive(), "the write deadlocked on a lock its own thread holds"
    return box["result"]


def test_one_lock_file_reached_by_two_spellings_is_taken_once(fleet, tmp_path, monkeypatch):
    """A skill's folder lock can be the profile's own lock under another spelling (a symlinked skills
    dir). Taking it twice would deadlock this thread on itself; it must be taken once."""
    locks = _folder_lock_module()
    _spelled_twice(tmp_path, fleet, monkeypatch, locks, "shared-skill")
    assert len(locks._skill_lock_paths("shared-skill")) == 1
    result = _finishes(lambda: _patch("Step 1: Do the thing.", "Step 1: Once."))
    assert result["success"] is True, result


def test_an_operation_inside_its_batch_does_not_retake_a_held_lock_under_another_spelling(
        fleet, tmp_path, monkeypatch):
    """The batch holds shared-skill's own lock under its plain spelling. Its second operation's folder
    lock is that same file reached through a symlinked dir: the operation must see it as held."""
    locks = _folder_lock_module()
    from tools.skill_manager_tool import skill_manage
    assert json.loads(skill_manage(action="create", name="second-skill", content=SKILL.replace(
        "shared-skill", "second-skill")))["success"] is True
    _spelled_twice(tmp_path, fleet, monkeypatch, locks, "shared-skill")
    result = _finishes(lambda: json.loads(skill_manage(action="patch", name="", operations=[
        {"action": "patch", "name": "shared-skill", "old_string": "Step 1: Do the thing.", "new_string": "S."},
        {"action": "patch", "name": "second-skill", "old_string": "Step 1: Do the thing.", "new_string": "T."}])))
    assert result["success"] is True, result


NESTED = textwrap.dedent("""
    import sys, time, pathlib
    import tools.skill_manager_locks as locks
    from tools.skill_manager_tool import _skill_mutation_lock
    locks._OUT_OF_ORDER_WAIT = 1.0
    ready, go, out = map(pathlib.Path, sys.argv[1:4])
    with _skill_mutation_lock("actual-skill"):
        ready.write_text("held")
        deadline = time.monotonic() + 30
        while not go.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        try:
            with _skill_mutation_lock("team/foo"):  # what a lifecycle hook's write of team/foo does
                out.write_text("acquired")
        except TimeoutError:
            out.write_text("busy")
""")


@pytest.mark.skipif(sys.platform == "win32", reason="probes with fcntl")
def test_a_nested_write_needing_a_lock_out_of_order_fails_instead_of_deadlocking(fleet, tmp_path):
    """One profile; local folder actual-skill has display name team/foo. Process A writes actual-skill
    (its folder lock). This process writes team/foo: it takes foo's own lock, then waits for A. A's
    lifecycle hook now writes team/foo from inside A's lock and needs foo's own lock, which sorts
    BEFORE the one A holds. Waiting would deadlock both; the nested write must give up instead."""
    actual = fleet["b"] / "skills" / "actual-skill"
    actual.mkdir()
    (actual / "SKILL.md").write_text(SKILL.replace("name: shared-skill", "name: team/foo"), encoding="utf-8")
    ready, go, out = tmp_path / "ready", tmp_path / "go", tmp_path / "out"
    env = {**os.environ, "HERMES_HOME": str(fleet["b"]), "PYTHONPATH": str(REPO)}
    holder = subprocess.Popen([sys.executable, "-c", NESTED, str(ready), str(go), str(out)], cwd=REPO, env=env)
    box = {}
    try:
        _wait_for(ready, holder)
        from tools.skill_manager_tool import _skill_mutation_lock

        def write_team_foo():
            with _skill_mutation_lock("team/foo"):
                box["done"] = True
        writer = threading.Thread(target=write_team_foo, daemon=True)
        writer.start()
        own_foo = _lock(fleet["b"] / "skills", "foo")
        for _ in range(250):  # until this process holds foo's own lock and waits for A
            probe = subprocess.run([sys.executable, "-c", PROBE, str(own_foo)], capture_output=True, text=True,
                                   check=True)
            if probe.stdout.split() == ["held"]:
                break
            threading.Event().wait(0.02)
        assert probe.stdout.split() == ["held"] and writer.is_alive()
        go.write_text("go")
        try:
            holder.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pytest.fail("the nested write deadlocked against this process")
    finally:
        go.write_text("go")
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=10)
    writer.join(timeout=20)
    assert out.read_text() == "busy" and box.get("done") is True, (out.read_text(), box)


def _probe(*paths):
    out = subprocess.run([sys.executable, "-c", PROBE, *map(str, paths)], capture_output=True, text=True, check=True)
    return out.stdout.split()


@pytest.mark.skipif(sys.platform == "win32", reason="probes with fcntl")
def test_a_free_lock_taken_out_of_order_is_held_reentered_and_released(fleet, monkeypatch):
    """The nested (out-of-order) path when the lock is free: it is really held (another process sees
    it), the same thread can re-enter it at once, and it is released and forgotten afterwards, on a
    normal exit and on an exception, so a later acquisition takes it for real."""
    import tools.skill_manager_locks as locks
    from tools.skill_manager_tool import _skill_mutation_lock
    monkeypatch.setattr(locks, "_OUT_OF_ORDER_WAIT", 1.0)
    skills = fleet["b"] / "skills"
    high, low = _lock(skills, "actual-skill"), _lock(skills, "foo")  # sha256: 9a9e... > 2c26...
    assert str(low) < str(high)
    with _skill_mutation_lock("actual-skill"):
        with _skill_mutation_lock("foo"):  # sorts before a lock this thread holds: the bounded path
            assert _probe(low) == ["held"]
            with _skill_mutation_lock("foo"):  # re-entry by the same thread must not wait on itself
                assert _probe(low) == ["held"]
        assert _probe(low) == ["free"]
        with pytest.raises(RuntimeError):
            with _skill_mutation_lock("foo"):
                raise RuntimeError("the write failed")
        assert _probe(low) == ["free"]
    with _skill_mutation_lock("foo"):  # in order now; a stale held entry would skip the real lock
        assert _probe(low) == ["held"]
    assert _probe(low, high) == ["free", "free"]


def test_a_local_skill_keeps_its_lock_in_the_local_root(fleet):
    from tools.skill_manager_tool import skill_manage
    local = json.loads(skill_manage(action="create", name="local-skill", content=SKILL.replace(
        "shared-skill", "local-skill")))
    assert local["success"] is True, local
    assert json.loads(skill_manage(action="patch", name="local-skill", old_string="Step 1: Do the thing.",
                                   new_string="Step 1: Local."))["success"] is True
    assert _lock(fleet["b"] / "skills", "local-skill").exists()
    assert not _lock(fleet["shared"], "local-skill").exists()


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions, non-root")
def test_a_read_only_shared_root_falls_back_to_the_local_lock(fleet):
    """A root whose top level cannot hold .locks/ (the skill dir itself is still writable) still
    gets its writes, locked in the profile's own root as before."""
    fleet["shared"].chmod(0o555)
    try:
        result = _patch("Step 1: Do the thing.", "Step 1: Read-only root.")
    finally:
        fleet["shared"].chmod(0o755)
    assert result["success"] is True, result
    assert not (fleet["shared"] / ".locks").exists()
    assert _lock(fleet["b"] / "skills", "shared-skill").exists()


def test_a_create_in_a_shared_create_dir_locks_in_that_root(fleet, tmp_path):
    brain = tmp_path / "brain"
    (fleet["b"] / "config.yaml").write_text(
        "skills:\n  write_approval: false\n  create_dir: " + str(brain) + "\n", encoding="utf-8")
    from tools.skill_manager_tool import skill_manage
    created = json.loads(skill_manage(action="create", name="brain-skill", content=SKILL.replace(
        "shared-skill", "brain-skill")))
    assert created["success"] is True, created
    assert (brain / "brain-skill" / "SKILL.md").exists()
    assert _lock(brain, "brain-skill").exists()
    assert _lock(fleet["b"] / "skills", "brain-skill").exists(), "the profile's own lock is still taken"


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions, non-root")
def test_a_lock_file_another_user_owns_falls_back_to_the_local_lock(fleet):
    """On a root shared by several OS users, a lock file this user cannot open for writing must not
    turn every write to that skill into an error; the write is locked in the local root instead."""
    lock = _lock(fleet["shared"], "shared-skill")
    lock.parent.mkdir()
    lock.write_text(" ", encoding="utf-8")
    lock.chmod(0o444)
    try:
        result = _patch("Step 1: Do the thing.", "Step 1: Shared by users.")
    finally:
        lock.chmod(0o644)
    assert result["success"] is True, result
    assert _lock(fleet["b"] / "skills", "shared-skill").exists()


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions, non-root")
def test_an_unwritable_locks_folder_in_the_shared_root_falls_back_to_the_local_lock(fleet):
    locks = fleet["shared"] / ".locks"
    locks.mkdir()
    locks.chmod(0o555)
    try:
        result = _patch("Step 1: Do the thing.", "Step 1: Locks folder read-only.")
    finally:
        locks.chmod(0o755)
    assert result["success"] is True, result
    assert _lock(fleet["b"] / "skills", "shared-skill").exists()


def test_a_failed_root_lookup_falls_back_to_the_local_lock(fleet, monkeypatch):
    import tools.skill_manager_guards as guards

    def broken(_path):
        raise RuntimeError("root lookup failed")

    monkeypatch.setattr(guards, "_containing_skills_root", broken)
    assert _patch("Step 1: Do the thing.", "Step 1: Lookup failed.")["success"] is True
    assert _lock(fleet["b"] / "skills", "shared-skill").exists()
    assert not (fleet["shared"] / ".locks").exists()


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions, non-root")
def test_a_write_only_lock_file_falls_back_to_the_local_lock(fleet):
    """The lock is opened for reading and writing ("a+"); a lock file this user may write but not read
    (another user's 0622 file) must fall back, not fail the write when the lock is taken."""
    lock = _lock(fleet["shared"], "shared-skill")
    lock.parent.mkdir()
    lock.write_text(" ", encoding="utf-8")
    lock.chmod(0o222)
    try:
        result = _patch("Step 1: Do the thing.", "Step 1: Write-only lock.")
    finally:
        lock.chmod(0o644)
    assert result["success"] is True, result
    assert _lock(fleet["b"] / "skills", "shared-skill").exists()


def test_a_batch_across_roots_locks_each_skill_in_its_root(fleet):
    from tools.skill_manager_tool import skill_manage
    assert json.loads(skill_manage(action="create", name="local-skill", content=SKILL.replace(
        "shared-skill", "local-skill")))["success"] is True
    result = json.loads(skill_manage(action="patch", name="", operations=[
        {"action": "patch", "name": "local-skill", "old_string": "Step 1: Do the thing.", "new_string": "L."},
        {"action": "patch", "name": "shared-skill", "old_string": "Step 1: Do the thing.", "new_string": "S."}]))
    assert result["success"] is True, result
    assert _lock(fleet["shared"], "shared-skill").exists()
    assert _lock(fleet["b"] / "skills", "local-skill").exists()


PROBE = textwrap.dedent("""
    import fcntl, sys
    for path in sys.argv[1:]:
        with open(path, "a+") as fd:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("held")
            else:
                print("free")
""")


@pytest.mark.skipif(sys.platform == "win32", reason="probes with fcntl")
def test_a_batch_holds_every_lock_from_before_its_snapshot_through_its_rollback(fleet, monkeypatch):
    """The batch's own locks (not just the per-op locks inside it) must be every lock of every skill:
    another process sees each of them held when the snapshot is taken AND while a failed batch rolls
    back."""
    import tools.skill_manager_batch as batch
    from tools.skill_manager_tool import skill_manage
    assert json.loads(skill_manage(action="create", name="local-skill", content=SKILL.replace(
        "shared-skill", "local-skill")))["success"] is True
    paths = [_lock(fleet["shared"], "shared-skill"), _lock(fleet["b"] / "skills", "shared-skill"),
             _lock(fleet["b"] / "skills", "local-skill")]
    seen = {}

    def probe(label):
        out = subprocess.run([sys.executable, "-c", PROBE, *map(str, paths)], capture_output=True, text=True,
                             check=True)
        seen[label] = out.stdout.split()

    real_snapshot, real_rollback = batch._snapshot_skills, batch._rollback

    def probe_then_snapshot(*a, **kw):
        probe("snapshot")
        return real_snapshot(*a, **kw)

    def probe_then_rollback(*a, **kw):
        probe("rollback")
        return real_rollback(*a, **kw)

    monkeypatch.setattr(batch, "_snapshot_skills", probe_then_snapshot)
    monkeypatch.setattr(batch, "_rollback", probe_then_rollback)
    result = json.loads(skill_manage(action="patch", name="", operations=[
        {"action": "patch", "name": "local-skill", "old_string": "Step 1: Do the thing.", "new_string": "L."},
        {"action": "patch", "name": "shared-skill", "old_string": "Step 1: Do the thing.", "new_string": "S."},
        {"action": "patch", "name": "shared-skill", "old_string": "not in the file", "new_string": "X."}]))
    assert result["success"] is False, result
    assert "Step 1: Do the thing." in (fleet["shared"] / "shared-skill" / "SKILL.md").read_text()
    assert seen == {"snapshot": ["held"] * 3, "rollback": ["held"] * 3}, seen
