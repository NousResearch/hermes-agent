from __future__ import annotations

import inspect
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

import tools.skill_publish_guard as _spg


SKILL_MD = "---\nname: {name}\ndescription: test skill.\n---\n# Test\n"


@pytest.fixture(autouse=True)
def isolated_skill_roots(tmp_path, monkeypatch):
    home = tmp_path / "home"
    hermes = tmp_path / "hermes"
    home.mkdir()
    hermes.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(hermes))
    monkeypatch.setenv("HERMES_DISABLE_SELF_IMPROVEMENT", "1")
    root = tmp_path / "skills"
    root.mkdir()

    import agent.skill_utils as skill_utils

    monkeypatch.setattr(skill_utils, "get_all_skills_dirs", lambda: [root])
    monkeypatch.setattr(skill_utils, "is_excluded_skill_path", lambda *_a, **_kw: False)
    return root


def _skill(path: Path, frontmatter_name: str | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if frontmatter_name is not None:
        (path / "SKILL.md").write_text(SKILL_MD.format(name=frontmatter_name), encoding="utf-8")
    return path


class RecordingLockFactory:
    def __init__(self, *, fail_on_enter: set[Path] | None = None, fail_on_release: set[Path] | None = None):
        self.fail_on_enter = {Path(p) for p in (fail_on_enter or set())}
        self.fail_on_release = {Path(p) for p in (fail_on_release or set())}
        self.acquired: list[tuple[Path, Path]] = []
        self.released: list[tuple[Path, Path]] = []
        self._locks: dict[Path, threading.Lock] = {}
        self.entered_events: dict[Path, threading.Event] = {}
        self.release_events: dict[Path, threading.Event] = {}

    def __call__(self, *, lock_path, canonical_skill_path):
        factory = self
        lock_path = Path(lock_path)
        canonical_skill_path = Path(canonical_skill_path)
        lock = self._locks.setdefault(lock_path, threading.Lock())
        entered = self.entered_events.setdefault(lock_path, threading.Event())
        released = self.release_events.setdefault(lock_path, threading.Event())

        class _Lock:
            def __enter__(self):
                if lock_path in factory.fail_on_enter:
                    raise PermissionError("blocked for test")
                lock.acquire()
                factory.acquired.append((lock_path, canonical_skill_path))
                entered.set()
                return None

            def __exit__(self, exc_type, exc, tb):
                factory.released.append((lock_path, canonical_skill_path))
                released.set()
                if lock_path in factory.fail_on_release:
                    if lock.locked():
                        lock.release()
                    raise _spg.SkillMutationLockReleaseFailure(
                        canonical_skill_path=canonical_skill_path,
                        lock_path=lock_path,
                        platform="test",
                        release_error=RuntimeError("release failed for test"),
                    )
                if lock.locked():
                    lock.release()
                return False

        return _Lock()


def _path_acquisitions(factory: RecordingLockFactory) -> list[Path]:
    return [canonical for _lock, canonical in factory.acquired[1:]]


def _path_releases(factory: RecordingLockFactory) -> list[Path]:
    return [
        canonical
        for lock, canonical in factory.released
        if lock.name.startswith(".hermes-skill-mutex-")
    ]


def test_g1_normal_publish_guard_still_rejects_multiple_duplicates(isolated_skill_roots, tmp_path):
    root = isolated_skill_roots
    _skill(root / "alpha", "alpha")
    _skill(root / "cat" / "alpha", "alpha")

    with pytest.raises(_spg.SkillMutationLockAcquireFailure):
        with _spg.live_skill_publish_guard("alpha", target=tmp_path / "new" / "alpha"):
            pass


def test_g2_repair_accepts_one_explicitly_approved_noncanonical_path(isolated_skill_roots):
    existing = _skill(isolated_skill_roots / "category" / "beta", "beta")
    with _spg.live_skill_repair_guard(
        "beta",
        target=existing,
        approved_existing_paths=[existing],
        mutation_paths=[existing],
    ):
        pass


def test_g3_repair_accepts_multiple_explicitly_approved_same_name_paths(isolated_skill_roots):
    a = _skill(isolated_skill_roots / "one" / "gamma", "gamma")
    b = _skill(isolated_skill_roots / "two" / "gamma", "gamma")
    with _spg.live_skill_repair_guard(
        "gamma",
        target=a,
        approved_existing_paths=[a, b],
        mutation_paths=[a, b],
    ):
        pass


def test_g4_unexpected_extra_same_name_path_fails_closed(isolated_skill_roots):
    approved = _skill(isolated_skill_roots / "one" / "delta", "delta")
    _skill(isolated_skill_roots / "two" / "delta", "delta")
    with pytest.raises(_spg.SkillMutationLockAcquireFailure):
        with _spg.live_skill_repair_guard(
            "delta",
            target=approved,
            approved_existing_paths=[approved],
            mutation_paths=[approved],
        ):
            pass


def test_g5_repair_set_changing_between_scans_fails_closed(isolated_skill_roots, monkeypatch):
    approved = _skill(isolated_skill_roots / "epsilon", "epsilon")
    extra = isolated_skill_roots / "other" / "epsilon"
    original = _spg._maintenance_duplicate_scan
    calls = 0

    def changing_scan(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            _skill(extra, "epsilon")
        return original(*args, **kwargs)

    monkeypatch.setattr(_spg, "_maintenance_duplicate_scan", changing_scan)
    with pytest.raises(_spg.SkillMutationLockAcquireFailure):
        with _spg.live_skill_repair_guard(
            "epsilon",
            target=approved,
            approved_existing_paths=[approved],
            mutation_paths=[approved],
        ):
            pass


def test_g6_and_input_order_path_locks_are_deterministically_sorted_and_deduped(isolated_skill_roots, monkeypatch):
    a = _skill(isolated_skill_roots / "zeta", "zeta")
    b = isolated_skill_roots / "prospective" / "zeta"
    c = isolated_skill_roots / "aaa" / "zeta"
    factory = RecordingLockFactory()
    monkeypatch.setattr(_spg, "_acquire_lock_at_path", factory)

    with _spg.live_skill_repair_guard(
        "zeta",
        target=a,
        approved_existing_paths=[a],
        mutation_paths=[b, a, c, a],
    ):
        pass

    expected = sorted({_spg._canonical_path(p) for p in [a, b, c]}, key=_spg._path_sort_key)
    assert _path_acquisitions(factory) == expected


def test_g7_partial_path_lock_acquisition_failure_releases_already_acquired(isolated_skill_roots, monkeypatch):
    a = _skill(isolated_skill_roots / "eta", "eta")
    b = isolated_skill_roots / "b" / "eta"
    ordered = sorted([_spg._canonical_path(a), _spg._canonical_path(b)], key=_spg._path_sort_key)
    failing_lock_path = _spg._target_lock_path(ordered[1])
    factory = RecordingLockFactory(fail_on_enter={failing_lock_path})
    monkeypatch.setattr(_spg, "_acquire_lock_at_path", factory)

    with pytest.raises(_spg.SkillMutationLockAcquireFailure):
        with _spg.live_skill_repair_guard(
            "eta",
            target=a,
            approved_existing_paths=[a],
            mutation_paths=[a, b],
        ):
            pass

    assert _path_acquisitions(factory) == [ordered[0]]
    assert _path_releases(factory) == [ordered[0]]


def test_g8_release_order_is_reverse_acquisition_order(isolated_skill_roots, monkeypatch):
    a = _skill(isolated_skill_roots / "theta", "theta")
    b = isolated_skill_roots / "b" / "theta"
    c = isolated_skill_roots / "c" / "theta"
    factory = RecordingLockFactory()
    monkeypatch.setattr(_spg, "_acquire_lock_at_path", factory)

    with _spg.live_skill_repair_guard(
        "theta",
        target=a,
        approved_existing_paths=[a],
        mutation_paths=[a, b, c],
    ):
        pass

    acquired_paths = _path_acquisitions(factory)
    released_paths = _path_releases(factory)
    assert released_paths == list(reversed(acquired_paths))


def test_g9_same_name_ordinary_publisher_serialized_behind_repair(isolated_skill_roots, monkeypatch):
    existing = _skill(isolated_skill_roots / "iota", "iota")
    factory = RecordingLockFactory()
    monkeypatch.setattr(_spg, "_acquire_lock_at_path", factory)
    repair_entered = threading.Event()
    release_repair = threading.Event()
    publisher_finished = threading.Event()

    def repair():
        with _spg.live_skill_repair_guard(
            "iota",
            target=existing,
            approved_existing_paths=[existing],
            mutation_paths=[existing],
        ):
            repair_entered.set()
            assert release_repair.wait(5)

    t = threading.Thread(target=repair)
    t.start()
    assert repair_entered.wait(5)

    def publish():
        with pytest.raises(_spg.SkillMutationLockAcquireFailure):
            with _spg.live_skill_publish_guard("iota", target=existing, replacement_policy="new_only"):
                pass
        publisher_finished.set()

    p = threading.Thread(target=publish)
    p.start()
    time.sleep(0.1)
    assert not publisher_finished.is_set()
    release_repair.set()
    t.join(5)
    p.join(5)
    assert publisher_finished.is_set()


def test_g10_same_name_repair_serialized_behind_repair(isolated_skill_roots, monkeypatch):
    existing = _skill(isolated_skill_roots / "kappa", "kappa")
    factory = RecordingLockFactory()
    monkeypatch.setattr(_spg, "_acquire_lock_at_path", factory)
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def guarded(evt_entered, evt_release=None):
        with _spg.live_skill_repair_guard(
            "kappa",
            target=existing,
            approved_existing_paths=[existing],
            mutation_paths=[existing],
        ):
            evt_entered.set()
            if evt_release is not None:
                assert evt_release.wait(5)

    t1 = threading.Thread(target=guarded, args=(first_entered, release_first))
    t2 = threading.Thread(target=guarded, args=(second_entered, None))
    t1.start()
    assert first_entered.wait(5)
    t2.start()
    time.sleep(0.1)
    assert not second_entered.is_set()
    release_first.set()
    t1.join(5)
    t2.join(5)
    assert second_entered.is_set()


def test_g11_unrelated_skill_names_remain_independent(isolated_skill_roots, monkeypatch):
    a = _skill(isolated_skill_roots / "lambda", "lambda")
    b = _skill(isolated_skill_roots / "mu", "mu")
    factory = RecordingLockFactory()
    monkeypatch.setattr(_spg, "_acquire_lock_at_path", factory)
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def first():
        with _spg.live_skill_repair_guard("lambda", target=a, approved_existing_paths=[a], mutation_paths=[a]):
            first_entered.set()
            assert release_first.wait(5)

    def second():
        with _spg.live_skill_repair_guard("mu", target=b, approved_existing_paths=[b], mutation_paths=[b]):
            second_entered.set()

    t1 = threading.Thread(target=first)
    t2 = threading.Thread(target=second)
    t1.start()
    assert first_entered.wait(5)
    t2.start()
    assert second_entered.wait(5)
    release_first.set()
    t1.join(5)
    t2.join(5)


def test_g12_frontmatter_name_matching_repair_only_normal_semantics_unchanged(isolated_skill_roots, tmp_path):
    frontmatter_only = _skill(isolated_skill_roots / "directory-name", "nu")
    with _spg.live_skill_repair_guard(
        "nu",
        target=frontmatter_only,
        approved_existing_paths=[frontmatter_only],
        mutation_paths=[frontmatter_only],
    ):
        pass

    # Ordinary duplicate scan still keys by live path layout/basename, not frontmatter name.
    with _spg.live_skill_publish_guard("nu", target=tmp_path / "new" / "nu"):
        pass


def test_g13_approved_path_replacement_symlink_or_disappearance_before_yield_fails_closed(isolated_skill_roots, monkeypatch, tmp_path):
    approved = _skill(isolated_skill_roots / "xi", "xi")
    original_identity = _spg._path_identity
    calls = 0

    def replacing_identity(path, **kwargs):
        nonlocal calls
        calls += 1
        result = original_identity(path, **kwargs)
        if calls == 1:
            for child in approved.iterdir():
                child.unlink()
            approved.rmdir()
            os.symlink(tmp_path, approved)
        return result

    monkeypatch.setattr(_spg, "_path_identity", replacing_identity)
    with pytest.raises(_spg.SkillMutationLockAcquireFailure):
        with _spg.live_skill_repair_guard(
            "xi",
            target=approved,
            approved_existing_paths=[approved],
            mutation_paths=[approved],
        ):
            pass


def test_g14_normal_public_signature_and_same_target_policy_behavior(isolated_skill_roots):
    sig = inspect.signature(_spg.live_skill_publish_guard)
    assert list(sig.parameters) == ["name", "target", "replacement_policy"]
    assert sig.parameters["target"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["replacement_policy"].default == "new_only"

    existing = _skill(isolated_skill_roots / "omicron", "omicron")
    with pytest.raises(_spg.SkillMutationLockAcquireFailure):
        with _spg.live_skill_publish_guard("omicron", target=existing):
            pass
    with _spg.live_skill_publish_guard("omicron", target=existing, replacement_policy="replace_same_target"):
        pass


def test_mutation_paths_may_include_nonexistent_destination(isolated_skill_roots):
    existing = _skill(isolated_skill_roots / "pi", "pi")
    future = isolated_skill_roots / "future" / "pi"
    with _spg.live_skill_repair_guard(
        "pi",
        target=existing,
        approved_existing_paths=[existing],
        mutation_paths=[existing, future],
    ):
        assert not future.exists()


def test_approved_external_survivor_is_not_path_locked(isolated_skill_roots, monkeypatch):
    survivor = _skill(isolated_skill_roots / "rho", "rho")
    destination = isolated_skill_roots / "managed" / "rho"
    factory = RecordingLockFactory()
    monkeypatch.setattr(_spg, "_acquire_lock_at_path", factory)

    with _spg.live_skill_repair_guard(
        "rho",
        target=destination,
        approved_existing_paths=[survivor],
        mutation_paths=[destination],
    ):
        pass

    assert _spg._canonical_path(survivor) not in _path_acquisitions(factory)
    assert _spg._canonical_path(destination) in _path_acquisitions(factory)


def test_identity_names_do_not_create_another_global_lock_key(isolated_skill_roots, monkeypatch):
    approved = _skill(isolated_skill_roots / "sigma", "alias-name")
    factory = RecordingLockFactory()
    monkeypatch.setattr(_spg, "_acquire_lock_at_path", factory)

    with _spg.live_skill_repair_guard(
        "sigma",
        target=approved,
        approved_existing_paths=[approved],
        mutation_paths=[approved],
        identity_names=("alias-name",),
    ):
        pass

    global_locks = [lock for lock, _canonical in factory.acquired if lock.name.startswith(".hermes-skill-name-mutex-")]
    assert len(global_locks) == 1
    assert global_locks[0] == _spg.normalized_name_lock_target("sigma", anchor=approved)
    assert global_locks[0] != _spg.normalized_name_lock_target("alias-name", anchor=approved)


def test_release_failure_during_multi_lock_release_attempts_remaining_releases(isolated_skill_roots, monkeypatch):
    a = _skill(isolated_skill_roots / "tau", "tau")
    b = isolated_skill_roots / "b" / "tau"
    ordered = sorted([_spg._canonical_path(a), _spg._canonical_path(b)], key=_spg._path_sort_key)
    failing_lock_path = _spg._target_lock_path(ordered[1])
    factory = RecordingLockFactory(fail_on_release={failing_lock_path})
    monkeypatch.setattr(_spg, "_acquire_lock_at_path", factory)

    with pytest.raises(_spg.SkillMutationLockReleaseFailure):
        with _spg.live_skill_repair_guard(
            "tau",
            target=a,
            approved_existing_paths=[a],
            mutation_paths=[a, b],
        ):
            pass

    released_canonicals = _path_releases(factory)
    assert released_canonicals == list(reversed(ordered))


def test_windows_case_aliases_collapse_to_one_mutation_lock_identity(isolated_skill_roots, monkeypatch):
    existing = _skill(isolated_skill_roots / "upsilon", "upsilon")
    upper = isolated_skill_roots / "Future" / "Upsilon"
    lower = isolated_skill_roots / "future" / "upsilon"
    factory = RecordingLockFactory()
    monkeypatch.setattr(_spg, "_IS_WINDOWS", True)
    monkeypatch.setattr(_spg, "_acquire_lock_at_path", factory)

    with _spg.live_skill_repair_guard(
        "upsilon",
        target=existing,
        approved_existing_paths=[existing],
        mutation_paths=[upper, lower, existing],
    ):
        pass

    acquired_texts = [str(path).replace(os.sep, "/").lower() for path in _path_acquisitions(factory)]
    assert acquired_texts.count(str(lower).replace(os.sep, "/").lower()) == 1


def test_windows_case_alias_lock_order_is_input_order_independent(isolated_skill_roots, monkeypatch):
    existing = _skill(isolated_skill_roots / "phi", "phi")
    b = isolated_skill_roots / "B" / "phi"
    a = isolated_skill_roots / "a" / "phi"

    def run(paths):
        factory = RecordingLockFactory()
        monkeypatch.setattr(_spg, "_acquire_lock_at_path", factory)
        with _spg.live_skill_repair_guard(
            "phi",
            target=existing,
            approved_existing_paths=[existing],
            mutation_paths=paths,
        ):
            pass
        return [str(path).replace(os.sep, "/").lower() for path in _path_acquisitions(factory)]

    monkeypatch.setattr(_spg, "_IS_WINDOWS", True)
    expected = sorted({str(p).replace(os.sep, "/").lower() for p in [a, b, existing]})
    assert run([b, existing, a]) == expected
    assert run([a, b, existing]) == expected


def test_posix_case_distinct_mutation_paths_remain_distinct(isolated_skill_roots, monkeypatch):
    existing = _skill(isolated_skill_roots / "chi", "chi")
    upper = isolated_skill_roots / "Future" / "chi"
    lower = isolated_skill_roots / "future" / "chi"
    factory = RecordingLockFactory()
    monkeypatch.setattr(_spg, "_IS_WINDOWS", False)
    monkeypatch.setattr(_spg, "_acquire_lock_at_path", factory)

    with _spg.live_skill_repair_guard(
        "chi",
        target=existing,
        approved_existing_paths=[existing],
        mutation_paths=[upper, lower, existing],
    ):
        pass

    acquired = set(_path_acquisitions(factory))
    assert _spg._canonical_path(upper) in acquired
    assert _spg._canonical_path(lower) in acquired


def test_existing_approved_symlink_fails_closed_before_body(isolated_skill_roots, tmp_path):
    target = _skill(tmp_path / "outside" / "psi", "psi")
    link = isolated_skill_roots / "psi"
    os.symlink(target, link)
    body_ran = False

    with pytest.raises(_spg.SkillMutationLockAcquireFailure) as excinfo:
        with _spg.live_skill_repair_guard(
            "psi",
            target=link,
            approved_existing_paths=[link],
            mutation_paths=[link],
        ):
            body_ran = True

    assert body_ran is False
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "symlink/junction" in str(excinfo.value.__cause__)


def test_approved_directory_replaced_by_symlink_between_phases_fails_closed(isolated_skill_roots, monkeypatch, tmp_path):
    approved = _skill(isolated_skill_roots / "omega", "omega")
    replacement_target = _skill(tmp_path / "replacement" / "omega", "omega")
    factory = RecordingLockFactory()
    replaced = False

    def replacing_acquire(*, lock_path, canonical_skill_path):
        ctx = factory(lock_path=lock_path, canonical_skill_path=canonical_skill_path)

        class _ReplacingLock:
            def __enter__(self):
                nonlocal replaced
                result = ctx.__enter__()
                if lock_path.name.startswith(".hermes-skill-mutex-") and not replaced:
                    for child in approved.iterdir():
                        child.unlink()
                    approved.rmdir()
                    os.symlink(replacement_target, approved)
                    replaced = True
                return result

            def __exit__(self, exc_type, exc, tb):
                return ctx.__exit__(exc_type, exc, tb)

        return _ReplacingLock()

    monkeypatch.setattr(_spg, "_acquire_lock_at_path", replacing_acquire)
    with pytest.raises(_spg.SkillMutationLockAcquireFailure) as excinfo:
        with _spg.live_skill_repair_guard(
            "omega",
            target=approved,
            approved_existing_paths=[approved],
            mutation_paths=[approved],
        ):
            pass

    assert replaced is True
    assert "symlink/junction" in str(excinfo.value.__cause__)


def test_lexical_symlink_and_resolved_target_cannot_bypass_identity_checks(isolated_skill_roots, tmp_path):
    target = _skill(tmp_path / "target" / "aardvark", "aardvark")
    link = isolated_skill_roots / "aardvark"
    os.symlink(target, link)

    with pytest.raises(_spg.SkillMutationLockAcquireFailure) as excinfo:
        with _spg.live_skill_repair_guard(
            "aardvark",
            target=link,
            approved_existing_paths=[link],
            mutation_paths=[target],
        ):
            pass

    assert "symlink/junction" in str(excinfo.value.__cause__)


def test_windows_junction_reparse_rejection_path_can_be_simulated(isolated_skill_roots, monkeypatch):
    approved = _skill(isolated_skill_roots / "badger", "badger")

    def fake_is_junction(self):
        return self == approved

    monkeypatch.setattr(Path, "is_junction", fake_is_junction, raising=False)
    with pytest.raises(_spg.SkillMutationLockAcquireFailure) as excinfo:
        with _spg.live_skill_repair_guard(
            "badger",
            target=approved,
            approved_existing_paths=[approved],
            mutation_paths=[approved],
        ):
            pass

    assert "symlink/junction" in str(excinfo.value.__cause__)


def test_initial_identity_disappearance_is_classified_with_cause(isolated_skill_roots, monkeypatch):
    approved = _skill(isolated_skill_roots / "capybara", "capybara")
    original_lstat = Path.lstat

    def disappearing_lstat(self):
        if self == approved:
            raise FileNotFoundError("gone for test")
        return original_lstat(self)

    monkeypatch.setattr(Path, "lstat", disappearing_lstat)
    with pytest.raises(_spg.SkillMutationLockAcquireFailure) as excinfo:
        with _spg.live_skill_repair_guard(
            "capybara",
            target=approved,
            approved_existing_paths=[approved],
            mutation_paths=[approved],
        ):
            pass

    assert isinstance(excinfo.value.__cause__, ValueError)
    assert isinstance(excinfo.value.__cause__.__cause__, FileNotFoundError)


def test_initial_identity_forbidden_link_is_classified_with_cause(isolated_skill_roots, tmp_path):
    target = _skill(tmp_path / "target" / "dingo", "dingo")
    link = isolated_skill_roots / "dingo"
    os.symlink(target, link)

    with pytest.raises(_spg.SkillMutationLockAcquireFailure) as excinfo:
        with _spg.live_skill_repair_guard(
            "dingo",
            target=link,
            approved_existing_paths=[link],
            mutation_paths=[link],
        ):
            pass

    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "symlink/junction" in str(excinfo.value.__cause__)


def test_initial_identity_read_failure_is_classified_with_cause(isolated_skill_roots, monkeypatch):
    approved = _skill(isolated_skill_roots / "echidna", "echidna")
    original_lstat = Path.lstat

    def failing_lstat(self):
        if self == approved:
            raise PermissionError("identity denied for test")
        return original_lstat(self)

    monkeypatch.setattr(Path, "lstat", failing_lstat)
    with pytest.raises(_spg.SkillMutationLockAcquireFailure) as excinfo:
        with _spg.live_skill_repair_guard(
            "echidna",
            target=approved,
            approved_existing_paths=[approved],
            mutation_paths=[approved],
        ):
            pass

    assert isinstance(excinfo.value.__cause__, ValueError)
    assert isinstance(excinfo.value.__cause__.__cause__, PermissionError)


def test_body_exception_remains_context_for_mutation_release_failure(isolated_skill_roots, monkeypatch):
    approved = _skill(isolated_skill_roots / "ferret", "ferret")
    failing_lock_path = _spg._target_lock_path(approved)
    factory = RecordingLockFactory(fail_on_release={failing_lock_path})
    monkeypatch.setattr(_spg, "_acquire_lock_at_path", factory)
    body_error = RuntimeError("body failed for test")

    with pytest.raises(_spg.SkillMutationLockReleaseFailure) as excinfo:
        with _spg.live_skill_repair_guard(
            "ferret",
            target=approved,
            approved_existing_paths=[approved],
            mutation_paths=[approved],
        ):
            raise body_error

    assert excinfo.value.__context__ is body_error
    assert _path_releases(factory) == [approved.resolve(strict=False)]


def test_body_exception_is_passed_to_global_release_failure(isolated_skill_roots, monkeypatch):
    approved = _skill(isolated_skill_roots / "goat", "goat")
    global_lock_path = _spg.normalized_name_lock_target("goat", anchor=approved)
    factory = RecordingLockFactory(fail_on_release={global_lock_path})
    monkeypatch.setattr(_spg, "_acquire_lock_at_path", factory)
    body_error = RuntimeError("body failed globally")

    with pytest.raises(_spg.SkillMutationLockReleaseFailure) as excinfo:
        with _spg.live_skill_repair_guard(
            "goat",
            target=approved,
            approved_existing_paths=[approved],
            mutation_paths=[approved],
        ):
            raise body_error

    assert excinfo.value.lock_path == global_lock_path
    assert excinfo.value.__context__ is body_error


def test_mutation_release_failure_remains_context_for_global_release_failure(isolated_skill_roots, monkeypatch):
    approved = _skill(isolated_skill_roots / "hyena", "hyena")
    mutation_lock_path = _spg._target_lock_path(approved)
    global_lock_path = _spg.normalized_name_lock_target("hyena", anchor=approved)
    factory = RecordingLockFactory(fail_on_release={mutation_lock_path, global_lock_path})
    monkeypatch.setattr(_spg, "_acquire_lock_at_path", factory)

    with pytest.raises(_spg.SkillMutationLockReleaseFailure) as excinfo:
        with _spg.live_skill_repair_guard(
            "hyena",
            target=approved,
            approved_existing_paths=[approved],
            mutation_paths=[approved],
        ):
            pass

    assert excinfo.value.lock_path == global_lock_path
    assert isinstance(excinfo.value.__context__, _spg.SkillMutationLockReleaseFailure)
    assert excinfo.value.__context__.lock_path == mutation_lock_path


def test_multiple_mutation_release_failures_retain_secondary_diagnostics(isolated_skill_roots, monkeypatch):
    approved = _skill(isolated_skill_roots / "ibex", "ibex")
    b = isolated_skill_roots / "b" / "ibex"
    c = isolated_skill_roots / "c" / "ibex"
    ordered = sorted([_spg._canonical_path(approved), _spg._canonical_path(b), _spg._canonical_path(c)], key=_spg._path_sort_key)
    failing = {_spg._target_lock_path(ordered[-1]), _spg._target_lock_path(ordered[-2])}
    factory = RecordingLockFactory(fail_on_release=failing)
    monkeypatch.setattr(_spg, "_acquire_lock_at_path", factory)

    with pytest.raises(_spg.SkillMutationLockReleaseFailure) as excinfo:
        with _spg.live_skill_repair_guard(
            "ibex",
            target=approved,
            approved_existing_paths=[approved],
            mutation_paths=[approved, b, c],
        ):
            pass

    assert _path_releases(factory) == list(reversed(ordered))
    assert excinfo.value.lock_path == _spg._target_lock_path(ordered[-1])
    assert [failure.lock_path for failure in excinfo.value.secondary_failures] == [
        _spg._target_lock_path(ordered[-2])
    ]


def test_successful_body_and_releases_remain_unchanged(isolated_skill_roots, monkeypatch):
    approved = _skill(isolated_skill_roots / "jaguar", "jaguar")
    factory = RecordingLockFactory()
    monkeypatch.setattr(_spg, "_acquire_lock_at_path", factory)

    with _spg.live_skill_repair_guard(
        "jaguar",
        target=approved,
        approved_existing_paths=[approved],
        mutation_paths=[approved],
    ):
        pass

    assert len(factory.acquired) == 2
    assert len(factory.released) == 2


def test_delete_guard_public_signature_and_locking_match_repair_path(isolated_skill_roots, monkeypatch):
    sig = inspect.signature(_spg.live_skill_delete_guard)
    assert list(sig.parameters) == [
        "name", "target", "approved_existing_paths", "mutation_paths", "identity_names",
    ]
    assert sig.parameters["target"].kind is inspect.Parameter.KEYWORD_ONLY

    approved = _skill(isolated_skill_roots / "lemur", "lemur")
    future = isolated_skill_roots / "future" / "lemur"
    factory = RecordingLockFactory()
    monkeypatch.setattr(_spg, "_acquire_lock_at_path", factory)

    with _spg.live_skill_delete_guard(
        "lemur",
        target=approved,
        approved_existing_paths=[approved],
        mutation_paths=[future, approved],
    ):
        pass

    expected = sorted({_spg._canonical_path(approved), _spg._canonical_path(future)}, key=_spg._path_sort_key)
    assert _path_acquisitions(factory) == expected


def test_delete_guard_unexpected_same_name_state_fails_closed(isolated_skill_roots):
    approved = _skill(isolated_skill_roots / "marten", "marten")
    _skill(isolated_skill_roots / "other" / "marten", "marten")

    with pytest.raises(_spg.SkillMutationLockAcquireFailure):
        with _spg.live_skill_delete_guard(
            "marten",
            target=approved,
            approved_existing_paths=[approved],
            mutation_paths=[approved],
        ):
            pass
