"""Tombstoned named-profile homes must never be recreated by runtime mkdir writers.

Regression for the ghost-profile class (#69934): two ungated
``mkdir(parents=True, exist_ok=True)`` writers — ``hermes_state.divert_session_transcript_jsonl``
(the state.db-replaced interrupt flush) and ``hermes_constants.get_scratch_dir`` (reached from
the child-env scratch export) — rematerialized a deleted profile's home tree, so a live cron
child then ran under a home ``profile delete`` had already retired. The gated twin
``mkdir_under_hermes_home`` already refuses; these writers must route through it.
"""

import shutil

import pytest

import hermes_constants
import hermes_state


def _tombstoned_home(tmp_path, *, keep_home=False):
    """A profiles/ghost home retired the way ``profile delete`` does it (tombstone first)."""
    root = tmp_path / ".hermes"
    home = root / "profiles" / "ghost"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("{}\n")
    tombstones = root / "profiles" / ".deleted"
    tombstones.mkdir(parents=True, exist_ok=True)
    (tombstones / "ghost").write_text("deleted\n")
    if not keep_home:
        shutil.rmtree(home)  # profile delete: tombstone written, home removed
    return root, home


class TestDivertRefusesTombstonedHome:
    def test_interrupt_flush_does_not_recreate_the_home(self, tmp_path, monkeypatch):
        root, home = _tombstoned_home(tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(home))
        with pytest.raises(FileNotFoundError):
            hermes_state.divert_session_transcript_jsonl(
                "20260928_030941_83dc6c", [{"role": "assistant", "content": "interrupted"}])
        assert not home.exists(), "tombstoned profile home was rematerialized"

    def test_tombstone_alone_refuses_even_while_rmtree_is_in_flight(self, tmp_path, monkeypatch):
        # profile delete writes the tombstone BEFORE rmtree: the delete-in-progress arm is
        # the only case that exercises named_profile_is_deleted rather than not exists().
        root, home = _tombstoned_home(tmp_path, keep_home=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        with pytest.raises(FileNotFoundError):
            hermes_state.divert_session_transcript_jsonl(
                "sid", [{"role": "assistant", "content": "interrupted"}])

    def test_live_home_still_diverts(self, tmp_path, monkeypatch):
        root = tmp_path / ".hermes"
        (root / "profiles").mkdir(parents=True)
        home = root / "profiles" / "alive"
        home.mkdir()
        (home / "config.yaml").write_text("{}\n")
        monkeypatch.setenv("HERMES_HOME", str(home))
        path = hermes_state.divert_session_transcript_jsonl(
            "sid", [{"role": "assistant", "content": "hello"}])
        assert path is not None and path.exists()
        assert "hello" in path.read_text()


class TestScratchDirRefusesTombstonedHome:
    def test_get_scratch_dir_does_not_recreate_the_home(self, tmp_path):
        root, home = _tombstoned_home(tmp_path)
        with pytest.raises(FileNotFoundError):
            hermes_constants.get_scratch_dir(home)
        assert not home.exists(), "tombstoned profile home was rematerialized"

    def test_healthy_named_profile_still_gets_scratch(self, tmp_path):
        root = tmp_path / ".hermes"
        (root / "profiles").mkdir(parents=True)
        home = root / "profiles" / "alive"
        home.mkdir()
        (home / "config.yaml").write_text("{}\n")
        scratch = hermes_constants.get_scratch_dir(home)
        assert scratch == home / "cache" / "scratch"
        assert scratch.is_dir()

    def test_bare_mkdir_failure_keeps_the_old_swallow_contract(self, tmp_path, monkeypatch):
        # Only the gate's refusal propagates; an ordinary mkdir OSError (EROFS etc.)
        # must still be swallowed and the path returned, as before this change.
        root = tmp_path / ".hermes"
        (root / "profiles").mkdir(parents=True)
        home = root / "profiles" / "alive"
        home.mkdir()
        (home / "config.yaml").write_text("{}\n")
        real_mkdir = hermes_constants.Path.mkdir

        def fake_mkdir(self, *a, **k):
            raise OSError("simulated EROFS")

        monkeypatch.setattr(hermes_constants.Path, "mkdir", fake_mkdir)
        try:
            scratch = hermes_constants.get_scratch_dir(home)
        finally:
            monkeypatch.setattr(hermes_constants.Path, "mkdir", real_mkdir)
        assert scratch == home / "cache" / "scratch"

    def test_child_env_export_survives_a_dead_routed_home(self, tmp_path, monkeypatch):
        # apply_scratch_tmp_env points TMPDIR for children routed to another home; a dead
        # routed home must leave the child on the OS default, never resurrect the home —
        # and must not leave a stale Hermes-exported TMPDIR from the previous home behind.
        root, home = _tombstoned_home(tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("HERMES_SCRATCH_DIR", raising=False)
        for key in hermes_constants.SCRATCH_TMP_ENV_VARS:
            monkeypatch.delenv(key, raising=False)
        stale = str(tmp_path / "previous-home" / "cache" / "scratch")
        env = {"HERMES_HOME": str(home),
               hermes_constants.SCRATCH_DIR_MARKER_ENV: stale,
               **{k: stale for k in hermes_constants.SCRATCH_TMP_ENV_VARS}}
        assert hermes_constants.apply_scratch_tmp_env(env) is False
        assert not home.exists(), "tombstoned profile home was rematerialized"
        for key in hermes_constants.SCRATCH_TMP_ENV_VARS:
            assert env.get(key, "") == "", f"stale Hermes-owned {key} survived the failure"
        assert env.get(hermes_constants.SCRATCH_DIR_MARKER_ENV, "") == ""

    def test_child_env_export_leaves_user_set_temp_alone(self, tmp_path, monkeypatch):
        root, home = _tombstoned_home(tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(home))
        env = {"HERMES_HOME": str(home), "TMPDIR": "/home/user/mytemp"}
        assert hermes_constants.apply_scratch_tmp_env(env) is False
        assert env["TMPDIR"] == "/home/user/mytemp"
        assert not home.exists()
