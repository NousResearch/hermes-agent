"""Invariant tests: the sanitized (symlink-stripped) skills copy must be daemon-visible.

When a skills tree contains symlinks, ``credential_files._safe_skills_path`` stages a
regular-files-only copy. That copy is a Docker bind source, so it must live under
HERMES_HOME — the only tree ``docker_host_mapping`` can translate for the host daemon;
a copy staged under the process tempdir resolves to an empty host sibling and the
skills silently vanish inside the sandbox container.

The staging copy is also random-per-process, so the reuse fingerprint must keep
placeholdering it (basename-keyed now, not tempdir-keyed) or cross-process container
reuse never matches for symlinked-skill users.
"""

import os
import tempfile
import types
from pathlib import Path

import pytest

import tools.credential_files as cf
import tools.environments.docker as docker_env


def _make_symlink(link, target) -> None:
    """Create *link -> target*, skipping the test where the platform forbids symlinks."""
    try:
        os.symlink(str(target), str(link))
    except OSError as exc:  # Windows without SeCreateSymbolicLinkPrivilege (EPERM/EACCES)
        pytest.skip(f"platform does not allow unprivileged symlinks: {exc}")


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    """A HERMES_HOME with a real skills dir; the credential_files seam stays patchable."""
    home = tmp_path / "hermes-home"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setattr(cf, "get_hermes_home", lambda: home)
    return home


def _seed_skill_tree(skills_dir, tmp_path):
    """Regular script + a symlink pointing at a secret outside the tree."""
    payload = b"print('skill payload')\n"
    (skills_dir / "script.py").write_bytes(payload)
    secret = tmp_path / "outside-secret.txt"
    secret.write_bytes(b"DO-NOT-MOUNT\n")
    _make_symlink(skills_dir / "link-to-secret", secret)
    return payload


class TestSafeSkillsStaging:
    def test_symlink_free_tree_returned_unchanged(self, fake_home):
        """Zero-overhead direct mount is preserved for the common (symlink-free) case."""
        (fake_home / "skills" / "plain.py").write_bytes(b"x\n")
        assert cf._safe_skills_path(fake_home / "skills") == str(fake_home / "skills")

    def test_staging_path_lives_under_hermes_home(self, fake_home, tmp_path):
        _seed_skill_tree(fake_home / "skills", tmp_path)
        staged = Path(cf._safe_skills_path(fake_home / "skills"))
        assert staged != fake_home / "skills"
        assert staged.is_relative_to(fake_home)
        assert staged.name.startswith(cf.SAFE_SKILLS_PREFIX)

    def test_symlinks_absent_from_sanitized_copy(self, fake_home, tmp_path):
        _seed_skill_tree(fake_home / "skills", tmp_path)
        staged = Path(cf._safe_skills_path(fake_home / "skills"))
        link = staged / "link-to-secret"
        assert not link.exists() and not link.is_symlink()
        assert not any(p.is_symlink() for p in staged.rglob("*"))

    def test_regular_file_bytes_match_original(self, fake_home, tmp_path):
        payload = _seed_skill_tree(fake_home / "skills", tmp_path)
        staged = Path(cf._safe_skills_path(fake_home / "skills"))
        assert (staged / "script.py").read_bytes() == payload

    def test_staging_dir_traversable_by_container_user(self, fake_home, tmp_path):
        if os.name != "posix":
            pytest.skip("POSIX mode bits are the bind-mount visibility contract")
        _seed_skill_tree(fake_home / "skills", tmp_path)
        staged = Path(cf._safe_skills_path(fake_home / "skills"))
        assert staged.stat().st_mode & 0o777 == 0o755


class TestDockerMountResolution:
    def _mount_args_for_staged_skills(self, fake_home, monkeypatch):
        """Drive the real sanitizer + the real mount-arg builder; only the mapping
        resolution (which probes the gateway's own container) is stubbed. Returns
        (mount_specs, staged_path) from ONE staging call — a second sanitizer call
        would replace the staged tree and desynchronize the assertion."""
        staged = Path(cf._safe_skills_path(fake_home / "skills"))
        monkeypatch.setattr(
            cf, "get_skills_directory_mount",
            lambda container_base="/root/.hermes": [
                cf._mount(staged, f"{container_base.rstrip('/')}/skills")])
        monkeypatch.setattr(cf, "get_credential_file_mounts", list)
        monkeypatch.setattr(cf, "get_cache_directory_mounts", list)
        monkeypatch.setattr(
            docker_env.docker_host_mapping, "resolve_host_mapping",
            lambda: {str(fake_home): "/host/data"})
        args = docker_env._readonly_skill_mount_args()
        specs = [args[i + 1] for i, a in enumerate(args) if a == "-v"]
        return specs, staged

    def test_sanitized_staging_maps_to_host_view_and_is_ro(self, fake_home, tmp_path, monkeypatch):
        _seed_skill_tree(fake_home / "skills", tmp_path)
        specs, staged = self._mount_args_for_staged_skills(fake_home, monkeypatch)
        skills_specs = [s for s in specs if s.endswith(":/root/.hermes/skills:ro")]
        assert len(skills_specs) == 1
        source = skills_specs[0].split(":")[0]
        assert source == f"/host/data/{staged.name}"
        assert not source.startswith("/tmp"), "unmapped tempdir source reaches the daemon empty"

    def test_plain_skills_dir_still_maps_as_before(self, fake_home, monkeypatch):
        (fake_home / "skills" / "plain.py").write_bytes(b"x\n")
        specs, _ = self._mount_args_for_staged_skills(fake_home, monkeypatch)
        assert "/host/data/skills:/root/.hermes/skills:ro" in specs


class TestReuseLabelStability:
    """The staged copy is random-per-process; the fingerprint must keep placeholdering
    it — now via the staging prefix under HERMES_HOME, not only the legacy tempdir."""

    @pytest.fixture(autouse=True)
    def _outside_real_tempdir(self, monkeypatch, tmp_path):
        """Neutralize the legacy tempdir rule so the outcome tests exercise path
        identity, not where pytest happened to root tmp_path."""
        monkeypatch.setattr(
            docker_env, "tempfile",
            types.SimpleNamespace(gettempdir=lambda: str(tmp_path / "definitely-not-a-temp")))

    def _fingerprint(self, home, mount):
        return docker_env._reuse_environment_fingerprint(
            image="img", mount_args=["-v", mount], hermes_home=home)

    def test_volatile_staging_keeps_fingerprint_identical(self, fake_home):
        home = str(fake_home)
        fp_a = self._fingerprint(home, f"{home}/.hermes-skills-safe-aaa:/root/.hermes/skills:ro")
        fp_b = self._fingerprint(home, f"{home}/.hermes-skills-safe-bbb:/root/.hermes/skills:ro")
        assert fp_a == fp_b

    def test_stable_mount_path_change_still_forces_fresh_container(self, fake_home):
        home = str(fake_home)
        fp_a = self._fingerprint(home, f"{home}/skills:/root/.hermes/skills:ro")
        fp_b = self._fingerprint(home, f"{home}/other:/root/.hermes/skills:ro")
        assert fp_a != fp_b


def test_legacy_tempdir_staging_still_recognized():
    """Backward-compat: pre-relocation staging dirs under the process tempdir must
    keep their volatile placeholder (runs against the real tempfile module)."""
    staged = tempfile.mkdtemp(prefix="hermes-skills-safe-")
    try:
        assert docker_env._is_volatile_mount_spec(f"{staged}:/root/.hermes/skills:ro") is True
    finally:
        os.rmdir(staged)
