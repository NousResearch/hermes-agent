"""FileSyncManager keeps the owning profile's HERMES_HOME across thread hops.

Remote backends tear down from the inactive-env reaper thread, which carries no profile scope.
Without the binding, get_files_fn (``iter_sync_files`` resolves ``get_hermes_home()`` lazily),
the config read and the lock path all fall back to the launch profile, so a secondary profile's
sync-back lands in the default profile's skills tree.
"""

import io
import tarfile
import threading
from pathlib import Path
from unittest.mock import MagicMock

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from tools.environments.file_sync import FileSyncManager

REMOTE = "/root/.hermes"


def _lazy_skill_files():
    """Like iter_skills_files: the host side is resolved from get_hermes_home() at call time."""
    skills = get_hermes_home() / "skills"
    return [(str(p), f"{REMOTE}/skills/{p.relative_to(skills).as_posix()}") for p in sorted(skills.rglob("*.md"))]


def _download_of(files: dict[str, bytes]):
    def download(dest: Path):
        with tarfile.open(dest, "w") as tar:
            for arcname, content in files.items():
                info = tarfile.TarInfo(name=arcname)
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
    return download


def _owner_home(tmp_path: Path) -> Path:
    home = tmp_path / "profiles" / "reviewer"
    (home / "skills" / "review").mkdir(parents=True)
    (home / "skills" / "review" / "SKILL.md").write_text("pushed", encoding="utf-8")
    return home


def _in_fresh_thread(fn):
    """Run *fn* on a new thread, as the reaper does: no ContextVar is inherited."""
    errors = []

    def run():
        try:
            fn()
        except Exception as exc:  # surfaced to the test below
            errors.append(exc)

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout=30)
    assert not errors, errors


def test_sync_back_from_unscoped_thread_writes_into_owning_profile(tmp_path):
    owner = _owner_home(tmp_path)
    default_home = get_hermes_home()
    assert default_home != owner

    token = set_hermes_home_override(owner)
    try:
        mgr = FileSyncManager(
            get_files_fn=_lazy_skill_files, upload_fn=MagicMock(), delete_fn=MagicMock(),
            bulk_download_fn=_download_of({
                "root/.hermes/skills/review/SKILL.md": b"pushed",
                "root/.hermes/skills/review/references/new.md": b"learned on the remote",
            }))
        mgr.sync(force=True)
    finally:
        reset_hermes_home_override(token)

    _in_fresh_thread(mgr.sync_back)

    assert (owner / "skills" / "review" / "references" / "new.md").read_bytes() == b"learned on the remote"
    assert not (default_home / "skills" / "review").exists()
    assert (owner / ".sync.lock").exists()


def test_sync_from_unscoped_thread_reads_owning_profile_files(tmp_path):
    owner = _owner_home(tmp_path)
    upload = MagicMock()

    token = set_hermes_home_override(owner)
    try:
        mgr = FileSyncManager(get_files_fn=_lazy_skill_files, upload_fn=upload, delete_fn=MagicMock())
    finally:
        reset_hermes_home_override(token)

    _in_fresh_thread(lambda: mgr.sync(force=True))

    # Uploads go through a staged snapshot, so only the remote side names the source file.
    assert [c.args[1] for c in upload.call_args_list] == [f"{REMOTE}/skills/review/SKILL.md"]


def test_binding_does_not_leak_into_the_calling_thread(tmp_path):
    owner = _owner_home(tmp_path)
    before = get_hermes_home()

    token = set_hermes_home_override(owner)
    try:
        mgr = FileSyncManager(get_files_fn=_lazy_skill_files, upload_fn=MagicMock(), delete_fn=MagicMock())
    finally:
        reset_hermes_home_override(token)

    mgr.sync(force=True)

    assert get_hermes_home() == before
