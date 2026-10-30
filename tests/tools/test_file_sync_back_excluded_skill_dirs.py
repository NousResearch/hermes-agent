"""Sync-back must not bring back skill trees the upload never sends.

The upload prunes ``EXCLUDED_SKILL_DIRS`` (``.hub``, ``.curator_backups``, ``node_modules``, ...)
from every skills directory, so a copy of those trees on the remote is never the host's: it was
left by an older build or by a Hermes install on that machine. Sync-back infers a host path for
any new remote file under a mapped directory, so without a guard it copied those trees into the
host's skills directory and replaced the host's ``.hub/lock.json`` with the remote's.
"""

import tarfile
from pathlib import Path

from tools.environments.file_sync import FileSyncManager

REMOTE_HOME = "/home/agent/.hermes"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_sync_back_skips_skill_bookkeeping_and_dependency_trees(tmp_path):
    skills = tmp_path / "home" / "skills"
    manifest = _write(skills / ".bundled_manifest", "manifest")
    skill = _write(skills / "tool" / "SKILL.md", "---\nname: tool\n---\n")
    hub_lock = _write(skills / ".hub" / "lock.json", '{"installed": ["host"]}')
    mapping = [(str(manifest), f"{REMOTE_HOME}/skills/.bundled_manifest"),
               (str(skill), f"{REMOTE_HOME}/skills/tool/SKILL.md")]

    remote = {
        "skills/.hub/lock.json": '{"installed": ["stale remote copy"]}',
        "skills/.curator_backups/2026-01-01/skills.tar.gz": "old backup",
        "skills/tool/node_modules/dep/index.js": "dependency",
        "skills/tool/scripts/new.py": "print('written on the remote')",
    }

    def download(destination: Path) -> None:
        source = tmp_path / "remote"
        with tarfile.open(destination, "w") as archive:
            for relative, text in remote.items():
                file = _write(source / relative, text)
                archive.add(file, arcname=f"{REMOTE_HOME.lstrip('/')}/{relative}")

    manager = FileSyncManager(get_files_fn=lambda: mapping, upload_fn=lambda *args: None,
                              delete_fn=lambda paths: None, bulk_download_fn=download)
    manager.sync(force=True)
    manager.sync_back(hermes_home=tmp_path / "home")

    assert hub_lock.read_text(encoding="utf-8") == '{"installed": ["host"]}'
    assert not (skills / ".curator_backups").exists()
    assert not (skills / "tool" / "node_modules").exists()
    assert (skills / "tool" / "scripts" / "new.py").read_text(encoding="utf-8") == "print('written on the remote')"
