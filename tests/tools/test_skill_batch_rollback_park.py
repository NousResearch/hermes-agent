"""A batch rollback must not park anything loadable inside a skills root.

The failure path moves the half-applied skill aside while it copies the snapshot back.
That park used to be a SIBLING of the entry (``<name>.rollback-broken``) inside the
skills root and was deleted with ``shutil.rmtree`` — which REFUSES a symlink. So

* a process killed between the park and the copy-back left an entry inside the root
  that the loader offers as a skill (it walks a root following symlinked entries and
  prunes only ``EXCLUDED_SKILL_DIRS``); when the park's SKILL.md declares the same
  frontmatter ``name`` as the live skill, that name resolves twice in one root, and
* a park that WAS a symlink (a profile farm linking each skill into a shared tree)
  could never be removed at all — the rmtree raised and ``ignore_errors=True`` ate it.

Both parks also stayed forever: nothing ever collected them.
"""

import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SKILL_MD = "---\nname: probe\ndescription: probe skill\n---\n\n# Probe\n\nStep 1.\n"

# Runs a failing batch with ``copytree`` turned into a SIGKILL on the copy-back, so the
# park is left behind exactly as a killed host or a power cut would leave it.
_KILL_ON_RESTORE = '''
import os, shutil, signal, sys

repo, home = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
os.environ["HERMES_HOME"] = home
os.environ["HERMES_YOLO_MODE"] = "1"

import tools.skill_manager_tool as smt

_real_copytree = shutil.copytree
_calls = {"n": 0}


def kill_on_restore(src, dst, *a, **k):
    _calls["n"] += 1
    if _calls["n"] > 1:  # call 1 snapshots; the copy-back never finishes
        os.kill(os.getpid(), signal.SIGKILL)
    return _real_copytree(src, dst, *a, **k)


shutil.copytree = kill_on_restore
smt.skill_manage(action="", name="", operations=[
    {"name": "probe", "action": "patch", "old_string": "Step 1.", "new_string": "Step ONE."},
    {"name": "probe", "action": "write_file", "file_path": "bad/nope.md", "file_content": "x"},
])
sys.exit(3)  # unreachable: the batch reaches its rollback
'''


class TestRollbackParkLocation(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp(prefix="skmbatch_park_"))
        self.addCleanup(shutil.rmtree, self.home, True)
        self._env = {k: os.environ.get(k) for k in ("HERMES_HOME", "HERMES_YOLO_MODE")}
        self.addCleanup(self._restore_env)
        os.environ["HERMES_HOME"] = str(self.home)
        os.environ["HERMES_YOLO_MODE"] = "1"
        self.root = self.home / "skills"
        (self.root / "yoyodine").mkdir(parents=True)
        import tools.skill_manager_tool as smt
        importlib.reload(smt)
        self.smt = smt
        self.asides = self.home / ".skill-rollback-asides"

    def _restore_env(self):
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _loadable(self):
        """What the loader's own walk offers as a skill inside the root."""
        from agent.skill_utils import iter_skill_index_files
        return sorted(str(p.relative_to(self.root))
                      for p in iter_skill_index_files(self.root, "SKILL.md"))

    def _skill(self, name="probe"):
        entry = self.root / "yoyodine" / name
        entry.mkdir(parents=True)
        (entry / "SKILL.md").write_text(SKILL_MD)
        return entry

    def _parked(self):
        return sorted(p.name for p in self.asides.iterdir()) if self.asides.is_dir() else []

    def _failing_batch(self):
        """Patch the skill, then fail an op: the batch rolls every touched skill back."""
        return json.loads(self.smt.skill_manage(action="", name="", operations=[
            {"name": "probe", "action": "patch",
             "old_string": "Step 1.", "new_string": "Step ONE."},
            {"name": "probe", "action": "write_file",
             "file_path": "bad/nope.md", "file_content": "x"},
        ]))

    def test_rollback_restores_the_skill_and_leaves_only_the_skill_loadable(self):
        entry = self._skill()
        before = (entry / "SKILL.md").read_text()

        result = self._failing_batch()

        self.assertFalse(result["success"], result)
        self.assertEqual(self._loadable(), ["yoyodine/probe/SKILL.md"])
        self.assertEqual((entry / "SKILL.md").read_text(), before)
        self.assertEqual(self._parked(), [])

    def test_park_is_created_outside_every_skills_root(self):
        from tools import skill_manager_batch as smb

        entry = self._skill()

        park = smb._park_entry_aside(entry)

        self.assertTrue(park.exists(), park)
        self.assertFalse(park.is_relative_to(self.root), park)
        self.assertEqual(self._loadable(), [])
        self.assertEqual(self._parked(), [park.name])

    def test_a_symlinked_entry_park_lands_outside_the_root_and_removes(self):
        """A farm links each skill into a shared tree; rmtree could not delete that park."""
        from tools import skill_manager_batch as smb

        shared = self.home / "shared" / "probe"
        shared.mkdir(parents=True)
        (shared / "SKILL.md").write_text(SKILL_MD)
        entry = self.root / "yoyodine" / "probe"
        try:
            entry.symlink_to(shared)
        except OSError:  # pragma: no cover — Windows without developer mode
            self.skipTest("directory symlinks unavailable on this host")

        park = smb._park_entry_aside(entry)

        self.assertTrue(park.is_symlink(), park)
        self.assertFalse(park.is_relative_to(self.root), park)
        self.assertEqual(self._loadable(), [])
        smb._remove_path(park)
        self.assertFalse(park.is_symlink(), "the park survived a removal")

    def test_stale_park_is_swept_while_a_fresh_one_survives(self):
        from tools import skill_manager_batch as smb

        stale = smb._park_entry_aside(self._skill("probe"))
        old = time.time() - 3 * 24 * 3600
        os.utime(stale, (old, old), follow_symlinks=False)

        fresh = smb._park_entry_aside(self._skill("other"))

        self.assertFalse(stale.exists(), "an abandoned park must not stay forever")
        self.assertTrue(fresh.exists(), fresh)

    def test_interrupted_rollback_leaves_nothing_loadable_inside_the_root(self):
        self._skill()
        script = self.home / "kill_restore.py"
        script.write_text(_KILL_ON_RESTORE)
        proc = subprocess.run(
            [sys.executable, str(script), str(REPO_ROOT), str(self.home)],
            capture_output=True, text=True, timeout=180,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        )

        # A hard kill mid-restore: -SIGKILL on POSIX, TerminateProcess on Windows.
        self.assertNotEqual(proc.returncode, 0, proc.stderr)
        self.assertNotEqual(proc.returncode, 3, "the batch finished instead of dying mid-restore")
        if os.name == "posix":
            self.assertEqual(proc.returncode, -signal.SIGKILL, proc.stderr)
        self.assertEqual(self._loadable(), [],
                         "a killed restore left a loadable entry parked in the root")
        self.assertEqual(len(self._parked()), 1, self._parked())


if __name__ == "__main__":
    unittest.main()
