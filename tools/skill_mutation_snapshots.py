"""Batch before-images for actual physical publishes and lexical unlinks, including external links."""

from contextvars import ContextVar
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

_active_journal: ContextVar["BeforeImages | None"] = ContextVar("skill_batch_before_images", default=None)


class BeforeImages:
    def __init__(self, root, protected=()):
        self.root = Path(tempfile.mkdtemp(prefix=".publishes-", dir=root))
        self.files = {}
        self.protected = tuple(Path(p).resolve() for p in protected)

    def remember(self, path, *, follow_leaf):
        path = Path(path)
        target = path.resolve() if follow_leaf else path.parent.resolve() / path.name
        if any(target.is_relative_to(root) for root in self.protected):
            return  # Package snapshots already own these bytes/topology, including failed-copy recovery.
        if target in self.files:
            return
        stop = target.parent
        while not stop.exists() and stop.parent != stop:
            stop = stop.parent
        backup = None
        if target.exists() or target.is_symlink():
            backup = self.root / hashlib.sha256(os.fsencode(target)).hexdigest()
            shutil.copy2(target, backup, follow_symlinks=False)
        self.files[target] = backup, stop

    def restore(self):
        errors = []
        for target, (backup, stop) in self.files.items():
            try:
                if backup is None:
                    target.unlink(missing_ok=True)
                    parent = target.parent
                    while parent != stop and parent.parent != parent:
                        if parent.exists():
                            try:
                                parent.rmdir()
                            except OSError:
                                break  # Never delete another consumer's files.
                        parent = parent.parent
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryDirectory(prefix=".rollback-", dir=target.parent) as tmp:
                        restored = Path(tmp) / "entry"
                        shutil.copy2(backup, restored, follow_symlinks=False)
                        os.replace(restored, target)  # Restore the pinned entry, not a newly resolved alias.
            except OSError as exc:
                errors.append(f"ROLLBACK FAILED for physical target '{target}': {exc}; backup '{backup}'")
        return "; ".join(errors)


def record_before_mutation(path, *, follow_leaf=True):
    journal = _active_journal.get()
    if journal is not None:
        journal.remember(path, follow_leaf=follow_leaf)
