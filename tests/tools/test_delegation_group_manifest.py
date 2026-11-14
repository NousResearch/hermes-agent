"""A group finishing must not mark its still-running siblings complete."""
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from tools.delegation_live_log import create_live_transcripts, update_manifest_statuses


@pytest.mark.parametrize("last_status", ["completed", "error", "interrupted"])
def test_manifest_completes_only_after_every_group_settles(tmp_path, last_status):
    ident, _, paths = create_live_transcripts([{"goal": "first group"}, {"goal": "second group"}], home=tmp_path)
    manifest = Path(paths[0]).parent / "manifest.json"
    update_manifest_statuses(ident, [{"task_index": 0, "status": "completed"}], home=tmp_path)
    first = json.loads(manifest.read_text(encoding="utf-8"))
    assert [t["status"] for t in first["tasks"]] == ["completed", "running"]
    assert "completed" not in first

    update_manifest_statuses(ident, [{"task_index": 1, "status": last_status}], home=tmp_path)
    final = json.loads(manifest.read_text(encoding="utf-8"))
    assert final["completed"]
    assert [t["status"] for t in final["tasks"]] == ["completed", last_status]


def test_concurrent_groups_preserve_every_sibling_status(tmp_path):
    count = 12
    ident, _, paths = create_live_transcripts([{"goal": f"group {i}"} for i in range(count)], home=tmp_path)
    ready = threading.Barrier(count)

    def finish(index):
        ready.wait(timeout=5)
        update_manifest_statuses(ident, [{"task_index": index, "status": "completed"}], home=tmp_path)

    with ThreadPoolExecutor(max_workers=count) as executor:
        list(executor.map(finish, range(count)))
    final = json.loads((Path(paths[0]).parent / "manifest.json").read_text(encoding="utf-8"))
    assert all(t["status"] == "completed" for t in final["tasks"])
    assert final["completed"]
