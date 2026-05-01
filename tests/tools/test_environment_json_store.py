import threading
import time
from pathlib import Path

from tools.environments import base as base_env
from tools.environments import modal, singularity, vercel_sandbox


def test_update_json_store_preserves_concurrent_updates(tmp_path):
    store = tmp_path / "snapshots.json"
    start = threading.Barrier(4)

    def worker(index: int) -> None:
        start.wait(timeout=5)

        def _mutate(data: dict) -> bool:
            time.sleep(0.01)
            data[f"task-{index}"] = f"snap-{index}"
            return True

        base_env._update_json_store(store, _mutate)

    threads = [
        threading.Thread(target=worker, args=(index,))
        for index in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert base_env._load_json_store(store) == {
        f"task-{index}": f"snap-{index}"
        for index in range(4)
    }


def test_modal_store_direct_snapshot_uses_locked_json_update(tmp_path, monkeypatch):
    store = tmp_path / "modal_snapshots.json"
    calls: list[Path] = []

    def fake_update(path: Path, update_fn):
        calls.append(path)
        data = {"task-a": "im-legacy"}
        changed = update_fn(data)
        assert changed is True
        assert data == {"direct:task-a": "im-fresh"}
        return changed

    monkeypatch.setattr(modal, "_SNAPSHOT_STORE", store)
    monkeypatch.setattr(modal, "_update_json_store", fake_update)

    modal._store_direct_snapshot("task-a", "im-fresh")

    assert calls == [store]


def test_vercel_store_snapshot_uses_locked_json_update(tmp_path, monkeypatch):
    store = tmp_path / "vercel_snapshots.json"
    calls: list[Path] = []

    def fake_update(path: Path, update_fn):
        calls.append(path)
        data = {"task-a": "snap-old"}
        changed = update_fn(data)
        assert changed is True
        assert data == {"task-a": "snap-new"}
        return changed

    monkeypatch.setattr(vercel_sandbox, "_snapshot_store_path", lambda: store)
    monkeypatch.setattr(vercel_sandbox, "_update_json_store", fake_update)

    vercel_sandbox._store_snapshot("task-a", "snap-new")

    assert calls == [store]


def test_singularity_store_snapshot_uses_locked_json_update(tmp_path, monkeypatch):
    store = tmp_path / "singularity_snapshots.json"
    calls: list[Path] = []

    def fake_update(path: Path, update_fn):
        calls.append(path)
        data = {"other-task": "/scratch/keep"}
        changed = update_fn(data)
        assert changed is True
        assert data == {
            "other-task": "/scratch/keep",
            "task-a": "/scratch/overlay-a",
        }
        return changed

    monkeypatch.setattr(singularity, "_SNAPSHOT_STORE", store)
    monkeypatch.setattr(singularity, "_update_json_store", fake_update)

    singularity._store_snapshot("task-a", "/scratch/overlay-a")

    assert calls == [store]
