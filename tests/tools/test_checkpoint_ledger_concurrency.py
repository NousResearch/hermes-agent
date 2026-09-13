import threading

from tools import checkpoint_manager as checkpoints


def test_concurrent_agent_writes_preserve_every_ledger_entry(tmp_path, monkeypatch):
    manager = checkpoints.CheckpointManager(enabled=True)
    monkeypatch.setattr(manager, "get_working_dir_for_path", lambda _path: str(tmp_path))
    monkeypatch.setattr(checkpoints, "_store_path", lambda: tmp_path / "store")
    monkeypatch.setattr(checkpoints, "_hash_file", lambda path: f"hash:{path.name}")
    original_load = checkpoints._load_ledger
    first_loaded = threading.Event()
    release_first = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def delayed_first_load(store, directory_hash):
        nonlocal calls
        data = original_load(store, directory_hash)
        with calls_lock:
            calls += 1
            first = calls == 1
        if first:
            first_loaded.set()
            assert release_first.wait(timeout=5)
        return data

    monkeypatch.setattr(checkpoints, "_load_ledger", delayed_first_load)
    first = threading.Thread(target=manager.record_agent_write, args=(str(tmp_path / "first.py"),))
    second = threading.Thread(target=manager.record_agent_write, args=(str(tmp_path / "second.py"),))
    first.start()
    assert first_loaded.wait(timeout=5)
    second.start()
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    ledger = original_load(checkpoints._store_path(), checkpoints._project_hash(str(tmp_path)))
    assert set(ledger) == {str(tmp_path / "first.py"), str(tmp_path / "second.py")}
