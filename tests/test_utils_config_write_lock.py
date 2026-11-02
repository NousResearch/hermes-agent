"""Concurrent config.yaml writers must not drop each other's keys.

Regression for #147939: single-key writes (``write_config_key`` /
``atomic_roundtrip_yaml_update``) had no lock at all — neither thread nor
cross-process — and a grouped write applied its keys one replace at a
time, so a writer racing a grouped write could lose a key or leave a
half-applied group behind. The first test forces exactly that
interleaving; the second is a real-subprocess stress check.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading

import pytest
import hermes_yaml as yaml


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    """A config.yaml under the isolated HERMES_HOME, so the writer seam
    accepts its parent directory."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    path = home / "config.yaml"
    path.write_text(
        "model:\n  provider: old-provider\n  default: old-model\n"
        "display:\n  personality: old\n",
        encoding="utf-8",
    )
    return path


def _read(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _race_second_writer_into_first_writers_dump(monkeypatch, first, second, timeout: float = 1.5):
    """Deterministic lost-write interleaving, mirroring
    tests/hermes_cli/test_config_rmw_lock.py: ``first`` runs until its
    first ``_roundtrip_dump``, then ``second`` is started and ``first``
    waits for ``second`` to load before dumping. Unlocked, ``second``
    loads the stale document and its dump is erased by ``first``'s
    stale dump. With the write lock, ``second`` blocks before its load,
    ``first``'s wait times out, and the two writes serialize."""
    import utils as utils_mod

    real_dump, real_load = utils_mod._roundtrip_dump, utils_mod._roundtrip_load
    first_at_dump, second_loaded = threading.Event(), threading.Event()
    gate = threading.Lock()

    def gated_dump(*args, **kwargs):
        if not first_at_dump.is_set():
            first_at_dump.set()
            second_loaded.wait(timeout)
            with gate:
                return real_dump(*args, **kwargs)
        return real_dump(*args, **kwargs)

    def spied_load(*args, **kwargs):
        result = real_load(*args, **kwargs)
        if first_at_dump.is_set() and not gate.locked():
            second_loaded.set()
        return result

    monkeypatch.setattr(utils_mod, "_roundtrip_dump", gated_dump)
    monkeypatch.setattr(utils_mod, "_roundtrip_load", spied_load)

    results: list = [None, None]
    threads = [threading.Thread(target=lambda: results.__setitem__(0, first())),
               threading.Thread(target=lambda: results.__setitem__(1, second()))]
    threads[0].start()
    assert first_at_dump.wait(30), "first writer never reached _roundtrip_dump"
    threads[1].start()
    for t in threads:
        t.join(timeout=60)
    return results


def test_grouped_model_write_and_single_key_writer_both_survive(config_path, monkeypatch):
    """A grouped model switch (default + provider, one write) racing a
    single-key write: all three keys survive, and the model route is
    never half-applied (both new, never one old + one new)."""
    from hermes_cli.config_backend import write_config_key, write_config_keys

    grouped = lambda: write_config_keys(
        config_path, {"model.default": "new-model", "model.provider": "new-provider"})
    single = lambda: write_config_key(config_path, "display.personality", "canary")

    _race_second_writer_into_first_writers_dump(monkeypatch, grouped, single)

    on_disk = _read(config_path)
    assert on_disk["display"]["personality"] == "canary"
    assert on_disk["model"]["default"] == "new-model"
    assert on_disk["model"]["provider"] == "new-provider"


def test_subprocess_single_key_writers_all_survive(config_path, monkeypatch):
    """Eight real subprocesses writing distinct keys at once: every key
    lands. A thread lock alone cannot pass this — it needs the
    cross-process lock."""
    import hermes_cli.config_backend as backend_mod

    root = str(backend_mod.__file__).rsplit("/hermes_cli/", 1)[0]
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from hermes_cli.config_backend import write_config_key; "
        "write_config_key(sys.argv[2], sys.argv[3], sys.argv[4])"
    )
    env = dict(os.environ, HERMES_HOME=str(config_path.parent), PYTHONPATH=root)
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", code, root, str(config_path), f"race.key{i}", f"value{i}"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        for i in range(8)
    ]
    for p in procs:
        _, err = p.communicate(timeout=120)
        assert p.returncode == 0, err

    on_disk = _read(config_path)
    for i in range(8):
        assert on_disk["race"][f"key{i}"] == f"value{i}"
