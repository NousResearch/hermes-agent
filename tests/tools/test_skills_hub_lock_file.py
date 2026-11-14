"""skills/.hub/lock.json and taps.json survive concurrent writers from different processes.

The desktop/TUI skill install RPC and the CLI ``hermes skills install/uninstall`` rewrite the same
lock.json from different processes. An in-place rewrite left it empty between truncate and write
(a torn read: the next ``record_install`` loaded ``EMPTY``); an unlocked load → mutate → save let
the last writer put back the snapshot it loaded, dropping the provenance of every skill another
process recorded meanwhile.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from tests.file_write_observer import observe_file_writes
from tools.skills_hub import HubLockFile, TapsManager

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WRITES_PER_PROCESS = 150

_WRITER = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    from tools.skills_hub import HubLockFile, TapsManager

    hub, tag, count, go = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    while not go.exists():
        time.sleep(0.001)
    lock, taps = HubLockFile(hub / "lock.json"), TapsManager(hub / "taps.json")
    for i in range(count):
        name = f"{tag}-{i}"
        lock.record_install(
            name=name, source="github", identifier=f"o/r/{name}", trust_level="community",
            scan_verdict="safe", skill_hash="sha256:0", install_path=name, files=["SKILL.md"],
        )
        taps.add(f"o/{name}")
""")


def _seed(lock: HubLockFile, names: list[str]) -> None:
    lock.save({"version": 1, "installed": {
        name: {"source": "github", "identifier": f"o/r/{name}", "install_path": name, "files": ["SKILL.md"]}
        for name in names
    }})


def test_concurrent_processes_keep_every_install_and_tap(tmp_path):
    hub = tmp_path / "skills" / ".hub"
    hub.mkdir(parents=True)
    go = tmp_path / "go"
    writers = [
        subprocess.Popen(
            [sys.executable, "-c", _WRITER, str(hub), tag, str(_WRITES_PER_PROCESS), str(go)],
            cwd=_REPO_ROOT, stderr=subprocess.PIPE, text=True,
        )
        for tag in ("cli", "desktop")
    ]
    go.touch()
    for writer in writers:
        _, stderr = writer.communicate(timeout=120)
        assert writer.returncode == 0, stderr

    expected = sorted(f"{tag}-{i}" for tag in ("cli", "desktop") for i in range(_WRITES_PER_PROCESS))
    assert sorted(HubLockFile(hub / "lock.json").load()["installed"]) == expected
    assert sorted(t["repo"] for t in TapsManager(hub / "taps.json").load()) == [f"o/{name}" for name in expected]


def test_reader_mid_save_sees_complete_hub_state(tmp_path):
    lock_path = tmp_path / "lock.json"
    taps_path = tmp_path / "taps.json"
    lock = HubLockFile(lock_path)
    taps = TapsManager(taps_path)
    _seed(lock, ["alpha", "beta"])
    taps.save([{"repo": "o/one", "path": "skills/"}])
    seen = []

    with observe_file_writes(lambda: seen.append((sorted(HubLockFile(lock_path).load()["installed"]),
                                                  [t["repo"] for t in TapsManager(taps_path).load()]))):
        lock.record_uninstall("beta")
        taps.add("o/two")

    assert seen
    assert all(installed and repos for installed, repos in seen), seen
    assert sorted(lock.load()["installed"]) == ["alpha"]
    assert [t["repo"] for t in taps.load()] == ["o/one", "o/two"]
