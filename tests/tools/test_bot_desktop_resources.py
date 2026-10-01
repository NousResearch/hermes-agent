"""Bot Desktop memory gate: the screen is refused, with a reason clients can show, while the host
cannot spare ``bot_desktop.min_free_memory_mb``; a host we cannot read is never treated as small."""

from __future__ import annotations

import pytest

from tools.bot_desktop import resources, runtime


def test_start_refuses_and_status_explains_when_memory_is_short(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "state_dir", lambda: tmp_path / "bd")
    monkeypatch.setattr(runtime, "missing_binaries", lambda: [])
    monkeypatch.setattr(runtime, "_launcher_pid", lambda: None)
    monkeypatch.setattr(resources, "min_free_mb", lambda: 1536)
    monkeypatch.setattr(resources, "memory_info", lambda: resources.MemoryInfo(available_mb=900, limit_mb=4096))
    spawned = []
    monkeypatch.setattr(runtime, "_spawn_and_wait", lambda *a, **k: spawned.append(a))

    st = runtime.status()
    assert st.running is False and st.blocker and "900 MB" in st.blocker and "4096 MB" in st.blocker
    assert st.memory_available_mb == 900 and st.memory_limit_mb == 4096
    with pytest.raises(RuntimeError, match="Not enough free memory"):
        runtime.start()
    assert spawned == [], "the launcher must not be spawned on a host that cannot hold it"


@pytest.mark.parametrize("info", [
    resources.MemoryInfo(available_mb=None, limit_mb=None),  # unreadable: not evidence of a small host
    resources.MemoryInfo(available_mb=1536, limit_mb=2048),  # exactly the threshold is enough
])
def test_unknown_or_sufficient_memory_never_blocks(monkeypatch, info):
    monkeypatch.setattr(resources, "min_free_mb", lambda: 1536)
    assert resources.memory_blocker(info) is None


def test_running_screen_is_not_reported_blocked_by_later_pressure(tmp_path, monkeypatch):
    """The gate guards the allocation; once the desktop is up, memory pressure is the browser's problem,
    not a reason to tell the pane its running screen is blocked."""
    monkeypatch.setattr(runtime, "state_dir", lambda: tmp_path / "bd")
    monkeypatch.setattr(runtime, "missing_binaries", lambda: [])
    monkeypatch.setattr(runtime, "_launcher_pid", lambda: 4242)
    monkeypatch.setattr(runtime, "published_env", lambda: {"DISPLAY": ":20"})
    monkeypatch.setattr(resources, "min_free_mb", lambda: 1536)
    monkeypatch.setattr(resources, "memory_info", lambda: resources.MemoryInfo(available_mb=100, limit_mb=4096))
    st = runtime.status()
    assert st.running is True and st.blocker is None


def test_memory_info_takes_the_tighter_of_cgroup_and_host(tmp_path, monkeypatch):
    """A 16 GB cgroup limit on a 4 GB box is still a 4 GB box; a 2 GB limit on a 64 GB box is 2 GB."""
    v2 = tmp_path / "cg"; v2.mkdir()
    monkeypatch.setattr(resources, "_CGROUP_V2", v2)
    monkeypatch.setattr(resources, "_CGROUP_V1", tmp_path / "nope")
    meminfo = tmp_path / "meminfo"
    monkeypatch.setattr(resources, "_MEMINFO", meminfo)
    meminfo.write_text("MemTotal:       4194304 kB\nMemAvailable:   3145728 kB\n")
    (v2 / "memory.max").write_text(str(16 * 1024 ** 3)); (v2 / "memory.current").write_text(str(1024 ** 3))
    assert resources.memory_info().available_mb == 3072
    (v2 / "memory.max").write_text(str(2 * 1024 ** 3))
    info = resources.memory_info()
    assert info.available_mb == 1024 and info.limit_mb == 2048
    (v2 / "memory.max").write_text("max")  # no limit: host numbers
    assert resources.memory_info() == resources.MemoryInfo(available_mb=3072, limit_mb=4096)


def _cgroup(monkeypatch, tmp_path, *, v2=True, limit, usage, cache):
    """A cgroup tree: ``usage`` consumed, ``cache`` of it reclaimable."""
    root = tmp_path / "cg"
    root.mkdir(exist_ok=True)
    monkeypatch.setattr(resources, "_MEMINFO", tmp_path / "no-meminfo")  # cgroup numbers only
    if v2:
        monkeypatch.setattr(resources, "_CGROUP_V2", root)
        monkeypatch.setattr(resources, "_CGROUP_V1", tmp_path / "nope")
        (root / "memory.max").write_text(str(limit))
        (root / "memory.current").write_text(str(usage))
        (root / "memory.stat").write_text(f"anon 123\ninactive_file {cache}\nslab 7\n")
    else:
        monkeypatch.setattr(resources, "_CGROUP_V2", tmp_path / "nope")
        monkeypatch.setattr(resources, "_CGROUP_V1", root)
        (root / "memory.limit_in_bytes").write_text(str(limit))
        (root / "memory.usage_in_bytes").write_text(str(usage))
        (root / "memory.stat").write_text(f"total_inactive_file {cache}\n")


@pytest.mark.parametrize("v2", [True, False], ids=["cgroup-v2", "cgroup-v1"])
def test_page_cache_does_not_count_against_the_limit(tmp_path, monkeypatch, v2):
    """643 MiB of mostly page cache must not read as 643 MiB consumed: charging it would tighten the gate
    over uptime, and refuse to restart a screen the idle auto-stop had just stopped."""
    MB = 1024 * 1024
    _cgroup(monkeypatch, tmp_path, v2=v2, limit=4096 * MB, usage=643 * MB, cache=340 * MB)
    assert resources.memory_info().available_mb == 4096 - (643 - 340)


def test_a_zero_floor_disables_the_gate(monkeypatch):
    """config_defaults documents "0 disables the check"."""
    monkeypatch.setattr(resources, "min_free_mb", lambda: 0)
    assert resources.memory_blocker(resources.MemoryInfo(available_mb=10, limit_mb=4096)) is None
