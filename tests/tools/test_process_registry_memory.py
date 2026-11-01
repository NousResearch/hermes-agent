"""Per-worker systemd scope ``MemoryMax`` bound (tools/process_registry_memory.py)."""

import pytest

import tools.process_registry_memory as prm
from tools.process_registry import _build_systemd_scope_argv

_GIB = 1024 * 1024 * 1024


def _host(monkeypatch, *, ram_gib, cgroup_gib=None):
    """Pin the host facts: physical RAM and the gateway's enclosing memory.max."""
    monkeypatch.delenv("TERMINAL_LOCAL_MEMORY_MAX_MB", raising=False)
    pages = ram_gib * _GIB // 4096
    monkeypatch.setattr(
        prm.os, "sysconf", lambda name: pages if name == "SC_PHYS_PAGES" else 4096
    )
    monkeypatch.setattr(
        prm,
        "_enclosing_cgroup_memory_max_bytes",
        lambda: None if cgroup_gib is None else cgroup_gib * _GIB,
    )


def _write_config(body: str) -> None:
    from hermes_cli.config import get_config_path

    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_worker_memory_limit_honors_local_guard_mb_override(monkeypatch):
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "123")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

    argv = _build_systemd_scope_argv(["/bin/bash", "-lc", "true"], unit_suffix="test")

    assert f"MemoryMax={123 * 1024 * 1024}" in argv


def test_worker_memory_limit_caps_oversized_local_guard_override(monkeypatch):
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "999999")
    monkeypatch.setattr(
        prm.Path,
        "read_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no cgroup")),
    )
    monkeypatch.setattr(
        prm.os,
        "sysconf",
        lambda *_args: (_ for _ in ()).throw(OSError("no sysconf")),
    )

    assert prm._worker_memory_max_bytes() == prm._DEFAULT_WORKER_MEMORY_MAX_BYTES


def test_worker_memory_limit_auto_keeps_4gib_cap_on_large_host(monkeypatch):
    _host(monkeypatch, ram_gib=128)
    _write_config("terminal:\n  worker_memory_max_mb: auto\n")

    assert prm._worker_memory_max_bytes() == prm._WORKER_MEMORY_MAX_CAP_BYTES


@pytest.mark.parametrize("value", ["8192", "'8192'"])
def test_worker_memory_limit_explicit_config_raises_auto_cap(monkeypatch, value):
    """A config.yaml value (YAML int or the quoted string `hermes config set`
    writes) lifts the scope past the 4 GiB auto cap, and reaches the real argv."""
    _host(monkeypatch, ram_gib=128)
    _write_config(f"terminal:\n  worker_memory_max_mb: {value}\n")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

    argv = _build_systemd_scope_argv(["/bin/bash", "-lc", "true"], unit_suffix="t")

    assert f"MemoryMax={8 * _GIB}" in argv


def test_worker_memory_limit_explicit_config_is_clamped_by_enclosing_cgroup(monkeypatch):
    _host(monkeypatch, ram_gib=128, cgroup_gib=6)
    _write_config("terminal:\n  worker_memory_max_mb: 8192\n")

    assert prm._worker_memory_max_bytes() == 6 * _GIB


def test_worker_memory_limit_explicit_config_is_clamped_by_physical_ram(monkeypatch):
    _host(monkeypatch, ram_gib=4)
    _write_config("terminal:\n  worker_memory_max_mb: 999999\n")

    assert prm._worker_memory_max_bytes() == 4 * _GIB


def test_worker_memory_limit_local_guard_still_only_tightens_explicit_config(monkeypatch):
    _host(monkeypatch, ram_gib=128)
    _write_config("terminal:\n  worker_memory_max_mb: 8192\n")

    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "2048")
    assert prm._worker_memory_max_bytes() == 2 * _GIB
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "16384")
    assert prm._worker_memory_max_bytes() == 8 * _GIB


@pytest.mark.parametrize(
    "value", ["invalid", "0", "63", "true", "8192.5", "'8192.5'", "[8192]", "-1"]
)
def test_worker_memory_limit_invalid_config_falls_back_to_auto_bound(monkeypatch, value):
    """Fractional, boolean, sub-floor or non-scalar values never widen the cap."""
    _host(monkeypatch, ram_gib=128)
    _write_config(f"terminal:\n  worker_memory_max_mb: {value}\n")

    assert prm._worker_memory_max_bytes() == prm._WORKER_MEMORY_MAX_CAP_BYTES
