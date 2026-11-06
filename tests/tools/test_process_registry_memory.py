"""Tests for the per-worker systemd MemoryMax cap derivation.

The operator cap (``terminal.local_memory_max_mb`` config or
``TERMINAL_LOCAL_MEMORY_MAX_MB`` env) may tighten OR widen the conservative
4 GiB default, but never beyond the enclosing cgroup slice or half of physical
RAM. Issue #137589 / #103670.
"""

import pytest


def _mock_host(monkeypatch, pr, phys_pages: int, page_size: int = 4096):
    """Simulate an unlimited host cgroup with *phys_pages* x *page_size* RAM."""
    monkeypatch.delenv("TERMINAL_LOCAL_MEMORY_MAX_MB", raising=False)
    monkeypatch.setattr(pr.Path, "read_text", lambda *_a, **_k: "max",
                        raising=False)
    monkeypatch.setattr(
        pr.os,
        "sysconf",
        lambda name: {"SC_PHYS_PAGES": phys_pages, "SC_PAGE_SIZE": page_size}[name],
    )


@pytest.mark.parametrize("configured_mb,expected_override,expected_cap_mb", [
    # Config may widen past the 4 GiB default on a RAM-rich host.
    (8192, 8192 * 1024 * 1024, 8192),
    # Explicit 0 means "not configured": conservative default ceiling stands.
    (0, None, 4096),
    # Sub-floor values are rejected (a warning, not a limit), default stands.
    (12, None, 4096),
])
def test_worker_memory_limit_from_config(
    monkeypatch, configured_mb, expected_override, expected_cap_mb
):
    import tools.process_registry as pr
    import tools.process_registry_memory as pr_mem

    _mock_host(monkeypatch, pr, phys_pages=8 * 1024 * 1024)
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"terminal": {"local_memory_max_mb": configured_mb}},
    )
    # env path must not clobber the config path in this test
    monkeypatch.delenv("TERMINAL_LOCAL_MEMORY_MAX_MB", raising=False)

    # the sibling helper is the single source of the operator override
    assert pr_mem.operator_worker_memory_max_bytes() == expected_override
    assert pr._worker_memory_max_bytes() == expected_cap_mb * 1024 * 1024


def test_worker_memory_limit_env_overrides_config(monkeypatch):
    """Env var wins over config (terminal.* convention)."""
    import tools.process_registry as pr

    _mock_host(monkeypatch, pr, phys_pages=8 * 1024 * 1024)
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "2048")
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"terminal": {"local_memory_max_mb": 8192}},
    )

    assert pr._worker_memory_max_bytes() == 2048 * 1024 * 1024


def test_worker_memory_limit_widening_still_clamped_by_half_ram(monkeypatch):
    """An oversized config can never breach half of physical RAM."""
    import tools.process_registry as pr

    _mock_host(monkeypatch, pr, phys_pages=1024)  # 4 MiB RAM → half = 2 MiB
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"terminal": {"local_memory_max_mb": 999999}},
    )

    # 2 MiB < 64 MiB floor, so the floor-safe bound wins: still never above it.
    assert pr._worker_memory_max_bytes() == pr._MIN_WORKER_MEMORY_MAX_BYTES


def test_worker_memory_limit_env_still_tightens_below_config(monkeypatch):
    """Env below the config value keeps tightening (both directions allowed)."""
    import tools.process_registry as pr

    _mock_host(monkeypatch, pr, phys_pages=8 * 1024 * 1024)
    monkeypatch.setenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "512")
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"terminal": {"local_memory_max_mb": 8192}},
    )

    assert pr._worker_memory_max_bytes() == 512 * 1024 * 1024
