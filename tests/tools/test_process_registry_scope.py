"""Contract tests for the gateway worker scope cgroup argv.

A gateway-spawned worker must never spill unbounded swap: a 2026-09-15 incident on
a 32 GiB host had a single pytest worker fill 3.9 GiB of an 8 GiB host swap file,
driving the whole user slice into a file-backed refault livelock while ~11 GiB RAM
sat free. The scope therefore carries a MemorySwapMax bound alongside MemoryMax,
and that bound can never be wider than the RAM bound.
"""

from tools.process_registry import (
    _WORKER_MEMORY_SWAP_MAX_BYTES,
    _systemd_scope_argv,
    _worker_memory_swap_max_bytes,
)


def _scope_properties(argv):
    props = {}
    for i, token in enumerate(argv):
        if token == "--property":
            key, _, value = argv[i + 1].partition("=")
            props[key] = value
    return props


def test_worker_scope_bounds_swap_spillover():
    argv = _systemd_scope_argv("/usr/bin/systemd-run", "hermes-worker-test", "/bin/true")
    props = _scope_properties(argv)
    assert "MemorySwapMax" in props
    assert int(props["MemorySwapMax"]) == _WORKER_MEMORY_SWAP_MAX_BYTES


def test_worker_scope_swap_bound_never_exceeds_memory_bound():
    assert _worker_memory_swap_max_bytes() <= _WORKER_MEMORY_SWAP_MAX_BYTES
    argv = _systemd_scope_argv("/usr/bin/systemd-run", "hermes-worker-test", "/bin/true")
    props = _scope_properties(argv)
    assert int(props["MemorySwapMax"]) <= int(props["MemoryMax"])
