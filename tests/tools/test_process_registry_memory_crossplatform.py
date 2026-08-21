"""Tests for cross-platform memory bound detection in tools/process_registry.py."""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

import tools.process_registry as pr


def test_worker_memory_max_bytes_runs_without_raising():
    """_worker_memory_max_bytes should execute cleanly across all platforms."""
    bound = pr._worker_memory_max_bytes()
    assert isinstance(bound, int)
    assert bound >= pr._MIN_WORKER_MEMORY_MAX_BYTES
    assert bound <= pr._WORKER_MEMORY_MAX_CAP_BYTES


def test_worker_memory_max_bytes_windows_attribute_error_handling(monkeypatch):
    """When os.sysconf does not exist (AttributeError) and psutil fails, fallback gracefully."""
    # Ensure cgroup is ignored
    monkeypatch.setattr(
        pr.Path,
        "read_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no cgroup")),
    )
    
    # Simulate missing sysconf attribute on os
    if hasattr(os, "sysconf"):
        monkeypatch.delattr(os, "sysconf")

    # Simulate psutil raising ImportError
    with patch.dict("sys.modules", {"psutil": None}):
        bound = pr._worker_memory_max_bytes()
        assert bound == pr._DEFAULT_WORKER_MEMORY_MAX_BYTES


def test_worker_memory_max_bytes_psutil_fallback(monkeypatch):
    """When sysconf is absent, psutil should be used to compute physical memory."""
    # Ensure cgroup is ignored
    monkeypatch.setattr(
        pr.Path,
        "read_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no cgroup")),
    )
    
    if hasattr(os, "sysconf"):
        monkeypatch.delattr(os, "sysconf")

    fake_memory = MagicMock()
    fake_memory.total = 4 * 1024 * 1024 * 1024  # 4 GiB
    
    with patch("psutil.virtual_memory", return_value=fake_memory):
        bound = pr._worker_memory_max_bytes()
        # 4 GiB // 2 = 2 GiB
        assert bound == 2 * 1024 * 1024 * 1024
