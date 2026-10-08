"""Physical skill fences must serialize overlaps without serializing disjoint profiles."""

from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path

import pytest

from tools.skill_resource_fences import acquire_resources, covered, resources_held


@pytest.fixture(autouse=True)
def namespace(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_REAL_HOME", str(tmp_path))


@pytest.mark.parametrize("parent_first", [True, False])
def test_nested_catalog_roots_contend_in_both_directions(tmp_path, parent_first):
    parent, child = tmp_path / "absent-shared", tmp_path / "absent-shared" / "category"
    held, waiting = (parent, child) if parent_first else (child, parent)
    with acquire_resources({held}, time.monotonic() + 5):
        with ThreadPoolExecutor(max_workers=1) as pool:
            def other():
                with acquire_resources({waiting}, time.monotonic() + 0.2):
                    pytest.fail("Overlapping physical subtrees cannot acquire concurrently")
            with pytest.raises(TimeoutError):
                pool.submit(other).result(timeout=3)
    assert not parent.exists(), "Lock acquisition must not activate an absent skill catalog"
    with acquire_resources({waiting}, time.monotonic() + 2):
        assert covered({waiting})
    assert not resources_held()


def test_disjoint_physical_subtrees_share_ancestors_without_waiting(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    with acquire_resources({a}, time.monotonic() + 5):
        with ThreadPoolExecutor(max_workers=1) as pool:
            def other():
                with acquire_resources({b}, time.monotonic() + 0.2):
                    return covered({b})
            assert pool.submit(other).result(timeout=3)
    assert not a.exists() and not b.exists()


def test_partial_acquisition_failure_releases_earlier_owned_resources(tmp_path):
    parent = tmp_path / "shared"
    with acquire_resources({parent / "child"}, time.monotonic() + 5):
        with pytest.raises(OSError, match="upgrade"):
            with acquire_resources({parent}, time.monotonic() + 1):
                pytest.fail("A shared-ancestor lock cannot be upgraded in place")
        assert covered({parent / "child"})
    with acquire_resources({parent}, time.monotonic() + 1):
        assert covered({parent, parent / "child"})


def test_reverse_requested_order_and_reentrant_descendants(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    with acquire_resources({b, a}, time.monotonic() + 5):
        with acquire_resources({a / "child", b}, time.monotonic() + 1):
            assert covered({a / "child", b})
    assert not resources_held()
