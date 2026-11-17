"""The fake hosted artifacts server answers bad versions the way the contract does."""

from __future__ import annotations

import pytest

from tests.fakes.hosted_artifacts_server import Store


@pytest.fixture
def store():
    store = Store(lambda: 9000, "http://portal.localhost")
    created = store.create("t", [{"path": "index.html", "content": "<p>1</p>", "encoding": "utf8"}])
    store.update(created["id"], 1, None, None, [{"path": "index.html", "content": "<p>2</p>", "encoding": "utf8"}])
    return store, created["id"]


@pytest.mark.parametrize("read", ["read", "show"])
def test_existing_versions_and_the_head(store, read):
    store, artifact_id = store
    assert getattr(store, read)(artifact_id, None)["version"] == 2
    assert getattr(store, read)(artifact_id, 1)["version"] == 1
    if read == "read":
        assert store.read(artifact_id, 1)["files"][0]["content"] == "<p>1</p>"


@pytest.mark.parametrize("read", ["read", "show"])
@pytest.mark.parametrize("version", [3, 99])
def test_out_of_range_version_is_not_found(store, read, version):
    store, artifact_id = store
    assert getattr(store, read)(artifact_id, version)["error"] == "not_found"


@pytest.mark.parametrize("read", ["read", "show"])
@pytest.mark.parametrize("version", [0, -1, True])
def test_non_positive_version_is_invalid(store, read, version):
    store, artifact_id = store
    assert getattr(store, read)(artifact_id, version)["error"] == "invalid_request"


def test_unknown_artifact_is_not_found(store):
    store, _ = store
    assert store.read("0" * 26, None)["error"] == "not_found"
