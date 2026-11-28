"""The checkout guard's advice has to survive being followed.

A ``--shared`` clone of a partial clone looks fine and then cannot fetch: it borrows
promisor packs without the promised blobs, so ``git fetch`` there dies with "pack has N
unresolved deltas". The guard recommended exactly that on every checkout, including the
partial clones Hermes itself ships with.
"""

from tools.self_repo_guard import _block_message, _is_partial_clone


def _repo(tmp_path, config_extra=""):
    root = tmp_path / "hermes-agent"
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "config").write_text(
        "[core]\n\trepositoryformatversion = 0\n" + config_extra, encoding="utf-8")
    return root


def test_partial_clone_is_detected(tmp_path):
    root = _repo(tmp_path, '[remote "origin"]\n\tpromisor = true\n')
    assert _is_partial_clone(root) is True


def test_plain_clone_is_not_partial(tmp_path):
    assert _is_partial_clone(_repo(tmp_path)) is False


def test_missing_config_is_not_partial(tmp_path):
    root = tmp_path / "bare-dir"
    root.mkdir()
    assert _is_partial_clone(root) is False


def test_partial_clone_gets_depth_advice_not_shared(tmp_path):
    root = _repo(tmp_path, "[extensions]\n\tpartialClone = origin\n")
    message = _block_message("git checkout -b topic", root)
    assert "partial clone" in message
    assert "--depth 1" in message
    assert "--shared" not in message, "a shared clone of a partial clone cannot fetch"
    assert "unresolved deltas" in message
    assert "restart Hermes" in message


def test_plain_clone_still_gets_shared_advice(tmp_path):
    message = _block_message("git checkout -b topic", _repo(tmp_path))
    assert "--shared" in message
    assert "--depth 1" not in message
