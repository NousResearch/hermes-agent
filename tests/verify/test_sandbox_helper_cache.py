import importlib.machinery
import importlib.util
import os
import pwd
from pathlib import Path
from types import SimpleNamespace

import pytest


HELPER = (
    Path(__file__).resolve().parents[2]
    / "scripts/security/hermes-verify-sandbox"
)


@pytest.fixture
def helper(monkeypatch):
    user = SimpleNamespace(pw_uid=os.getuid(), pw_gid=os.getgid())
    monkeypatch.setattr(pwd, "getpwnam", lambda _name: user)

    name = "verify_sandbox_helper_cache_test"
    loader = importlib.machinery.SourceFileLoader(name, str(HELPER))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_valid_cache_is_opened_without_following_path_later(
    helper, tmp_path
):
    cache = tmp_path / "npm"
    cache.mkdir(mode=0o750)
    cache.chmod(0o750)
    helper.NPM_CACHE_HOST = str(cache)
    helper.HERMES_UID = os.getuid()
    helper.VERIFY = SimpleNamespace(pw_gid=os.getgid())

    fd = helper.open_dependency_cache()
    try:
        assert fd is not None
        assert os.fstat(fd).st_ino == cache.stat().st_ino
    finally:
        os.close(fd)


def test_missing_cache_is_optional(helper, tmp_path):
    helper.NPM_CACHE_HOST = str(tmp_path / "missing")
    assert helper.open_dependency_cache() is None


def test_symlink_cache_is_rejected(helper, tmp_path):
    target = tmp_path / "real-cache"
    target.mkdir()
    link = tmp_path / "cache-link"
    link.symlink_to(target, target_is_directory=True)
    helper.NPM_CACHE_HOST = str(link)

    with pytest.raises(SystemExit) as exc:
        helper.open_dependency_cache()
    assert exc.value.code != 0


def test_world_writable_cache_is_rejected(helper, tmp_path):
    cache = tmp_path / "world-writable"
    cache.mkdir(mode=0o757)
    cache.chmod(0o757)
    helper.NPM_CACHE_HOST = str(cache)
    helper.HERMES_UID = os.getuid()
    helper.VERIFY = SimpleNamespace(pw_gid=os.getgid())

    with pytest.raises(SystemExit) as exc:
        helper.open_dependency_cache()
    assert exc.value.code != 0
