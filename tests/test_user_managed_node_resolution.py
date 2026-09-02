"""User-managed runtime respect: Hermes must not shadow the user's Node toolchain.

``find_node_executable`` and ``with_hermes_node_path`` honour a suitable
user-managed Node/npm on PATH (outside the Hermes-managed tree) instead of
unconditionally preferring/prepending the managed tree. Suitability is
fail-closed: the binary must run and its version must satisfy the checkout's
``engines``, so an outdated user Node can never be handed to Hermes-owned
subprocesses that need a supported one.

These tests build hermetic PATHs with stub binaries so a real Node on the CI
runner or dev machine cannot contaminate the matrix.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import hermes_constants


def _stub(path: Path, body: str, mode: int = 0o755) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(mode)
    return path


def _clear_cache():
    hermes_constants._user_managed_node_cache.clear()


@pytest.fixture(autouse=True)
def _reset_detection_cache():
    _clear_cache()
    yield
    _clear_cache()


class TestEngineSpec:
    """The engines.node / engines.npm parser and evaluator."""

    def test_node_engines_from_package_json(self):
        clauses, supported = hermes_constants._engines_spec("node")
        assert supported is True
        assert clauses

    def test_npm_engines_from_package_json(self):
        clauses, supported = hermes_constants._engines_spec("npm")
        assert supported is True
        assert clauses

    def test_version_satisfies_caret(self):
        clauses, supported = hermes_constants._engines_spec("node")
        assert hermes_constants._version_satisfies("24.19.0", clauses, supported) is True
        assert hermes_constants._version_satisfies("22.22.0", clauses, supported) is True
        assert hermes_constants._version_satisfies("22.21.9", clauses, supported) is False
        assert hermes_constants._version_satisfies("25.0.0", clauses, supported) is False

    def test_version_satisfies_gte(self):
        clauses: list[tuple[str | None, tuple[int, int, int]]] = [
            (None, (26, 0, 0)),
            (">=", (26, 0, 0)),
        ]
        assert hermes_constants._version_satisfies("26.1.0", clauses, True) is True
        assert hermes_constants._version_satisfies("25.9.9", clauses, True) is False

    def test_version_satisfies_lt(self):
        clauses: list[tuple[str | None, tuple[int, int, int]]] = [
            ("<", (11, 10, 0)),
            (">=", (11, 17, 0)),
        ]
        assert hermes_constants._version_satisfies("11.9.0", clauses, True) is True
        assert hermes_constants._version_satisfies("11.10.0", clauses, True) is False
        assert hermes_constants._version_satisfies("11.17.0", clauses, True) is True

    def test_unknown_grammar_fails_closed(self):
        assert hermes_constants._version_satisfies("99.0.0", [], False) is False
        assert hermes_constants._version_satisfies("not-a-version", [(">=", (1, 0, 0))], True) is False


class TestUserManagedDetection:
    """user_managed_node_detected() across the runtime matrix."""

    def test_suitable_user_node_detected(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        user_bin = tmp_path / "user-bin"
        _stub(user_bin / "node", "#!/bin/sh\necho 'v24.19.0'\nexit 0\n")
        _stub(user_bin / "npm", "#!/bin/sh\necho '11.17.0'\nexit 0\n")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("PATH", str(user_bin))

        assert hermes_constants.user_managed_node_detected() is True

    def test_outdated_user_node_not_detected(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        user_bin = tmp_path / "user-bin"
        _stub(user_bin / "node", "#!/bin/sh\necho 'v18.20.0'\nexit 0\n")
        # 11.12.0 sits in the engines.npm dead zone: >=11.10.0, <11.17.0.
        _stub(user_bin / "npm", "#!/bin/sh\necho '11.12.0'\nexit 0\n")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("PATH", str(user_bin))

        assert hermes_constants.user_managed_node_detected() is False

    def test_managed_tree_symlink_not_mistaken_for_user_runtime(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        managed_bin = home / "node" / "bin"
        _stub(managed_bin / "node", "#!/bin/sh\necho 'v24.19.0'\nexit 0\n")
        link_bin = tmp_path / "fake-local" / "bin"
        link_bin.mkdir(parents=True)
        (link_bin / "node").symlink_to(managed_bin / "node")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("PATH", str(link_bin))

        assert hermes_constants.user_managed_node_detected() is False

    def test_symlinked_candidate_returned_as_candidate(self, tmp_path, monkeypatch):
        """A shim symlink must be returned as the shim, not its resolved target.

        Version managers (mise/nvm) dispatch on argv[0]; returning the
        resolved target would hand callers the manager binary instead of the
        tool shim.
        """
        home = tmp_path / "hermes"
        user_bin = tmp_path / "user-bin"
        target = _stub(tmp_path / "real" / "npm", "#!/bin/sh\necho '11.17.0'\nexit 0\n")
        shim = user_bin / "npm"
        shim.parent.mkdir(parents=True)
        shim.symlink_to(target)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("PATH", str(user_bin))

        assert hermes_constants._user_managed_node_executable("npm") == str(shim)

    def test_empty_path_detects_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        monkeypatch.setenv("PATH", "")

        assert hermes_constants.user_managed_node_detected() is False

    def test_broken_user_binary_not_detected(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        user_bin = tmp_path / "user-bin"
        _stub(user_bin / "node", "#!/bin/sh\nexit 1\n")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("PATH", str(user_bin))

        assert hermes_constants.user_managed_node_detected() is False


class TestDetectionCache:
    """The detection verdict cache must not go stale on a changing runtime."""

    def test_cache_reevaluates_when_path_changes(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        user_bin = tmp_path / "user-bin"
        _stub(user_bin / "node", "#!/bin/sh\necho 'v24.19.0'\nexit 0\n")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("PATH", str(user_bin))

        assert hermes_constants.user_managed_node_detected() is True
        # Same process, no manual cache clear: a PATH change must re-key the
        # verdict immediately.
        monkeypatch.setenv("PATH", "")
        assert hermes_constants.user_managed_node_detected() is False

    def test_cache_ttl_bounds_staleness(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        user_bin = tmp_path / "user-bin"
        node_stub = _stub(user_bin / "node", "#!/bin/sh\necho 'v24.19.0'\nexit 0\n")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("PATH", str(user_bin))

        assert hermes_constants.user_managed_node_detected() is True
        # The runtime vanishes without touching PATH or the managed tree
        # (a manager swapping shim targets). The verdict is only bounded by
        # the TTL; a zero TTL forces re-evaluation on the next call.
        node_stub.unlink()
        monkeypatch.setattr(hermes_constants, "_USER_MANAGED_NODE_CACHE_TTL_SECONDS", 0.0)
        assert hermes_constants.user_managed_node_detected() is False

    def test_cache_key_fingerprints_the_requested_home(self, tmp_path):
        """A profile-scoped key must fingerprint THAT home's tree state, not
        the default home's — otherwise two profiles share cache verdicts."""
        home_a = tmp_path / "a"
        home_b = tmp_path / "b"
        (home_a / "node").mkdir(parents=True)
        (home_b / "node").mkdir(parents=True)

        key_a = hermes_constants._user_managed_node_cache_key(home_a)
        key_b = hermes_constants._user_managed_node_cache_key(home_b)
        assert key_a != key_b
        assert str(home_a) in key_a
        assert str(home_b) in key_b


class TestWindowsCandidateOrdering:
    """The win32 shim ordering the detector relies on, pinned platform-free."""

    def test_windows_shim_ordering(self, monkeypatch):
        monkeypatch.setattr(hermes_constants.sys, "platform", "win32")
        assert hermes_constants._candidate_node_command_names("npm") == [
            "npm.cmd",
            "npm.exe",
            "npm",
        ]
        assert hermes_constants._candidate_node_command_names("npx") == [
            "npx.cmd",
            "npx.exe",
            "npx",
        ]
        assert hermes_constants._candidate_node_command_names("node") == [
            "node.exe",
            "node",
        ]

    def test_posix_uses_bare_name(self, monkeypatch):
        monkeypatch.setattr(hermes_constants.sys, "platform", "darwin")
        assert hermes_constants._candidate_node_command_names("npm") == ["npm"]
        assert hermes_constants._candidate_node_command_names("node") == ["node"]


class TestFindNodeExecutable:
    """find_node_executable() honours the user runtime first."""

    def test_user_node_wins_over_managed_tree(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        managed_bin = home / "node" / "bin"
        _stub(managed_bin / "npm", "#!/bin/sh\necho '11.17.0'\nexit 0\n")
        user_bin = tmp_path / "user-bin"
        user_npm = _stub(user_bin / "npm", "#!/bin/sh\necho '11.17.0'\nexit 0\n")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("PATH", str(user_bin))

        assert hermes_constants.find_node_executable("npm") == str(user_npm)

    def test_broken_managed_tree_falls_back_to_suitable_user_npm(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        managed_bin = home / "node" / "bin"
        _stub(managed_bin / "npm", "#!/bin/sh\nexit 1\n")
        user_bin = tmp_path / "user-bin"
        user_npm = _stub(user_bin / "npm", "#!/bin/sh\necho '11.17.0'\nexit 0\n")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("PATH", str(user_bin))
        monkeypatch.setattr(hermes_constants, "_managed_node_heal_attempted", False)
        monkeypatch.setattr(hermes_constants, "heal_hermes_managed_node", lambda: False)

        # Previously this returned None (managed tree present but broken);
        # with a suitable user runtime the user's npm is the answer.
        assert hermes_constants.find_node_executable("npm") == str(user_npm)

    def test_outdated_user_node_defers_to_managed_tree(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        managed_bin = home / "node" / "bin"
        managed_npm = _stub(managed_bin / "npm", "#!/bin/sh\necho '11.17.0'\nexit 0\n")
        user_bin = tmp_path / "user-bin"
        # 11.12.0 sits in the engines.npm dead zone: >=11.10.0, <11.17.0.
        _stub(user_bin / "npm", "#!/bin/sh\necho '11.12.0'\nexit 0\n")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("PATH", str(user_bin))
        monkeypatch.setattr(hermes_constants, "_managed_node_heal_attempted", False)

        # The user npm fails engines.npm; the managed tree is used instead.
        assert hermes_constants.find_node_executable("npm") == str(managed_npm)

    def test_nothing_anywhere_returns_none(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        (home / "node").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("PATH", "")
        monkeypatch.setattr(hermes_constants, "_managed_node_heal_attempted", False)
        monkeypatch.setattr(hermes_constants, "heal_hermes_managed_node", lambda: False)

        assert hermes_constants.find_node_executable("npm") is None


class TestWithHermesNodePath:
    """with_hermes_node_path() stops prepending when a user runtime exists."""

    def test_no_prepend_when_user_managed_detected(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        (home / "node" / "bin").mkdir(parents=True)
        user_bin = tmp_path / "user-bin"
        _stub(user_bin / "node", "#!/bin/sh\necho 'v24.19.0'\nexit 0\n")
        _stub(user_bin / "npm", "#!/bin/sh\necho '11.17.0'\nexit 0\n")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("PATH", str(user_bin))

        env = hermes_constants.with_hermes_node_path({"PATH": str(user_bin)})
        assert env["PATH"] == str(user_bin)

    def test_prepend_preserved_for_managed_only_install(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        managed_bin = home / "node" / "bin"
        managed_bin.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("PATH", "")

        env = hermes_constants.with_hermes_node_path({"PATH": "/usr/bin"})
        assert env["PATH"].split(os.pathsep)[0] == str(managed_bin)
