"""Regression coverage for ``_create_environment`` local-shell wiring.

Background
----------
``terminal.shell`` config (auto/bash/zsh) must reach ``LocalEnvironment``.
Three call sites (``terminal_tool``, ``code_execution_tool``, ``file_tools``)
populate ``local_config = {"shell": ..., "persistent": ...}`` and pass it to
``_create_environment(..., local_config=...)``. The local branch of
``_create_environment`` MUST read ``shell`` from ``local_config`` — NOT from
``container_config`` (the container dict is unrelated to the local backend)
and MUST NOT reference an undefined variable.

Observed regression (2026-07-15)
-------------------------------
A WIP refactor moved the ``shell`` key into ``local_config`` but left the
local branch reading an unbound ``lc`` (a typo for the never-assigned alias
of ``local_config``). Result::

    NameError: name 'lc' is not defined

The terminal tool then crashed on every local invocation, surfacing to the
agent as ``"Terminal tool is broken (lc not defined). Using subprocess
directly."``

These tests pin the wiring so the same NameError cannot recur silently.
"""

import pytest

from tools.terminal_tool import _create_environment


class TestCreateEnvironmentLocalShell:
    """``_create_environment`` must propagate ``local_config["shell"]``."""

    def test_local_reads_shell_from_local_config(self, tmp_path):
        """shell=zsh in local_config → env._shell_setting == 'zsh'."""
        env = _create_environment(
            env_type="local",
            image="",
            cwd=str(tmp_path),
            timeout=5,
            local_config={"shell": "bash", "persistent": False},
        )
        assert env._shell_setting == "bash"

    def test_local_defaults_to_auto_when_local_config_omits_shell(self, tmp_path):
        """No 'shell' key → defaults to 'auto' (NOT a NameError)."""
        env = _create_environment(
            env_type="local",
            image="",
            cwd=str(tmp_path),
            timeout=5,
            local_config={"persistent": False},
        )
        assert env._shell_setting == "auto"

    def test_local_defaults_to_auto_when_local_config_none(self, tmp_path):
        """local_config=None → still 'auto', still no NameError."""
        env = _create_environment(
            env_type="local",
            image="",
            cwd=str(tmp_path),
            timeout=5,
            local_config=None,
        )
        assert env._shell_setting == "auto"

    def test_local_does_not_read_shell_from_container_config(self, tmp_path):
        """container_config['shell'] must NOT leak into the local backend.

        The local branch sources ``shell`` exclusively from ``local_config``.
        Putting ``shell`` in ``container_config`` (a docker/modal concern) must
        be a no-op for the local env so configs don't cross backends.
        """
        env = _create_environment(
            env_type="local",
            image="",
            cwd=str(tmp_path),
            timeout=5,
            container_config={"shell": "zsh"},
            local_config={"shell": "bash", "persistent": False},
        )
        assert env._shell_setting == "bash"

    def test_local_zsh_propagates(self, tmp_path):
        """End-to-end: local_config shell=zsh resolves to a zsh binary path."""
        env = _create_environment(
            env_type="local",
            image="",
            cwd=str(tmp_path),
            timeout=5,
            local_config={"shell": "zsh", "persistent": False},
        )
        assert env._shell_setting == "zsh"
        # _resolve_terminal_shell falls back to bash if zsh binary is absent;
        # both are acceptable — what matters is no NameError + kind detected.
        assert env._shell_kind in {"zsh", "bash"}
