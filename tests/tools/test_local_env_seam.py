"""Tests for the two subclass seams in LocalEnvironment._run_bash.

``_wrap_popen_args`` lets a subclass wrap the shell argv (a sandbox prefix)
and ``_popen_preexec`` lets it supply a ``preexec_fn`` (resource limits).
LocalEnvironment itself must keep its identity behavior.
"""

import os
import resource
import shutil
import sys
from unittest.mock import MagicMock, patch

import pytest

import tools.environments.local as local_mod
from tools.environments.local import LocalEnvironment

FAKE_BASH = "/opt/fake/bash"


def _sentinel_preexec():
    pass


class WrappingEnvironment(LocalEnvironment):
    def _wrap_popen_args(self, args):
        return ["wrapper", "--flag", "--", *args]

    def _popen_preexec(self):
        return _sentinel_preexec


@pytest.fixture
def popen_spy():
    """Replace subprocess.Popen in the local module and record its call."""
    fake_proc = MagicMock()
    fake_proc.pid = os.getpid()
    with patch.object(local_mod.subprocess, "Popen", return_value=fake_proc) as spy, \
            patch.object(local_mod, "_find_bash", return_value=FAKE_BASH):
        yield spy


class TestDefaultSeams:
    def test_wrap_is_identity(self):
        env = LocalEnvironment()
        assert env._wrap_popen_args(["a", "b"]) == ["a", "b"]

    def test_preexec_is_none(self):
        assert LocalEnvironment()._popen_preexec() is None

    def test_popen_receives_plain_bash_args(self, popen_spy, tmp_path):
        env = LocalEnvironment(cwd=str(tmp_path))
        env._run_bash("echo hi")
        args, kwargs = popen_spy.call_args
        assert args[0] == [FAKE_BASH, "-c", "echo hi"]
        assert "preexec_fn" not in kwargs

    def test_login_args_unchanged(self, popen_spy, tmp_path):
        env = LocalEnvironment(cwd=str(tmp_path))
        with patch.object(local_mod, "_resolve_shell_init_files", return_value=[]):
            env._run_bash("true", login=True)
        args, _ = popen_spy.call_args
        assert args[0] == [FAKE_BASH, "-l", "-c", "true"]


class TestSubclassSeams:
    def test_popen_receives_wrapped_args_and_preexec(self, popen_spy, tmp_path):
        env = WrappingEnvironment(cwd=str(tmp_path))
        env._run_bash("echo hi")
        args, kwargs = popen_spy.call_args
        assert args[0] == ["wrapper", "--flag", "--", FAKE_BASH, "-c", "echo hi"]
        assert kwargs["preexec_fn"] is _sentinel_preexec

    def test_login_invocation_is_wrapped_too(self, popen_spy, tmp_path):
        env = WrappingEnvironment(cwd=str(tmp_path))
        with patch.object(local_mod, "_resolve_shell_init_files", return_value=[]):
            env._run_bash("true", login=True)
        args, _ = popen_spy.call_args
        assert args[0][:3] == ["wrapper", "--flag", "--"]
        assert args[0][3:] == [FAKE_BASH, "-l", "-c", "true"]

    def test_wrapper_sees_the_full_shell_argv(self, popen_spy, tmp_path):
        seen = []

        class Recording(LocalEnvironment):
            def _wrap_popen_args(self, args):
                seen.append(list(args))
                return args

        Recording(cwd=str(tmp_path))._run_bash("pwd")
        # __init__ runs init_session's login bootstrap through _run_bash
        # first; every spawn, including that one, goes through the wrapper.
        assert seen[-1] == [FAKE_BASH, "-c", "pwd"]
        assert all(args[0] == FAKE_BASH for args in seen)
        assert any(args[:2] == [FAKE_BASH, "-l"] for args in seen)


@pytest.mark.skipif(sys.platform == "win32", reason="preexec_fn and env(1) are POSIX only")
@pytest.mark.skipif(shutil.which("bash") is None or shutil.which("env") is None, reason="needs bash and env")
class TestSeamsEndToEnd:
    def test_wrapper_and_preexec_reach_the_child(self, tmp_path):
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(64, soft)

        class Limited(LocalEnvironment):
            def _wrap_popen_args(self, args):
                return [shutil.which("env"), "HERMES_SEAM_MARK=wrapped", *args]

            def _popen_preexec(self):
                def _limits():
                    resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
                return _limits

        proc = Limited(cwd=str(tmp_path))._run_bash('echo "$HERMES_SEAM_MARK $(ulimit -n)"')
        out, _ = proc.communicate(timeout=30)
        assert out.strip() == f"wrapped {target}"
