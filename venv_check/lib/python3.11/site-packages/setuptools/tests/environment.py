import os
import subprocess
import sys
import unicodedata
from subprocess import PIPE as _PIPE, Popen as _Popen

import jaraco.envs


class VirtualEnv(jaraco.envs.VirtualEnv):
    name = '.env'
    # Some version of PyPy will import distutils on startup, implicitly
    # importing setuptools, and thus leading to BackendInvalid errors
    # when upgrading Setuptools. Bypass this behavior by avoiding the
    # early availability and need to upgrade.
    create_opts = ['--no-setuptools']

    def run(self, cmd, *args, **kwargs):
        cmd = [self.exe(cmd[0])] + cmd[1:]
        kwargs = {"cwd": self.root, "encoding": "utf-8", **kwargs}  # Allow overriding
        # In some environments (eg. downstream distro packaging), where:
        # - tox isn't used to run tests and
        # - PYTHONPATH is set to point to a specific setuptools codebase and
        # - no custom env is explicitly set by a test
        # PYTHONPATH will leak into the spawned processes.
        # In that case tests look for module in the wrong place (on PYTHONPATH).
        # Unless the test sets its own special env, pass a copy of the existing
        # environment with removed PYTHONPATH to the subprocesses.
        if "env" not in kwargs:
            env = dict(os.environ)
            if "PYTHONPATH" in env:
                del env["PYTHONPATH"]
            kwargs["env"] = env
        return subprocess.check_output(cmd, *args, **kwargs)


def _which_dirs(cmd):
    result = set()
    for path in os.environ.get('PATH', '').split(os.pathsep):
        filename = os.path.join(path, cmd)
        if os.access(filename, os.X_OK):
            result.add(path)
    return result


def run_setup_py(cmd, pypath=None, path=None, data_stream=0, env=None):
    """
    Execution command for tests, separate from those used by the
    code directly to prevent accidental behavior issues
    """
    if env is None:
        env = dict()
        for envname in os.environ:
            env[envname] = os.environ[envname]

    # override the python path if needed
    if pypath is not None:
        env["PYTHONPATH"] = pypath

    # override the execution path if needed
    if path is not None:
        env["PATH"] = path
    if not env.get("PATH", ""):
        env["PATH"] = _which_dirs("tar").union(_which_dirs("gzip"))
        env["PATH"] = os.pathsep.join(env["PATH"])

    cmd = [sys.executable, "setup.py"] + list(cmd)

    # https://bugs.python.org/issue8557
    shell = sys.platform == 'win32'

    try:
        proc = _Popen(
            cmd,
            stdout=_PIPE,
            stderr=_PIPE,
            shell=shell,
            env=env,
            encoding="utf-8",
        )

        if isinstance(data_stream, tuple):
            data_stream = slice(*data_stream)
        data = proc.communicate()[data_stream]
    except OSError:
        return 1, ''

    # decode the console string if needed
    if hasattr(data, "decode"):
        # use the default encoding
        data = data.decode()
        data = unicodedata.normalize('NFC', data)

    # communicate calls wait()
    return proc.returncode, data
