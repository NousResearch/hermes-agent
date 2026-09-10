import contextlib
import os
import subprocess
import sys
from pathlib import Path

import path
import pytest

from . import contexts, environment


@pytest.fixture
def user_override(monkeypatch):
    """
    Override site.USER_BASE and site.USER_SITE with temporary directories in
    a context.
    """
    with contexts.tempdir() as user_base:
        monkeypatch.setattr('site.USER_BASE', user_base)
        with contexts.tempdir() as user_site:
            monkeypatch.setattr('site.USER_SITE', user_site)
            with contexts.save_user_site_setting():
                yield


@pytest.fixture
def tmpdir_cwd(tmpdir):
    with tmpdir.as_cwd() as orig:
        yield orig


@pytest.fixture(autouse=True, scope="session")
def workaround_xdist_376(request):
    """
    Workaround pytest-dev/pytest-xdist#376

    ``pytest-xdist`` tends to inject '' into ``sys.path``,
    which may break certain isolation expectations.
    Remove the entry so the import
    machinery behaves the same irrespective of xdist.
    """
    if not request.config.pluginmanager.has_plugin('xdist'):
        return

    with contextlib.suppress(ValueError):
        sys.path.remove('')


@pytest.fixture
def sample_project(tmp_path):
    """
    Clone the 'sampleproject' and return a path to it.
    """
    cmd = ['git', 'clone', 'https://github.com/pypa/sampleproject']
    try:
        subprocess.check_call(cmd, cwd=str(tmp_path))
    except Exception:
        pytest.skip("Unable to clone sampleproject")
    return tmp_path / 'sampleproject'


# sdist and wheel artifacts should be stable across a round of tests
# so we can build them once per session and use the files as "readonly"

# In the case of setuptools, building the wheel without sdist may cause
# it to contain the `build` directory, and therefore create situations with
# `setuptools/build/lib/build/lib/...`. To avoid that, build both artifacts at once.


def _build_distributions(tmp_path_factory, request):
    with contexts.session_locked_tmp_dir(
        request, tmp_path_factory, "dist_build"
    ) as tmp:  # pragma: no cover
        sdist = next(tmp.glob("*.tar.gz"), None)
        wheel = next(tmp.glob("*.whl"), None)
        if sdist and wheel:
            return (sdist, wheel)

        # Sanity check: should not create recursive setuptools/build/lib/build/lib/...
        assert not Path(request.config.rootdir, "build/lib/build").exists()

        subprocess.check_output([
            sys.executable,
            "-m",
            "build",
            "--outdir",
            str(tmp),
            str(request.config.rootdir),
        ])

        # Sanity check: should not create recursive setuptools/build/lib/build/lib/...
        assert not Path(request.config.rootdir, "build/lib/build").exists()

        return next(tmp.glob("*.tar.gz")), next(tmp.glob("*.whl"))


@pytest.fixture(scope="session")
def setuptools_sdist(tmp_path_factory, request):
    prebuilt = os.getenv("PRE_BUILT_SETUPTOOLS_SDIST")
    if prebuilt and os.path.exists(prebuilt):  # pragma: no cover
        return Path(prebuilt).resolve()

    sdist, _ = _build_distributions(tmp_path_factory, request)
    return sdist


@pytest.fixture(scope="session")
def setuptools_wheel(tmp_path_factory, request):
    prebuilt = os.getenv("PRE_BUILT_SETUPTOOLS_WHEEL")
    if prebuilt and os.path.exists(prebuilt):  # pragma: no cover
        return Path(prebuilt).resolve()

    _, wheel = _build_distributions(tmp_path_factory, request)
    return wheel


@pytest.fixture
def venv(tmp_path, setuptools_wheel):
    """Virtual env with the version of setuptools under test installed"""
    env = environment.VirtualEnv()
    env.root = path.Path(tmp_path / 'venv')
    env.create_opts = ['--no-setuptools', '--wheel=bundle']
    # TODO: Use `--no-wheel` when setuptools implements its own bdist_wheel
    env.req = str(setuptools_wheel)
    # In some environments (eg. downstream distro packaging),
    # where tox isn't used to run tests and PYTHONPATH is set to point to
    # a specific setuptools codebase, PYTHONPATH will leak into the spawned
    # processes.
    # env.create() should install the just created setuptools
    # wheel, but it doesn't if it finds another existing matching setuptools
    # installation present on PYTHONPATH:
    # `setuptools is already installed with the same version as the provided
    # wheel. Use --force-reinstall to force an installation of the wheel.`
    # This prevents leaking PYTHONPATH to the created environment.
    with contexts.environment(PYTHONPATH=None):
        return env.create()


@pytest.fixture
def venv_without_setuptools(tmp_path):
    """Virtual env without any version of setuptools installed"""
    env = environment.VirtualEnv()
    env.root = path.Path(tmp_path / 'venv_without_setuptools')
    env.create_opts = ['--no-setuptools', '--no-wheel']
    env.ensure_env()
    return env


@pytest.fixture
def bare_venv(tmp_path):
    """Virtual env without any common packages installed"""
    env = environment.VirtualEnv()
    env.root = path.Path(tmp_path / 'bare_venv')
    env.create_opts = ['--no-setuptools', '--no-pip', '--no-wheel', '--no-seed']
    env.ensure_env()
    return env
