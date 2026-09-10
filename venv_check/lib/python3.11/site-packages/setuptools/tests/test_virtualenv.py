import os
import subprocess
import sys
from urllib.error import URLError
from urllib.request import urlopen

import pytest


@pytest.fixture(autouse=True)
def pytest_virtualenv_works(venv):
    """
    pytest_virtualenv may not work. if it doesn't, skip these
    tests. See #1284.
    """
    venv_prefix = venv.run(["python", "-c", "import sys; print(sys.prefix)"]).strip()
    if venv_prefix == sys.prefix:
        pytest.skip("virtualenv is broken (see pypa/setuptools#1284)")


def test_clean_env_install(venv_without_setuptools, setuptools_wheel):
    """
    Check setuptools can be installed in a clean environment.
    """
    cmd = ["python", "-m", "pip", "install", str(setuptools_wheel)]
    venv_without_setuptools.run(cmd)


def access_pypi():
    # Detect if tests are being run without connectivity
    if not os.environ.get('NETWORK_REQUIRED', False):  # pragma: nocover
        try:
            urlopen('https://pypi.org', timeout=1)
        except URLError:
            # No network, disable most of these tests
            return False

    return True


@pytest.mark.skipif(
    'platform.python_implementation() == "PyPy"',
    reason="https://github.com/pypa/setuptools/pull/2865#issuecomment-965834995",
)
@pytest.mark.skipif(not access_pypi(), reason="no network")
# ^-- Even when it is not necessary to install a different version of `pip`
#     the build process will still try to download `wheel`, see #3147 and #2986.
@pytest.mark.parametrize(
    'pip_version',
    [
        None,
        pytest.param(
            'pip<20.1',
            marks=pytest.mark.xfail(
                'sys.version_info >= (3, 12)',
                reason="pip 23.1.2 required for Python 3.12 and later",
            ),
        ),
        pytest.param(
            'pip<21',
            marks=pytest.mark.xfail(
                'sys.version_info >= (3, 12)',
                reason="pip 23.1.2 required for Python 3.12 and later",
            ),
        ),
        pytest.param(
            'pip<22',
            marks=pytest.mark.xfail(
                'sys.version_info >= (3, 12)',
                reason="pip 23.1.2 required for Python 3.12 and later",
            ),
        ),
        pytest.param(
            'pip<23',
            marks=pytest.mark.xfail(
                'sys.version_info >= (3, 12)',
                reason="pip 23.1.2 required for Python 3.12 and later",
            ),
        ),
        pytest.param(
            'https://github.com/pypa/pip/archive/main.zip',
            marks=pytest.mark.xfail(reason='#2975'),
        ),
    ],
)
def test_pip_upgrade_from_source(
    pip_version, venv_without_setuptools, setuptools_wheel, setuptools_sdist
):
    """
    Check pip can upgrade setuptools from source.
    """
    # Install pip/wheel, in a venv without setuptools (as it
    # should not be needed for bootstrapping from source)
    venv = venv_without_setuptools
    venv.run(["pip", "install", "-U", "wheel"])
    if pip_version is not None:
        venv.run(["python", "-m", "pip", "install", "-U", pip_version, "--retries=1"])
    with pytest.raises(subprocess.CalledProcessError):
        # Meta-test to make sure setuptools is not installed
        venv.run(["python", "-c", "import setuptools"])

    # Then install from wheel.
    venv.run(["pip", "install", str(setuptools_wheel)])
    # And finally try to upgrade from source.
    venv.run(["pip", "install", "--no-cache-dir", "--upgrade", str(setuptools_sdist)])


def test_no_missing_dependencies(bare_venv, request):
    """
    Quick and dirty test to ensure all external dependencies are vendored.
    """
    setuptools_dir = request.config.rootdir
    bare_venv.run(['python', 'setup.py', '--help'], cwd=setuptools_dir)
