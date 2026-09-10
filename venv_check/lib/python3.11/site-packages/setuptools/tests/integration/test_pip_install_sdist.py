# https://github.com/python/mypy/issues/16936
# mypy: disable-error-code="has-type"
"""Integration tests for setuptools that focus on building packages via pip.

The idea behind these tests is not to exhaustively check all the possible
combinations of packages, operating systems, supporting libraries, etc, but
rather check a limited number of popular packages and how they interact with
the exposed public API. This way if any change in API is introduced, we hope to
identify backward compatibility problems before publishing a release.

The number of tested packages is purposefully kept small, to minimise duration
and the associated maintenance cost (changes in the way these packages define
their build process may require changes in the tests).
"""

import json
import os
import shutil
import sys
from enum import Enum
from glob import glob
from hashlib import md5
from urllib.request import urlopen

import pytest
from packaging.requirements import Requirement

from .helpers import Archive, run

pytestmark = pytest.mark.integration


(LATEST,) = Enum("v", "LATEST")  # type: ignore[misc] # https://github.com/python/mypy/issues/16936
"""Default version to be checked"""
# There are positive and negative aspects of checking the latest version of the
# packages.
# The main positive aspect is that the latest version might have already
# removed the use of APIs deprecated in previous releases of setuptools.


# Packages to be tested:
# (Please notice the test environment cannot support EVERY library required for
# compiling binary extensions. In Ubuntu/Debian nomenclature, we only assume
# that `build-essential`, `gfortran` and `libopenblas-dev` are installed,
# due to their relevance to the numerical/scientific programming ecosystem)
EXAMPLES = [
    ("pip", LATEST),  # just in case...
    ("pytest", LATEST),  # uses setuptools_scm
    ("mypy", LATEST),  # custom build_py + ext_modules
    # --- Popular packages: https://hugovk.github.io/top-pypi-packages/ ---
    ("botocore", LATEST),
    ("kiwisolver", LATEST),  # build_ext
    ("brotli", LATEST),  # not in the list but used by urllib3
    ("pyyaml", LATEST),  # cython + custom build_ext + custom distclass
    ("charset-normalizer", LATEST),  # uses mypyc, used by aiohttp
    ("protobuf", LATEST),
    # ("requests", LATEST),  # XXX: https://github.com/psf/requests/pull/6920
    ("celery", LATEST),
    # When adding packages to this list, make sure they expose a `__version__`
    # attribute, or modify the tests below
]


# Some packages have "optional" dependencies that modify their build behaviour
# and are not listed in pyproject.toml, others still use `setup_requires`
EXTRA_BUILD_DEPS = {
    "pyyaml": ("Cython<3.0",),  # constraint to avoid errors
    "charset-normalizer": ("mypy>=1.4.1",),  # no pyproject.toml available
}

EXTRA_ENV_VARS = {
    "pyyaml": {"PYYAML_FORCE_CYTHON": "1"},
    "charset-normalizer": {"CHARSET_NORMALIZER_USE_MYPYC": "1"},
}

IMPORT_NAME = {
    "pyyaml": "yaml",
    "protobuf": "google.protobuf",
}


VIRTUALENV = (sys.executable, "-m", "virtualenv")


# By default, pip will try to build packages in isolation (PEP 517), which
# means it will download the previous stable version of setuptools.
# `pip` flags can avoid that (the version of setuptools under test
# should be the one to be used)
INSTALL_OPTIONS = (
    "--ignore-installed",
    "--no-build-isolation",
    # Omit "--no-binary :all:" the sdist is supplied directly.
    # Allows dependencies as wheels.
)
# The downside of `--no-build-isolation` is that pip will not download build
# dependencies. The test script will have to also handle that.


@pytest.fixture
def venv_python(tmp_path):
    run([*VIRTUALENV, str(tmp_path / ".venv")])
    possible_path = (str(p.parent) for p in tmp_path.glob(".venv/*/python*"))
    return shutil.which("python", path=os.pathsep.join(possible_path))


@pytest.fixture(autouse=True)
def _prepare(tmp_path, venv_python, monkeypatch):
    download_path = os.getenv("DOWNLOAD_PATH", str(tmp_path))
    os.makedirs(download_path, exist_ok=True)

    # Environment vars used for building some of the packages
    monkeypatch.setenv("USE_MYPYC", "1")

    yield

    # Let's provide the maximum amount of information possible in the case
    # it is necessary to debug the tests directly from the CI logs.
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Temporary directory:")
    map(print, tmp_path.glob("*"))
    print("Virtual environment:")
    run([venv_python, "-m", "pip", "freeze"])


@pytest.mark.parametrize(("package", "version"), EXAMPLES)
@pytest.mark.uses_network
def test_install_sdist(package, version, tmp_path, venv_python, setuptools_wheel):
    venv_pip = (venv_python, "-m", "pip")
    sdist = retrieve_sdist(package, version, tmp_path)
    deps = build_deps(package, sdist)
    if deps:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Dependencies:", deps)
        run([*venv_pip, "install", *deps])

    # Use a virtualenv to simulate PEP 517 isolation
    # but install fresh setuptools wheel to ensure the version under development
    env = EXTRA_ENV_VARS.get(package, {})
    run([*venv_pip, "install", "--force-reinstall", setuptools_wheel])
    run([*venv_pip, "install", *INSTALL_OPTIONS, sdist], env)

    # Execute a simple script to make sure the package was installed correctly
    pkg = IMPORT_NAME.get(package, package).replace("-", "_")
    script = f"import {pkg}; print(getattr({pkg}, '__version__', 0))"
    run([venv_python, "-c", script])


# ---- Helper Functions ----


def retrieve_sdist(package, version, tmp_path):
    """Either use cached sdist file or download it from PyPI"""
    # `pip download` cannot be used due to
    # https://github.com/pypa/pip/issues/1884
    # https://discuss.python.org/t/pep-625-file-name-of-a-source-distribution/4686
    # We have to find the correct distribution file and download it
    download_path = os.getenv("DOWNLOAD_PATH", str(tmp_path))
    dist = retrieve_pypi_sdist_metadata(package, version)

    # Remove old files to prevent cache to grow indefinitely
    for file in glob(os.path.join(download_path, f"{package}*")):
        if dist["filename"] != file:
            os.unlink(file)

    dist_file = os.path.join(download_path, dist["filename"])
    if not os.path.exists(dist_file):
        download(dist["url"], dist_file, dist["md5_digest"])
    return dist_file


def retrieve_pypi_sdist_metadata(package, version):
    # https://warehouse.pypa.io/api-reference/json.html
    id_ = package if version is LATEST else f"{package}/{version}"
    with urlopen(f"https://pypi.org/pypi/{id_}/json") as f:
        metadata = json.load(f)

    if metadata["info"]["yanked"]:
        raise ValueError(f"Release for {package} {version} was yanked")

    version = metadata["info"]["version"]
    release = metadata["releases"][version] if version is LATEST else metadata["urls"]
    (sdist,) = filter(lambda d: d["packagetype"] == "sdist", release)
    return sdist


def download(url, dest, md5_digest):
    with urlopen(url) as f:
        data = f.read()

    assert md5(data).hexdigest() == md5_digest

    with open(dest, "wb") as f:
        f.write(data)

    assert os.path.exists(dest)


def build_deps(package, sdist_file):
    """Find out what are the build dependencies for a package.

    "Manually" install them, since pip will not install build
    deps with `--no-build-isolation`.
    """
    # delay importing, since pytest discovery phase may hit this file from a
    # testenv without tomli
    from setuptools.compat.py310 import tomllib

    archive = Archive(sdist_file)
    info = tomllib.loads(_read_pyproject(archive))
    deps = info.get("build-system", {}).get("requires", [])
    deps += EXTRA_BUILD_DEPS.get(package, [])
    # Remove setuptools from requirements (and deduplicate)
    requirements = {Requirement(d).name: d for d in deps}
    return [v for k, v in requirements.items() if k != "setuptools"]


def _read_pyproject(archive):
    contents = (
        archive.get_content(member)
        for member in archive
        if os.path.basename(archive.get_name(member)) == "pyproject.toml"
    )
    return next(contents, "")
