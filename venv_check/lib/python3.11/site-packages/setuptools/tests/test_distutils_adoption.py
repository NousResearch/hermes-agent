import os
import platform
import sys
import textwrap

import pytest

IS_PYPY = '__pypy__' in sys.builtin_module_names

_TEXT_KWARGS = {"text": True, "encoding": "utf-8"}  # For subprocess.run


def win_sr(env):
    """
    On Windows, SYSTEMROOT must be present to avoid

    > Fatal Python error: _Py_HashRandomization_Init: failed to
    > get random numbers to initialize Python
    """
    if env and platform.system() == 'Windows':
        env['SYSTEMROOT'] = os.environ['SYSTEMROOT']
    return env


def find_distutils(venv, imports='distutils', env=None, **kwargs):
    py_cmd = 'import {imports}; print(distutils.__file__)'.format(**locals())
    cmd = ['python', '-c', py_cmd]
    return venv.run(cmd, env=win_sr(env), **_TEXT_KWARGS, **kwargs)


def count_meta_path(venv, env=None):
    py_cmd = textwrap.dedent(
        """
        import sys
        is_distutils = lambda finder: finder.__class__.__name__ == "DistutilsMetaFinder"
        print(len(list(filter(is_distutils, sys.meta_path))))
        """
    )
    cmd = ['python', '-c', py_cmd]
    return int(venv.run(cmd, env=win_sr(env), **_TEXT_KWARGS))


skip_without_stdlib_distutils = pytest.mark.skipif(
    sys.version_info >= (3, 12),
    reason='stdlib distutils is removed from Python 3.12+',
)


@skip_without_stdlib_distutils
def test_distutils_stdlib(venv):
    """
    Ensure stdlib distutils is used when appropriate.
    """
    env = dict(SETUPTOOLS_USE_DISTUTILS='stdlib')
    assert venv.name not in find_distutils(venv, env=env).split(os.sep)
    assert count_meta_path(venv, env=env) == 0


def test_distutils_local_with_setuptools(venv):
    """
    Ensure local distutils is used when appropriate.
    """
    env = dict(SETUPTOOLS_USE_DISTUTILS='local')
    loc = find_distutils(venv, imports='setuptools, distutils', env=env)
    assert venv.name in loc.split(os.sep)
    assert count_meta_path(venv, env=env) <= 1


@pytest.mark.xfail('IS_PYPY', reason='pypy imports distutils on startup')
def test_distutils_local(venv):
    """
    Even without importing, the setuptools-local copy of distutils is
    preferred.
    """
    env = dict(SETUPTOOLS_USE_DISTUTILS='local')
    assert venv.name in find_distutils(venv, env=env).split(os.sep)
    assert count_meta_path(venv, env=env) <= 1


def test_pip_import(venv):
    """
    Ensure pip can be imported.
    Regression test for #3002.
    """
    cmd = ['python', '-c', 'import pip']
    venv.run(cmd, **_TEXT_KWARGS)


def test_distutils_has_origin():
    """
    Distutils module spec should have an origin. #2990.
    """
    assert __import__('distutils').__spec__.origin


ENSURE_IMPORTS_ARE_NOT_DUPLICATED = r"""
# Depending on the importlib machinery and _distutils_hack, some imports are
# duplicated resulting in different module objects being loaded, which prevents
# patches as shown in #3042.
# This script provides a way of verifying if this duplication is happening.

from distutils import cmd
import distutils.command.sdist as sdist

# import last to prevent caching
from distutils import {imported_module}

for mod in (cmd, sdist):
    assert mod.{imported_module} == {imported_module}, (
        f"\n{{mod.dir_util}}\n!=\n{{{imported_module}}}"
    )

print("success")
"""


@pytest.mark.usefixtures("tmpdir_cwd")
@pytest.mark.parametrize(
    ('distutils_version', 'imported_module'),
    [
        pytest.param("stdlib", "dir_util", marks=skip_without_stdlib_distutils),
        pytest.param("stdlib", "file_util", marks=skip_without_stdlib_distutils),
        pytest.param("stdlib", "archive_util", marks=skip_without_stdlib_distutils),
        ("local", "dir_util"),
        ("local", "file_util"),
        ("local", "archive_util"),
    ],
)
def test_modules_are_not_duplicated_on_import(distutils_version, imported_module, venv):
    env = dict(SETUPTOOLS_USE_DISTUTILS=distutils_version)
    script = ENSURE_IMPORTS_ARE_NOT_DUPLICATED.format(imported_module=imported_module)
    cmd = ['python', '-c', script]
    output = venv.run(cmd, env=win_sr(env), **_TEXT_KWARGS).strip()
    assert output == "success"


ENSURE_LOG_IMPORT_IS_NOT_DUPLICATED = r"""
import types
import distutils.dist as dist
from distutils import log
if isinstance(dist.log, types.ModuleType):
    assert dist.log == log, f"\n{dist.log}\n!=\n{log}"
print("success")
"""


@pytest.mark.usefixtures("tmpdir_cwd")
@pytest.mark.parametrize(
    "distutils_version",
    [
        "local",
        pytest.param("stdlib", marks=skip_without_stdlib_distutils),
    ],
)
def test_log_module_is_not_duplicated_on_import(distutils_version, venv):
    env = dict(SETUPTOOLS_USE_DISTUTILS=distutils_version)
    cmd = ['python', '-c', ENSURE_LOG_IMPORT_IS_NOT_DUPLICATED]
    output = venv.run(cmd, env=win_sr(env), **_TEXT_KWARGS).strip()
    assert output == "success"


ENSURE_CONSISTENT_ERROR_FROM_MODIFIED_PY = r"""
from setuptools.modified import newer
from {imported_module}.errors import DistutilsError

# Can't use pytest.raises in this context
try:
    newer("", "")
except DistutilsError:
    print("success")
else:
    raise AssertionError("Expected to raise")
"""


@pytest.mark.usefixtures("tmpdir_cwd")
@pytest.mark.parametrize(
    ('distutils_version', 'imported_module'),
    [
        ("local", "distutils"),
        # Unfortunately we still get ._distutils.errors.DistutilsError with SETUPTOOLS_USE_DISTUTILS=stdlib
        # But that's a deprecated use-case we don't mind not fully supporting in newer code
        pytest.param(
            "stdlib", "setuptools._distutils", marks=skip_without_stdlib_distutils
        ),
    ],
)
def test_consistent_error_from_modified_py(distutils_version, imported_module, venv):
    env = dict(SETUPTOOLS_USE_DISTUTILS=distutils_version)
    cmd = [
        'python',
        '-c',
        ENSURE_CONSISTENT_ERROR_FROM_MODIFIED_PY.format(
            imported_module=imported_module
        ),
    ]
    output = venv.run(cmd, env=win_sr(env), **_TEXT_KWARGS).strip()
    assert output == "success"
