"""
Backward compatibility for homebrew builds on macOS.
"""

import functools
import os
import subprocess
import sys
import sysconfig


@functools.lru_cache
def enabled():
    """
    Only enabled for Python 3.9 framework homebrew builds
    except ensurepip and venv.
    """
    PY39 = (3, 9) < sys.version_info < (3, 10)
    framework = sys.platform == 'darwin' and sys._framework
    homebrew = "Cellar" in sysconfig.get_config_var('projectbase')
    venv = sys.prefix != sys.base_prefix
    ensurepip = os.environ.get("ENSUREPIP_OPTIONS")
    return PY39 and framework and homebrew and not venv and not ensurepip


schemes = dict(
    osx_framework_library=dict(
        stdlib='{installed_base}/{platlibdir}/python{py_version_short}',
        platstdlib='{platbase}/{platlibdir}/python{py_version_short}',
        purelib='{homebrew_prefix}/lib/python{py_version_short}/site-packages',
        platlib='{homebrew_prefix}/{platlibdir}/python{py_version_short}/site-packages',
        include='{installed_base}/include/python{py_version_short}{abiflags}',
        platinclude='{installed_platbase}/include/python{py_version_short}{abiflags}',
        scripts='{homebrew_prefix}/bin',
        data='{homebrew_prefix}',
    )
)


@functools.lru_cache
def vars():
    if not enabled():
        return {}
    homebrew_prefix = subprocess.check_output(['brew', '--prefix'], text=True).strip()
    return locals()


def scheme(name):
    """
    Override the selected scheme for posix_prefix.
    """
    if not enabled() or not name.endswith('_prefix'):
        return name
    return 'osx_framework_library'
