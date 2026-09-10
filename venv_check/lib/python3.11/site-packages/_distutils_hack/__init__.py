# don't import any costly modules
import os
import sys

report_url = (
    "https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml"
)


def warn_distutils_present():
    if 'distutils' not in sys.modules:
        return
    import warnings

    warnings.warn(
        "Distutils was imported before Setuptools, but importing Setuptools "
        "also replaces the `distutils` module in `sys.modules`. This may lead "
        "to undesirable behaviors or errors. To avoid these issues, avoid "
        "using distutils directly, ensure that setuptools is installed in the "
        "traditional way (e.g. not an editable install), and/or make sure "
        "that setuptools is always imported before distutils."
    )


def clear_distutils():
    if 'distutils' not in sys.modules:
        return
    import warnings

    warnings.warn(
        "Setuptools is replacing distutils. Support for replacing "
        "an already imported distutils is deprecated. In the future, "
        "this condition will fail. "
        f"Register concerns at {report_url}"
    )
    mods = [
        name
        for name in sys.modules
        if name == "distutils" or name.startswith("distutils.")
    ]
    for name in mods:
        del sys.modules[name]


def enabled():
    """
    Allow selection of distutils by environment variable.
    """
    which = os.environ.get('SETUPTOOLS_USE_DISTUTILS', 'local')
    if which == 'stdlib':
        import warnings

        warnings.warn(
            "Reliance on distutils from stdlib is deprecated. Users "
            "must rely on setuptools to provide the distutils module. "
            "Avoid importing distutils or import setuptools first, "
            "and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. "
            f"Register concerns at {report_url}"
        )
    return which == 'local'


def ensure_local_distutils():
    import importlib

    clear_distutils()

    # With the DistutilsMetaFinder in place,
    # perform an import to cause distutils to be
    # loaded from setuptools._distutils. Ref #2906.
    with shim():
        importlib.import_module('distutils')

    # check that submodules load as expected
    core = importlib.import_module('distutils.core')
    assert '_distutils' in core.__file__, core.__file__
    assert 'setuptools._distutils.log' not in sys.modules


def do_override():
    """
    Ensure that the local copy of distutils is preferred over stdlib.

    See https://github.com/pypa/setuptools/issues/417#issuecomment-392298401
    for more motivation.
    """
    if enabled():
        warn_distutils_present()
        ensure_local_distutils()


class _TrivialRe:
    def __init__(self, *patterns) -> None:
        self._patterns = patterns

    def match(self, string):
        return all(pat in string for pat in self._patterns)


class DistutilsMetaFinder:
    def find_spec(self, fullname, path, target=None):
        # optimization: only consider top level modules and those
        # found in the CPython test suite.
        if path is not None and not fullname.startswith('test.'):
            return None

        method_name = 'spec_for_{fullname}'.format(**locals())
        method = getattr(self, method_name, lambda: None)
        return method()

    def spec_for_distutils(self):
        if self.is_cpython():
            return None

        import importlib
        import importlib.abc
        import importlib.util

        try:
            mod = importlib.import_module('setuptools._distutils')
        except Exception:
            # There are a couple of cases where setuptools._distutils
            # may not be present:
            # - An older Setuptools without a local distutils is
            #   taking precedence. Ref #2957.
            # - Path manipulation during sitecustomize removes
            #   setuptools from the path but only after the hook
            #   has been loaded. Ref #2980.
            # In either case, fall back to stdlib behavior.
            return None

        class DistutilsLoader(importlib.abc.Loader):
            def create_module(self, spec):
                mod.__name__ = 'distutils'
                return mod

            def exec_module(self, module):
                pass

        return importlib.util.spec_from_loader(
            'distutils', DistutilsLoader(), origin=mod.__file__
        )

    @staticmethod
    def is_cpython():
        """
        Suppress supplying distutils for CPython (build and tests).
        Ref #2965 and #3007.
        """
        return os.path.isfile('pybuilddir.txt')

    def spec_for_pip(self):
        """
        Ensure stdlib distutils when running under pip.
        See pypa/pip#8761 for rationale.
        """
        if sys.version_info >= (3, 12) or self.pip_imported_during_build():
            return
        clear_distutils()
        self.spec_for_distutils = lambda: None

    @classmethod
    def pip_imported_during_build(cls):
        """
        Detect if pip is being imported in a build script. Ref #2355.
        """
        import traceback

        return any(
            cls.frame_file_is_setup(frame) for frame, line in traceback.walk_stack(None)
        )

    @staticmethod
    def frame_file_is_setup(frame):
        """
        Return True if the indicated frame suggests a setup.py file.
        """
        # some frames may not have __file__ (#2940)
        return frame.f_globals.get('__file__', '').endswith('setup.py')

    def spec_for_sensitive_tests(self):
        """
        Ensure stdlib distutils when running select tests under CPython.

        python/cpython#91169
        """
        clear_distutils()
        self.spec_for_distutils = lambda: None

    sensitive_tests = (
        [
            'test.test_distutils',
            'test.test_peg_generator',
            'test.test_importlib',
        ]
        if sys.version_info < (3, 10)
        else [
            'test.test_distutils',
        ]
    )


for name in DistutilsMetaFinder.sensitive_tests:
    setattr(
        DistutilsMetaFinder,
        f'spec_for_{name}',
        DistutilsMetaFinder.spec_for_sensitive_tests,
    )


DISTUTILS_FINDER = DistutilsMetaFinder()


def add_shim():
    DISTUTILS_FINDER in sys.meta_path or insert_shim()


class shim:
    def __enter__(self) -> None:
        insert_shim()

    def __exit__(self, exc: object, value: object, tb: object) -> None:
        _remove_shim()


def insert_shim():
    sys.meta_path.insert(0, DISTUTILS_FINDER)


def _remove_shim():
    try:
        sys.meta_path.remove(DISTUTILS_FINDER)
    except ValueError:
        pass


if sys.version_info < (3, 12):
    # DistutilsMetaFinder can only be disabled in Python < 3.12 (PEP 632)
    remove_shim = _remove_shim
