import contextlib
import glob
import importlib
import os.path
import platform
import re
import shutil
import site
import subprocess
import sys
import tempfile
import textwrap
import time
from distutils import sysconfig
from distutils.command.build_ext import build_ext
from distutils.core import Distribution
from distutils.errors import (
    CompileError,
    DistutilsPlatformError,
    DistutilsSetupError,
    UnknownFileError,
)
from distutils.extension import Extension
from distutils.tests import missing_compiler_executable
from distutils.tests.support import TempdirManager, copy_xxmodule_c, fixup_build_ext
from io import StringIO

import jaraco.path
import path
import pytest
from test import support

from .compat import py39 as import_helper


@pytest.fixture()
def user_site_dir(request):
    self = request.instance
    self.tmp_dir = self.mkdtemp()
    self.tmp_path = path.Path(self.tmp_dir)
    from distutils.command import build_ext

    orig_user_base = site.USER_BASE

    site.USER_BASE = self.mkdtemp()
    build_ext.USER_BASE = site.USER_BASE

    # bpo-30132: On Windows, a .pdb file may be created in the current
    # working directory. Create a temporary working directory to cleanup
    # everything at the end of the test.
    with self.tmp_path:
        yield

    site.USER_BASE = orig_user_base
    build_ext.USER_BASE = orig_user_base

    if sys.platform == 'cygwin':
        time.sleep(1)


@contextlib.contextmanager
def safe_extension_import(name, path):
    with import_helper.CleanImport(name):
        with extension_redirect(name, path) as new_path:
            with import_helper.DirsOnSysPath(new_path):
                yield


@contextlib.contextmanager
def extension_redirect(mod, path):
    """
    Tests will fail to tear down an extension module if it's been imported.

    Before importing, copy the file to a temporary directory that won't
    be cleaned up. Yield the new path.
    """
    if platform.system() != "Windows" and sys.platform != "cygwin":
        yield path
        return
    with import_helper.DirsOnSysPath(path):
        spec = importlib.util.find_spec(mod)
    filename = os.path.basename(spec.origin)
    trash_dir = tempfile.mkdtemp(prefix='deleteme')
    dest = os.path.join(trash_dir, os.path.basename(filename))
    shutil.copy(spec.origin, dest)
    yield trash_dir
    # TODO: can the file be scheduled for deletion?


@pytest.mark.usefixtures('user_site_dir')
class TestBuildExt(TempdirManager):
    def build_ext(self, *args, **kwargs):
        return build_ext(*args, **kwargs)

    @pytest.mark.parametrize("copy_so", [False])
    def test_build_ext(self, copy_so):
        missing_compiler_executable()
        copy_xxmodule_c(self.tmp_dir)
        xx_c = os.path.join(self.tmp_dir, 'xxmodule.c')
        xx_ext = Extension('xx', [xx_c])
        if sys.platform != "win32":
            if not copy_so:
                xx_ext = Extension(
                    'xx',
                    [xx_c],
                    library_dirs=['/usr/lib'],
                    libraries=['z'],
                    runtime_library_dirs=['/usr/lib'],
                )
            elif sys.platform == 'linux':
                libz_so = {
                    os.path.realpath(name) for name in glob.iglob('/usr/lib*/libz.so*')
                }
                libz_so = sorted(libz_so, key=lambda lib_path: len(lib_path))
                shutil.copyfile(libz_so[-1], '/tmp/libxx_z.so')

                xx_ext = Extension(
                    'xx',
                    [xx_c],
                    library_dirs=['/tmp'],
                    libraries=['xx_z'],
                    runtime_library_dirs=['/tmp'],
                )
        dist = Distribution({'name': 'xx', 'ext_modules': [xx_ext]})
        dist.package_dir = self.tmp_dir
        cmd = self.build_ext(dist)
        fixup_build_ext(cmd)
        cmd.build_lib = self.tmp_dir
        cmd.build_temp = self.tmp_dir

        old_stdout = sys.stdout
        if not support.verbose:
            # silence compiler output
            sys.stdout = StringIO()
        try:
            cmd.ensure_finalized()
            cmd.run()
        finally:
            sys.stdout = old_stdout

        with safe_extension_import('xx', self.tmp_dir):
            self._test_xx(copy_so)

        if sys.platform == 'linux' and copy_so:
            os.unlink('/tmp/libxx_z.so')

    @staticmethod
    def _test_xx(copy_so):
        import xx  # type: ignore[import-not-found] # Module generated for tests

        for attr in ('error', 'foo', 'new', 'roj'):
            assert hasattr(xx, attr)

        assert xx.foo(2, 5) == 7
        assert xx.foo(13, 15) == 28
        assert xx.new().demo() is None
        if support.HAVE_DOCSTRINGS:
            doc = 'This is a template module just for instruction.'
            assert xx.__doc__ == doc
        assert isinstance(xx.Null(), xx.Null)
        assert isinstance(xx.Str(), xx.Str)

        if sys.platform == 'linux':
            so_headers = subprocess.check_output(
                ["readelf", "-d", xx.__file__], universal_newlines=True
            )
            import pprint

            pprint.pprint(so_headers)
            rpaths = [
                rpath
                for line in so_headers.split("\n")
                if "RPATH" in line or "RUNPATH" in line
                for rpath in line.split()[2][1:-1].split(":")
            ]
            if not copy_so:
                pprint.pprint(rpaths)
                # Linked against a library in /usr/lib{,64}
                assert "/usr/lib" not in rpaths and "/usr/lib64" not in rpaths
            else:
                # Linked against a library in /tmp
                assert "/tmp" in rpaths
                # The import is the real test here

    def test_solaris_enable_shared(self):
        dist = Distribution({'name': 'xx'})
        cmd = self.build_ext(dist)
        old = sys.platform

        sys.platform = 'sunos'  # fooling finalize_options
        from distutils.sysconfig import _config_vars

        old_var = _config_vars.get('Py_ENABLE_SHARED')
        _config_vars['Py_ENABLE_SHARED'] = True
        try:
            cmd.ensure_finalized()
        finally:
            sys.platform = old
            if old_var is None:
                del _config_vars['Py_ENABLE_SHARED']
            else:
                _config_vars['Py_ENABLE_SHARED'] = old_var

        # make sure we get some library dirs under solaris
        assert len(cmd.library_dirs) > 0

    def test_user_site(self):
        import site

        dist = Distribution({'name': 'xx'})
        cmd = self.build_ext(dist)

        # making sure the user option is there
        options = [name for name, short, label in cmd.user_options]
        assert 'user' in options

        # setting a value
        cmd.user = True

        # setting user based lib and include
        lib = os.path.join(site.USER_BASE, 'lib')
        incl = os.path.join(site.USER_BASE, 'include')
        os.mkdir(lib)
        os.mkdir(incl)

        # let's run finalize
        cmd.ensure_finalized()

        # see if include_dirs and library_dirs
        # were set
        assert lib in cmd.library_dirs
        assert lib in cmd.rpath
        assert incl in cmd.include_dirs

    def test_optional_extension(self):
        # this extension will fail, but let's ignore this failure
        # with the optional argument.
        modules = [Extension('foo', ['xxx'], optional=False)]
        dist = Distribution({'name': 'xx', 'ext_modules': modules})
        cmd = self.build_ext(dist)
        cmd.ensure_finalized()
        with pytest.raises((UnknownFileError, CompileError)):
            cmd.run()  # should raise an error

        modules = [Extension('foo', ['xxx'], optional=True)]
        dist = Distribution({'name': 'xx', 'ext_modules': modules})
        cmd = self.build_ext(dist)
        cmd.ensure_finalized()
        cmd.run()  # should pass

    def test_finalize_options(self):
        # Make sure Python's include directories (for Python.h, pyconfig.h,
        # etc.) are in the include search path.
        modules = [Extension('foo', ['xxx'], optional=False)]
        dist = Distribution({'name': 'xx', 'ext_modules': modules})
        cmd = self.build_ext(dist)
        cmd.finalize_options()

        py_include = sysconfig.get_python_inc()
        for p in py_include.split(os.path.pathsep):
            assert p in cmd.include_dirs

        plat_py_include = sysconfig.get_python_inc(plat_specific=True)
        for p in plat_py_include.split(os.path.pathsep):
            assert p in cmd.include_dirs

        # make sure cmd.libraries is turned into a list
        # if it's a string
        cmd = self.build_ext(dist)
        cmd.libraries = 'my_lib, other_lib lastlib'
        cmd.finalize_options()
        assert cmd.libraries == ['my_lib', 'other_lib', 'lastlib']

        # make sure cmd.library_dirs is turned into a list
        # if it's a string
        cmd = self.build_ext(dist)
        cmd.library_dirs = f'my_lib_dir{os.pathsep}other_lib_dir'
        cmd.finalize_options()
        assert 'my_lib_dir' in cmd.library_dirs
        assert 'other_lib_dir' in cmd.library_dirs

        # make sure rpath is turned into a list
        # if it's a string
        cmd = self.build_ext(dist)
        cmd.rpath = f'one{os.pathsep}two'
        cmd.finalize_options()
        assert cmd.rpath == ['one', 'two']

        # make sure cmd.link_objects is turned into a list
        # if it's a string
        cmd = build_ext(dist)
        cmd.link_objects = 'one two,three'
        cmd.finalize_options()
        assert cmd.link_objects == ['one', 'two', 'three']

        # XXX more tests to perform for win32

        # make sure define is turned into 2-tuples
        # strings if they are ','-separated strings
        cmd = self.build_ext(dist)
        cmd.define = 'one,two'
        cmd.finalize_options()
        assert cmd.define == [('one', '1'), ('two', '1')]

        # make sure undef is turned into a list of
        # strings if they are ','-separated strings
        cmd = self.build_ext(dist)
        cmd.undef = 'one,two'
        cmd.finalize_options()
        assert cmd.undef == ['one', 'two']

        # make sure swig_opts is turned into a list
        cmd = self.build_ext(dist)
        cmd.swig_opts = None
        cmd.finalize_options()
        assert cmd.swig_opts == []

        cmd = self.build_ext(dist)
        cmd.swig_opts = '1 2'
        cmd.finalize_options()
        assert cmd.swig_opts == ['1', '2']

    def test_check_extensions_list(self):
        dist = Distribution()
        cmd = self.build_ext(dist)
        cmd.finalize_options()

        # 'extensions' option must be a list of Extension instances
        with pytest.raises(DistutilsSetupError):
            cmd.check_extensions_list('foo')

        # each element of 'ext_modules' option must be an
        # Extension instance or 2-tuple
        exts = [('bar', 'foo', 'bar'), 'foo']
        with pytest.raises(DistutilsSetupError):
            cmd.check_extensions_list(exts)

        # first element of each tuple in 'ext_modules'
        # must be the extension name (a string) and match
        # a python dotted-separated name
        exts = [('foo-bar', '')]
        with pytest.raises(DistutilsSetupError):
            cmd.check_extensions_list(exts)

        # second element of each tuple in 'ext_modules'
        # must be a dictionary (build info)
        exts = [('foo.bar', '')]
        with pytest.raises(DistutilsSetupError):
            cmd.check_extensions_list(exts)

        # ok this one should pass
        exts = [('foo.bar', {'sources': [''], 'libraries': 'foo', 'some': 'bar'})]
        cmd.check_extensions_list(exts)
        ext = exts[0]
        assert isinstance(ext, Extension)

        # check_extensions_list adds in ext the values passed
        # when they are in ('include_dirs', 'library_dirs', 'libraries'
        # 'extra_objects', 'extra_compile_args', 'extra_link_args')
        assert ext.libraries == 'foo'
        assert not hasattr(ext, 'some')

        # 'macros' element of build info dict must be 1- or 2-tuple
        exts = [
            (
                'foo.bar',
                {
                    'sources': [''],
                    'libraries': 'foo',
                    'some': 'bar',
                    'macros': [('1', '2', '3'), 'foo'],
                },
            )
        ]
        with pytest.raises(DistutilsSetupError):
            cmd.check_extensions_list(exts)

        exts[0][1]['macros'] = [('1', '2'), ('3',)]
        cmd.check_extensions_list(exts)
        assert exts[0].undef_macros == ['3']
        assert exts[0].define_macros == [('1', '2')]

    def test_get_source_files(self):
        modules = [Extension('foo', ['xxx'], optional=False)]
        dist = Distribution({'name': 'xx', 'ext_modules': modules})
        cmd = self.build_ext(dist)
        cmd.ensure_finalized()
        assert cmd.get_source_files() == ['xxx']

    def test_unicode_module_names(self):
        modules = [
            Extension('foo', ['aaa'], optional=False),
            Extension('föö', ['uuu'], optional=False),
        ]
        dist = Distribution({'name': 'xx', 'ext_modules': modules})
        cmd = self.build_ext(dist)
        cmd.ensure_finalized()
        assert re.search(r'foo(_d)?\..*', cmd.get_ext_filename(modules[0].name))
        assert re.search(r'föö(_d)?\..*', cmd.get_ext_filename(modules[1].name))
        assert cmd.get_export_symbols(modules[0]) == ['PyInit_foo']
        assert cmd.get_export_symbols(modules[1]) == ['PyInitU_f_1gaa']

    def test_export_symbols__init__(self):
        # https://github.com/python/cpython/issues/80074
        # https://github.com/pypa/setuptools/issues/4826
        modules = [
            Extension('foo.__init__', ['aaa']),
            Extension('föö.__init__', ['uuu']),
        ]
        dist = Distribution({'name': 'xx', 'ext_modules': modules})
        cmd = self.build_ext(dist)
        cmd.ensure_finalized()
        assert cmd.get_export_symbols(modules[0]) == ['PyInit_foo']
        assert cmd.get_export_symbols(modules[1]) == ['PyInitU_f_1gaa']

    def test_compiler_option(self):
        # cmd.compiler is an option and
        # should not be overridden by a compiler instance
        # when the command is run
        dist = Distribution()
        cmd = self.build_ext(dist)
        cmd.compiler = 'unix'
        cmd.ensure_finalized()
        cmd.run()
        assert cmd.compiler == 'unix'

    def test_get_outputs(self):
        missing_compiler_executable()
        tmp_dir = self.mkdtemp()
        c_file = os.path.join(tmp_dir, 'foo.c')
        self.write_file(c_file, 'void PyInit_foo(void) {}\n')
        ext = Extension('foo', [c_file], optional=False)
        dist = Distribution({'name': 'xx', 'ext_modules': [ext]})
        cmd = self.build_ext(dist)
        fixup_build_ext(cmd)
        cmd.ensure_finalized()
        assert len(cmd.get_outputs()) == 1

        cmd.build_lib = os.path.join(self.tmp_dir, 'build')
        cmd.build_temp = os.path.join(self.tmp_dir, 'tempt')

        # issue #5977 : distutils build_ext.get_outputs
        # returns wrong result with --inplace
        other_tmp_dir = os.path.realpath(self.mkdtemp())
        old_wd = os.getcwd()
        os.chdir(other_tmp_dir)
        try:
            cmd.inplace = True
            cmd.run()
            so_file = cmd.get_outputs()[0]
        finally:
            os.chdir(old_wd)
        assert os.path.exists(so_file)
        ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
        assert so_file.endswith(ext_suffix)
        so_dir = os.path.dirname(so_file)
        assert so_dir == other_tmp_dir

        cmd.inplace = False
        cmd.compiler = None
        cmd.run()
        so_file = cmd.get_outputs()[0]
        assert os.path.exists(so_file)
        assert so_file.endswith(ext_suffix)
        so_dir = os.path.dirname(so_file)
        assert so_dir == cmd.build_lib

        # inplace = False, cmd.package = 'bar'
        build_py = cmd.get_finalized_command('build_py')
        build_py.package_dir = {'': 'bar'}
        path = cmd.get_ext_fullpath('foo')
        # checking that the last directory is the build_dir
        path = os.path.split(path)[0]
        assert path == cmd.build_lib

        # inplace = True, cmd.package = 'bar'
        cmd.inplace = True
        other_tmp_dir = os.path.realpath(self.mkdtemp())
        old_wd = os.getcwd()
        os.chdir(other_tmp_dir)
        try:
            path = cmd.get_ext_fullpath('foo')
        finally:
            os.chdir(old_wd)
        # checking that the last directory is bar
        path = os.path.split(path)[0]
        lastdir = os.path.split(path)[-1]
        assert lastdir == 'bar'

    def test_ext_fullpath(self):
        ext = sysconfig.get_config_var('EXT_SUFFIX')
        # building lxml.etree inplace
        # etree_c = os.path.join(self.tmp_dir, 'lxml.etree.c')
        # etree_ext = Extension('lxml.etree', [etree_c])
        # dist = Distribution({'name': 'lxml', 'ext_modules': [etree_ext]})
        dist = Distribution()
        cmd = self.build_ext(dist)
        cmd.inplace = True
        cmd.distribution.package_dir = {'': 'src'}
        cmd.distribution.packages = ['lxml', 'lxml.html']
        curdir = os.getcwd()
        wanted = os.path.join(curdir, 'src', 'lxml', 'etree' + ext)
        path = cmd.get_ext_fullpath('lxml.etree')
        assert wanted == path

        # building lxml.etree not inplace
        cmd.inplace = False
        cmd.build_lib = os.path.join(curdir, 'tmpdir')
        wanted = os.path.join(curdir, 'tmpdir', 'lxml', 'etree' + ext)
        path = cmd.get_ext_fullpath('lxml.etree')
        assert wanted == path

        # building twisted.runner.portmap not inplace
        build_py = cmd.get_finalized_command('build_py')
        build_py.package_dir = {}
        cmd.distribution.packages = ['twisted', 'twisted.runner.portmap']
        path = cmd.get_ext_fullpath('twisted.runner.portmap')
        wanted = os.path.join(curdir, 'tmpdir', 'twisted', 'runner', 'portmap' + ext)
        assert wanted == path

        # building twisted.runner.portmap inplace
        cmd.inplace = True
        path = cmd.get_ext_fullpath('twisted.runner.portmap')
        wanted = os.path.join(curdir, 'twisted', 'runner', 'portmap' + ext)
        assert wanted == path

    @pytest.mark.skipif('platform.system() != "Darwin"')
    @pytest.mark.usefixtures('save_env')
    def test_deployment_target_default(self):
        # Issue 9516: Test that, in the absence of the environment variable,
        # an extension module is compiled with the same deployment target as
        #  the interpreter.
        self._try_compile_deployment_target('==', None)

    @pytest.mark.skipif('platform.system() != "Darwin"')
    @pytest.mark.usefixtures('save_env')
    def test_deployment_target_too_low(self):
        # Issue 9516: Test that an extension module is not allowed to be
        # compiled with a deployment target less than that of the interpreter.
        with pytest.raises(DistutilsPlatformError):
            self._try_compile_deployment_target('>', '10.1')

    @pytest.mark.skipif('platform.system() != "Darwin"')
    @pytest.mark.usefixtures('save_env')
    def test_deployment_target_higher_ok(self):  # pragma: no cover
        # Issue 9516: Test that an extension module can be compiled with a
        # deployment target higher than that of the interpreter: the ext
        # module may depend on some newer OS feature.
        deptarget = sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET')
        if deptarget:
            # increment the minor version number (i.e. 10.6 -> 10.7)
            deptarget = [int(x) for x in deptarget.split('.')]
            deptarget[-1] += 1
            deptarget = '.'.join(str(i) for i in deptarget)
            self._try_compile_deployment_target('<', deptarget)

    def _try_compile_deployment_target(self, operator, target):  # pragma: no cover
        if target is None:
            if os.environ.get('MACOSX_DEPLOYMENT_TARGET'):
                del os.environ['MACOSX_DEPLOYMENT_TARGET']
        else:
            os.environ['MACOSX_DEPLOYMENT_TARGET'] = target

        jaraco.path.build(
            {
                'deptargetmodule.c': textwrap.dedent(f"""\
                    #include <AvailabilityMacros.h>

                    int dummy;

                    #if TARGET {operator} MAC_OS_X_VERSION_MIN_REQUIRED
                    #else
                    #error "Unexpected target"
                    #endif

                    """),
            },
            self.tmp_path,
        )

        # get the deployment target that the interpreter was built with
        target = sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET')
        target = tuple(map(int, target.split('.')[0:2]))
        # format the target value as defined in the Apple
        # Availability Macros.  We can't use the macro names since
        # at least one value we test with will not exist yet.
        if target[:2] < (10, 10):
            # for 10.1 through 10.9.x -> "10n0"
            tmpl = '{:02}{:01}0'
        else:
            # for 10.10 and beyond -> "10nn00"
            if len(target) >= 2:
                tmpl = '{:02}{:02}00'
            else:
                # 11 and later can have no minor version (11 instead of 11.0)
                tmpl = '{:02}0000'
        target = tmpl.format(*target)
        deptarget_ext = Extension(
            'deptarget',
            [self.tmp_path / 'deptargetmodule.c'],
            extra_compile_args=[f'-DTARGET={target}'],
        )
        dist = Distribution({'name': 'deptarget', 'ext_modules': [deptarget_ext]})
        dist.package_dir = self.tmp_dir
        cmd = self.build_ext(dist)
        cmd.build_lib = self.tmp_dir
        cmd.build_temp = self.tmp_dir

        try:
            old_stdout = sys.stdout
            if not support.verbose:
                # silence compiler output
                sys.stdout = StringIO()
            try:
                cmd.ensure_finalized()
                cmd.run()
            finally:
                sys.stdout = old_stdout

        except CompileError:
            self.fail("Wrong deployment target during compilation")


class TestParallelBuildExt(TestBuildExt):
    def build_ext(self, *args, **kwargs):
        build_ext = super().build_ext(*args, **kwargs)
        build_ext.parallel = True
        return build_ext
