"""Tests for distutils.sysconfig."""

import contextlib
import distutils
import os
import pathlib
import subprocess
import sys
from distutils import sysconfig
from distutils.ccompiler import new_compiler  # noqa: F401
from distutils.unixccompiler import UnixCCompiler

import jaraco.envs
import path
import pytest
from jaraco.text import trim
from test.support import swap_item


def _gen_makefile(root, contents):
    jaraco.path.build({'Makefile': trim(contents)}, root)
    return root / 'Makefile'


@pytest.mark.usefixtures('save_env')
class TestSysconfig:
    def test_get_config_h_filename(self):
        config_h = sysconfig.get_config_h_filename()
        assert os.path.isfile(config_h)

    @pytest.mark.skipif("platform.system() == 'Windows'")
    @pytest.mark.skipif("sys.implementation.name != 'cpython'")
    def test_get_makefile_filename(self):
        makefile = sysconfig.get_makefile_filename()
        assert os.path.isfile(makefile)

    def test_get_python_lib(self, tmp_path):
        assert sysconfig.get_python_lib() != sysconfig.get_python_lib(prefix=tmp_path)

    def test_get_config_vars(self):
        cvars = sysconfig.get_config_vars()
        assert isinstance(cvars, dict)
        assert cvars

    @pytest.mark.skipif('sysconfig.IS_PYPY')
    @pytest.mark.skipif('sysconfig.python_build')
    @pytest.mark.xfail('platform.system() == "Windows"')
    def test_srcdir_simple(self):
        # See #15364.
        srcdir = pathlib.Path(sysconfig.get_config_var('srcdir'))

        assert srcdir.absolute()
        assert srcdir.is_dir()

        makefile = pathlib.Path(sysconfig.get_makefile_filename())
        assert makefile.parent.samefile(srcdir)

    @pytest.mark.skipif('sysconfig.IS_PYPY')
    @pytest.mark.skipif('not sysconfig.python_build')
    def test_srcdir_python_build(self):
        # See #15364.
        srcdir = pathlib.Path(sysconfig.get_config_var('srcdir'))

        # The python executable has not been installed so srcdir
        # should be a full source checkout.
        Python_h = srcdir.joinpath('Include', 'Python.h')
        assert Python_h.is_file()
        assert sysconfig._is_python_source_dir(srcdir)
        assert sysconfig._is_python_source_dir(str(srcdir))

    def test_srcdir_independent_of_cwd(self):
        """
        srcdir should be independent of the current working directory
        """
        # See #15364.
        srcdir = sysconfig.get_config_var('srcdir')
        with path.Path('..'):
            srcdir2 = sysconfig.get_config_var('srcdir')
        assert srcdir == srcdir2

    def customize_compiler(self):
        # make sure AR gets caught
        class compiler:
            compiler_type = 'unix'
            executables = UnixCCompiler.executables

            def __init__(self):
                self.exes = {}

            def set_executables(self, **kw):
                for k, v in kw.items():
                    self.exes[k] = v

        sysconfig_vars = {
            'AR': 'sc_ar',
            'CC': 'sc_cc',
            'CXX': 'sc_cxx',
            'ARFLAGS': '--sc-arflags',
            'CFLAGS': '--sc-cflags',
            'CCSHARED': '--sc-ccshared',
            'LDSHARED': 'sc_ldshared',
            'SHLIB_SUFFIX': 'sc_shutil_suffix',
        }

        comp = compiler()
        with contextlib.ExitStack() as cm:
            for key, value in sysconfig_vars.items():
                cm.enter_context(swap_item(sysconfig._config_vars, key, value))
            sysconfig.customize_compiler(comp)

        return comp

    @pytest.mark.skipif("not isinstance(new_compiler(), UnixCCompiler)")
    @pytest.mark.usefixtures('disable_macos_customization')
    def test_customize_compiler(self):
        # Make sure that sysconfig._config_vars is initialized
        sysconfig.get_config_vars()

        os.environ['AR'] = 'env_ar'
        os.environ['CC'] = 'env_cc'
        os.environ['CPP'] = 'env_cpp'
        os.environ['CXX'] = 'env_cxx --env-cxx-flags'
        os.environ['LDSHARED'] = 'env_ldshared'
        os.environ['LDFLAGS'] = '--env-ldflags'
        os.environ['ARFLAGS'] = '--env-arflags'
        os.environ['CFLAGS'] = '--env-cflags'
        os.environ['CPPFLAGS'] = '--env-cppflags'
        os.environ['RANLIB'] = 'env_ranlib'

        comp = self.customize_compiler()
        assert comp.exes['archiver'] == 'env_ar --env-arflags'
        assert comp.exes['preprocessor'] == 'env_cpp --env-cppflags'
        assert comp.exes['compiler'] == 'env_cc --env-cflags --env-cppflags'
        assert comp.exes['compiler_so'] == (
            'env_cc --env-cflags --env-cppflags --sc-ccshared'
        )
        assert (
            comp.exes['compiler_cxx']
            == 'env_cxx --env-cxx-flags --sc-cflags --env-cppflags'
        )
        assert comp.exes['linker_exe'] == 'env_cc'
        assert comp.exes['linker_so'] == (
            'env_ldshared --env-ldflags --env-cflags --env-cppflags'
        )
        assert comp.shared_lib_extension == 'sc_shutil_suffix'

        if sys.platform == "darwin":
            assert comp.exes['ranlib'] == 'env_ranlib'
        else:
            assert 'ranlib' not in comp.exes

        del os.environ['AR']
        del os.environ['CC']
        del os.environ['CPP']
        del os.environ['CXX']
        del os.environ['LDSHARED']
        del os.environ['LDFLAGS']
        del os.environ['ARFLAGS']
        del os.environ['CFLAGS']
        del os.environ['CPPFLAGS']
        del os.environ['RANLIB']

        comp = self.customize_compiler()
        assert comp.exes['archiver'] == 'sc_ar --sc-arflags'
        assert comp.exes['preprocessor'] == 'sc_cc -E'
        assert comp.exes['compiler'] == 'sc_cc --sc-cflags'
        assert comp.exes['compiler_so'] == 'sc_cc --sc-cflags --sc-ccshared'
        assert comp.exes['compiler_cxx'] == 'sc_cxx --sc-cflags'
        assert comp.exes['linker_exe'] == 'sc_cc'
        assert comp.exes['linker_so'] == 'sc_ldshared'
        assert comp.shared_lib_extension == 'sc_shutil_suffix'
        assert 'ranlib' not in comp.exes

    def test_parse_makefile_base(self, tmp_path):
        makefile = _gen_makefile(
            tmp_path,
            """
            CONFIG_ARGS=  '--arg1=optarg1' 'ENV=LIB'
            VAR=$OTHER
            OTHER=foo
            """,
        )
        d = sysconfig.parse_makefile(makefile)
        assert d == {'CONFIG_ARGS': "'--arg1=optarg1' 'ENV=LIB'", 'OTHER': 'foo'}

    def test_parse_makefile_literal_dollar(self, tmp_path):
        makefile = _gen_makefile(
            tmp_path,
            """
            CONFIG_ARGS=  '--arg1=optarg1' 'ENV=\\$$LIB'
            VAR=$OTHER
            OTHER=foo
            """,
        )
        d = sysconfig.parse_makefile(makefile)
        assert d == {'CONFIG_ARGS': r"'--arg1=optarg1' 'ENV=\$LIB'", 'OTHER': 'foo'}

    def test_sysconfig_module(self):
        import sysconfig as global_sysconfig

        assert global_sysconfig.get_config_var('CFLAGS') == sysconfig.get_config_var(
            'CFLAGS'
        )
        assert global_sysconfig.get_config_var('LDFLAGS') == sysconfig.get_config_var(
            'LDFLAGS'
        )

    # On macOS, binary installers support extension module building on
    # various levels of the operating system with differing Xcode
    # configurations, requiring customization of some of the
    # compiler configuration directives to suit the environment on
    # the installed machine. Some of these customizations may require
    # running external programs and are thus deferred until needed by
    # the first extension module build. Only
    # the Distutils version of sysconfig is used for extension module
    # builds, which happens earlier in the Distutils tests. This may
    # cause the following tests to fail since no tests have caused
    # the global version of sysconfig to call the customization yet.
    # The solution for now is to simply skip this test in this case.
    # The longer-term solution is to only have one version of sysconfig.
    @pytest.mark.skipif("sysconfig.get_config_var('CUSTOMIZED_OSX_COMPILER')")
    def test_sysconfig_compiler_vars(self):
        import sysconfig as global_sysconfig

        if sysconfig.get_config_var('CUSTOMIZED_OSX_COMPILER'):
            pytest.skip('compiler flags customized')
        assert global_sysconfig.get_config_var('LDSHARED') == sysconfig.get_config_var(
            'LDSHARED'
        )
        assert global_sysconfig.get_config_var('CC') == sysconfig.get_config_var('CC')

    @pytest.mark.skipif("not sysconfig.get_config_var('EXT_SUFFIX')")
    def test_SO_deprecation(self):
        with pytest.warns(DeprecationWarning):
            sysconfig.get_config_var('SO')

    def test_customize_compiler_before_get_config_vars(self, tmp_path):
        # Issue #21923: test that a Distribution compiler
        # instance can be called without an explicit call to
        # get_config_vars().
        jaraco.path.build(
            {
                'file': trim("""
                    from distutils.core import Distribution
                    config = Distribution().get_command_obj('config')
                    # try_compile may pass or it may fail if no compiler
                    # is found but it should not raise an exception.
                    rc = config.try_compile('int x;')
                    """)
            },
            tmp_path,
        )
        p = subprocess.Popen(
            [sys.executable, tmp_path / 'file'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding='utf-8',
        )
        outs, errs = p.communicate()
        assert 0 == p.returncode, "Subprocess failed: " + outs

    def test_parse_config_h(self):
        config_h = sysconfig.get_config_h_filename()
        input = {}
        with open(config_h, encoding="utf-8") as f:
            result = sysconfig.parse_config_h(f, g=input)
        assert input is result
        with open(config_h, encoding="utf-8") as f:
            result = sysconfig.parse_config_h(f)
        assert isinstance(result, dict)

    @pytest.mark.skipif("platform.system() != 'Windows'")
    @pytest.mark.skipif("sys.implementation.name != 'cpython'")
    def test_win_ext_suffix(self):
        assert sysconfig.get_config_var("EXT_SUFFIX").endswith(".pyd")
        assert sysconfig.get_config_var("EXT_SUFFIX") != ".pyd"

    @pytest.mark.skipif("platform.system() != 'Windows'")
    @pytest.mark.skipif("sys.implementation.name != 'cpython'")
    @pytest.mark.skipif(
        '\\PCbuild\\'.casefold() not in sys.executable.casefold(),
        reason='Need sys.executable to be in a source tree',
    )
    def test_win_build_venv_from_source_tree(self, tmp_path):
        """Ensure distutils.sysconfig detects venvs from source tree builds."""
        env = jaraco.envs.VEnv()
        env.create_opts = env.clean_opts
        env.root = tmp_path
        env.ensure_env()
        cmd = [
            env.exe(),
            "-c",
            "import distutils.sysconfig; print(distutils.sysconfig.python_build)",
        ]
        distutils_path = os.path.dirname(os.path.dirname(distutils.__file__))
        out = subprocess.check_output(
            cmd, env={**os.environ, "PYTHONPATH": distutils_path}
        )
        assert out == "True"

    def test_get_python_inc_missing_config_dir(self, monkeypatch):
        """
        In portable Python installations, the sysconfig will be broken,
        pointing to the directories where the installation was built and
        not where it currently is. In this case, ensure that the missing
        directory isn't used for get_python_inc.

        See pypa/distutils#178.
        """

        def override(name):
            if name == 'INCLUDEPY':
                return '/does-not-exist'
            return sysconfig.get_config_var(name)

        monkeypatch.setattr(sysconfig, 'get_config_var', override)

        assert os.path.exists(sysconfig.get_python_inc())
