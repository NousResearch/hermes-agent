"""Easy install Tests"""

import contextlib
import io
import itertools
import logging
import os
import pathlib
import re
import site
import subprocess
import sys
import tarfile
import tempfile
import time
import warnings
import zipfile
from pathlib import Path
from typing import NamedTuple
from unittest import mock

import pytest
from jaraco import path

import pkg_resources
import setuptools.command.easy_install as ei
from pkg_resources import Distribution as PRDistribution, normalize_path, working_set
from setuptools import sandbox
from setuptools._normalization import safer_name
from setuptools.command.easy_install import PthDistributions
from setuptools.dist import Distribution
from setuptools.sandbox import run_setup
from setuptools.tests import fail_on_ascii
from setuptools.tests.server import MockServer, path_to_url

from . import contexts
from .textwrap import DALS

import distutils.errors


@pytest.fixture(autouse=True)
def pip_disable_index(monkeypatch):
    """
    Important: Disable the default index for pip to avoid
    querying packages in the index and potentially resolving
    and installing packages there.
    """
    monkeypatch.setenv('PIP_NO_INDEX', 'true')


class FakeDist:
    def get_entry_map(self, group):
        if group != 'console_scripts':
            return {}
        return {'name': 'ep'}

    def as_requirement(self):
        return 'spec'


SETUP_PY = DALS(
    """
    from setuptools import setup

    setup()
    """
)


class TestEasyInstallTest:
    def test_get_script_args(self):
        header = ei.CommandSpec.best().from_environment().as_header()
        dist = FakeDist()
        args = next(ei.ScriptWriter.get_args(dist))
        _name, script = itertools.islice(args, 2)
        assert script.startswith(header)
        assert "'spec'" in script
        assert "'console_scripts'" in script
        assert "'name'" in script
        assert re.search('^# EASY-INSTALL-ENTRY-SCRIPT', script, flags=re.MULTILINE)

    def test_no_find_links(self):
        # new option '--no-find-links', that blocks find-links added at
        # the project level
        dist = Distribution()
        cmd = ei.easy_install(dist)
        cmd.check_pth_processing = lambda: True
        cmd.no_find_links = True
        cmd.find_links = ['link1', 'link2']
        cmd.install_dir = os.path.join(tempfile.mkdtemp(), 'ok')
        cmd.args = ['ok']
        cmd.ensure_finalized()
        assert cmd.package_index.scanned_urls == {}

        # let's try without it (default behavior)
        cmd = ei.easy_install(dist)
        cmd.check_pth_processing = lambda: True
        cmd.find_links = ['link1', 'link2']
        cmd.install_dir = os.path.join(tempfile.mkdtemp(), 'ok')
        cmd.args = ['ok']
        cmd.ensure_finalized()
        keys = sorted(cmd.package_index.scanned_urls.keys())
        assert keys == ['link1', 'link2']

    def test_write_exception(self):
        """
        Test that `cant_write_to_target` is rendered as a DistutilsError.
        """
        dist = Distribution()
        cmd = ei.easy_install(dist)
        cmd.install_dir = os.getcwd()
        with pytest.raises(distutils.errors.DistutilsError):
            cmd.cant_write_to_target()

    def test_all_site_dirs(self, monkeypatch):
        """
        get_site_dirs should always return site dirs reported by
        site.getsitepackages.
        """
        path = normalize_path('/setuptools/test/site-packages')

        def mock_gsp():
            return [path]

        monkeypatch.setattr(site, 'getsitepackages', mock_gsp, raising=False)
        assert path in ei.get_site_dirs()

    def test_all_site_dirs_works_without_getsitepackages(self, monkeypatch):
        monkeypatch.delattr(site, 'getsitepackages', raising=False)
        assert ei.get_site_dirs()

    @pytest.fixture
    def sdist_unicode(self, tmpdir):
        files = [
            (
                'setup.py',
                DALS(
                    """
                    import setuptools
                    setuptools.setup(
                        name="setuptools-test-unicode",
                        version="1.0",
                        packages=["mypkg"],
                        include_package_data=True,
                    )
                    """
                ),
            ),
            (
                'mypkg/__init__.py',
                "",
            ),
            (
                'mypkg/☃.txt',
                "",
            ),
        ]
        sdist_name = 'setuptools-test-unicode-1.0.zip'
        sdist = tmpdir / sdist_name
        # can't use make_sdist, because the issue only occurs
        #  with zip sdists.
        sdist_zip = zipfile.ZipFile(str(sdist), 'w')
        for filename, content in files:
            sdist_zip.writestr(filename, content)
        sdist_zip.close()
        return str(sdist)

    @fail_on_ascii
    def test_unicode_filename_in_sdist(self, sdist_unicode, tmpdir, monkeypatch):
        """
        The install command should execute correctly even if
        the package has unicode filenames.
        """
        dist = Distribution({'script_args': ['easy_install']})
        target = (tmpdir / 'target').ensure_dir()
        cmd = ei.easy_install(
            dist,
            install_dir=str(target),
            args=['x'],
        )
        monkeypatch.setitem(os.environ, 'PYTHONPATH', str(target))
        cmd.ensure_finalized()
        cmd.easy_install(sdist_unicode)

    @pytest.fixture
    def sdist_unicode_in_script(self, tmpdir):
        files = [
            (
                "setup.py",
                DALS(
                    """
                    import setuptools
                    setuptools.setup(
                        name="setuptools-test-unicode",
                        version="1.0",
                        packages=["mypkg"],
                        include_package_data=True,
                        scripts=['mypkg/unicode_in_script'],
                    )
                    """
                ),
            ),
            ("mypkg/__init__.py", ""),
            (
                "mypkg/unicode_in_script",
                DALS(
                    """
                    #!/bin/sh
                    # á

                    non_python_fn() {
                    }
                """
                ),
            ),
        ]
        sdist_name = "setuptools-test-unicode-script-1.0.zip"
        sdist = tmpdir / sdist_name
        # can't use make_sdist, because the issue only occurs
        #  with zip sdists.
        sdist_zip = zipfile.ZipFile(str(sdist), "w")
        for filename, content in files:
            sdist_zip.writestr(filename, content.encode('utf-8'))
        sdist_zip.close()
        return str(sdist)

    @fail_on_ascii
    def test_unicode_content_in_sdist(
        self, sdist_unicode_in_script, tmpdir, monkeypatch
    ):
        """
        The install command should execute correctly even if
        the package has unicode in scripts.
        """
        dist = Distribution({"script_args": ["easy_install"]})
        target = (tmpdir / "target").ensure_dir()
        cmd = ei.easy_install(dist, install_dir=str(target), args=["x"])
        monkeypatch.setitem(os.environ, "PYTHONPATH", str(target))
        cmd.ensure_finalized()
        cmd.easy_install(sdist_unicode_in_script)

    @pytest.fixture
    def sdist_script(self, tmpdir):
        files = [
            (
                'setup.py',
                DALS(
                    """
                    import setuptools
                    setuptools.setup(
                        name="setuptools-test-script",
                        version="1.0",
                        scripts=["mypkg_script"],
                    )
                    """
                ),
            ),
            (
                'mypkg_script',
                DALS(
                    """
                     #/usr/bin/python
                     print('mypkg_script')
                     """
                ),
            ),
        ]
        sdist_name = 'setuptools-test-script-1.0.zip'
        sdist = str(tmpdir / sdist_name)
        make_sdist(sdist, files)
        return sdist

    @pytest.mark.skipif(
        not sys.platform.startswith('linux'), reason="Test can only be run on Linux"
    )
    def test_script_install(self, sdist_script, tmpdir, monkeypatch):
        """
        Check scripts are installed.
        """
        dist = Distribution({'script_args': ['easy_install']})
        target = (tmpdir / 'target').ensure_dir()
        cmd = ei.easy_install(
            dist,
            install_dir=str(target),
            args=['x'],
        )
        monkeypatch.setitem(os.environ, 'PYTHONPATH', str(target))
        cmd.ensure_finalized()
        cmd.easy_install(sdist_script)
        assert (target / 'mypkg_script').exists()


@pytest.mark.filterwarnings('ignore:Unbuilt egg')
class TestPTHFileWriter:
    def test_add_from_cwd_site_sets_dirty(self):
        """a pth file manager should set dirty
        if a distribution is in site but also the cwd
        """
        pth = PthDistributions('does-not_exist', [os.getcwd()])
        assert not pth.dirty
        pth.add(PRDistribution(os.getcwd()))
        assert pth.dirty

    def test_add_from_site_is_ignored(self):
        location = '/test/location/does-not-have-to-exist'
        # PthDistributions expects all locations to be normalized
        location = pkg_resources.normalize_path(location)
        pth = PthDistributions(
            'does-not_exist',
            [
                location,
            ],
        )
        assert not pth.dirty
        pth.add(PRDistribution(location))
        assert not pth.dirty

    def test_many_pth_distributions_merge_together(self, tmpdir):
        """
        If the pth file is modified under the hood, then PthDistribution
        will refresh its content before saving, merging contents when
        necessary.
        """
        # putting the pth file in a dedicated sub-folder,
        pth_subdir = tmpdir.join("pth_subdir")
        pth_subdir.mkdir()
        pth_path = str(pth_subdir.join("file1.pth"))
        pth1 = PthDistributions(pth_path)
        pth2 = PthDistributions(pth_path)
        assert pth1.paths == pth2.paths == [], (
            "unless there would be some default added at some point"
        )
        # and so putting the src_subdir in folder distinct than the pth one,
        # so to keep it absolute by PthDistributions
        new_src_path = tmpdir.join("src_subdir")
        new_src_path.mkdir()  # must exist to be accounted
        new_src_path_str = str(new_src_path)
        pth1.paths.append(new_src_path_str)
        pth1.save()
        assert pth1.paths, (
            "the new_src_path added must still be present/valid in pth1 after save"
        )
        # now,
        assert new_src_path_str not in pth2.paths, (
            "right before we save the entry should still not be present"
        )
        pth2.save()
        assert new_src_path_str in pth2.paths, (
            "the new_src_path entry should have been added by pth2 with its save() call"
        )
        assert pth2.paths[-1] == new_src_path, (
            "and it should match exactly on the last entry actually "
            "given we append to it in save()"
        )
        # finally,
        assert PthDistributions(pth_path).paths == pth2.paths, (
            "and we should have the exact same list at the end "
            "with a fresh PthDistributions instance"
        )


@pytest.fixture
def setup_context(tmpdir):
    with (tmpdir / 'setup.py').open('w', encoding="utf-8") as f:
        f.write(SETUP_PY)
    with tmpdir.as_cwd():
        yield tmpdir


@pytest.mark.usefixtures("user_override")
@pytest.mark.usefixtures("setup_context")
class TestUserInstallTest:
    # prevent check that site-packages is writable. easy_install
    # shouldn't be writing to system site-packages during finalize
    # options, but while it does, bypass the behavior.
    prev_sp_write = mock.patch(
        'setuptools.command.easy_install.easy_install.check_site_dir',
        mock.Mock(),
    )

    # simulate setuptools installed in user site packages
    @mock.patch('setuptools.command.easy_install.__file__', site.USER_SITE)
    @mock.patch('site.ENABLE_USER_SITE', True)
    @prev_sp_write
    def test_user_install_not_implied_user_site_enabled(self):
        self.assert_not_user_site()

    @mock.patch('site.ENABLE_USER_SITE', False)
    @prev_sp_write
    def test_user_install_not_implied_user_site_disabled(self):
        self.assert_not_user_site()

    @staticmethod
    def assert_not_user_site():
        # create a finalized easy_install command
        dist = Distribution()
        dist.script_name = 'setup.py'
        cmd = ei.easy_install(dist)
        cmd.args = ['py']
        cmd.ensure_finalized()
        assert not cmd.user, 'user should not be implied'

    def test_multiproc_atexit(self):
        pytest.importorskip('multiprocessing')

        log = logging.getLogger('test_easy_install')
        logging.basicConfig(level=logging.INFO, stream=sys.stderr)
        log.info('this should not break')

    @pytest.fixture
    def foo_package(self, tmpdir):
        egg_file = tmpdir / 'foo-1.0.egg-info'
        with egg_file.open('w') as f:
            f.write('Name: foo\n')
        return str(tmpdir)

    @pytest.fixture
    def install_target(self, tmpdir):
        target = str(tmpdir)
        with mock.patch('sys.path', sys.path + [target]):
            python_path = os.path.pathsep.join(sys.path)
            with mock.patch.dict(os.environ, PYTHONPATH=python_path):
                yield target

    def test_local_index(self, foo_package, install_target):
        """
        The local index must be used when easy_install locates installed
        packages.
        """
        dist = Distribution()
        dist.script_name = 'setup.py'
        cmd = ei.easy_install(dist)
        cmd.install_dir = install_target
        cmd.args = ['foo']
        cmd.ensure_finalized()
        cmd.local_index.scan([foo_package])
        res = cmd.easy_install('foo')
        actual = os.path.normcase(os.path.realpath(res.location))
        expected = os.path.normcase(os.path.realpath(foo_package))
        assert actual == expected

    @contextlib.contextmanager
    def user_install_setup_context(self, *args, **kwargs):
        """
        Wrap sandbox.setup_context to patch easy_install in that context to
        appear as user-installed.
        """
        with self.orig_context(*args, **kwargs):
            import setuptools.command.easy_install as ei

            ei.__file__ = site.USER_SITE
            yield

    def patched_setup_context(self):
        self.orig_context = sandbox.setup_context

        return mock.patch(
            'setuptools.sandbox.setup_context',
            self.user_install_setup_context,
        )


@pytest.fixture
def distutils_package():
    distutils_setup_py = SETUP_PY.replace(
        'from setuptools import setup',
        'from distutils.core import setup',
    )
    with contexts.tempdir(cd=os.chdir):
        with open('setup.py', 'w', encoding="utf-8") as f:
            f.write(distutils_setup_py)
        yield


@pytest.mark.usefixtures("distutils_package")
class TestDistutilsPackage:
    def test_bdist_egg_available_on_distutils_pkg(self):
        run_setup('setup.py', ['bdist_egg'])


@pytest.fixture
def mock_index():
    # set up a server which will simulate an alternate package index.
    p_index = MockServer()
    if p_index.server_port == 0:
        # Some platforms (Jython) don't find a port to which to bind,
        # so skip test for them.
        pytest.skip("could not find a valid port")
    p_index.start()
    return p_index


class TestInstallRequires:
    def test_setup_install_includes_dependencies(self, tmp_path, mock_index):
        """
        When ``python setup.py install`` is called directly, it will use easy_install
        to fetch dependencies.
        """
        # TODO: Remove these tests once `setup.py install` is completely removed
        project_root = tmp_path / "project"
        project_root.mkdir(exist_ok=True)
        install_root = tmp_path / "install"
        install_root.mkdir(exist_ok=True)

        self.create_project(project_root)
        cmd = [
            sys.executable,
            '-c',
            '__import__("setuptools").setup()',
            'install',
            '--install-base',
            str(install_root),
            '--install-lib',
            str(install_root),
            '--install-headers',
            str(install_root),
            '--install-scripts',
            str(install_root),
            '--install-data',
            str(install_root),
            '--install-purelib',
            str(install_root),
            '--install-platlib',
            str(install_root),
        ]
        env = {**os.environ, "__EASYINSTALL_INDEX": mock_index.url}
        cp = subprocess.run(
            cmd,
            cwd=str(project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        assert cp.returncode != 0
        try:
            assert '/does-not-exist/' in {r.path for r in mock_index.requests}
            assert next(
                line
                for line in cp.stdout.splitlines()
                if "not find suitable distribution for" in line
                and "does-not-exist" in line
            )
        except Exception:
            if "failed to get random numbers" in cp.stdout:
                pytest.xfail(f"{sys.platform} failure - {cp.stdout}")
            raise

    def create_project(self, root):
        config = """
        [metadata]
        name = project
        version = 42

        [options]
        install_requires = does-not-exist
        py_modules = mod
        """
        (root / 'setup.cfg').write_text(DALS(config), encoding="utf-8")
        (root / 'mod.py').touch()


class TestSetupRequires:
    def test_setup_requires_honors_fetch_params(self, mock_index, monkeypatch):
        """
        When easy_install installs a source distribution which specifies
        setup_requires, it should honor the fetch parameters (such as
        index-url, and find-links).
        """
        monkeypatch.setenv('PIP_RETRIES', '0')
        monkeypatch.setenv('PIP_TIMEOUT', '0')
        monkeypatch.setenv('PIP_NO_INDEX', 'false')
        with contexts.quiet():
            # create an sdist that has a build-time dependency.
            with TestSetupRequires.create_sdist() as dist_file:
                with contexts.tempdir() as temp_install_dir:
                    with contexts.environment(PYTHONPATH=temp_install_dir):
                        cmd = [
                            sys.executable,
                            '-c',
                            '__import__("setuptools").setup()',
                            'easy_install',
                            '--index-url',
                            mock_index.url,
                            '--exclude-scripts',
                            '--install-dir',
                            temp_install_dir,
                            dist_file,
                        ]
                        subprocess.Popen(cmd).wait()
        # there should have been one requests to the server
        assert [r.path for r in mock_index.requests] == ['/does-not-exist/']

    @staticmethod
    @contextlib.contextmanager
    def create_sdist():
        """
        Return an sdist with a setup_requires dependency (of something that
        doesn't exist)
        """
        with contexts.tempdir() as dir:
            dist_path = os.path.join(dir, 'setuptools-test-fetcher-1.0.tar.gz')
            make_sdist(
                dist_path,
                [
                    (
                        'setup.py',
                        DALS(
                            """
                    import setuptools
                    setuptools.setup(
                        name="setuptools-test-fetcher",
                        version="1.0",
                        setup_requires = ['does-not-exist'],
                    )
                """
                        ),
                    ),
                    ('setup.cfg', ''),
                ],
            )
            yield dist_path

    use_setup_cfg = (
        (),
        ('dependency_links',),
        ('setup_requires',),
        ('dependency_links', 'setup_requires'),
    )

    @pytest.mark.parametrize('use_setup_cfg', use_setup_cfg)
    def test_setup_requires_overrides_version_conflict(self, use_setup_cfg):
        """
        Regression test for distribution issue 323:
        https://bitbucket.org/tarek/distribute/issues/323

        Ensures that a distribution's setup_requires requirements can still be
        installed and used locally even if a conflicting version of that
        requirement is already on the path.
        """

        fake_dist = PRDistribution(
            'does-not-matter', project_name='foobar', version='0.0'
        )
        working_set.add(fake_dist)

        with contexts.save_pkg_resources_state():
            with contexts.tempdir() as temp_dir:
                test_pkg = create_setup_requires_package(
                    temp_dir, use_setup_cfg=use_setup_cfg
                )
                test_setup_py = os.path.join(test_pkg, 'setup.py')
                with contexts.quiet() as (stdout, _stderr):
                    # Don't even need to install the package, just
                    # running the setup.py at all is sufficient
                    run_setup(test_setup_py, ['--name'])

                lines = stdout.readlines()
                assert len(lines) > 0
                assert lines[-1].strip() == 'test_pkg'

    @pytest.mark.parametrize('use_setup_cfg', use_setup_cfg)
    def test_setup_requires_override_nspkg(self, use_setup_cfg):
        """
        Like ``test_setup_requires_overrides_version_conflict`` but where the
        ``setup_requires`` package is part of a namespace package that has
        *already* been imported.
        """

        with contexts.save_pkg_resources_state():
            with contexts.tempdir() as temp_dir:
                foobar_1_archive = os.path.join(temp_dir, 'foo_bar-0.1.tar.gz')
                make_nspkg_sdist(foobar_1_archive, 'foo.bar', '0.1')
                # Now actually go ahead an extract to the temp dir and add the
                # extracted path to sys.path so foo.bar v0.1 is importable
                foobar_1_dir = os.path.join(temp_dir, 'foo_bar-0.1')
                os.mkdir(foobar_1_dir)
                with tarfile.open(foobar_1_archive) as tf:
                    tf.extraction_filter = lambda member, path: member
                    tf.extractall(foobar_1_dir)
                sys.path.insert(1, foobar_1_dir)

                dist = PRDistribution(
                    foobar_1_dir, project_name='foo.bar', version='0.1'
                )
                working_set.add(dist)

                template = DALS(
                    """\
                    import foo  # Even with foo imported first the
                                # setup_requires package should override
                    import setuptools
                    setuptools.setup(**%r)

                    if not (hasattr(foo, '__path__') and
                            len(foo.__path__) == 2):
                        print('FAIL')

                    if 'foo_bar-0.2' not in foo.__path__[0]:
                        print('FAIL')
                """
                )

                test_pkg = create_setup_requires_package(
                    temp_dir,
                    'foo.bar',
                    '0.2',
                    make_nspkg_sdist,
                    template,
                    use_setup_cfg=use_setup_cfg,
                )

                test_setup_py = os.path.join(test_pkg, 'setup.py')

                with contexts.quiet() as (stdout, _stderr):
                    try:
                        # Don't even need to install the package, just
                        # running the setup.py at all is sufficient
                        run_setup(test_setup_py, ['--name'])
                    except pkg_resources.VersionConflict:  # pragma: nocover
                        pytest.fail(
                            'Installing setup.py requirements caused a VersionConflict'
                        )

                assert 'FAIL' not in stdout.getvalue()
                lines = stdout.readlines()
                assert len(lines) > 0
                assert lines[-1].strip() == 'test_pkg'

    @pytest.mark.parametrize('use_setup_cfg', use_setup_cfg)
    def test_setup_requires_with_attr_version(self, use_setup_cfg):
        def make_dependency_sdist(dist_path, distname, version):
            files = [
                (
                    'setup.py',
                    DALS(
                        f"""
                    import setuptools
                    setuptools.setup(
                        name={distname!r},
                        version={version!r},
                        py_modules=[{distname!r}],
                    )
                    """
                    ),
                ),
                (
                    distname + '.py',
                    DALS(
                        """
                    version = 42
                    """
                    ),
                ),
            ]
            make_sdist(dist_path, files)

        with contexts.save_pkg_resources_state():
            with contexts.tempdir() as temp_dir:
                test_pkg = create_setup_requires_package(
                    temp_dir,
                    setup_attrs=dict(version='attr: foobar.version'),
                    make_package=make_dependency_sdist,
                    use_setup_cfg=use_setup_cfg + ('version',),
                )
                test_setup_py = os.path.join(test_pkg, 'setup.py')
                with contexts.quiet() as (stdout, _stderr):
                    run_setup(test_setup_py, ['--version'])
                lines = stdout.readlines()
                assert len(lines) > 0
                assert lines[-1].strip() == '42'

    def test_setup_requires_honors_pip_env(self, mock_index, monkeypatch):
        monkeypatch.setenv('PIP_RETRIES', '0')
        monkeypatch.setenv('PIP_TIMEOUT', '0')
        monkeypatch.setenv('PIP_NO_INDEX', 'false')
        monkeypatch.setenv('PIP_INDEX_URL', mock_index.url)
        with contexts.save_pkg_resources_state():
            with contexts.tempdir() as temp_dir:
                test_pkg = create_setup_requires_package(
                    temp_dir,
                    'python-xlib',
                    '0.19',
                    setup_attrs=dict(dependency_links=[]),
                )
                test_setup_cfg = os.path.join(test_pkg, 'setup.cfg')
                with open(test_setup_cfg, 'w', encoding="utf-8") as fp:
                    fp.write(
                        DALS(
                            """
                        [easy_install]
                        index_url = https://pypi.org/legacy/
                        """
                        )
                    )
                test_setup_py = os.path.join(test_pkg, 'setup.py')
                with pytest.raises(distutils.errors.DistutilsError):
                    run_setup(test_setup_py, ['--version'])
        assert len(mock_index.requests) == 1
        assert mock_index.requests[0].path == '/python-xlib/'

    def test_setup_requires_with_pep508_url(self, mock_index, monkeypatch):
        monkeypatch.setenv('PIP_RETRIES', '0')
        monkeypatch.setenv('PIP_TIMEOUT', '0')
        monkeypatch.setenv('PIP_INDEX_URL', mock_index.url)
        with contexts.save_pkg_resources_state():
            with contexts.tempdir() as temp_dir:
                dep_sdist = os.path.join(temp_dir, 'dep.tar.gz')
                make_trivial_sdist(dep_sdist, 'dependency', '42')
                dep_url = path_to_url(dep_sdist, authority='localhost')
                test_pkg = create_setup_requires_package(
                    temp_dir,
                    # Ignored (overridden by setup_attrs)
                    'python-xlib',
                    '0.19',
                    setup_attrs=dict(setup_requires=f'dependency @ {dep_url}'),
                )
                test_setup_py = os.path.join(test_pkg, 'setup.py')
                run_setup(test_setup_py, ['--version'])
        assert len(mock_index.requests) == 0

    def test_setup_requires_with_allow_hosts(self, mock_index):
        """The `allow-hosts` option in not supported anymore."""
        files = {
            'test_pkg': {
                'setup.py': DALS(
                    """
                    from setuptools import setup
                    setup(setup_requires='python-xlib')
                    """
                ),
                'setup.cfg': DALS(
                    """
                    [easy_install]
                    allow_hosts = *
                    """
                ),
            }
        }
        with contexts.save_pkg_resources_state():
            with contexts.tempdir() as temp_dir:
                path.build(files, prefix=temp_dir)
                setup_py = str(pathlib.Path(temp_dir, 'test_pkg', 'setup.py'))
                with pytest.raises(distutils.errors.DistutilsError):
                    run_setup(setup_py, ['--version'])
        assert len(mock_index.requests) == 0

    def test_setup_requires_with_python_requires(self, monkeypatch, tmpdir):
        """Check `python_requires` is honored."""
        monkeypatch.setenv('PIP_RETRIES', '0')
        monkeypatch.setenv('PIP_TIMEOUT', '0')
        monkeypatch.setenv('PIP_NO_INDEX', '1')
        monkeypatch.setenv('PIP_VERBOSE', '1')
        dep_1_0_sdist = 'dep-1.0.tar.gz'
        dep_1_0_url = path_to_url(str(tmpdir / dep_1_0_sdist))
        dep_1_0_python_requires = '>=2.7'
        make_python_requires_sdist(
            str(tmpdir / dep_1_0_sdist), 'dep', '1.0', dep_1_0_python_requires
        )
        dep_2_0_sdist = 'dep-2.0.tar.gz'
        dep_2_0_url = path_to_url(str(tmpdir / dep_2_0_sdist))
        dep_2_0_python_requires = (
            f'!={sys.version_info.major}.{sys.version_info.minor}.*'
        )
        make_python_requires_sdist(
            str(tmpdir / dep_2_0_sdist), 'dep', '2.0', dep_2_0_python_requires
        )
        index = tmpdir / 'index.html'
        index.write_text(
            DALS(
                """
            <!DOCTYPE html>
            <html><head><title>Links for dep</title></head>
            <body>
                <h1>Links for dep</h1>
                <a href="{dep_1_0_url}"\
data-requires-python="{dep_1_0_python_requires}">{dep_1_0_sdist}</a><br/>
                <a href="{dep_2_0_url}"\
data-requires-python="{dep_2_0_python_requires}">{dep_2_0_sdist}</a><br/>
            </body>
            </html>
            """
            ).format(
                dep_1_0_url=dep_1_0_url,
                dep_1_0_sdist=dep_1_0_sdist,
                dep_1_0_python_requires=dep_1_0_python_requires,
                dep_2_0_url=dep_2_0_url,
                dep_2_0_sdist=dep_2_0_sdist,
                dep_2_0_python_requires=dep_2_0_python_requires,
            ),
            'utf-8',
        )
        index_url = path_to_url(str(index))
        with contexts.save_pkg_resources_state():
            test_pkg = create_setup_requires_package(
                str(tmpdir),
                'python-xlib',
                '0.19',  # Ignored (overridden by setup_attrs).
                setup_attrs=dict(setup_requires='dep', dependency_links=[index_url]),
            )
            test_setup_py = os.path.join(test_pkg, 'setup.py')
            run_setup(test_setup_py, ['--version'])
        eggs = list(
            map(str, pkg_resources.find_distributions(os.path.join(test_pkg, '.eggs')))
        )
        assert eggs == ['dep 1.0']

    @pytest.mark.parametrize('with_dependency_links_in_setup_py', (False, True))
    def test_setup_requires_with_find_links_in_setup_cfg(
        self, monkeypatch, with_dependency_links_in_setup_py
    ):
        monkeypatch.setenv('PIP_RETRIES', '0')
        monkeypatch.setenv('PIP_TIMEOUT', '0')
        with contexts.save_pkg_resources_state():
            with contexts.tempdir() as temp_dir:
                make_trivial_sdist(
                    os.path.join(temp_dir, 'python-xlib-42.tar.gz'), 'python-xlib', '42'
                )
                test_pkg = os.path.join(temp_dir, 'test_pkg')
                test_setup_py = os.path.join(test_pkg, 'setup.py')
                test_setup_cfg = os.path.join(test_pkg, 'setup.cfg')
                os.mkdir(test_pkg)
                with open(test_setup_py, 'w', encoding="utf-8") as fp:
                    if with_dependency_links_in_setup_py:
                        dependency_links = [os.path.join(temp_dir, 'links')]
                    else:
                        dependency_links = []
                    fp.write(
                        DALS(
                            """
                        from setuptools import installer, setup
                        setup(setup_requires='python-xlib==42',
                        dependency_links={dependency_links!r})
                        """
                        ).format(dependency_links=dependency_links)
                    )
                with open(test_setup_cfg, 'w', encoding="utf-8") as fp:
                    fp.write(
                        DALS(
                            """
                        [easy_install]
                        index_url = {index_url}
                        find_links = {find_links}
                        """
                        ).format(
                            index_url=os.path.join(temp_dir, 'index'),
                            find_links=temp_dir,
                        )
                    )
                run_setup(test_setup_py, ['--version'])

    def test_setup_requires_with_transitive_extra_dependency(self, monkeypatch):
        """
        Use case: installing a package with a build dependency on
        an already installed `dep[extra]`, which in turn depends
        on `extra_dep` (whose is not already installed).
        """
        with contexts.save_pkg_resources_state():
            with contexts.tempdir() as temp_dir:
                # Create source distribution for `extra_dep`.
                make_trivial_sdist(
                    os.path.join(temp_dir, 'extra_dep-1.0.tar.gz'), 'extra_dep', '1.0'
                )
                # Create source tree for `dep`.
                dep_pkg = os.path.join(temp_dir, 'dep')
                os.mkdir(dep_pkg)
                path.build(
                    {
                        'setup.py': DALS(
                            """
                          import setuptools
                          setuptools.setup(
                              name='dep', version='2.0',
                              extras_require={'extra': ['extra_dep']},
                          )
                         """
                        ),
                        'setup.cfg': '',
                    },
                    prefix=dep_pkg,
                )
                # "Install" dep.
                run_setup(os.path.join(dep_pkg, 'setup.py'), ['dist_info'])
                working_set.add_entry(dep_pkg)
                # Create source tree for test package.
                test_pkg = os.path.join(temp_dir, 'test_pkg')
                test_setup_py = os.path.join(test_pkg, 'setup.py')
                os.mkdir(test_pkg)
                with open(test_setup_py, 'w', encoding="utf-8") as fp:
                    fp.write(
                        DALS(
                            """
                        from setuptools import installer, setup
                        setup(setup_requires='dep[extra]')
                        """
                        )
                    )
                # Check...
                monkeypatch.setenv('PIP_FIND_LINKS', str(temp_dir))
                monkeypatch.setenv('PIP_NO_INDEX', '1')
                monkeypatch.setenv('PIP_RETRIES', '0')
                monkeypatch.setenv('PIP_TIMEOUT', '0')
                run_setup(test_setup_py, ['--version'])

    def test_setup_requires_with_distutils_command_dep(self, monkeypatch):
        """
        Use case: ensure build requirements' extras
        are properly installed and activated.
        """
        with contexts.save_pkg_resources_state():
            with contexts.tempdir() as temp_dir:
                # Create source distribution for `extra_dep`.
                make_sdist(
                    os.path.join(temp_dir, 'extra_dep-1.0.tar.gz'),
                    [
                        (
                            'setup.py',
                            DALS(
                                """
                          import setuptools
                          setuptools.setup(
                              name='extra_dep',
                              version='1.0',
                              py_modules=['extra_dep'],
                          )
                          """
                            ),
                        ),
                        ('setup.cfg', ''),
                        ('extra_dep.py', ''),
                    ],
                )
                # Create source tree for `epdep`.
                dep_pkg = os.path.join(temp_dir, 'epdep')
                os.mkdir(dep_pkg)
                path.build(
                    {
                        'setup.py': DALS(
                            """
                          import setuptools
                          setuptools.setup(
                              name='dep', version='2.0',
                              py_modules=['epcmd'],
                              extras_require={'extra': ['extra_dep']},
                              entry_points='''
                                           [distutils.commands]
                                           epcmd = epcmd:epcmd [extra]
                                           ''',
                          )
                         """
                        ),
                        'setup.cfg': '',
                        'epcmd.py': DALS(
                            """
                                     from distutils.command.build_py import build_py

                                     import extra_dep

                                     class epcmd(build_py):
                                         pass
                                     """
                        ),
                    },
                    prefix=dep_pkg,
                )
                # "Install" dep.
                run_setup(os.path.join(dep_pkg, 'setup.py'), ['dist_info'])
                working_set.add_entry(dep_pkg)
                # Create source tree for test package.
                test_pkg = os.path.join(temp_dir, 'test_pkg')
                test_setup_py = os.path.join(test_pkg, 'setup.py')
                os.mkdir(test_pkg)
                with open(test_setup_py, 'w', encoding="utf-8") as fp:
                    fp.write(
                        DALS(
                            """
                        from setuptools import installer, setup
                        setup(setup_requires='dep[extra]')
                        """
                        )
                    )
                # Check...
                monkeypatch.setenv('PIP_FIND_LINKS', str(temp_dir))
                monkeypatch.setenv('PIP_NO_INDEX', '1')
                monkeypatch.setenv('PIP_RETRIES', '0')
                monkeypatch.setenv('PIP_TIMEOUT', '0')
                run_setup(test_setup_py, ['epcmd'])


def make_trivial_sdist(dist_path, distname, version):
    """
    Create a simple sdist tarball at dist_path, containing just a simple
    setup.py.
    """

    make_sdist(
        dist_path,
        [
            (
                'setup.py',
                DALS(
                    f"""\
             import setuptools
             setuptools.setup(
                 name={distname!r},
                 version={version!r}
             )
         """
                ),
            ),
            ('setup.cfg', ''),
        ],
    )


def make_nspkg_sdist(dist_path, distname, version):
    """
    Make an sdist tarball with distname and version which also contains one
    package with the same name as distname.  The top-level package is
    designated a namespace package).
    """
    # Assert that the distname contains at least one period
    assert '.' in distname

    parts = distname.split('.')
    nspackage = parts[0]

    packages = ['.'.join(parts[:idx]) for idx in range(1, len(parts) + 1)]

    setup_py = DALS(
        f"""\
        import setuptools
        setuptools.setup(
            name={distname!r},
            version={version!r},
            packages={packages!r},
            namespace_packages=[{nspackage!r}]
        )
    """
    )

    init = "__import__('pkg_resources').declare_namespace(__name__)"

    files = [('setup.py', setup_py), (os.path.join(nspackage, '__init__.py'), init)]
    for package in packages[1:]:
        filename = os.path.join(*(package.split('.') + ['__init__.py']))
        files.append((filename, ''))

    make_sdist(dist_path, files)


def make_python_requires_sdist(dist_path, distname, version, python_requires):
    make_sdist(
        dist_path,
        [
            (
                'setup.py',
                DALS(
                    """\
                import setuptools
                setuptools.setup(
                  name={name!r},
                  version={version!r},
                  python_requires={python_requires!r},
                )
                """
                ).format(
                    name=distname, version=version, python_requires=python_requires
                ),
            ),
            ('setup.cfg', ''),
        ],
    )


def make_sdist(dist_path, files):
    """
    Create a simple sdist tarball at dist_path, containing the files
    listed in ``files`` as ``(filename, content)`` tuples.
    """

    # Distributions with only one file don't play well with pip.
    assert len(files) > 1
    with tarfile.open(dist_path, 'w:gz') as dist:
        for filename, content in files:
            file_bytes = io.BytesIO(content.encode('utf-8'))
            file_info = tarfile.TarInfo(name=filename)
            file_info.size = len(file_bytes.getvalue())
            file_info.mtime = int(time.time())
            dist.addfile(file_info, fileobj=file_bytes)


def create_setup_requires_package(
    path,
    distname='foobar',
    version='0.1',
    make_package=make_trivial_sdist,
    setup_py_template=None,
    setup_attrs=None,
    use_setup_cfg=(),
):
    """Creates a source tree under path for a trivial test package that has a
    single requirement in setup_requires--a tarball for that requirement is
    also created and added to the dependency_links argument.

    ``distname`` and ``version`` refer to the name/version of the package that
    the test package requires via ``setup_requires``.  The name of the test
    package itself is just 'test_pkg'.
    """

    normalized_distname = safer_name(distname)
    test_setup_attrs = {
        'name': 'test_pkg',
        'version': '0.0',
        'setup_requires': [f'{normalized_distname}=={version}'],
        'dependency_links': [os.path.abspath(path)],
    }
    if setup_attrs:
        test_setup_attrs.update(setup_attrs)

    test_pkg = os.path.join(path, 'test_pkg')
    os.mkdir(test_pkg)

    # setup.cfg
    if use_setup_cfg:
        options = []
        metadata = []
        for name in use_setup_cfg:
            value = test_setup_attrs.pop(name)
            if name in 'name version'.split():
                section = metadata
            else:
                section = options
            if isinstance(value, (tuple, list)):
                value = ';'.join(value)
            section.append(f'{name}: {value}')
        test_setup_cfg_contents = DALS(
            """
            [metadata]
            {metadata}
            [options]
            {options}
            """
        ).format(
            options='\n'.join(options),
            metadata='\n'.join(metadata),
        )
    else:
        test_setup_cfg_contents = ''
    with open(os.path.join(test_pkg, 'setup.cfg'), 'w', encoding="utf-8") as f:
        f.write(test_setup_cfg_contents)

    # setup.py
    if setup_py_template is None:
        setup_py_template = DALS(
            """\
            import setuptools
            setuptools.setup(**%r)
        """
        )
    with open(os.path.join(test_pkg, 'setup.py'), 'w', encoding="utf-8") as f:
        f.write(setup_py_template % test_setup_attrs)

    foobar_path = os.path.join(path, f'{normalized_distname}-{version}.tar.gz')
    make_package(foobar_path, distname, version)

    return test_pkg


@pytest.mark.skipif(
    sys.platform.startswith('java') and ei.is_sh(sys.executable),
    reason="Test cannot run under java when executable is sh",
)
class TestScriptHeader:
    non_ascii_exe = '/Users/José/bin/python'
    exe_with_spaces = r'C:\Program Files\Python36\python.exe'

    def test_get_script_header(self):
        expected = f'#!{ei.nt_quote_arg(os.path.normpath(sys.executable))}\n'
        actual = ei.ScriptWriter.get_header('#!/usr/local/bin/python')
        assert actual == expected

    def test_get_script_header_args(self):
        expected = f'#!{ei.nt_quote_arg(os.path.normpath(sys.executable))} -x\n'
        actual = ei.ScriptWriter.get_header('#!/usr/bin/python -x')
        assert actual == expected

    def test_get_script_header_non_ascii_exe(self):
        actual = ei.ScriptWriter.get_header(
            '#!/usr/bin/python', executable=self.non_ascii_exe
        )
        expected = f'#!{self.non_ascii_exe} -x\n'
        assert actual == expected

    def test_get_script_header_exe_with_spaces(self):
        actual = ei.ScriptWriter.get_header(
            '#!/usr/bin/python', executable='"' + self.exe_with_spaces + '"'
        )
        expected = f'#!"{self.exe_with_spaces}"\n'
        assert actual == expected


class TestCommandSpec:
    def test_custom_launch_command(self):
        """
        Show how a custom CommandSpec could be used to specify a #! executable
        which takes parameters.
        """
        cmd = ei.CommandSpec(['/usr/bin/env', 'python3'])
        assert cmd.as_header() == '#!/usr/bin/env python3\n'

    def test_from_param_for_CommandSpec_is_passthrough(self):
        """
        from_param should return an instance of a CommandSpec
        """
        cmd = ei.CommandSpec(['python'])
        cmd_new = ei.CommandSpec.from_param(cmd)
        assert cmd is cmd_new

    @mock.patch('sys.executable', TestScriptHeader.exe_with_spaces)
    @mock.patch.dict(os.environ)
    def test_from_environment_with_spaces_in_executable(self):
        os.environ.pop('__PYVENV_LAUNCHER__', None)
        cmd = ei.CommandSpec.from_environment()
        assert len(cmd) == 1
        assert cmd.as_header().startswith('#!"')

    def test_from_simple_string_uses_shlex(self):
        """
        In order to support `executable = /usr/bin/env my-python`, make sure
        from_param invokes shlex on that input.
        """
        cmd = ei.CommandSpec.from_param('/usr/bin/env my-python')
        assert len(cmd) == 2
        assert '"' not in cmd.as_header()

    def test_from_param_raises_expected_error(self) -> None:
        """
        from_param should raise its own TypeError when the argument's type is unsupported
        """
        with pytest.raises(TypeError) as exc_info:
            ei.CommandSpec.from_param(object())  # type: ignore[arg-type] # We want a type error here
        assert (
            str(exc_info.value) == "Argument has an unsupported type <class 'object'>"
        ), exc_info.value


class TestWindowsScriptWriter:
    def test_header(self):
        hdr = ei.WindowsScriptWriter.get_header('')
        assert hdr.startswith('#!')
        assert hdr.endswith('\n')
        hdr = hdr.lstrip('#!')
        hdr = hdr.rstrip('\n')
        # header should not start with an escaped quote
        assert not hdr.startswith('\\"')


class VersionStub(NamedTuple):
    major: int
    minor: int
    micro: int
    releaselevel: str
    serial: int


def test_use_correct_python_version_string(tmpdir, tmpdir_cwd, monkeypatch):
    # In issue #3001, easy_install wrongly uses the `python3.1` directory
    # when the interpreter is `python3.10` and the `--user` option is given.
    # See pypa/setuptools#3001.
    dist = Distribution()
    cmd = dist.get_command_obj('easy_install')
    cmd.args = ['ok']
    cmd.optimize = 0
    cmd.user = True
    cmd.install_userbase = str(tmpdir)
    cmd.install_usersite = None
    install_cmd = dist.get_command_obj('install')
    install_cmd.install_userbase = str(tmpdir)
    install_cmd.install_usersite = None

    with monkeypatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        version = '3.10.1 (main, Dec 21 2021, 09:17:12) [GCC 10.2.1 20210110]'
        info = VersionStub(3, 10, 1, "final", 0)
        patch.setattr('site.ENABLE_USER_SITE', True)
        patch.setattr('sys.version', version)
        patch.setattr('sys.version_info', info)
        patch.setattr(cmd, 'create_home_path', mock.Mock())
        cmd.finalize_options()

    name = "pypy" if hasattr(sys, 'pypy_version_info') else "python"
    install_dir = cmd.install_dir.lower()

    # In some platforms (e.g. Windows), install_dir is mostly determined
    # via `sysconfig`, which define constants eagerly at module creation.
    # This means that monkeypatching `sys.version` to emulate 3.10 for testing
    # may have no effect.
    # The safest test here is to rely on the fact that 3.1 is no longer
    # supported/tested, and make sure that if 'python3.1' ever appears in the string
    # it is followed by another digit (e.g. 'python3.10').
    if re.search(name + r'3\.?1', install_dir):
        assert re.search(name + r'3\.?1\d', install_dir)

    # The following "variables" are used for interpolation in distutils
    # installation schemes, so it should be fair to treat them as "semi-public",
    # or at least public enough so we can have a test to make sure they are correct
    assert cmd.config_vars['py_version'] == '3.10.1'
    assert cmd.config_vars['py_version_short'] == '3.10'
    assert cmd.config_vars['py_version_nodot'] == '310'


@pytest.mark.xfail(
    sys.platform == "darwin",
    reason="https://github.com/pypa/setuptools/pull/4716#issuecomment-2447624418",
)
def test_editable_user_and_build_isolation(setup_context, monkeypatch, tmp_path):
    """`setup.py develop` should honor `--user` even under build isolation"""

    # == Arrange ==
    # Pretend that build isolation was enabled
    # e.g pip sets the environment variable PYTHONNOUSERSITE=1
    monkeypatch.setattr('site.ENABLE_USER_SITE', False)

    # Patching $HOME for 2 reasons:
    # 1. setuptools/command/easy_install.py:create_home_path
    #    tries creating directories in $HOME.
    #    Given::
    #        self.config_vars['DESTDIRS'] = (
    #            "/home/user/.pyenv/versions/3.9.10 "
    #            "/home/user/.pyenv/versions/3.9.10/lib "
    #            "/home/user/.pyenv/versions/3.9.10/lib/python3.9 "
    #            "/home/user/.pyenv/versions/3.9.10/lib/python3.9/lib-dynload")
    #    `create_home_path` will::
    #        makedirs(
    #            "/home/user/.pyenv/versions/3.9.10 "
    #            "/home/user/.pyenv/versions/3.9.10/lib "
    #            "/home/user/.pyenv/versions/3.9.10/lib/python3.9 "
    #            "/home/user/.pyenv/versions/3.9.10/lib/python3.9/lib-dynload")
    #
    # 2. We are going to force `site` to update site.USER_BASE and site.USER_SITE
    #    To point inside our new home
    monkeypatch.setenv('HOME', str(tmp_path / '.home'))
    monkeypatch.setenv('USERPROFILE', str(tmp_path / '.home'))
    monkeypatch.setenv('APPDATA', str(tmp_path / '.home'))
    monkeypatch.setattr('site.USER_BASE', None)
    monkeypatch.setattr('site.USER_SITE', None)
    user_site = Path(site.getusersitepackages())
    user_site.mkdir(parents=True, exist_ok=True)

    sys_prefix = tmp_path / '.sys_prefix'
    sys_prefix.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr('sys.prefix', str(sys_prefix))

    setup_script = (
        "__import__('setuptools').setup(name='aproj', version=42, packages=[])\n"
    )
    (tmp_path / "setup.py").write_text(setup_script, encoding="utf-8")

    # == Sanity check ==
    assert list(sys_prefix.glob("*")) == []
    assert list(user_site.glob("*")) == []

    # == Act ==
    run_setup('setup.py', ['develop', '--user'])

    # == Assert ==
    # Should not install to sys.prefix
    assert list(sys_prefix.glob("*")) == []
    # Should install to user site
    installed = {f.name for f in user_site.glob("*")}
    # sometimes easy-install.pth is created and sometimes not
    installed = installed - {"easy-install.pth"}
    assert installed == {'aproj.egg-link'}
