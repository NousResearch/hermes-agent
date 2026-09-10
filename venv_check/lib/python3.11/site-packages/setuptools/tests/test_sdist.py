"""sdist tests"""

import contextlib
import io
import logging
import os
import pathlib
import sys
import tarfile
import tempfile
import unicodedata
from inspect import cleandoc
from pathlib import Path
from unittest import mock

import jaraco.path
import pytest

from setuptools import Command, SetuptoolsDeprecationWarning
from setuptools._importlib import metadata
from setuptools.command.egg_info import manifest_maker
from setuptools.command.sdist import sdist
from setuptools.dist import Distribution
from setuptools.extension import Extension
from setuptools.tests import fail_on_ascii

from .text import Filenames

import distutils
from distutils.core import run_setup

SETUP_ATTRS = {
    'name': 'sdist_test',
    'version': '0.0',
    'packages': ['sdist_test'],
    'package_data': {'sdist_test': ['*.txt']},
    'data_files': [("data", [os.path.join("d", "e.dat")])],
}

SETUP_PY = f"""\
from setuptools import setup

setup(**{SETUP_ATTRS!r})
"""

EXTENSION = Extension(
    name="sdist_test.f",
    sources=[os.path.join("sdist_test", "f.c")],
    depends=[os.path.join("sdist_test", "f.h")],
)
EXTENSION_SOURCES = EXTENSION.sources + EXTENSION.depends


@contextlib.contextmanager
def quiet():
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


# Convert to POSIX path
def posix(path):
    if not isinstance(path, str):
        return path.replace(os.sep.encode('ascii'), b'/')
    else:
        return path.replace(os.sep, '/')


# HFS Plus uses decomposed UTF-8
def decompose(path):
    if isinstance(path, str):
        return unicodedata.normalize('NFD', path)
    try:
        path = path.decode('utf-8')
        path = unicodedata.normalize('NFD', path)
        path = path.encode('utf-8')
    except UnicodeError:
        pass  # Not UTF-8
    return path


def read_all_bytes(filename):
    with open(filename, 'rb') as fp:
        return fp.read()


def latin1_fail():
    try:
        desc, filename = tempfile.mkstemp(suffix=Filenames.latin_1)
        os.close(desc)
        os.remove(filename)
    except Exception:
        return True


fail_on_latin1_encoded_filenames = pytest.mark.xfail(
    latin1_fail(),
    reason="System does not support latin-1 filenames",
)


skip_under_xdist = pytest.mark.skipif(
    "os.environ.get('PYTEST_XDIST_WORKER')",
    reason="pytest-dev/pytest-xdist#843",
)
skip_under_stdlib_distutils = pytest.mark.skipif(
    not distutils.__package__.startswith('setuptools'),
    reason="the test is not supported with stdlib distutils",
)


def touch(path):
    open(path, 'wb').close()
    return path


def symlink_or_skip_test(src, dst):
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError):
        pytest.skip("symlink not supported in OS")
        return None
    return dst


class TestSdistTest:
    @pytest.fixture(autouse=True)
    def source_dir(self, tmpdir):
        tmpdir = tmpdir / "project_root"
        tmpdir.mkdir()

        (tmpdir / 'setup.py').write_text(SETUP_PY, encoding='utf-8')

        # Set up the rest of the test package
        test_pkg = tmpdir / 'sdist_test'
        test_pkg.mkdir()
        data_folder = tmpdir / 'd'
        data_folder.mkdir()
        # *.rst was not included in package_data, so c.rst should not be
        # automatically added to the manifest when not under version control
        for fname in ['__init__.py', 'a.txt', 'b.txt', 'c.rst']:
            touch(test_pkg / fname)
        touch(data_folder / 'e.dat')
        # C sources are not included by default, but they will be,
        # if an extension module uses them as sources or depends
        for fname in EXTENSION_SOURCES:
            touch(tmpdir / fname)

        with tmpdir.as_cwd():
            yield tmpdir

    def assert_package_data_in_manifest(self, cmd):
        manifest = cmd.filelist.files
        assert os.path.join('sdist_test', 'a.txt') in manifest
        assert os.path.join('sdist_test', 'b.txt') in manifest
        assert os.path.join('sdist_test', 'c.rst') not in manifest
        assert os.path.join('d', 'e.dat') in manifest

    def setup_with_extension(self):
        setup_attrs = {**SETUP_ATTRS, 'ext_modules': [EXTENSION]}

        dist = Distribution(setup_attrs)
        dist.script_name = 'setup.py'
        cmd = sdist(dist)
        cmd.ensure_finalized()

        with quiet():
            cmd.run()

        return cmd

    def test_package_data_in_sdist(self):
        """Regression test for pull request #4: ensures that files listed in
        package_data are included in the manifest even if they're not added to
        version control.
        """

        dist = Distribution(SETUP_ATTRS)
        dist.script_name = 'setup.py'
        cmd = sdist(dist)
        cmd.ensure_finalized()

        with quiet():
            cmd.run()

        self.assert_package_data_in_manifest(cmd)

    def test_package_data_and_include_package_data_in_sdist(self):
        """
        Ensure package_data and include_package_data work
        together.
        """
        setup_attrs = {**SETUP_ATTRS, 'include_package_data': True}
        assert setup_attrs['package_data']

        dist = Distribution(setup_attrs)
        dist.script_name = 'setup.py'
        cmd = sdist(dist)
        cmd.ensure_finalized()

        with quiet():
            cmd.run()

        self.assert_package_data_in_manifest(cmd)

    def test_extension_sources_in_sdist(self):
        """
        Ensure that the files listed in Extension.sources and Extension.depends
        are automatically included in the manifest.
        """
        cmd = self.setup_with_extension()
        self.assert_package_data_in_manifest(cmd)
        manifest = cmd.filelist.files
        for path in EXTENSION_SOURCES:
            assert path in manifest

    def test_missing_extension_sources(self):
        """
        Similar to test_extension_sources_in_sdist but the referenced files don't exist.
        Missing files should not be included in distribution (with no error raised).
        """
        for path in EXTENSION_SOURCES:
            os.remove(path)

        cmd = self.setup_with_extension()
        self.assert_package_data_in_manifest(cmd)
        manifest = cmd.filelist.files
        for path in EXTENSION_SOURCES:
            assert path not in manifest

    def test_symlinked_extension_sources(self):
        """
        Similar to test_extension_sources_in_sdist but the referenced files are
        instead symbolic links to project-local files. Referenced file paths
        should be included. Symlink targets themselves should NOT be included.
        """
        symlinked = []
        for path in EXTENSION_SOURCES:
            base, ext = os.path.splitext(path)
            target = base + "_target." + ext

            os.rename(path, target)
            symlink_or_skip_test(os.path.basename(target), path)
            symlinked.append(target)

        cmd = self.setup_with_extension()
        self.assert_package_data_in_manifest(cmd)
        manifest = cmd.filelist.files
        for path in EXTENSION_SOURCES:
            assert path in manifest
        for path in symlinked:
            assert path not in manifest

    _INVALID_PATHS = {
        "must be relative": lambda: (
            os.path.abspath(os.path.join("sdist_test", "f.h"))
        ),
        "can't have `..` segments": lambda: (
            os.path.join("sdist_test", "..", "sdist_test", "f.h")
        ),
        "doesn't exist": lambda: (
            os.path.join("sdist_test", "this_file_does_not_exist.h")
        ),
        "must be inside the project root": lambda: (
            symlink_or_skip_test(
                touch(os.path.join("..", "outside_of_project_root.h")),
                "symlink.h",
            )
        ),
    }

    @skip_under_stdlib_distutils
    @pytest.mark.parametrize("reason", _INVALID_PATHS.keys())
    def test_invalid_extension_depends(self, reason, caplog):
        """
        Due to backwards compatibility reasons, `Extension.depends` should accept
        invalid/weird paths, but then ignore them when building a sdist.

        This test verifies that the source distribution is still built
        successfully with such paths, but that instead of adding these paths to
        the manifest, we emit an informational message, notifying the user that
        the invalid path won't be automatically included.
        """
        invalid_path = self._INVALID_PATHS[reason]()
        extension = Extension(
            name="sdist_test.f",
            sources=[],
            depends=[invalid_path],
        )
        setup_attrs = {**SETUP_ATTRS, 'ext_modules': [extension]}

        dist = Distribution(setup_attrs)
        dist.script_name = 'setup.py'
        cmd = sdist(dist)
        cmd.ensure_finalized()

        with quiet(), caplog.at_level(logging.INFO):
            cmd.run()

        self.assert_package_data_in_manifest(cmd)
        manifest = cmd.filelist.files
        assert invalid_path not in manifest

        expected_message = [
            message
            for (logger, level, message) in caplog.record_tuples
            if (
                logger == "root"  #
                and level == logging.INFO  #
                and invalid_path in message  #
            )
        ]
        assert len(expected_message) == 1
        (expected_message,) = expected_message
        assert reason in expected_message

    def test_custom_build_py(self):
        """
        Ensure projects defining custom build_py don't break
        when creating sdists (issue #2849)
        """
        from distutils.command.build_py import build_py as OrigBuildPy

        using_custom_command_guard = mock.Mock()

        class CustomBuildPy(OrigBuildPy):
            """
            Some projects have custom commands inheriting from `distutils`
            """

            def get_data_files(self):
                using_custom_command_guard()
                return super().get_data_files()

        setup_attrs = {**SETUP_ATTRS, 'include_package_data': True}
        assert setup_attrs['package_data']

        dist = Distribution(setup_attrs)
        dist.script_name = 'setup.py'
        cmd = sdist(dist)
        cmd.ensure_finalized()

        # Make sure we use the custom command
        cmd.cmdclass = {'build_py': CustomBuildPy}
        cmd.distribution.cmdclass = {'build_py': CustomBuildPy}
        assert cmd.distribution.get_command_class('build_py') == CustomBuildPy

        msg = "setuptools instead of distutils"
        with quiet(), pytest.warns(SetuptoolsDeprecationWarning, match=msg):
            cmd.run()

        using_custom_command_guard.assert_called()
        self.assert_package_data_in_manifest(cmd)

    def test_setup_py_exists(self):
        dist = Distribution(SETUP_ATTRS)
        dist.script_name = 'foo.py'
        cmd = sdist(dist)
        cmd.ensure_finalized()

        with quiet():
            cmd.run()

        manifest = cmd.filelist.files
        assert 'setup.py' in manifest

    def test_setup_py_missing(self):
        dist = Distribution(SETUP_ATTRS)
        dist.script_name = 'foo.py'
        cmd = sdist(dist)
        cmd.ensure_finalized()

        if os.path.exists("setup.py"):
            os.remove("setup.py")
        with quiet():
            cmd.run()

        manifest = cmd.filelist.files
        assert 'setup.py' not in manifest

    def test_setup_py_excluded(self):
        with open("MANIFEST.in", "w", encoding="utf-8") as manifest_file:
            manifest_file.write("exclude setup.py")

        dist = Distribution(SETUP_ATTRS)
        dist.script_name = 'foo.py'
        cmd = sdist(dist)
        cmd.ensure_finalized()

        with quiet():
            cmd.run()

        manifest = cmd.filelist.files
        assert 'setup.py' not in manifest

    def test_defaults_case_sensitivity(self, source_dir):
        """
        Make sure default files (README.*, etc.) are added in a case-sensitive
        way to avoid problems with packages built on Windows.
        """

        touch(source_dir / 'readme.rst')
        touch(source_dir / 'SETUP.cfg')

        dist = Distribution(SETUP_ATTRS)
        # the extension deliberately capitalized for this test
        # to make sure the actual filename (not capitalized) gets added
        # to the manifest
        dist.script_name = 'setup.PY'
        cmd = sdist(dist)
        cmd.ensure_finalized()

        with quiet():
            cmd.run()

        # lowercase all names so we can test in a
        # case-insensitive way to make sure the files
        # are not included.
        manifest = map(lambda x: x.lower(), cmd.filelist.files)
        assert 'readme.rst' not in manifest, manifest
        assert 'setup.py' not in manifest, manifest
        assert 'setup.cfg' not in manifest, manifest

    def test_exclude_dev_only_cache_folders(self, source_dir):
        included = {
            # Emulate problem in https://github.com/pypa/setuptools/issues/4601
            "MANIFEST.in": (
                "global-include LICEN[CS]E* COPYING* NOTICE* AUTHORS*\n"
                "global-include *.txt\n"
            ),
            # For the sake of being conservative and limiting unforeseen side-effects
            # we just exclude dev-only cache folders at the root of the repository:
            "test/.venv/lib/python3.9/site-packages/bar-2.dist-info/AUTHORS.rst": "",
            "src/.nox/py/lib/python3.12/site-packages/bar-2.dist-info/COPYING.txt": "",
            "doc/.tox/default/lib/python3.11/site-packages/foo-4.dist-info/LICENSE": "",
            # Let's test against false positives with similarly named files:
            ".venv-requirements.txt": "",
            ".tox-coveragerc.txt": "",
            ".noxy/coveragerc.txt": "",
        }

        excluded = {
            # .tox/.nox/.venv are well-know folders present at the root of Python repos
            # and therefore should be excluded
            ".tox/release/lib/python3.11/site-packages/foo-4.dist-info/LICENSE": "",
            ".nox/py/lib/python3.12/site-packages/bar-2.dist-info/COPYING.txt": "",
            ".venv/lib/python3.9/site-packages/bar-2.dist-info/AUTHORS.rst": "",
        }

        for file, content in {**excluded, **included}.items():
            Path(source_dir, file).parent.mkdir(parents=True, exist_ok=True)
            Path(source_dir, file).write_text(content, encoding="utf-8")

        cmd = self.setup_with_extension()
        self.assert_package_data_in_manifest(cmd)
        manifest = {f.replace(os.sep, '/') for f in cmd.filelist.files}
        for path in excluded:
            assert os.path.exists(path)
            assert path not in manifest, (path, manifest)
        for path in included:
            assert os.path.exists(path)
            assert path in manifest, (path, manifest)

    @fail_on_ascii
    def test_manifest_is_written_with_utf8_encoding(self):
        # Test for #303.
        dist = Distribution(SETUP_ATTRS)
        dist.script_name = 'setup.py'
        mm = manifest_maker(dist)
        mm.manifest = os.path.join('sdist_test.egg-info', 'SOURCES.txt')
        os.mkdir('sdist_test.egg-info')

        # UTF-8 filename
        filename = os.path.join('sdist_test', 'smörbröd.py')

        # Must create the file or it will get stripped.
        touch(filename)

        # Add UTF-8 filename and write manifest
        with quiet():
            mm.run()
            mm.filelist.append(filename)
            mm.write_manifest()

        contents = read_all_bytes(mm.manifest)

        # The manifest should be UTF-8 encoded
        u_contents = contents.decode('UTF-8')

        # The manifest should contain the UTF-8 filename
        assert posix(filename) in u_contents

    @fail_on_ascii
    def test_write_manifest_allows_utf8_filenames(self):
        # Test for #303.
        dist = Distribution(SETUP_ATTRS)
        dist.script_name = 'setup.py'
        mm = manifest_maker(dist)
        mm.manifest = os.path.join('sdist_test.egg-info', 'SOURCES.txt')
        os.mkdir('sdist_test.egg-info')

        filename = os.path.join(b'sdist_test', Filenames.utf_8)

        # Must touch the file or risk removal
        touch(filename)

        # Add filename and write manifest
        with quiet():
            mm.run()
            u_filename = filename.decode('utf-8')
            mm.filelist.files.append(u_filename)
            # Re-write manifest
            mm.write_manifest()

        contents = read_all_bytes(mm.manifest)

        # The manifest should be UTF-8 encoded
        contents.decode('UTF-8')

        # The manifest should contain the UTF-8 filename
        assert posix(filename) in contents

        # The filelist should have been updated as well
        assert u_filename in mm.filelist.files

    @skip_under_xdist
    def test_write_manifest_skips_non_utf8_filenames(self):
        """
        Files that cannot be encoded to UTF-8 (specifically, those that
        weren't originally successfully decoded and have surrogate
        escapes) should be omitted from the manifest.
        See https://bitbucket.org/tarek/distribute/issue/303 for history.
        """
        dist = Distribution(SETUP_ATTRS)
        dist.script_name = 'setup.py'
        mm = manifest_maker(dist)
        mm.manifest = os.path.join('sdist_test.egg-info', 'SOURCES.txt')
        os.mkdir('sdist_test.egg-info')

        # Latin-1 filename
        filename = os.path.join(b'sdist_test', Filenames.latin_1)

        # Add filename with surrogates and write manifest
        with quiet():
            mm.run()
            u_filename = filename.decode('utf-8', 'surrogateescape')
            mm.filelist.append(u_filename)
            # Re-write manifest
            mm.write_manifest()

        contents = read_all_bytes(mm.manifest)

        # The manifest should be UTF-8 encoded
        contents.decode('UTF-8')

        # The Latin-1 filename should have been skipped
        assert posix(filename) not in contents

        # The filelist should have been updated as well
        assert u_filename not in mm.filelist.files

    @fail_on_ascii
    def test_manifest_is_read_with_utf8_encoding(self):
        # Test for #303.
        dist = Distribution(SETUP_ATTRS)
        dist.script_name = 'setup.py'
        cmd = sdist(dist)
        cmd.ensure_finalized()

        # Create manifest
        with quiet():
            cmd.run()

        # Add UTF-8 filename to manifest
        filename = os.path.join(b'sdist_test', Filenames.utf_8)
        cmd.manifest = os.path.join('sdist_test.egg-info', 'SOURCES.txt')
        manifest = open(cmd.manifest, 'ab')
        manifest.write(b'\n' + filename)
        manifest.close()

        # The file must exist to be included in the filelist
        touch(filename)

        # Re-read manifest
        cmd.filelist.files = []
        with quiet():
            cmd.read_manifest()

        # The filelist should contain the UTF-8 filename
        filename = filename.decode('utf-8')
        assert filename in cmd.filelist.files

    @fail_on_latin1_encoded_filenames
    def test_read_manifest_skips_non_utf8_filenames(self):
        # Test for #303.
        dist = Distribution(SETUP_ATTRS)
        dist.script_name = 'setup.py'
        cmd = sdist(dist)
        cmd.ensure_finalized()

        # Create manifest
        with quiet():
            cmd.run()

        # Add Latin-1 filename to manifest
        filename = os.path.join(b'sdist_test', Filenames.latin_1)
        cmd.manifest = os.path.join('sdist_test.egg-info', 'SOURCES.txt')
        manifest = open(cmd.manifest, 'ab')
        manifest.write(b'\n' + filename)
        manifest.close()

        # The file must exist to be included in the filelist
        touch(filename)

        # Re-read manifest
        cmd.filelist.files = []
        with quiet():
            cmd.read_manifest()

        # The Latin-1 filename should have been skipped
        filename = filename.decode('latin-1')
        assert filename not in cmd.filelist.files

    @fail_on_ascii
    @fail_on_latin1_encoded_filenames
    def test_sdist_with_utf8_encoded_filename(self):
        # Test for #303.
        dist = Distribution(self.make_strings(SETUP_ATTRS))
        dist.script_name = 'setup.py'
        cmd = sdist(dist)
        cmd.ensure_finalized()

        filename = os.path.join(b'sdist_test', Filenames.utf_8)
        touch(filename)

        with quiet():
            cmd.run()

        if sys.platform == 'darwin':
            filename = decompose(filename)

        fs_enc = sys.getfilesystemencoding()

        if sys.platform == 'win32':
            if fs_enc == 'cp1252':
                # Python mangles the UTF-8 filename
                filename = filename.decode('cp1252')
                assert filename in cmd.filelist.files
            else:
                filename = filename.decode('mbcs')
                assert filename in cmd.filelist.files
        else:
            filename = filename.decode('utf-8')
            assert filename in cmd.filelist.files

    @classmethod
    def make_strings(cls, item):
        if isinstance(item, dict):
            return {key: cls.make_strings(value) for key, value in item.items()}
        if isinstance(item, list):
            return list(map(cls.make_strings, item))
        return str(item)

    @fail_on_latin1_encoded_filenames
    @skip_under_xdist
    def test_sdist_with_latin1_encoded_filename(self):
        # Test for #303.
        dist = Distribution(self.make_strings(SETUP_ATTRS))
        dist.script_name = 'setup.py'
        cmd = sdist(dist)
        cmd.ensure_finalized()

        # Latin-1 filename
        filename = os.path.join(b'sdist_test', Filenames.latin_1)
        touch(filename)
        assert os.path.isfile(filename)

        with quiet():
            cmd.run()

        # not all windows systems have a default FS encoding of cp1252
        if sys.platform == 'win32':
            # Latin-1 is similar to Windows-1252 however
            # on mbcs filesys it is not in latin-1 encoding
            fs_enc = sys.getfilesystemencoding()
            if fs_enc != 'mbcs':
                fs_enc = 'latin-1'
            filename = filename.decode(fs_enc)

            assert filename in cmd.filelist.files
        else:
            # The Latin-1 filename should have been skipped
            filename = filename.decode('latin-1')
            assert filename not in cmd.filelist.files

    _EXAMPLE_DIRECTIVES = {
        "setup.cfg - long_description and version": """
            [metadata]
            name = testing
            version = file: src/VERSION.txt
            license_files = DOWHATYOUWANT
            long_description = file: README.rst, USAGE.rst
            """,
        "pyproject.toml - static readme/license files and dynamic version": """
            [project]
            name = "testing"
            readme = "USAGE.rst"
            license-files = ["DOWHATYOUWANT"]
            dynamic = ["version"]
            [tool.setuptools.dynamic]
            version = {file = ["src/VERSION.txt"]}
            """,
        "pyproject.toml - directive with str instead of list": """
            [project]
            name = "testing"
            readme = "USAGE.rst"
            license-files = ["DOWHATYOUWANT"]
            dynamic = ["version"]
            [tool.setuptools.dynamic]
            version = {file = "src/VERSION.txt"}
            """,
        "pyproject.toml - deprecated license table with file entry": """
            [project]
            name = "testing"
            readme = "USAGE.rst"
            license = {file = "DOWHATYOUWANT"}
            dynamic = ["version"]
            [tool.setuptools.dynamic]
            version = {file = "src/VERSION.txt"}
            """,
    }

    @pytest.mark.parametrize("config", _EXAMPLE_DIRECTIVES.keys())
    @pytest.mark.filterwarnings(
        "ignore:.project.license. as a TOML table is deprecated"
    )
    def test_add_files_referenced_by_config_directives(self, source_dir, config):
        config_file, _, _ = config.partition(" - ")
        config_text = self._EXAMPLE_DIRECTIVES[config]
        (source_dir / 'src').mkdir()
        (source_dir / 'src/VERSION.txt').write_text("0.42", encoding="utf-8")
        (source_dir / 'README.rst').write_text("hello world!", encoding="utf-8")
        (source_dir / 'USAGE.rst').write_text("hello world!", encoding="utf-8")
        (source_dir / 'DOWHATYOUWANT').write_text("hello world!", encoding="utf-8")
        (source_dir / config_file).write_text(config_text, encoding="utf-8")

        dist = Distribution({"packages": []})
        dist.script_name = 'setup.py'
        dist.parse_config_files()

        cmd = sdist(dist)
        cmd.ensure_finalized()
        with quiet():
            cmd.run()

        assert (
            'src/VERSION.txt' in cmd.filelist.files
            or 'src\\VERSION.txt' in cmd.filelist.files
        )
        assert 'USAGE.rst' in cmd.filelist.files
        assert 'DOWHATYOUWANT' in cmd.filelist.files
        assert '/' not in cmd.filelist.files
        assert '\\' not in cmd.filelist.files

    def test_pyproject_toml_in_sdist(self, source_dir):
        """
        Check if pyproject.toml is included in source distribution if present
        """
        touch(source_dir / 'pyproject.toml')
        dist = Distribution(SETUP_ATTRS)
        dist.script_name = 'setup.py'
        cmd = sdist(dist)
        cmd.ensure_finalized()
        with quiet():
            cmd.run()
        manifest = cmd.filelist.files
        assert 'pyproject.toml' in manifest

    def test_pyproject_toml_excluded(self, source_dir):
        """
        Check that pyproject.toml can excluded even if present
        """
        touch(source_dir / 'pyproject.toml')
        with open('MANIFEST.in', 'w', encoding="utf-8") as mts:
            print('exclude pyproject.toml', file=mts)
        dist = Distribution(SETUP_ATTRS)
        dist.script_name = 'setup.py'
        cmd = sdist(dist)
        cmd.ensure_finalized()
        with quiet():
            cmd.run()
        manifest = cmd.filelist.files
        assert 'pyproject.toml' not in manifest

    def test_build_subcommand_source_files(self, source_dir):
        touch(source_dir / '.myfile~')

        # Sanity check: without custom commands file list should not be affected
        dist = Distribution({**SETUP_ATTRS, "script_name": "setup.py"})
        cmd = sdist(dist)
        cmd.ensure_finalized()
        with quiet():
            cmd.run()
        manifest = cmd.filelist.files
        assert '.myfile~' not in manifest

        # Test: custom command should be able to augment file list
        dist = Distribution({**SETUP_ATTRS, "script_name": "setup.py"})
        build = dist.get_command_obj("build")
        build.sub_commands = [*build.sub_commands, ("build_custom", None)]

        class build_custom(Command):
            def initialize_options(self): ...

            def finalize_options(self): ...

            def run(self): ...

            def get_source_files(self):
                return ['.myfile~']

        dist.cmdclass.update(build_custom=build_custom)

        cmd = sdist(dist)
        cmd.use_defaults = True
        cmd.ensure_finalized()
        with quiet():
            cmd.run()
        manifest = cmd.filelist.files
        assert '.myfile~' in manifest

    @pytest.mark.skipif("os.environ.get('SETUPTOOLS_USE_DISTUTILS') == 'stdlib'")
    def test_build_base_pathlib(self, source_dir):
        """
        Ensure if build_base is a pathlib.Path, the build still succeeds.
        """
        dist = Distribution({
            **SETUP_ATTRS,
            "script_name": "setup.py",
            "options": {"build": {"build_base": pathlib.Path('build')}},
        })
        cmd = sdist(dist)
        cmd.ensure_finalized()
        with quiet():
            cmd.run()


def test_default_revctrl():
    """
    When _default_revctrl was removed from the `setuptools.command.sdist`
    module in 10.0, it broke some systems which keep an old install of
    setuptools (Distribute) around. Those old versions require that the
    setuptools package continue to implement that interface, so this
    function provides that interface, stubbed. See #320 for details.

    This interface must be maintained until Ubuntu 12.04 is no longer
    supported (by Setuptools).
    """
    (ep,) = metadata.EntryPoints._from_text(
        """
        [setuptools.file_finders]
        svn_cvs = setuptools.command.sdist:_default_revctrl
        """
    )
    res = ep.load()
    assert hasattr(res, '__iter__')


class TestRegressions:
    """
    Can be removed/changed if the project decides to change how it handles symlinks
    or external files.
    """

    @staticmethod
    def files_for_symlink_in_extension_depends(tmp_path, dep_path):
        return {
            "external": {
                "dir": {"file.h": ""},
            },
            "project": {
                "setup.py": cleandoc(
                    f"""
                    from setuptools import Extension, setup
                    setup(
                        name="myproj",
                        version="42",
                        ext_modules=[
                            Extension(
                                "hello", sources=["hello.pyx"],
                                depends=[{dep_path!r}]
                            )
                        ],
                    )
                    """
                ),
                "hello.pyx": "",
                "MANIFEST.in": "global-include *.h",
            },
        }

    @pytest.mark.parametrize(
        "dep_path", ("myheaders/dir/file.h", "myheaders/dir/../dir/file.h")
    )
    def test_symlink_in_extension_depends(self, monkeypatch, tmp_path, dep_path):
        # Given a project with a symlinked dir and a "depends" targeting that dir
        files = self.files_for_symlink_in_extension_depends(tmp_path, dep_path)
        jaraco.path.build(files, prefix=str(tmp_path))
        symlink_or_skip_test(tmp_path / "external", tmp_path / "project/myheaders")

        # When `sdist` runs, there should be no error
        members = run_sdist(monkeypatch, tmp_path / "project")
        # and the sdist should contain the symlinked files
        for expected in (
            "myproj-42/hello.pyx",
            "myproj-42/myheaders/dir/file.h",
        ):
            assert expected in members

    @staticmethod
    def files_for_external_path_in_extension_depends(tmp_path, dep_path):
        head, _, tail = dep_path.partition("$tmp_path$/")
        dep_path = tmp_path / tail if tail else head

        return {
            "external": {
                "dir": {"file.h": ""},
            },
            "project": {
                "setup.py": cleandoc(
                    f"""
                    from setuptools import Extension, setup
                    setup(
                        name="myproj",
                        version="42",
                        ext_modules=[
                            Extension(
                                "hello", sources=["hello.pyx"],
                                depends=[{str(dep_path)!r}]
                            )
                        ],
                    )
                    """
                ),
                "hello.pyx": "",
                "MANIFEST.in": "global-include *.h",
            },
        }

    @pytest.mark.parametrize(
        "dep_path", ("$tmp_path$/external/dir/file.h", "../external/dir/file.h")
    )
    def test_external_path_in_extension_depends(self, monkeypatch, tmp_path, dep_path):
        # Given a project with a "depends" targeting an external dir
        files = self.files_for_external_path_in_extension_depends(tmp_path, dep_path)
        jaraco.path.build(files, prefix=str(tmp_path))
        # When `sdist` runs, there should be no error
        members = run_sdist(monkeypatch, tmp_path / "project")
        # and the sdist should not contain the external file
        for name in members:
            assert "file.h" not in name


def run_sdist(monkeypatch, project):
    """Given a project directory, run the sdist and return its contents"""
    monkeypatch.chdir(project)
    with quiet():
        run_setup("setup.py", ["sdist"])

    archive = next((project / "dist").glob("*.tar.gz"))
    with tarfile.open(str(archive)) as tar:
        return set(tar.getnames())


def test_sanity_check_setuptools_own_sdist(setuptools_sdist):
    with tarfile.open(setuptools_sdist) as tar:
        files = tar.getnames()

    # setuptools sdist should not include the .tox folder
    tox_files = [name for name in files if ".tox" in name]
    assert len(tox_files) == 0, f"not empty {tox_files}"
