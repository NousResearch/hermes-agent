"""Tests for distutils.core."""

import distutils.core
import io
import os
import sys
from distutils.dist import Distribution

import pytest

# setup script that uses __file__
setup_using___file__ = """\

__file__

from distutils.core import setup
setup()
"""

setup_prints_cwd = """\

import os
print(os.getcwd())

from distutils.core import setup
setup()
"""

setup_does_nothing = """\
from distutils.core import setup
setup()
"""


setup_defines_subclass = """\
from distutils.core import setup
from distutils.command.install import install as _install

class install(_install):
    sub_commands = _install.sub_commands + ['cmd']

setup(cmdclass={'install': install})
"""

setup_within_if_main = """\
from distutils.core import setup

def main():
    return setup(name="setup_within_if_main")

if __name__ == "__main__":
    main()
"""


@pytest.fixture(autouse=True)
def save_stdout(monkeypatch):
    monkeypatch.setattr(sys, 'stdout', sys.stdout)


@pytest.fixture
def temp_file(tmp_path):
    return tmp_path / 'file'


@pytest.mark.usefixtures('save_env')
@pytest.mark.usefixtures('save_argv')
class TestCore:
    def test_run_setup_provides_file(self, temp_file):
        # Make sure the script can use __file__; if that's missing, the test
        # setup.py script will raise NameError.
        temp_file.write_text(setup_using___file__, encoding='utf-8')
        distutils.core.run_setup(temp_file)

    def test_run_setup_preserves_sys_argv(self, temp_file):
        # Make sure run_setup does not clobber sys.argv
        argv_copy = sys.argv.copy()
        temp_file.write_text(setup_does_nothing, encoding='utf-8')
        distutils.core.run_setup(temp_file)
        assert sys.argv == argv_copy

    def test_run_setup_defines_subclass(self, temp_file):
        # Make sure the script can use __file__; if that's missing, the test
        # setup.py script will raise NameError.
        temp_file.write_text(setup_defines_subclass, encoding='utf-8')
        dist = distutils.core.run_setup(temp_file)
        install = dist.get_command_obj('install')
        assert 'cmd' in install.sub_commands

    def test_run_setup_uses_current_dir(self, tmp_path):
        """
        Test that the setup script is run with the current directory
        as its own current directory.
        """
        sys.stdout = io.StringIO()
        cwd = os.getcwd()

        # Create a directory and write the setup.py file there:
        setup_py = tmp_path / 'setup.py'
        setup_py.write_text(setup_prints_cwd, encoding='utf-8')
        distutils.core.run_setup(setup_py)

        output = sys.stdout.getvalue()
        if output.endswith("\n"):
            output = output[:-1]
        assert cwd == output

    def test_run_setup_within_if_main(self, temp_file):
        temp_file.write_text(setup_within_if_main, encoding='utf-8')
        dist = distutils.core.run_setup(temp_file, stop_after="config")
        assert isinstance(dist, Distribution)
        assert dist.get_name() == "setup_within_if_main"

    def test_run_commands(self, temp_file):
        sys.argv = ['setup.py', 'build']
        temp_file.write_text(setup_within_if_main, encoding='utf-8')
        dist = distutils.core.run_setup(temp_file, stop_after="commandline")
        assert 'build' not in dist.have_run
        distutils.core.run_commands(dist)
        assert 'build' in dist.have_run

    def test_debug_mode(self, capsys, monkeypatch):
        # this covers the code called when DEBUG is set
        sys.argv = ['setup.py', '--name']
        distutils.core.setup(name='bar')
        assert capsys.readouterr().out == 'bar\n'
        monkeypatch.setattr(distutils.core, 'DEBUG', True)
        distutils.core.setup(name='bar')
        wanted = "options (after parsing config files):\n"
        assert capsys.readouterr().out.startswith(wanted)
