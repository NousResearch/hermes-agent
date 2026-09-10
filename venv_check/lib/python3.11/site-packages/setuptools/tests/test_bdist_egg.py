"""develop tests"""

import os
import re
import zipfile

import pytest

from setuptools.dist import Distribution

from . import contexts

SETUP_PY = """\
from setuptools import setup

setup(py_modules=['hi'])
"""


@pytest.fixture
def setup_context(tmpdir):
    with (tmpdir / 'setup.py').open('w') as f:
        f.write(SETUP_PY)
    with (tmpdir / 'hi.py').open('w') as f:
        f.write('1\n')
    with tmpdir.as_cwd():
        yield tmpdir


class Test:
    @pytest.mark.usefixtures("user_override")
    @pytest.mark.usefixtures("setup_context")
    def test_bdist_egg(self):
        dist = Distribution(
            dict(
                script_name='setup.py',
                script_args=['bdist_egg'],
                name='foo',
                py_modules=['hi'],
            )
        )
        os.makedirs(os.path.join('build', 'src'))
        with contexts.quiet():
            dist.parse_command_line()
            dist.run_commands()

        # let's see if we got our egg link at the right place
        [content] = os.listdir('dist')
        assert re.match(r'foo-0.0.0-py[23].\d+.egg$', content)

    @pytest.mark.xfail(
        os.environ.get('PYTHONDONTWRITEBYTECODE', False),
        reason="Byte code disabled",
    )
    @pytest.mark.usefixtures("user_override")
    @pytest.mark.usefixtures("setup_context")
    def test_exclude_source_files(self):
        dist = Distribution(
            dict(
                script_name='setup.py',
                script_args=['bdist_egg', '--exclude-source-files'],
                py_modules=['hi'],
            )
        )
        with contexts.quiet():
            dist.parse_command_line()
            dist.run_commands()
        [dist_name] = os.listdir('dist')
        dist_filename = os.path.join('dist', dist_name)
        zip = zipfile.ZipFile(dist_filename)
        names = list(zi.filename for zi in zip.filelist)
        assert 'hi.pyc' in names
        assert 'hi.py' not in names
