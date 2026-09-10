"""Tests for distutils.extension."""

import os
import pathlib
import warnings
from distutils.extension import Extension, read_setup_file

import pytest
from test.support.warnings_helper import check_warnings


class TestExtension:
    def test_read_setup_file(self):
        # trying to read a Setup file
        # (sample extracted from the PyGame project)
        setup = os.path.join(os.path.dirname(__file__), 'Setup.sample')

        exts = read_setup_file(setup)
        names = [ext.name for ext in exts]
        names.sort()

        # here are the extensions read_setup_file should have created
        # out of the file
        wanted = [
            '_arraysurfarray',
            '_camera',
            '_numericsndarray',
            '_numericsurfarray',
            'base',
            'bufferproxy',
            'cdrom',
            'color',
            'constants',
            'display',
            'draw',
            'event',
            'fastevent',
            'font',
            'gfxdraw',
            'image',
            'imageext',
            'joystick',
            'key',
            'mask',
            'mixer',
            'mixer_music',
            'mouse',
            'movie',
            'overlay',
            'pixelarray',
            'pypm',
            'rect',
            'rwobject',
            'scrap',
            'surface',
            'surflock',
            'time',
            'transform',
        ]

        assert names == wanted

    def test_extension_init(self):
        # the first argument, which is the name, must be a string
        with pytest.raises(TypeError):
            Extension(1, [])
        ext = Extension('name', [])
        assert ext.name == 'name'

        # the second argument, which is the list of files, must
        # be an iterable of strings or PathLike objects, and not a string
        with pytest.raises(TypeError):
            Extension('name', 'file')
        with pytest.raises(TypeError):
            Extension('name', ['file', 1])
        ext = Extension('name', ['file1', 'file2'])
        assert ext.sources == ['file1', 'file2']
        ext = Extension('name', [pathlib.Path('file1'), pathlib.Path('file2')])
        assert ext.sources == ['file1', 'file2']

        # any non-string iterable of strings or PathLike objects should work
        ext = Extension('name', ('file1', 'file2'))  # tuple
        assert ext.sources == ['file1', 'file2']
        ext = Extension('name', {'file1', 'file2'})  # set
        assert sorted(ext.sources) == ['file1', 'file2']
        ext = Extension('name', iter(['file1', 'file2']))  # iterator
        assert ext.sources == ['file1', 'file2']
        ext = Extension('name', [pathlib.Path('file1'), 'file2'])  # mixed types
        assert ext.sources == ['file1', 'file2']

        # others arguments have defaults
        for attr in (
            'include_dirs',
            'define_macros',
            'undef_macros',
            'library_dirs',
            'libraries',
            'runtime_library_dirs',
            'extra_objects',
            'extra_compile_args',
            'extra_link_args',
            'export_symbols',
            'swig_opts',
            'depends',
        ):
            assert getattr(ext, attr) == []

        assert ext.language is None
        assert ext.optional is None

        # if there are unknown keyword options, warn about them
        with check_warnings() as w:
            warnings.simplefilter('always')
            ext = Extension('name', ['file1', 'file2'], chic=True)

        assert len(w.warnings) == 1
        assert str(w.warnings[0].message) == "Unknown Extension options: 'chic'"
