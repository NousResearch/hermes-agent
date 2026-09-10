"""Tests for distutils.util."""

import email
import email.generator
import email.policy
import io
import os
import pathlib
import sys
import sysconfig as stdlib_sysconfig
import unittest.mock as mock
from copy import copy
from distutils import sysconfig, util
from distutils.errors import DistutilsByteCompileError, DistutilsPlatformError
from distutils.util import (
    byte_compile,
    change_root,
    check_environ,
    convert_path,
    get_host_platform,
    get_platform,
    grok_environment_error,
    rfc822_escape,
    split_quoted,
    strtobool,
)

import pytest


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setattr(os, 'name', os.name)
    monkeypatch.setattr(sys, 'platform', sys.platform)
    monkeypatch.setattr(sys, 'version', sys.version)
    monkeypatch.setattr(os, 'sep', os.sep)
    monkeypatch.setattr(os.path, 'join', os.path.join)
    monkeypatch.setattr(os.path, 'isabs', os.path.isabs)
    monkeypatch.setattr(os.path, 'splitdrive', os.path.splitdrive)
    monkeypatch.setattr(sysconfig, '_config_vars', copy(sysconfig._config_vars))


@pytest.mark.usefixtures('save_env')
class TestUtil:
    def test_get_host_platform(self):
        with mock.patch('os.name', 'nt'):
            with mock.patch('sys.version', '... [... (ARM64)]'):
                assert get_host_platform() == 'win-arm64'
            with mock.patch('sys.version', '... [... (ARM)]'):
                assert get_host_platform() == 'win-arm32'

        with mock.patch('sys.version_info', (3, 9, 0, 'final', 0)):
            assert get_host_platform() == stdlib_sysconfig.get_platform()

    def test_get_platform(self):
        with mock.patch('os.name', 'nt'):
            with mock.patch.dict('os.environ', {'VSCMD_ARG_TGT_ARCH': 'x86'}):
                assert get_platform() == 'win32'
            with mock.patch.dict('os.environ', {'VSCMD_ARG_TGT_ARCH': 'x64'}):
                assert get_platform() == 'win-amd64'
            with mock.patch.dict('os.environ', {'VSCMD_ARG_TGT_ARCH': 'arm'}):
                assert get_platform() == 'win-arm32'
            with mock.patch.dict('os.environ', {'VSCMD_ARG_TGT_ARCH': 'arm64'}):
                assert get_platform() == 'win-arm64'

    def test_convert_path(self):
        expected = os.sep.join(('', 'home', 'to', 'my', 'stuff'))
        assert convert_path('/home/to/my/stuff') == expected
        assert convert_path(pathlib.Path('/home/to/my/stuff')) == expected
        assert convert_path('.') == os.curdir

    def test_change_root(self):
        # linux/mac
        os.name = 'posix'

        def _isabs(path):
            return path[0] == '/'

        os.path.isabs = _isabs

        def _join(*path):
            return '/'.join(path)

        os.path.join = _join

        assert change_root('/root', '/old/its/here') == '/root/old/its/here'
        assert change_root('/root', 'its/here') == '/root/its/here'

        # windows
        os.name = 'nt'
        os.sep = '\\'

        def _isabs(path):
            return path.startswith('c:\\')

        os.path.isabs = _isabs

        def _splitdrive(path):
            if path.startswith('c:'):
                return ('', path.replace('c:', ''))
            return ('', path)

        os.path.splitdrive = _splitdrive

        def _join(*path):
            return '\\'.join(path)

        os.path.join = _join

        assert (
            change_root('c:\\root', 'c:\\old\\its\\here') == 'c:\\root\\old\\its\\here'
        )
        assert change_root('c:\\root', 'its\\here') == 'c:\\root\\its\\here'

        # BugsBunny os (it's a great os)
        os.name = 'BugsBunny'
        with pytest.raises(DistutilsPlatformError):
            change_root('c:\\root', 'its\\here')

        # XXX platforms to be covered: mac

    def test_check_environ(self):
        util.check_environ.cache_clear()
        os.environ.pop('HOME', None)

        check_environ()

        assert os.environ['PLAT'] == get_platform()

    @pytest.mark.skipif("os.name != 'posix'")
    def test_check_environ_getpwuid(self):
        util.check_environ.cache_clear()
        os.environ.pop('HOME', None)

        import pwd

        # only set pw_dir field, other fields are not used
        result = pwd.struct_passwd((
            None,
            None,
            None,
            None,
            None,
            '/home/distutils',
            None,
        ))
        with mock.patch.object(pwd, 'getpwuid', return_value=result):
            check_environ()
            assert os.environ['HOME'] == '/home/distutils'

        util.check_environ.cache_clear()
        os.environ.pop('HOME', None)

        # bpo-10496: Catch pwd.getpwuid() error
        with mock.patch.object(pwd, 'getpwuid', side_effect=KeyError):
            check_environ()
            assert 'HOME' not in os.environ

    def test_split_quoted(self):
        assert split_quoted('""one"" "two" \'three\' \\four') == [
            'one',
            'two',
            'three',
            'four',
        ]

    def test_strtobool(self):
        yes = ('y', 'Y', 'yes', 'True', 't', 'true', 'True', 'On', 'on', '1')
        no = ('n', 'no', 'f', 'false', 'off', '0', 'Off', 'No', 'N')

        for y in yes:
            assert strtobool(y)

        for n in no:
            assert not strtobool(n)

    indent = 8 * ' '

    @pytest.mark.parametrize(
        "given,wanted",
        [
            # 0x0b, 0x0c, ..., etc are also considered a line break by Python
            ("hello\x0b\nworld\n", f"hello\x0b{indent}\n{indent}world\n{indent}"),
            ("hello\x1eworld", f"hello\x1e{indent}world"),
            ("", ""),
            (
                "I am a\npoor\nlonesome\nheader\n",
                f"I am a\n{indent}poor\n{indent}lonesome\n{indent}header\n{indent}",
            ),
        ],
    )
    def test_rfc822_escape(self, given, wanted):
        """
        We want to ensure a multi-line header parses correctly.

        For interoperability, the escaped value should also "round-trip" over
        `email.generator.Generator.flatten` and `email.message_from_*`
        (see pypa/setuptools#4033).

        The main issue is that internally `email.policy.EmailPolicy` uses
        `splitlines` which will split on some control chars. If all the new lines
        are not prefixed with spaces, the parser will interrupt reading
        the current header and produce an incomplete value, while
        incorrectly interpreting the rest of the headers as part of the payload.
        """
        res = rfc822_escape(given)

        policy = email.policy.EmailPolicy(
            utf8=True,
            mangle_from_=False,
            max_line_length=0,
        )
        with io.StringIO() as buffer:
            raw = f"header: {res}\nother-header: 42\n\npayload\n"
            orig = email.message_from_string(raw)
            email.generator.Generator(buffer, policy=policy).flatten(orig)
            buffer.seek(0)
            regen = email.message_from_file(buffer)

        for msg in (orig, regen):
            assert msg.get_payload() == "payload\n"
            assert msg["other-header"] == "42"
            # Generator may replace control chars with `\n`
            assert set(msg["header"].splitlines()) == set(res.splitlines())

        assert res == wanted

    def test_dont_write_bytecode(self):
        # makes sure byte_compile raise a DistutilsError
        # if sys.dont_write_bytecode is True
        old_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            with pytest.raises(DistutilsByteCompileError):
                byte_compile([])
        finally:
            sys.dont_write_bytecode = old_dont_write_bytecode

    def test_grok_environment_error(self):
        # test obsolete function to ensure backward compat (#4931)
        exc = OSError("Unable to find batch file")
        msg = grok_environment_error(exc)
        assert msg == "error: Unable to find batch file"
