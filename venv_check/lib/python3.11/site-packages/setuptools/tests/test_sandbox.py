"""develop tests"""

import os
import types

import pytest

import pkg_resources
import setuptools.sandbox


class TestSandbox:
    def test_devnull(self, tmpdir):
        with setuptools.sandbox.DirectorySandbox(str(tmpdir)):
            self._file_writer(os.devnull)

    @staticmethod
    def _file_writer(path):
        def do_write():
            with open(path, 'w', encoding="utf-8") as f:
                f.write('xxx')

        return do_write

    def test_setup_py_with_BOM(self):
        """
        It should be possible to execute a setup.py with a Byte Order Mark
        """
        target = pkg_resources.resource_filename(__name__, 'script-with-bom.py')
        namespace = types.ModuleType('namespace')
        setuptools.sandbox._execfile(target, vars(namespace))
        assert namespace.result == 'passed'

    def test_setup_py_with_CRLF(self, tmpdir):
        setup_py = tmpdir / 'setup.py'
        with setup_py.open('wb') as stream:
            stream.write(b'"degenerate script"\r\n')
        setuptools.sandbox._execfile(str(setup_py), globals())


class TestExceptionSaver:
    def test_exception_trapped(self):
        with setuptools.sandbox.ExceptionSaver():
            raise ValueError("details")

    def test_exception_resumed(self):
        with setuptools.sandbox.ExceptionSaver() as saved_exc:
            raise ValueError("details")

        with pytest.raises(ValueError) as caught:
            saved_exc.resume()

        assert isinstance(caught.value, ValueError)
        assert str(caught.value) == 'details'

    def test_exception_reconstructed(self):
        orig_exc = ValueError("details")

        with setuptools.sandbox.ExceptionSaver() as saved_exc:
            raise orig_exc

        with pytest.raises(ValueError) as caught:
            saved_exc.resume()

        assert isinstance(caught.value, ValueError)
        assert caught.value is not orig_exc

    def test_no_exception_passes_quietly(self):
        with setuptools.sandbox.ExceptionSaver() as saved_exc:
            pass

        saved_exc.resume()

    def test_unpickleable_exception(self):
        class CantPickleThis(Exception):
            "This Exception is unpickleable because it's not in globals"

            def __repr__(self) -> str:
                return f'CantPickleThis{self.args!r}'

        with setuptools.sandbox.ExceptionSaver() as saved_exc:
            raise CantPickleThis('detail')

        with pytest.raises(setuptools.sandbox.UnpickleableException) as caught:
            saved_exc.resume()

        assert str(caught.value) == "CantPickleThis('detail',)"

    def test_unpickleable_exception_when_hiding_setuptools(self):
        """
        As revealed in #440, an infinite recursion can occur if an unpickleable
        exception while setuptools is hidden. Ensure this doesn't happen.
        """

        class ExceptionUnderTest(Exception):
            """
            An unpickleable exception (not in globals).
            """

        with pytest.raises(setuptools.sandbox.UnpickleableException) as caught:
            with setuptools.sandbox.save_modules():
                setuptools.sandbox.hide_setuptools()
                raise ExceptionUnderTest

        (msg,) = caught.value.args
        assert msg == 'ExceptionUnderTest()'

    def test_sandbox_violation_raised_hiding_setuptools(self, tmpdir):
        """
        When in a sandbox with setuptools hidden, a SandboxViolation
        should reflect a proper exception and not be wrapped in
        an UnpickleableException.
        """

        def write_file():
            "Trigger a SandboxViolation by writing outside the sandbox"
            with open('/etc/foo', 'w', encoding="utf-8"):
                pass

        with pytest.raises(setuptools.sandbox.SandboxViolation) as caught:
            with setuptools.sandbox.save_modules():
                setuptools.sandbox.hide_setuptools()
                with setuptools.sandbox.DirectorySandbox(str(tmpdir)):
                    write_file()

        cmd, args, kwargs = caught.value.args
        assert cmd == 'open'
        assert args == ('/etc/foo', 'w')
        assert kwargs == {"encoding": "utf-8"}

        msg = str(caught.value)
        assert 'open' in msg
        assert "('/etc/foo', 'w')" in msg
        assert "{'encoding': 'utf-8'}" in msg
