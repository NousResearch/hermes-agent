"""Regression tests for lossy browser output decoding (#PR-browser-encoding).

Before the fix, ``_run_browser_command`` read agent-browser stdout/stderr
with a strict ``encoding="utf-8"`` decoder. On Windows the subprocess can
emit output in the system code page (e.g. Cp1252 with accented
characters), which raised::

    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc6 in position 15

That exception bubbled up as a hard ``browser_navigate`` failure instead of
surfacing the real browser error. The fix reads bytes and decodes with
``errors="replace"`` so the content is always returned.
"""

from unittest.mock import mock_open, patch


class TestReadOutputFileLossy:
    def setup_method(self):
        from tools import browser_tool

        self.bt = browser_tool

    def test_decodes_valid_utf8(self, tmp_path):
        p = tmp_path / "out.txt"
        p.write_bytes("hello world\n".encode("utf-8"))
        assert self.bt._read_output_file_lossy(str(p)) == "hello world"

    def test_decodes_cp1252_bytes_with_errors_replace(self, tmp_path):
        """A Cp1252-encoded accented string must not raise."""
        p = tmp_path / "out.txt"
        # 0xc6 is 'Æ' in Cp1252; invalid as strict UTF-8.
        p.write_bytes(b"status: falha na conex\xc6o\n")
        result = self.bt._read_output_file_lossy(str(p))
        assert result == "status: falha na conex\ufffdo"

    def test_accepts_already_decoded_text_from_file_like_wrapper(self):
        """Preserve compatibility with wrappers/mocks that return text."""
        with patch("builtins.open", mock_open(read_data="already decoded\n")):
            assert self.bt._read_output_file_lossy("out.txt") == "already decoded"

    def test_missing_file_returns_empty(self, tmp_path):
        missing = tmp_path / "does_not_exist.txt"
        assert self.bt._read_output_file_lossy(str(missing)) == ""

    def test_read_command_output_files_never_raises_on_bad_bytes(self, tmp_path):
        """_read_command_output_files must tolerate non-UTF-8 stderr."""
        out = tmp_path / "stdout.txt"
        err = tmp_path / "stderr.txt"
        out.write_bytes(b"ok\n")
        err.write_bytes(b"erro: byte \xc6 invalido\n")
        stdout, stderr = self.bt._read_command_output_files(str(out), str(err))
        assert stdout == "ok"
        assert stderr == "erro: byte \ufffd invalido"
