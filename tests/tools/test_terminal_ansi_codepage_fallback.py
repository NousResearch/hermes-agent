"""Terminal output must not be destroyed by decoding ANSI-codepage bytes as UTF-8 (#89442).

Git Bash forwards a native Windows child's bytes without transcoding them, so on
a CJK install a ``powershell.exe`` error arrives as GBK inside a stream whose
MSYS tools write UTF-8. ``_wait_for_process`` decoded the whole stream strictly
as UTF-8 with ``errors="replace"``, which turned every one of those bytes into
U+FFFD *at decode time* - the original bytes were gone and the agent read a wall
of replacement characters instead of the message it needed.

The contract these tests pin, in order of how much it matters:

1. **Output that is valid UTF-8 decodes exactly as before.** The fallback is
   consulted only for byte sequences that strict UTF-8 rejects, which are
   already lost today. This change can recover text; it must not be able to
   corrupt text that was previously fine.
2. Decoding is per line, not per read. Legacy multi-byte encodings are not
   self-synchronising, so a fallback decode over an arbitrary byte range can
   split a two-byte GBK character. ``TestChunkBoundaries`` is that proof.
3. A backend that cannot know its writer's codepage keeps the old decoder. The
   default on ``BaseEnvironment`` is None, and only ``LocalEnvironment`` - where
   the writer is genuinely a child of this process - overrides it.
"""

import codecs
import locale
import os
from unittest.mock import MagicMock

from tools.environments.base import (
    BaseEnvironment,
    _system_ansi_encoding,
    _Utf8WithFallbackDecoder,
)
from tools.environments.local import LocalEnvironment

GBK_LINE = "找不到名为no-such-process的进程".encode("gbk")
UTF8_LINE = "réponse: 找不到".encode("utf-8")


class _TestableEnv(BaseEnvironment):
    """Concrete subclass so base-class methods can be exercised directly."""

    def __init__(self, cwd="/tmp", timeout=10):
        super().__init__(cwd=cwd, timeout=timeout)

    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        raise NotImplementedError("Use mock")

    def cleanup(self):
        pass


class TestValidUtf8IsUntouched:
    """The load-bearing half. If any of these break, the change is a net loss."""

    def test_ascii_is_unchanged(self):
        d = _Utf8WithFallbackDecoder("cp936")
        assert d.decode(b"hello world\n") == "hello world\n"

    def test_multibyte_utf8_is_unchanged(self):
        d = _Utf8WithFallbackDecoder("cp936")
        assert d.decode(UTF8_LINE + b"\n") == UTF8_LINE.decode("utf-8") + "\n"

    def test_utf8_that_is_also_valid_in_the_fallback_still_decodes_as_utf8(self):
        """The ordering matters, not just the outcome.

        A short UTF-8 sequence is usually *also* decodable as cp936, into
        different characters. Trying the fallback first, or on any error rather
        than only on a UTF-8 error, would silently mojibake ordinary Chinese
        output - a regression far worse than the bug.
        """
        text = "中文"
        raw = text.encode("utf-8")
        assert raw.decode("cp936", errors="ignore") != text  # premise of the test
        d = _Utf8WithFallbackDecoder("cp936")
        assert d.decode(raw + b"\n") == text + "\n"

    def test_a_pure_utf8_stream_matches_the_old_decoder_exactly(self):
        """Byte-for-byte agreement with the decoder this replaces."""
        payload = ("hello\n" + "réponse 中文\n" + "tail без переноса").encode("utf-8")
        baseline = codecs.getincrementaldecoder("utf-8")(errors="replace")
        old = baseline.decode(payload) + baseline.decode(b"", final=True)

        d = _Utf8WithFallbackDecoder("cp936")
        new = d.decode(payload) + d.decode(b"", final=True)

        assert new == old


class TestTheFallbackRecoversText:

    def test_a_gbk_line_is_recovered(self):
        d = _Utf8WithFallbackDecoder("cp936")
        out = d.decode(GBK_LINE + b"\n")
        assert out == GBK_LINE.decode("gbk") + "\n"
        assert "�" not in out

    def test_the_reported_powershell_shape(self):
        """The exact interleaving from #89442: MSYS line, then a native error."""
        d = _Utf8WithFallbackDecoder("cp936")
        stream = b"$ Stop-Process\n" + GBK_LINE + b"\r\n" + b"done\n"
        out = d.decode(stream) + d.decode(b"", final=True)
        assert out == "$ Stop-Process\n" + GBK_LINE.decode("gbk") + "\r\n" + "done\n"

    def test_without_the_fallback_the_same_bytes_are_destroyed(self):
        """Pins the bug itself, so a revert of the wiring is visible here."""
        baseline = codecs.getincrementaldecoder("utf-8")(errors="replace")
        out = baseline.decode(GBK_LINE + b"\n")
        assert "�" in out
        assert "找不到" not in out


class TestChunkBoundaries:
    """A 4096-byte read can split a character. Neither encoding may be damaged."""

    def test_a_utf8_character_split_across_reads_survives(self):
        raw = "中".encode("utf-8")
        d = _Utf8WithFallbackDecoder("cp936")
        first = d.decode(raw[:2])
        second = d.decode(raw[2:] + b"\n")
        assert first + second == "中\n"

    def test_a_gbk_character_split_across_reads_survives(self):
        """The case a per-chunk fallback cannot get right.

        GBK is two bytes per character with no lead/continuation distinction,
        so decoding an arbitrary byte range picks up half a character. Holding
        the line together is what makes this work.
        """
        raw = "进程".encode("gbk")
        assert len(raw) == 4
        d = _Utf8WithFallbackDecoder("cp936")
        out = d.decode(raw[:1]) + d.decode(raw[1:3]) + d.decode(raw[3:] + b"\n")
        assert out == "进程\n"

    def test_a_line_split_across_many_reads_survives(self):
        d = _Utf8WithFallbackDecoder("cp936")
        payload = GBK_LINE + b"\n"
        out = "".join(d.decode(payload[i:i + 1]) for i in range(len(payload)))
        assert out == GBK_LINE.decode("gbk") + "\n"


class TestBufferingContract:

    def test_an_unterminated_tail_is_held_until_final(self):
        d = _Utf8WithFallbackDecoder("cp936")
        assert d.decode(b"no newline yet") == ""
        assert d.decode(b"", final=True) == "no newline yet"

    def test_final_flush_recovers_an_unterminated_fallback_line(self):
        """``printf`` without a trailing newline is ordinary, not exotic."""
        d = _Utf8WithFallbackDecoder("cp936")
        assert d.decode(GBK_LINE) == ""
        assert d.decode(b"", final=True) == GBK_LINE.decode("gbk")

    def test_a_bare_carriage_return_is_a_boundary(self):
        """Progress output redraws with CR and would otherwise buffer to the cap."""
        d = _Utf8WithFallbackDecoder("cp936")
        assert d.decode(b"50%\r") == "50%\r"

    def test_output_with_no_boundary_is_flushed_at_the_cap(self):
        """A program that never emits a newline must not pin memory."""
        d = _Utf8WithFallbackDecoder("cp936")
        blob = b"x" * (_Utf8WithFallbackDecoder._MAX_BUFFERED_BYTES + 10)
        out = d.decode(blob)
        assert out == blob.decode("ascii")
        assert d.decode(b"", final=True) == ""

    def test_nothing_is_emitted_twice(self):
        d = _Utf8WithFallbackDecoder("cp936")
        first = d.decode(b"one\ntwo\n")
        second = d.decode(b"", final=True)
        assert first == "one\ntwo\n"
        assert second == ""


class TestItStillFailsSafe:

    def test_binary_with_a_nul_byte_keeps_the_replacement_behaviour(self):
        """Legacy codepages are byte-dense, so binary would come back as mojibake.

        cp1252 in particular decodes almost anything. A NUL is the cheap
        discriminator: no text encoding in use emits an interior NUL, and
        binary blobs almost always contain one.
        """
        d = _Utf8WithFallbackDecoder("cp1252")
        out = d.decode(b"\x00\xff\xfe\n")
        assert out == b"\x00\xff\xfe\n".decode("utf-8", errors="replace")
        assert "�" in out

    def test_an_unknown_fallback_codec_degrades_to_replacement(self):
        d = _Utf8WithFallbackDecoder("not-a-real-codec")
        out = d.decode(GBK_LINE + b"\n")
        assert "�" in out

    def test_bytes_invalid_in_both_encodings_degrade_to_replacement(self):
        """cp936 rejects a lone 0x80, so neither decode can succeed."""
        d = _Utf8WithFallbackDecoder("cp936")
        out = d.decode(b"\x80\n")
        assert out == b"\x80\n".decode("utf-8", errors="replace")


class TestSystemAnsiEncoding:

    def test_a_utf8_host_has_no_fallback(self, monkeypatch):
        """None is what makes this provably a no-op almost everywhere."""
        monkeypatch.setattr(locale, "getencoding", lambda: "utf-8")
        assert _system_ansi_encoding() is None

    def test_utf8_aliases_are_normalised(self, monkeypatch):
        monkeypatch.setattr(locale, "getencoding", lambda: "UTF8")
        assert _system_ansi_encoding() is None

    def test_a_cjk_host_reports_its_codepage(self, monkeypatch):
        monkeypatch.setattr(locale, "getencoding", lambda: "cp936")
        assert _system_ansi_encoding() == codecs.lookup("cp936").name

    def test_an_unknown_encoding_name_is_no_fallback(self, monkeypatch):
        monkeypatch.setattr(locale, "getencoding", lambda: "nonsense-codec")
        assert _system_ansi_encoding() is None

    def test_an_empty_encoding_name_is_no_fallback(self, monkeypatch):
        monkeypatch.setattr(locale, "getencoding", lambda: "")
        assert _system_ansi_encoding() is None

    def test_a_raising_locale_is_no_fallback(self, monkeypatch):
        def _boom():
            raise RuntimeError("locale is unavailable")

        monkeypatch.setattr(locale, "getencoding", _boom)
        assert _system_ansi_encoding() is None


class TestBackendScope:

    def test_the_base_default_is_no_fallback(self):
        """Remote backends must not guess a container's codepage from this host."""
        assert _TestableEnv()._output_fallback_encoding() is None

    def test_local_has_no_fallback_off_windows(self, monkeypatch):
        """On POSIX the shell and its children agree on the locale, so a
        non-UTF-8 line is far likelier to be binary than to be locale text."""
        monkeypatch.setattr("tools.environments.local._IS_WINDOWS", False)
        monkeypatch.setattr(locale, "getencoding", lambda: "cp936")
        env = LocalEnvironment.__new__(LocalEnvironment)
        assert env._output_fallback_encoding() is None

    def test_local_uses_the_ansi_codepage_on_windows(self, monkeypatch):
        monkeypatch.setattr("tools.environments.local._IS_WINDOWS", True)
        monkeypatch.setattr(locale, "getencoding", lambda: "cp936")
        env = LocalEnvironment.__new__(LocalEnvironment)
        assert env._output_fallback_encoding() == codecs.lookup("cp936").name

    def test_local_on_a_utf8_windows_host_has_no_fallback(self, monkeypatch):
        monkeypatch.setattr("tools.environments.local._IS_WINDOWS", True)
        monkeypatch.setattr(locale, "getencoding", lambda: "utf-8")
        env = LocalEnvironment.__new__(LocalEnvironment)
        assert env._output_fallback_encoding() is None


def _proc_emitting(payload: bytes):
    """A ProcessHandle whose stdout is a real pipe holding *payload*.

    A real fd rather than a mock: ``_wait_for_process`` resolves ``fileno()``
    and ``os.read()``s it, and a MagicMock ``fileno()`` sends the drain down the
    iterator fallback instead of the byte path under test.
    """
    read_fd, write_fd = os.pipe()
    os.write(write_fd, payload)
    os.close(write_fd)
    proc = MagicMock()
    proc.poll.return_value = 0
    proc.returncode = 0
    proc.stdout = os.fdopen(read_fd, "rb", buffering=0)
    # Real handles only carry this when _pipe_stdin recorded a failure; a bare
    # MagicMock would auto-create it and append a bogus "[stdin write failed]".
    proc._hermes_stdin_errors = []
    return proc


class TestEndToEndThroughTheDrain:
    """The wiring, exercised through ``_wait_for_process`` itself."""

    def test_a_backend_with_a_fallback_recovers_the_message(self):
        env = _TestableEnv()
        env._output_fallback_encoding = lambda: "cp936"
        proc = _proc_emitting(GBK_LINE + b"\n")
        try:
            result = env._wait_for_process(proc, timeout=5)
        finally:
            proc.stdout.close()
        assert GBK_LINE.decode("gbk") in result["output"]
        assert "�" not in result["output"]

    def test_a_backend_without_one_is_unchanged(self):
        env = _TestableEnv()
        proc = _proc_emitting(GBK_LINE + b"\n")
        try:
            result = env._wait_for_process(proc, timeout=5)
        finally:
            proc.stdout.close()
        assert "�" in result["output"]

    def test_utf8_output_is_identical_with_and_without_a_fallback(self):
        payload = "réponse 中文\nsecond line\n".encode("utf-8")

        plain = _TestableEnv()
        proc_a = _proc_emitting(payload)
        try:
            without = plain._wait_for_process(proc_a, timeout=5)["output"]
        finally:
            proc_a.stdout.close()

        fallback = _TestableEnv()
        fallback._output_fallback_encoding = lambda: "cp936"
        proc_b = _proc_emitting(payload)
        try:
            with_fb = fallback._wait_for_process(proc_b, timeout=5)["output"]
        finally:
            proc_b.stdout.close()

        assert with_fb == without

    def test_a_raising_hook_cannot_fail_the_command(self):
        """Resolving an encoding is not worth losing a command's output over."""
        env = _TestableEnv()

        def _boom():
            raise RuntimeError("no locale here")

        env._output_fallback_encoding = _boom
        proc = _proc_emitting(b"still here\n")
        try:
            result = env._wait_for_process(proc, timeout=5)
        finally:
            proc.stdout.close()
        assert "still here" in result["output"]
