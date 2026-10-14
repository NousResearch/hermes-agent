"""Native pipe readers, including adopted streams and Windows pipes."""
import codecs
import logging

from tools.process_registry_capture import read_local_streams

logger = logging.getLogger("tools.process_registry")


class ProcessReaderMixin:
    def _reader_loop(self, session):
        """Background thread: read stdout from a local Popen process.
        ``buffer.read1(4096)`` not ``TextIOWrapper.read(4096)``: on pipes the latter
        blocks until EOF, landing "live" output in one burst at exit. Orphaned-pipe
        guard: a backgrounded grandchild (``node server.js &``) inherits our pipe's write
        end so EOF never arrives while it lives, which would park this thread and never
        fire ``notify_on_complete``; on POSIX we ``select()`` and stop draining shortly
        after the direct child exits (mirrors ``environments/base.py::_wait_for_process``).
        Windows pipes lack select(), so the lazy ``_reconcile_local_exit`` is the net.

        Windows pipes don't support select(); the blocking path is kept there and the lazy reconcile in
        poll()/wait() remains the safety net. See #68915, #8340.
        """
        from tools.process_registry import _IS_WINDOWS, _FINAL_DRAIN_MAX_CHARS

        # ``bash -lic`` without a tty writes its startup warnings one write() per line, so the
        # reader can wake between them; strip leading noise from every chunk until the
        # process has produced real output, not just from the first read.
        if session.byte_capture is not None:
            return read_local_streams(self, session)
        head_noise = True
        # A split multibyte UTF-8 char would become U+FFFD with stateless decoding; the
        # incremental decoder holds the partial sequence until the rest arrives.
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        # Incremental decoder: raw pipe reads can split a multibyte UTF-8 character across two read1()
        # chunks. A stateless per-chunk ``bytes.decode(errors="replace")`` turns both halves into U+FFFD
        # mojibake. The incremental decoder holds the partial sequence until the continuation bytes arrive —
        # same treatment the foreground path already has in
        # ``tools/environments/base.py::_wait_for_process``. (Ported from openclaw/openclaw#112325.)
        def _append_chunk(chunk: str):
            nonlocal head_noise
            if head_noise:
                chunk = self._clean_shell_noise(chunk)
                head_noise = not chunk.strip()
            self._ingest_output(session, chunk)
        try:
            proc = session.process
            if proc is None or proc.stdout is None:
                return
            stdout = proc.stdout
            raw_read = getattr(getattr(stdout, "buffer", None), "read1", None)

            def _read_once():
                """One 4 KiB read: decoded text ('' for a partial multibyte tail), None at EOF."""
                if raw_read is None:  # mocked/alternate streams without a raw buffer: less "live"
                    return stdout.read(4096) or None
                raw = raw_read(4096)
                return decoder.decode(raw) if raw else None
            # select() needs a real OS fd; mocked streams (tests, adapters) may lack
            # fileno() and use the blocking read instead.
            try:
                fd = stdout.fileno() if raw_read is not None and not _IS_WINDOWS else None
            except (OSError, ValueError, AttributeError):
                fd = None
            if not (isinstance(fd, int) and fd >= 0):
                fd = None
            if fd is not None:
                import select as _select
                session._reader_selectable = True
            idle_after_exit = 0
            drained_after_request = 0
            while True:
                if fd is not None:
                    try:
                        # Once a finish is requested, read only what is already buffered: the exited
                        # child's tail, never an orphaned grandchild's future writes.
                        wait_s = 0 if session._reader_finish_requested.is_set() else 0.2
                        ready, _, _ = _select.select([fd], [], [], wait_s)
                    except (ValueError, OSError):
                        break  # fd already closed
                    if not ready:
                        if session._reader_finish_requested.is_set():
                            break
                        # Direct child gone and pipe idle ~200ms: a few more cycles for a
                        # buffered tail, then stop rather than wait forever on an orphaned
                        # grandchild's pipe.
                        if proc.poll() is not None:
                            # See #68915.
                            idle_after_exit += 1
                        if idle_after_exit >= 3:
                            break
                        continue
                chunk = _read_once()
                if chunk is None:
                    break  # true EOF — all writers closed
                if chunk:
                    _append_chunk(chunk)
                if session._reader_finish_requested.is_set():
                    # A single chunk is not the tail: the child can exit with a whole pipe buffer
                    # unread. The cap only stops a grandchild that never lets the pipe go empty.
                    drained_after_request += len(chunk)
                    if drained_after_request >= _FINAL_DRAIN_MAX_CHARS:
                        break
                idle_after_exit = 0
        except Exception as e:
            logger.debug("Process stdout reader ended: %s", e, exc_info=True)
        finally:
            self._finish_reader(
                session, decoder, _append_chunk, "Process",
                session.process.wait, lambda: session.process.returncode)

