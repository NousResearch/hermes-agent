"""Incremental sandbox log reads for ProcessRegistry's non-local poller."""


import shlex
import time


class ProcessEnvLogMixin:
    def _recover_launch_capture(self, session, env, pid_path: str, result: dict) -> dict:
        if not (result.get("output_incomplete") or result.get("capture_error")):
            return result
        from tools.environments.capture import OutputCaptureError, require_complete_output
        session.capture_failed("remote launch PID read")
        # The launch may have succeeded. Only reread its durable receipt, never launch again.
        try:
            recovered = env.execute(f"cat {shlex.quote(pid_path)} 2>/dev/null", timeout=5)
            require_complete_output(recovered)
        except Exception as exc:
            raise OutputCaptureError(
                "Background command may already have started; its PID could not be recovered. "
                "Inspect its effects before considering a retry.") from exc
        if not recovered.get("output", "").strip().isdigit():
            raise OutputCaptureError(
                "Background command may already have started; its PID could not be recovered. "
                "Inspect its effects before considering a retry.")
        return recovered

    def _env_poller_loop(self, session, env, log_path: str, pid_path: str, exit_path: str):
        """Poll durable log reads; an incomplete fetch never commits its byte offset."""
        q = shlex.quote
        prev_output_bytes = 0
        while not session.exited:
            time.sleep(2)
            try:
                prev_output_bytes = self._fetch_log_delta(session, env, log_path, prev_output_bytes)
                check = env.execute(
                    f"kill -0 \"$(cat {q(pid_path)} 2>/dev/null)\" 2>/dev/null; echo $?", timeout=5)
                if self._capture_fetch_failed(session, check, "remote status read"):
                    continue
                check_output = check.get("output", "").strip()
                if check_output and check_output.splitlines()[-1].strip() != "0":
                    exited = env.execute(f"cat {q(exit_path)} 2>/dev/null", timeout=5)
                    if self._capture_fetch_failed(session, exited, "remote exit status read"):
                        continue
                    exit_str = exited.get("output", "").strip()
                    try:
                        exit_code = int(exit_str.splitlines()[-1].strip())
                    except (ValueError, IndexError):
                        exit_code = -1
                    self._finish_remote_log(session, env, log_path, prev_output_bytes, exit_code)
                    return
            except Exception:  # health: allow BLE001 -- backend loss keeps existing lifecycle semantics, never exception text
                self._finish_exited(session, -1, reason="lost", source="backend_lost")
                return

    def _finish_remote_log(self, session, env, log_path, offset, exit_code):
        try:
            self._fetch_log_delta(session, env, log_path, offset)
        except Exception:  # health: allow BLE001 -- transport failure cannot overwrite an already observed child exit
            session.capture_failed("final remote log read")
        self._finish_exited(session, exit_code)

    def _fetch_log_delta(self, session, env, log_path: str, offset: int) -> int:
        fetched = env.execute(self._log_delta_command(shlex.quote(log_path), offset), timeout=10)
        if self._capture_fetch_failed(session, fetched, "remote log read"):
            return offset
        header, _, delta = fetched.get("output", "").partition("\n")
        try:
            size, used = map(int, header.split())
        except ValueError:
            session.capture_failed("remote log framing")
            return offset
        with session._lock:
            if session.exited:
                return offset
            if used < offset:
                session.output_buffer = ""
            session._append_locked(delta)
        if delta:
            self._check_watch_patterns(session, delta)
            self._emit_output(session, delta)
        return size

    def _capture_fetch_failed(self, session, result: dict, stage: str) -> bool:
        if not (result.get("output_incomplete") or result.get("capture_error")):
            return False
        with session._lock:
            changed = not session.exited and not session.capture_error
            if changed:
                session.capture_failed(stage)
        if changed:
            self._write_checkpoint()
        return True


def log_delta_command(quoted_log_path: str, offset: int) -> str:
    """Shell command that reads only the log bytes written since ``offset``
    (``cat``-ing the whole file every poll re-sends all output over docker/SSH).

    Prints one header line ``"<size> <offset>"`` then the bytes in [offset, size).
    The size is read first and the tail cut at that same size, so a growing file
    never sends a byte twice; a file that shrank was rotated/truncated, so the
    offset drops to 0 and the reader starts over. The window end is pulled back
    to a UTF-8 character boundary (the backend decodes each ``execute()`` result
    on its own, so a straddling multibyte char would become U+FFFD and break watch
    patterns at the seam): up to 3 trailing continuation bytes are held for the
    next poll and the header reports the trimmed size."""
    return (
        f"O={offset}; "
        f"S=$({{ wc -c < {quoted_log_path}; }} 2>/dev/null | tr -dc '0-9'); "
        f"S=${{S:-0}}; "
        f'if [ "$S" -lt "$O" ]; then O=0; fi; '
        # Scan back up to 3 continuation bytes (octal 200-277) to the lead byte; if
        # the lead's declared length (3xx=2, 34x-35x=3, 36x-37x=4) exceeds the bytes
        # present, trim to before it. Complete sequences and ASCII tails untouched.
        f'N=0; P=$S; while [ "$P" -gt "$O" ] && [ "$N" -lt 3 ]; do '
        f"B=$(tail -c +$P {quoted_log_path} 2>/dev/null | head -c 1 | od -An -to1 | tr -dc '0-9'); "
        f'case "$B" in 2[0-7][0-7]) P=$((P-1)); N=$((N+1));; *) break;; esac; done; '
        f'if [ "$N" -gt 0 ] || [ "$P" -eq "$S" ]; then '
        f"B=$(tail -c +$P {quoted_log_path} 2>/dev/null | head -c 1 | od -An -to1 | tr -dc '0-9'); "
        f'case "$B" in 3[0-3][0-7]) L=2;; 3[4-5][0-7]) L=3;; 3[6-7][0-7]) L=4;; *) L=1;; esac; '
        f'if [ "$L" -gt $((N+1)) ]; then S=$((P-1)); fi; fi; '
        f'echo "$S $O"; '
        f'if [ "$S" -gt "$O" ]; then '
        f"tail -c +$((O+1)) {quoted_log_path} 2>/dev/null | head -c $((S-O)); fi"
    )
