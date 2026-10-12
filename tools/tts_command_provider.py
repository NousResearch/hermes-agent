"""Shared runner for user-configured shell ("command") TTS/STT providers.

``tts.providers.<name>: {type: command, command: "piper -f {output_path} < {input_path}"}``
(and the ``stt.`` twin): ``{placeholders}`` are shell-quoted for their surrounding quote
context, ``{{``/``}}`` stay literal. Owns the quote-aware rendering, the idle-timeout
process runner and the generic ``<section>.providers.<name>`` readers, re-imported by
``tts_tool``/``transcription_tools`` under their historical private names. TTS placeholders:
``{input_path}``/``{text_path}``, ``{output_path}``, ``{format}``, ``{voice}``, ``{model}``,
``{speed}``. Built-in provider names always win over a same-named ``providers`` entry.
"""

from __future__ import annotations

import os
import logging
import queue
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional

from tools.tts_command_error_facts import (
    CommandFailureEvent,
    CommandFailureFacts,
    CommandOperation,
    CommandStream,
    command_failure_facts,
    set_command_failure_facts,
)
from tools.tts_command_process import CommandProcessScope
from utils import is_truthy_value

logger = logging.getLogger(__name__)

def shell_quote_context(command_template: str, position: int) -> Optional[str]:
    """Return the shell quote char (``'``/``"``) active right before *position*, or None."""
    quote: Optional[str] = None
    escaped = False
    i = 0
    while i < position:
        char = command_template[i]
        if quote == "'":
            if char == "'":
                quote = None
        elif quote == '"':
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = None
        elif char in ("'", '"'):
            quote = char
        elif char == "\\":
            i += 1
        i += 1
    return quote


def quote_command_placeholder(value: str, quote_context: Optional[str]) -> str:
    """Quote a placeholder value for its position in a shell command template."""
    if quote_context == "'":
        return value.replace("'", r"'\''")
    if quote_context == '"':
        return value.replace("\\", "\\\\").replace('"', r'\"').replace("$", r"\$").replace("`", r"\`")
    return subprocess.list2cmdline([value]) if os.name == "nt" else shlex.quote(value)


def render_command_template(command_template: str, placeholders: dict[str, str]) -> str:
    """Replace ``{name}`` placeholders (quote-aware) while preserving ``{{``/``}}``."""
    names = "|".join(re.escape(name) for name in placeholders)
    pattern = re.compile(rf"(?<!\$)(?:\{{\{{(?P<double>{names})\}}\}}|\{{(?P<single>{names})\}})")
    replacements: list[tuple[str, str]] = []

    def replace_match(match: re.Match[str]) -> str:
        name = match.group("double") or match.group("single")
        token = f"__HERMES_CMD_PLACEHOLDER_{len(replacements)}__"
        quoted = quote_command_placeholder(placeholders[name], shell_quote_context(command_template, match.start()))
        replacements.append((token, quoted))
        return token

    rendered = pattern.sub(replace_match, command_template).replace("{{", "{").replace("}}", "}")
    for token, value in replacements:
        rendered = rendered.replace(token, value)
    return rendered


def _signal_process_tree(psutil: Any, proc: subprocess.Popen, method: str) -> None:
    """Apply ``terminate``/``kill`` to *proc* and all descendants (best effort)."""
    try:
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            try:
                getattr(child, method)()
            except psutil.NoSuchProcess:
                pass
        getattr(parent, method)()
    except psutil.NoSuchProcess:
        return
    except Exception:
        getattr(proc, method)()


def terminate_command_process_tree(proc: subprocess.Popen) -> None:
    """Best-effort termination of a shell process and all of its children."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            # CREATE_NO_WINDOW: taskkill is a console-subprocess — the kill itself must not
            # flash a window on windowless hosts, same class of defect as the spawn above.
            from hermes_cli._subprocess_compat import windows_hide_flags
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=5, stdin=subprocess.DEVNULL,
                           creationflags=windows_hide_flags(), check=False)
        except Exception:
            proc.kill()
        return
    try:
        import psutil
    except ImportError:
        psutil = None
    # Without psutil only the shell itself is signalled (children may survive).
    signal = ((lambda m: getattr(proc, m)()) if psutil is None
              else (lambda m: _signal_process_tree(psutil, proc, m)))
    signal("terminate")
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        signal("kill")


def command_env_passthrough(config: dict[str, Any]) -> list:
    """``env_passthrough`` allowlist: parent env vars copied back into the secret-scrubbed child env."""
    raw = config.get("env_passthrough")
    return [str(x).strip() for x in raw if str(x).strip()] if isinstance(raw, (list, tuple)) else []


def command_failure_detail(exc: subprocess.CalledProcessError) -> str:
    """``stderr: ...; stdout: ...`` for a failed command provider, or ``no command output``."""
    parts = [f"{stream}: {text.strip()}" for stream, text in (("stderr", exc.stderr), ("stdout", exc.stdout)) if text]
    return "; ".join(parts) or "no command output"


def provider_popen_group_kwargs(os_name: str) -> dict[str, Any]:
    """Process-group spawn kwargs for a command-provider child, per platform.

    On Windows the shell=True spawn routes through cmd.exe, a console-subsystem
    shim: with only CREATE_NEW_PROCESS_GROUP it allocates a *visible* console
    window flash on every provider run from a windowless host (pythonw gateway,
    TUI, Desktop). The shared detach bundle (hermes_cli._subprocess_compat.
    windows_detach_flags_without_breakaway) adds CREATE_NO_WINDOW so the child
    owns a hidden console its descendants inherit instead — deliberately without
    CREATE_BREAKAWAY_FROM_JOB: this child is foreground-owned and killed by us on
    idle timeout, so job teardown propagation must keep working. POSIX keeps the
    session detach (setsid) so the tree can be signalled on idle timeout.
    """
    from hermes_cli._subprocess_compat import windows_detach_flags_without_breakaway
    if os_name == "nt":
        return {"creationflags": windows_detach_flags_without_breakaway()}
    return {"start_new_session": True}


class _CommandOutput:
    def __init__(self) -> None:
        self.queue: queue.Queue[tuple[str, str | CommandFailureEvent | None]] = queue.Queue()
        self.chunks: dict[str, list[str]] = {"stdout": [], "stderr": []}
        self.readers: dict[str, threading.Thread] = {}
        self.failures: list[CommandFailureEvent] = []

    def _read_stream(self, name: str, stream: Any) -> None:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        read1 = getattr(getattr(stream, "buffer", None), "read1", None)
        try:
            while True:
                chunk = (
                    stream.read(65536)
                    if read1 is None
                    else read1(65536).decode(encoding, errors="replace")
                )
                if not chunk:
                    break
                self.queue.put((name, chunk))
        except (OSError, ValueError) as exc:
            self.queue.put(
                (
                    name,
                    CommandFailureEvent(
                        exc, CommandOperation.READ, CommandStream(name)
                    ),
                )
            )
        finally:
            try:
                stream.close()
            except (OSError, ValueError) as exc:
                self.queue.put(
                    (
                        name,
                        CommandFailureEvent(
                            exc, CommandOperation.CLOSE, CommandStream(name)
                        ),
                    )
                )
            finally:
                self.queue.put((name, None))

    def _record(self, name: str, chunk: str | CommandFailureEvent | None) -> None:
        if isinstance(chunk, CommandFailureEvent):
            self.failures.append(chunk)
            return
        if chunk:
            self.chunks[name].append(chunk)

    def _raise_failure(self) -> None:
        if not self.failures:
            return
        first = self.failures[0]
        error = first.error
        previous = command_failure_facts(error)
        facts = CommandFailureFacts(
            first.operation,
            first.stream,
            previous.cleanup_events if previous is not None else (),
        ).with_cleanup(*self.failures[1:])
        set_command_failure_facts(error, facts)
        for failure in self.failures[1:]:
            note = f"Command-provider {failure.stream} {failure.operation} failed: {type(failure.error).__name__}"
            if failure.error is not error and note not in getattr(
                error, "__notes__", ()
            ):
                error.add_note(note)
        raise error

    def start(self, proc: subprocess.Popen) -> None:
        from agent.memory_provider import spawn_context_thread

        for name, stream in (("stdout", proc.stdout), ("stderr", proc.stderr)):
            reader = spawn_context_thread(self._read_stream, args=(name, stream), name=f"tts-command-{name}")
            self.readers[name] = reader
            reader.start()

    def receive(self, proc: subprocess.Popen, scope: Optional[CommandProcessScope], timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        open_streams = {"stdout", "stderr"}
        while open_streams:
            if scope is not None and scope.exited():
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            try:
                name, chunk = self.queue.get(timeout=min(0.05, remaining))
            except queue.Empty:
                continue
            self._record(name, chunk)
            self._raise_failure()
            if chunk is None:
                open_streams.discard(name)
                continue
            deadline = time.monotonic() + timeout
        if scope is None:
            try:
                proc.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                return True
            return False
        while not scope.exited():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            time.sleep(min(0.05, remaining))
        return False

    def finish(self, proc: subprocess.Popen, timeout: float) -> tuple[str, str]:
        deadline = time.monotonic() + timeout
        for name, stream in (("stdout", proc.stdout), ("stderr", proc.stderr)):
            reader = self.readers.get(name)
            if reader is not None and reader.ident is not None:
                reader.join(timeout=max(0.0, deadline - time.monotonic()))
            elif stream is not None:
                try:
                    stream.close()
                except (OSError, ValueError) as exc:
                    self.failures.append(
                        CommandFailureEvent(
                            exc, CommandOperation.CLOSE, CommandStream(name)
                        )
                    )
        stdout, stderr = self.result()
        self._raise_failure()
        if any(reader.is_alive() for reader in self.readers.values()):
            error = subprocess.TimeoutExpired(
                proc.args, timeout, output=stdout, stderr=stderr
            )
            set_command_failure_facts(
                error, CommandFailureFacts(CommandOperation.OUTPUT_DRAIN)
            )
            raise error
        return stdout, stderr

    def result(self) -> tuple[str, str]:
        while not self.queue.empty():
            name, chunk = self.queue.get_nowait()
            self._record(name, chunk)
        return "".join(self.chunks["stdout"]), "".join(self.chunks["stderr"])


def _command_failure(
    command: str,
    timeout: float,
    returncode: Optional[int],
    timed_out: bool,
    stdout: Optional[str] = None,
    stderr: Optional[str] = None,
) -> Optional[subprocess.TimeoutExpired | subprocess.CalledProcessError]:
    if timed_out:
        failure = subprocess.TimeoutExpired(
            command, timeout, output=stdout, stderr=stderr
        )
        failure.__cause__ = subprocess.TimeoutExpired(command, timeout)
        set_command_failure_facts(
            failure, CommandFailureFacts(CommandOperation.NATIVE_TIMEOUT)
        )
        return failure
    if returncode:
        failure = subprocess.CalledProcessError(
            returncode, command, output=stdout, stderr=stderr
        )
        set_command_failure_facts(
            failure, CommandFailureFacts(CommandOperation.NATIVE_EXIT)
        )
        return failure
    return None


def _record_command_cleanup(
    primary: BaseException, error: BaseException, operation: CommandOperation
) -> None:
    facts = command_failure_facts(primary) or CommandFailureFacts()
    observed = command_failure_facts(error)
    stream = None
    if operation == CommandOperation.OUTPUT_DRAIN and observed is not None:
        operation = observed.root_operation or operation
        stream = observed.root_stream
    event = CommandFailureEvent(error, operation, stream)
    set_command_failure_facts(primary, facts.with_cleanup(event))


def run_command_provider(
    command: str,
    timeout: float,
    env_passthrough: Optional[list] = None,
    *,
    keep_children: bool = False,
) -> subprocess.CompletedProcess:
    """Run a provider with an output-reset idle timeout and bounded cleanup.

    Finite POSIX commands clean their session group on every exit. After exit, output
    must drain within ``timeout`` before a result can be returned. Warm hooks may keep
    service children after normal completion; timeouts and interruptions still clean up.
    Child env is scrubbed of Hermes secrets and propagates delegated-child lineage markers.
    """
    from agent.delegation_context import delegated_child_subprocess_env
    from tools.env_passthrough import resolve_passthrough_value
    from tools.environments.local import hermes_subprocess_env

    scrubbed = hermes_subprocess_env(inherit_credentials=False)
    for key in env_passthrough or []:
        # Under the multiplexer os.environ is the LAUNCH profile's .env: resolve through the served
        # profile's secret scope so its own key is forwarded and never another profile's.
        value = resolve_passthrough_value(key, os.environ.get(key))
        if value is not None:
            scrubbed[key] = value
    # Own process group so the whole tree can be signalled on idle timeout; on Windows the
    # shared bundle also hides the console (shell=True spawns via cmd.exe, which otherwise
    # flashes a window on windowless hosts). Lossy UTF-8 decode: locale-mismatched bytes
    # must not raise in the reader threads.
    group = provider_popen_group_kwargs(os.name)
    output = _CommandOutput()
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=delegated_child_subprocess_env(scrubbed),
        stdin=subprocess.DEVNULL,
        **group,
    )
    scope = None
    completed = False
    timed_out = False
    failure = None
    try:
        if os.name == "posix" and proc.returncode is None:
            scope = CommandProcessScope(proc)
        output.start(proc)
        timed_out = output.receive(proc, scope, timeout)
        if keep_children and not timed_out:
            output.finish(proc, timeout)
        completed = not timed_out
    finally:
        primary_error = sys.exception()
        try:
            if scope is not None:
                scope.close(preserve_children=keep_children and completed)
            else:
                if not completed:
                    terminate_command_process_tree(proc)
                proc.wait(timeout=2)
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
            failure = _command_failure(command, timeout, proc.returncode, timed_out)
            if primary_error is None:
                primary_error = failure
            if primary_error is None:
                set_command_failure_facts(
                    exc, CommandFailureFacts(CommandOperation.PROCESS_CLEANUP)
                )
                raise
            _record_command_cleanup(
                primary_error, exc, CommandOperation.PROCESS_CLEANUP
            )
            primary_error.add_note(
                f"Command-provider process cleanup failed: {type(exc).__name__}"
            )
            logger.warning(
                "Command-provider process cleanup failed for PID %s: %s",
                proc.pid,
                type(exc).__name__,
            )
        finally:
            failure = failure or _command_failure(
                command, timeout, proc.returncode, timed_out
            )
            pending_error = sys.exception()
            if pending_error is None:
                pending_error = failure
            try:
                stdout, stderr = output.finish(proc, timeout)
            except (
                OSError,
                subprocess.SubprocessError,
                RuntimeError,
                ValueError,
            ) as exc:
                if pending_error is None:
                    if command_failure_facts(exc) is None:
                        set_command_failure_facts(
                            exc, CommandFailureFacts(CommandOperation.OUTPUT_DRAIN)
                        )
                    raise
                if exc is not pending_error:
                    _record_command_cleanup(
                        pending_error, exc, CommandOperation.OUTPUT_DRAIN
                    )
                    pending_error.add_note(
                        f"Command-provider output cleanup failed: {type(exc).__name__}"
                    )
                    logger.warning(
                        "Command-provider output cleanup failed for PID %s: %s",
                        proc.pid,
                        type(exc).__name__,
                    )
                stdout, stderr = output.result()
    if failure is not None:
        complete = _command_failure(
            command, timeout, proc.returncode, timed_out, stdout, stderr
        )
        assert complete is not None
        facts = command_failure_facts(failure)
        if facts is not None:
            set_command_failure_facts(complete, facts)
        for note in getattr(failure, "__notes__", ()):
            complete.add_note(note)
        raise complete
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)


# ---- Generic ``<section>.providers.<name>`` config layer (TTS and STT share it) ----
def _get_provider_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return ``config[name]`` if it's a dict, else an empty dict."""
    section = config.get(name) if isinstance(config, dict) else None
    return section if isinstance(section, dict) else {}


def _named_provider_config(config: dict[str, Any], name: str, builtins: frozenset[str]) -> dict[str, Any]:
    """``<section>.providers.<name>`` (canonical), else ``<section>.<name>`` for non-built-in names
    only — refused for built-ins so a user's ``openai:`` block still means OpenAI, not a command."""
    section = _get_provider_section(config, "providers").get(name)
    if isinstance(section, dict):
        return section
    return _get_provider_section(config, name) if name.lower() not in builtins else {}


def _is_command_provider_config(config: dict[str, Any]) -> bool:
    """True when *config* declares a command-type provider (has a non-empty ``command``)."""
    if not isinstance(config, dict):
        return False
    ptype = str(config.get("type") or "").strip().lower()
    command = config.get("command")
    return ptype in ("", "command") and isinstance(command, str) and bool(command.strip())


def _resolve_command_config(
    provider: str, config: dict[str, Any], reserved: frozenset[str]) -> Optional[dict[str, Any]]:
    """Config of a user-declared command provider; None for *reserved* names, unknown or non-command."""
    key = (provider or "").lower().strip()
    if not key or key in reserved:
        return None
    named = _named_provider_config(config, key, reserved)
    return named if _is_command_provider_config(named) else None


def _command_timeout(config: dict[str, Any], default: float) -> float:
    """Timeout in seconds (``timeout`` > ``timeout_seconds``); invalid or non-positive -> *default*."""
    raw = config.get("timeout", config.get("timeout_seconds", default))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    return value if value > 0 else float(default)


def _command_output_format(config: dict[str, Any], formats: frozenset[str], default: str) -> str:
    """Validated ``format``/``output_format`` from *config*, else *default*."""
    raw = config.get("format") or config.get("output_format") or default
    fmt = str(raw).lower().strip().lstrip(".")
    return fmt if fmt in formats else default


# ---- TTS ``tts.providers.<name>`` layer -----------------------------------

# Any ``tts.provider`` value NOT in this set refers to ``tts.providers.<name>``.
BUILTIN_TTS_PROVIDERS = frozenset({
    "edge", "elevenlabs", "openai", "minimax", "xai", "mistral", "gemini",
    "neutts", "kittentts", "piper", "deepinfra"})

DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS = 120
DEFAULT_COMMAND_TTS_OUTPUT_FORMAT = "mp3"
COMMAND_TTS_OUTPUT_FORMATS = frozenset({"mp3", "wav", "ogg", "flac", "m4a", "aac", "amr", "opus"})
DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH = 5000


_get_named_provider_config = partial(_named_provider_config, builtins=BUILTIN_TTS_PROVIDERS)
_resolve_command_provider_config = partial(_resolve_command_config, reserved=BUILTIN_TTS_PROVIDERS)
_get_command_tts_timeout = partial(_command_timeout, default=DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS)


def _iter_command_providers(tts_config: dict[str, Any]):
    """Yield (name, config) pairs for every declared command-type provider."""
    for name, cfg in _get_provider_section(tts_config, "providers").items():
        if isinstance(name, str) and name.lower() not in BUILTIN_TTS_PROVIDERS and _is_command_provider_config(cfg):
            yield name, cfg


def _get_command_tts_output_format(config: dict[str, Any], output_path: Optional[str] = None) -> str:
    """Validated output format: the output path's suffix wins, then ``format``/``output_format``."""
    suffix = Path(output_path).suffix.lower().strip().lstrip(".") if output_path else ""
    if suffix in COMMAND_TTS_OUTPUT_FORMATS:
        return suffix
    return _command_output_format(config, COMMAND_TTS_OUTPUT_FORMATS, DEFAULT_COMMAND_TTS_OUTPUT_FORMAT)


def _is_command_tts_voice_compatible(config: dict[str, Any]) -> bool:
    """True only when the user explicitly opted in to voice delivery."""
    return is_truthy_value(config.get("voice_compatible", False))


def _configured_command_tts_output_path(path: Path, config: dict[str, Any]) -> Path:
    """Return an output path whose extension matches the provider's output_format."""
    return path.with_suffix(f".{_get_command_tts_output_format(config)}")


def _generate_command_tts(
    text: str, output_path: str, provider_name: str, config: dict[str, Any], tts_config: dict[str, Any],
) -> str:
    """Generate speech by running a user-configured shell command; returns the audio path it wrote.
    Raises ``ValueError`` for bad provider config, ``RuntimeError`` for timeouts / bad exits / no output."""
    command_template = str(config.get("command") or "").strip()
    if not command_template:
        raise ValueError(f"tts.providers.{provider_name}.command is not configured")
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    timeout = _get_command_tts_timeout(config)
    with tempfile.TemporaryDirectory() as tmpdir:
        text_path = Path(tmpdir) / "input.txt"
        text_path.write_text(text, encoding="utf-8")
        placeholders = {
            "input_path": str(text_path), "text_path": str(text_path), "output_path": str(output),
            "format": _get_command_tts_output_format(config, str(output)),
            "voice": str(config.get("voice", "")), "model": str(config.get("model", "")),
            "speed": str(config.get("speed", tts_config.get("speed", ""))),
        }
        command = render_command_template(command_template, placeholders)
        try:
            run_command_provider(command, timeout, env_passthrough=command_env_passthrough(config))
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"TTS provider '{provider_name}' timed out after {timeout:g}s") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"TTS provider '{provider_name}' exited with code {exc.returncode}: {command_failure_detail(exc)}"
            ) from exc
    if not output.exists() or output.stat().st_size <= 0:
        raise RuntimeError(f"TTS provider '{provider_name}' produced no output at {output}")
    return str(output)
