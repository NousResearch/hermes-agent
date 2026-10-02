"""Shared runner for user-configured ("command") TTS/STT providers.

``tts.providers.<name>: {type: command, command: "voxcpm --text-file {input_path} --out {output_path}"}``
(and the ``stt.`` twin): templates are tokenized before ``{placeholders}`` are substituted,
so every value remains one argv element; ``{{``/``}}`` stay literal. Owns argv construction,
the idle-timeout process runner and the generic ``<section>.providers.<name>`` readers, re-imported by
``tts_tool``/``transcription_tools`` under their historical private names. TTS placeholders:
``{input_path}``/``{text_path}``, ``{output_path}``, ``{format}``, ``{voice}``, ``{model}``,
``{speed}``. Built-in provider names always win over a same-named ``providers`` entry.

Commands are tokenized into an argv list and executed WITHOUT a shell (Aikido #414):
shell operators (``|``, ``>``, ``<``, ``&&``, ``;``) and ``$VAR``/``%VAR%`` reach the
program as literal arguments, so stdin/stdout redirects and pipelines are not
available — engines that read stdin need a small wrapper script.
"""

from __future__ import annotations

import os
import queue
import re
import shlex
import subprocess
import tempfile
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Sequence

from utils import is_truthy_value


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


def render_command_template(command_template: str, placeholders: Dict[str, str]) -> str:
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


def _split_windows_commandline(command: str) -> list[str]:
    """Split a Microsoft C-runtime command line without invoking ``cmd.exe``."""
    argv: list[str] = []
    length = len(command)
    index = 0
    while index < length:
        while index < length and command[index] in " \t":
            index += 1
        if index >= length:
            break

        argument: list[str] = []
        in_quotes = False
        while index < length:
            if command[index] in " \t" and not in_quotes:
                break

            backslashes = 0
            while index < length and command[index] == "\\":
                backslashes += 1
                index += 1
            if index < length and command[index] == '"':
                argument.extend("\\" * (backslashes // 2))
                if backslashes % 2:
                    argument.append('"')
                    index += 1
                elif in_quotes and index + 1 < length and command[index + 1] == '"':
                    argument.append('"')
                    index += 2
                else:
                    in_quotes = not in_quotes
                    index += 1
                continue

            argument.extend("\\" * backslashes)
            if index >= length or (command[index] in " \t" and not in_quotes):
                break
            argument.append(command[index])
            index += 1

        argv.append("".join(argument))
        while index < length and command[index] in " \t":
            index += 1
    return argv


def build_command_argv(
    command_template: str, placeholders: Dict[str, str], *, windows: Optional[bool] = None,
) -> list[str]:
    """Parse a template first, then substitute each placeholder inside its argv element.

    Parsing before substitution keeps spaces, quotes, backslashes, empty values, and
    shell metacharacters in placeholder values as data instead of command syntax.
    ``windows`` is explicit for pure construction tests; production uses the host OS.
    """
    replacements: list[tuple[str, str]] = []
    protected = command_template
    if placeholders:
        names = "|".join(re.escape(name) for name in placeholders)
        pattern = re.compile(rf"(?<!\$)(?:\{{\{{(?P<double>{names})\}}\}}|\{{(?P<single>{names})\}})")

        def protect_match(match: re.Match[str]) -> str:
            name = match.group("double") or match.group("single")
            token = f"\x00HERMES_CMD_PLACEHOLDER_{len(replacements)}\x00"
            replacements.append((token, str(placeholders[name])))
            return token

        protected = pattern.sub(protect_match, protected)
    protected = protected.replace("{{", "{").replace("}}", "}")
    use_windows = os.name == "nt" if windows is None else windows
    argv = _split_windows_commandline(protected) if use_windows else shlex.split(protected, posix=True)
    for token, value in replacements:
        argv = [argument.replace(token, value) for argument in argv]
    return argv


def _resolved_command_argv(argv: Sequence[str], env: Dict[str, str]) -> list[str]:
    """Resolve argv[0] against the child's PATH and reject Windows batch shells."""
    if not argv or not argv[0]:
        raise OSError("Command provider command must name an executable")
    from hermes_platform.resolver import LookupContext, locate_command

    prepared = list(argv)
    executable = prepared[0]
    if os.sep not in executable and not (os.altsep and os.altsep in executable):
        resolution = locate_command(
            executable,
            LookupContext(path=env.get("PATH", ""), pathext=env.get("PATHEXT")),
        )
        if resolution.command:
            prepared[0] = resolution.command[0]
    if os.name == "nt" and Path(prepared[0]).suffix.lower() in {".bat", ".cmd"}:
        raise OSError(
            "Windows command providers cannot execute .cmd or .bat files without re-entering cmd.exe; "
            "configure an .exe or direct interpreter command instead"
        )
    return prepared


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
    """Best-effort termination of a provider process and all of its children."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=5, stdin=subprocess.DEVNULL)
        except Exception:
            proc.kill()
        return
    try:
        import psutil  # type: ignore
    except ImportError:
        psutil = None
    # Without psutil only the provider itself is signalled (children may survive).
    signal = ((lambda m: getattr(proc, m)()) if psutil is None
              else (lambda m: _signal_process_tree(psutil, proc, m)))
    signal("terminate")
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        signal("kill")


def command_env_passthrough(config: Dict[str, Any]) -> list:
    """``env_passthrough`` allowlist: parent env vars copied back into the secret-scrubbed child env."""
    raw = config.get("env_passthrough")
    return [str(x).strip() for x in raw if str(x).strip()] if isinstance(raw, (list, tuple)) else []


def command_failure_detail(exc: subprocess.CalledProcessError) -> str:
    """``stderr: ...; stdout: ...`` for a failed command provider, or ``no command output``."""
    parts = [f"{stream}: {text.strip()}" for stream, text in (("stderr", exc.stderr), ("stdout", exc.stdout)) if text]
    return "; ".join(parts) or "no command output"


def run_command_provider(
    command: str | Sequence[str], timeout: float, env_passthrough: Optional[list] = None,
) -> subprocess.CompletedProcess:
    """Run a command-provider command with process-tree idle cleanup.
    ``timeout`` is an IDLE timeout, reset whenever the command emits output — a slow-but-alive
    provider survives, a silently stalled one is killed. Child env is scrubbed of Hermes secrets
    while propagating delegated-child lineage markers.

    The command string is tokenized into an argv list and executed WITHOUT a shell
    (Aikido #414, same policy as the ``HERMES_LOCAL_STT_COMMAND`` path): shell
    operators and variable expansion are passed through as literal arguments.
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
    argv = build_command_argv(command, {}) if isinstance(command, str) else list(command)
    argv = _resolved_command_argv(argv, scrubbed)
    # Own process group so the whole tree can be signalled on idle timeout. Lossy UTF-8 decode:
    # locale-mismatched bytes must not raise in the reader threads.
    group = ({"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)} if os.name == "nt"
             else {"start_new_session": True})
    proc = subprocess.Popen(argv, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", env=delegated_child_subprocess_env(scrubbed),
                            stdin=subprocess.DEVNULL, **group)
    output_queue: "queue.Queue[tuple[str, Optional[str]]]" = queue.Queue()
    chunks: Dict[str, list[str]] = {"stdout": [], "stderr": []}
    open_streams = {"stdout", "stderr"}

    def read_stream(name: str, stream: Any) -> None:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        read1 = getattr(getattr(stream, "buffer", None), "read1", None)
        try:
            while True:
                chunk = stream.read(65536) if read1 is None else read1(65536).decode(encoding, errors="replace")
                if not chunk:
                    break
                output_queue.put((name, chunk))
        finally:
            output_queue.put((name, None))

    readers = [threading.Thread(target=read_stream, args=(name, stream), daemon=True)
               for name, stream in (("stdout", proc.stdout), ("stderr", proc.stderr))]
    for reader in readers:
        reader.start()
    deadline = time.monotonic() + timeout
    timed_out = False
    while open_streams:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        try:
            name, chunk = output_queue.get(timeout=min(0.05, remaining))
        except queue.Empty:
            continue
        if chunk is None:
            open_streams.discard(name)
            continue
        chunks[name].append(chunk)
        deadline = time.monotonic() + timeout
    if not timed_out:
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
    if timed_out:
        terminate_command_process_tree(proc)
        for reader in readers:
            reader.join(timeout=0.5)
        while not output_queue.empty():
            name, chunk = output_queue.get_nowait()
            if chunk:
                chunks[name].append(chunk)
    stdout, stderr = "".join(chunks["stdout"]), "".join(chunks["stderr"])
    if timed_out:
        raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr) from (
            subprocess.TimeoutExpired(command, timeout))
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, command, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)


# ---- Generic ``<section>.providers.<name>`` config layer (TTS and STT share it) ----
def _get_provider_section(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Return ``config[name]`` if it's a dict, else an empty dict."""
    section = config.get(name) if isinstance(config, dict) else None
    return section if isinstance(section, dict) else {}


def _named_provider_config(config: Dict[str, Any], name: str, builtins: FrozenSet[str]) -> Dict[str, Any]:
    """``<section>.providers.<name>`` (canonical), else ``<section>.<name>`` for non-built-in names
    only — refused for built-ins so a user's ``openai:`` block still means OpenAI, not a command."""
    section = _get_provider_section(config, "providers").get(name)
    if isinstance(section, dict):
        return section
    return _get_provider_section(config, name) if name.lower() not in builtins else {}


def _is_command_provider_config(config: Dict[str, Any]) -> bool:
    """True when *config* declares a command-type provider (has a non-empty ``command``)."""
    if not isinstance(config, dict):
        return False
    ptype = str(config.get("type") or "").strip().lower()
    command = config.get("command")
    return ptype in ("", "command") and isinstance(command, str) and bool(command.strip())


def _resolve_command_config(
    provider: str, config: Dict[str, Any], reserved: FrozenSet[str]) -> Optional[Dict[str, Any]]:
    """Config of a user-declared command provider; None for *reserved* names, unknown or non-command."""
    key = (provider or "").lower().strip()
    if not key or key in reserved:
        return None
    named = _named_provider_config(config, key, reserved)
    return named if _is_command_provider_config(named) else None


def _command_timeout(config: Dict[str, Any], default: float) -> float:
    """Timeout in seconds (``timeout`` > ``timeout_seconds``); invalid or non-positive -> *default*."""
    raw = config.get("timeout", config.get("timeout_seconds", default))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    return value if value > 0 else float(default)


def _command_output_format(config: Dict[str, Any], formats: FrozenSet[str], default: str) -> str:
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


def _iter_command_providers(tts_config: Dict[str, Any]):
    """Yield (name, config) pairs for every declared command-type provider."""
    for name, cfg in _get_provider_section(tts_config, "providers").items():
        if isinstance(name, str) and name.lower() not in BUILTIN_TTS_PROVIDERS and _is_command_provider_config(cfg):
            yield name, cfg


def _get_command_tts_output_format(config: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """Validated output format: the output path's suffix wins, then ``format``/``output_format``."""
    suffix = Path(output_path).suffix.lower().strip().lstrip(".") if output_path else ""
    if suffix in COMMAND_TTS_OUTPUT_FORMATS:
        return suffix
    return _command_output_format(config, COMMAND_TTS_OUTPUT_FORMATS, DEFAULT_COMMAND_TTS_OUTPUT_FORMAT)


def _is_command_tts_voice_compatible(config: Dict[str, Any]) -> bool:
    """True only when the user explicitly opted in to voice delivery."""
    return is_truthy_value(config.get("voice_compatible", False))


def _configured_command_tts_output_path(path: Path, config: Dict[str, Any]) -> Path:
    """Return an output path whose extension matches the provider's output_format."""
    return path.with_suffix(f".{_get_command_tts_output_format(config)}")


def _generate_command_tts(
    text: str, output_path: str, provider_name: str, config: Dict[str, Any], tts_config: Dict[str, Any],
) -> str:
    """Generate speech by running a user-configured command; returns the audio path it wrote.
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
        command = build_command_argv(command_template, placeholders)
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
