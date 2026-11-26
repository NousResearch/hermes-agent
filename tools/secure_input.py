#!/usr/bin/env python3
"""
Secure Input Tool — masked credential entry that never exposes secrets to the agent.

When the agent calls this tool with an env-var name, the user is prompted with
``****`` masked typing (or ``getpass`` fallback) and the value is written directly
to ``~/.hermes/.env``.  The agent NEVER sees the secret — the tool returns only
a success/error envelope.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional, Callable

from tools.registry import registry, tool_error

# Minimum masked-secret-prompt length to reject as likely-accidental.
_MIN_SECRET_LENGTH = 1


def _append_to_env_file(env_path: Path, key: str, value: str) -> None:
    """Upsert ``KEY=value`` into *env_path*, preserving existing order.

    Writes atomically via a temp file + rename so a crash during write
    cannot truncate the file.  Permissions are forced to ``0o600``.
    """
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.touch(mode=0o600, exist_ok=True)
    env_path.chmod(0o600)

    # atomically replace
    tmp = env_path.with_suffix(env_path.suffix + ".tmp" + _random_suffix())
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        lines = []

    new_line = f"{key}={value}\n"
    replaced = False
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            if "=" in stripped:
                existing_key = stripped.split("=", 1)[0]
                # Handle 'export KEY=value' syntax (common in .bashrc-style .env files)
                if existing_key == key or existing_key.removeprefix("export ") == key:
                    out.append(new_line)
                    replaced = True
                    continue
        out.append(line)
    if not replaced:
        # Guard against .env files missing trailing newline: if the last
        # line does not end in \n, prepend one so we do not concatenate
        # "OLD_VAR=123NEW_VAR=456\n".
        if out and not out[-1].endswith("\n"):
            out.append("\n")
        out.append(new_line)

    # Write atomically with strict 0600 FROM CREATION (no umask window).
    # os.open + mode=0o600 guarantees the temp file is never world-readable,
    # even for a microsecond, unlike write_text() + chmod() which races.
    data = "".join(out)
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data.encode("utf-8"))
    finally:
        os.close(fd)
    os.replace(tmp, env_path)


def _random_suffix() -> str:
    """Return a short random hex suffix for a temp filename."""
    import secrets
    return secrets.token_hex(4)


def _secure_prompt(prompt_text: str) -> str:
    """Read a secret from the user with ``****`` masked echo.

    Opens ``/dev/tty`` directly to bypass prompt_toolkit's stdin/stdout
    redirection.  Falls back to ``getpass.getpass`` if /dev/tty is unavailable.
    """
    # Try /dev/tty first — this bypasses prompt_toolkit's terminal control
    try:
        return _tty_masked_input(prompt_text)
    except Exception:
        pass

    # Fallback: masked_secret_prompt (uses sys.stdin/sys.stdout which may
    # be redirected by prompt_toolkit but works when those are real TTYs)
    try:
        from hermes_cli.secret_prompt import masked_secret_prompt
        return masked_secret_prompt(prompt_text, mask="*")
    except Exception:
        pass

    # Last resort: getpass (reads from /dev/tty internally if available)
    import getpass
    try:
        return getpass.getpass(prompt_text + ": " if not prompt_text.endswith(": ") else prompt_text)
    except Exception:
        # Absolute last resort — but at least the value won't appear in the tool return
        sys.stderr.write(prompt_text)
        sys.stderr.flush()
        return sys.stdin.readline().rstrip("\n")


def _tty_masked_input(prompt_text: str) -> str:
    """Read a line from /dev/tty with echo disabled (no asterisks, just hidden).

    This is the most reliable way to read a secret when prompt_toolkit has
    taken over sys.stdin/sys.stdout — it opens the controlling terminal
    directly and disables echo via termios.
    """
    try:
        import termios
        import tty as _tty_mod
    except ImportError:
        raise OSError("termios not available on this platform")

    tty_path = "/dev/tty"
    if not os.path.exists(tty_path):
        raise OSError("/dev/tty not available")

    fp = open(tty_path, "r+b", buffering=0)
    try:
        fd = fp.fileno()
        old = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)
        # Disable echo but keep canonical mode so we can read a line with backspace
        new[3] = new[3] & ~termios.ECHO
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, new)
            fp.write(prompt_text.encode())
            fp.write(b": " if not prompt_text.endswith(": ") else b"")
            fp.flush()
            line = fp.readline()
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            return line.rstrip("\r\n")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    finally:
        fp.close()


def secure_input_tool(
    key: str,
    prompt: Optional[str] = None,
    confirm: bool = True,
    callback: Optional[Callable] = None,
) -> str:
    """Prompt the user for a credential and store it in ``~/.hermes/.env``.

    The value is read via masked input (``****``) and written directly to disk.
    The agent receives only a success/error envelope — the secret itself never
    enters the agent's context, session database, or logs.

    Args:
        key:      Environment variable name to store (e.g. ``"OPENAI_API_KEY"``).
        prompt:   Custom prompt shown to the user.  Defaults to
                  ``"Enter value for <key>: "``.
        confirm:  When True (default), prompt a second time and require the two
                  entries to match.
        callback: Platform-provided secure-input callback — injected by the
                  agent runner (cli.py / gateway).  Signature:
                  ``callback(key, prompt_text) -> str``.
                  When omitted, ``masked_secret_prompt`` is used directly.

    Returns:
        JSON: ``{"stored": true, "key": "...", "file": "..."}`` on success,
              ``{"stored": false, "error": "..."}`` on failure.
    """
    # --- Validate arguments ---------------------------------------------------
    if not key or not key.strip():
        return tool_error("Missing required parameter: key (env var name).")

    key = key.strip()
    # Basic env-var name validation
    if not _is_valid_env_name(key):
        return tool_error(
            f"Invalid env var name: {key!r}. "
            "Must start with a letter/underscore and contain only "
            "A-Z, 0-9, and underscores."
        )

    prompt_text = (prompt or "").strip() or f"Enter value for {key}"

    # --- Read secret from user ------------------------------------------------
    try:
        if callback is not None:
            value = callback(key, prompt_text)
        else:
            # Add a trailing space to the prompt for readability
            value = _secure_prompt(f"{prompt_text}: ")

        if not value or not value.strip():
            return tool_error("No value entered — credential not stored.")

        value = value.strip()
        if len(value) < _MIN_SECRET_LENGTH:
            return tool_error("Value too short — credential not stored.")

        # Confirmation
        if confirm:
            confirm_text = f"Confirm {key}"
            if callback is not None:
                value2 = callback(key, confirm_text)
            else:
                value2 = _secure_prompt(f"{confirm_text}: ")

            if value != value2:
                return tool_error("Values do not match — credential not stored.")

    except (KeyboardInterrupt, EOFError):
        return tool_error("Input interrupted — credential not stored.")
    except Exception as exc:
        return tool_error(f"Failed to read input: {exc}")

    # --- Write to .env --------------------------------------------------------
    try:
        from hermes_constants import get_hermes_home
        env_path = get_hermes_home() / ".env"
        _append_to_env_file(env_path, key, value)
    except Exception as exc:
        return tool_error(f"Failed to write .env: {exc}")
    finally:
        # Python strings are immutable — true memory scrubbing requires
        # ctypes to overwrite the underlying buffer. The best defence is
        # that the value never entered the agent context, session DB, or
        # logs in the first place.
        try:
            del value
            if confirm:
                del value2
        except Exception:
            pass

    return json.dumps({
        "stored": True,
        "key": key,
        "file": str(env_path),
    }, ensure_ascii=False)


def _is_valid_env_name(name: str) -> bool:
    """Return True when *name* looks like a valid environment variable name."""
    if not name:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(c.isalnum() or c == "_" for c in name)


def check_secure_input_requirements() -> bool:
    """Secure input has no external API requirements — always available."""
    return True


# =============================================================================
# Schema
# =============================================================================

SECURE_INPUT_SCHEMA = {
    "name": "secure_input",
    "description": (
        "Prompt the user for a credential or API key with masked typing, "
        "then store it securely in the Hermes .env file. "
        "The agent NEVER sees the secret — only a success/error "
        "envelope is returned. Use when you need the user to provide "
        "a password, token, or API key that must not appear in chat "
        "history, logs, or the session database. "
        "Do NOT use for non-secret config values the agent can safely see."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": (
                    "Environment variable name to store the credential under "
                    "(e.g. 'OPENAI_API_KEY', 'DATABASE_PASSWORD'). Must be a "
                    "valid env-var name: letters, digits, underscores, starting "
                    "with a letter or underscore."
                ),
            },
            "prompt": {
                "type": "string",
                "description": (
                    "Optional custom prompt shown to the user before they type. "
                    "Default: 'Enter value for <key>'. Keep it short — the user "
                    "sees this on the masked-input line."
                ),
            },
            "confirm": {
                "type": "boolean",
                "description": (
                    "When true (default), prompt a second time to confirm the "
                    "value. Both entries must match. Set false for long tokens "
                    "the user is pasting."
                ),
                "default": True,
            },
        },
        "required": ["key"],
    },
}


# --- Registry ----------------------------------------------------------------

registry.register(
    name="secure_input",
    toolset="secure_input",
    schema=SECURE_INPUT_SCHEMA,
    handler=lambda args, **kw: secure_input_tool(
        key=args.get("key", ""),
        prompt=args.get("prompt"),
        confirm=args.get("confirm", True),
        callback=kw.get("callback"),
    ),
    check_fn=check_secure_input_requirements,
    emoji="🔐",
)
