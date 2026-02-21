#!/usr/bin/env python3
"""
Terminal Tool Module (mini-swe-agent backend)

A terminal tool that executes commands using mini-swe-agent's execution environments.
Supports local execution, Docker containers, and Modal cloud sandboxes.

Environment Selection (via TERMINAL_ENV environment variable):
- "local": Execute directly on the host machine (default, fastest)
- "docker": Execute in Docker containers (isolated, requires Docker)
- "modal": Execute in Modal cloud sandboxes (scalable, requires Modal account)

Features:
- Multiple execution backends (local, docker, modal)
- Background task support
- VM/container lifecycle management
- Automatic cleanup after inactivity

Usage:
    from terminal_tool import terminal_tool

    # Execute a simple command
    result = terminal_tool("ls -la")

    # Execute in background
    result = terminal_tool("python server.py", background=True)
"""

import json
import logging
import os
import signal
import sys
import time
import threading
import atexit
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global interrupt event: set by the agent when a user interrupt arrives.
# The terminal tool polls this during command execution so it can kill
# long-running subprocesses immediately instead of blocking until timeout.
# ---------------------------------------------------------------------------
_interrupt_event = threading.Event()


def set_interrupt_event(active: bool) -> None:
    """Called by the agent to signal or clear the interrupt."""
    if active:
        _interrupt_event.set()
    else:
        _interrupt_event.clear()


def is_interrupted() -> bool:
    """Check if an interrupt has been requested."""
    return _interrupt_event.is_set()


# Add mini-swe-agent to path if not installed
mini_swe_path = Path(__file__).parent.parent / "mini-swe-agent" / "src"
if mini_swe_path.exists():
    sys.path.insert(0, str(mini_swe_path))


# =============================================================================
# Custom Singularity Environment with more space
# =============================================================================

def _get_scratch_dir() -> Path:
    """Get the best directory for Singularity sandboxes - prefers /scratch if available."""
    # Check for configurable scratch directory first (highest priority)
    custom_scratch = os.getenv("TERMINAL_SCRATCH_DIR")
    if custom_scratch:
        scratch_path = Path(custom_scratch)
        scratch_path.mkdir(parents=True, exist_ok=True)
        return scratch_path
    
    # Check for /scratch (common on HPC clusters, especially GPU nodes)
    scratch = Path("/scratch")
    if scratch.exists() and os.access(scratch, os.W_OK):
        # Create user-specific subdirectory
        user_scratch = scratch / os.getenv("USER", "hermes") / "hermes-agent"
        user_scratch.mkdir(parents=True, exist_ok=True)
        logger.info("Using /scratch for sandboxes: %s", user_scratch)
        return user_scratch
    
    # Fall back to /tmp (only relevant for Singularity/HPC sandboxes)
    logger.debug("/scratch not available, using /tmp for sandboxes")
    return Path(tempfile.gettempdir())


def _get_apptainer_cache_dir() -> Path:
    """Get the Apptainer cache directory for SIF images."""
    # Check for APPTAINER_CACHEDIR env var
    cache_dir = os.getenv("APPTAINER_CACHEDIR")
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path
    
    # Use user-specific subdirectory in scratch for cache
    scratch = _get_scratch_dir()
    cache_path = scratch / ".apptainer"
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


# Lock for SIF building to prevent race conditions
_sif_build_lock = threading.Lock()


def _get_or_build_sif(image: str, executable: str = "apptainer") -> str:
    """
    Get or build a SIF image from a docker:// URL.
    
    If the image is already a .sif file, returns it as-is.
    If the image is a docker:// URL, checks for cached SIF and builds if needed.
    
    Args:
        image: Image path (docker://... URL or .sif path)
        executable: apptainer or singularity
        
    Returns:
        Path to SIF file, or original image if not a docker:// URL
    """
    # If already a .sif file, use it directly
    if image.endswith('.sif') and Path(image).exists():
        return image
    
    # If not a docker:// URL, return as-is (could be a local sandbox or other format)
    if not image.startswith('docker://'):
        return image
    
    # Generate SIF filename from docker image name
    # docker://nikolaik/python-nodejs:python3.11-nodejs20 -> python-nodejs-python3.11-nodejs20.sif
    image_name = image.replace('docker://', '').replace('/', '-').replace(':', '-')
    cache_dir = _get_apptainer_cache_dir()
    sif_path = cache_dir / f"{image_name}.sif"
    
    # Check if SIF already exists
    if sif_path.exists():
        return str(sif_path)
    
    # Build SIF with lock to prevent multiple workers building simultaneously
    with _sif_build_lock:
        # Double-check after acquiring lock (another thread may have built it)
        if sif_path.exists():
            return str(sif_path)
        
        logger.info("Building SIF image (one-time setup)...")
        logger.info("  Source: %s", image)
        logger.info("  Target: %s", sif_path)
        
        # Ensure tmp directory exists for build
        tmp_dir = cache_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        
        # Set APPTAINER_TMPDIR for the build
        env = os.environ.copy()
        env["APPTAINER_TMPDIR"] = str(tmp_dir)
        env["APPTAINER_CACHEDIR"] = str(cache_dir)
        
        try:
            result = subprocess.run(
                [executable, "build", str(sif_path), image],
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout for pulling and building
                env=env
            )
            if result.returncode != 0:
                logger.warning("SIF build failed, falling back to docker:// URL")
                logger.warning("  Error: %s", result.stderr[:500])
                return image
            
            logger.info("SIF image built successfully")
            return str(sif_path)
            
        except subprocess.TimeoutExpired:
            logger.warning("SIF build timed out, falling back to docker:// URL")
            # Clean up partial file
            if sif_path.exists():
                sif_path.unlink()
            return image
        except Exception as e:
            logger.warning("SIF build error: %s, falling back to docker:// URL", e)
            return image


# Disk usage warning threshold (in GB)
DISK_USAGE_WARNING_THRESHOLD_GB = float(os.getenv("TERMINAL_DISK_WARNING_GB", "500"))


def _check_disk_usage_warning():
    """Check if total disk usage exceeds warning threshold."""
    scratch_dir = _get_scratch_dir()
    
    try:
        # Get total size of hermes directories
        total_bytes = 0
        import glob
        for path in glob.glob(str(scratch_dir / "hermes-*")):
            for f in Path(path).rglob('*'):
                if f.is_file():
                    try:
                        total_bytes += f.stat().st_size
                    except OSError:
                        pass
        
        total_gb = total_bytes / (1024 ** 3)
        
        if total_gb > DISK_USAGE_WARNING_THRESHOLD_GB:
            logger.warning("Disk usage (%.1fGB) exceeds threshold (%.0fGB). Consider running cleanup_all_environments().",
                           total_gb, DISK_USAGE_WARNING_THRESHOLD_GB)
            return True
        
        return False
    except Exception as e:
        return False


# Session-cached sudo password (persists until CLI exits)
_cached_sudo_password: str = ""

# Optional UI callbacks for interactive prompts. When set, these are called
# instead of the default /dev/tty or input() readers. The CLI registers these
# so prompts route through prompt_toolkit's event loop.
#   _sudo_password_callback() -> str  (return password or "" to skip)
#   _approval_callback(command, description) -> str  ("once"/"session"/"always"/"deny")
_sudo_password_callback = None
_approval_callback = None


def set_sudo_password_callback(cb):
    """Register a callback for sudo password prompts (used by CLI)."""
    global _sudo_password_callback
    _sudo_password_callback = cb


def set_approval_callback(cb):
    """Register a callback for dangerous command approval prompts (used by CLI)."""
    global _approval_callback
    _approval_callback = cb

# =============================================================================
# Dangerous Command Approval System
# =============================================================================

# Session-cached dangerous command approvals (pattern -> approved)
_session_approved_patterns: set = set()

# Last approval-required command (for gateway to pick up)
# Set by _check_dangerous_command when in ask mode, read by gateway
_last_pending_approval: dict = {}

# Dangerous command patterns (regex, description)
DANGEROUS_PATTERNS = [
    (r'\brm\s+(-[^\s]*\s+)*/', "delete in root path"),
    (r'\brm\s+(-[^\s]*)?r', "recursive delete"),
    (r'\bchmod\s+(-[^\s]*\s+)*777\b', "world-writable permissions"),
    (r'\bchown\s+(-[^\s]*)?R\s+root', "recursive chown to root"),
    (r'\bmkfs\b', "format filesystem"),
    (r'\bdd\s+.*if=', "disk copy"),
    (r'>\s*/dev/sd', "write to block device"),
    (r'\bDROP\s+(TABLE|DATABASE)\b', "SQL DROP"),
    (r'\bDELETE\s+FROM\b(?!.*\bWHERE\b)', "SQL DELETE without WHERE"),
    (r'\bTRUNCATE\s+(TABLE)?\s*\w', "SQL TRUNCATE"),
    (r'>\s*/etc/', "overwrite system config"),
    (r'\bsystemctl\s+(stop|disable|mask)\b', "stop/disable system service"),
    (r'\bkill\s+-9\s+-1\b', "kill all processes"),
    (r'\bpkill\s+-9\b', "force kill processes"),
    (r':()\s*{\s*:\s*\|\s*:&\s*}\s*;:', "fork bomb"),
]


def _load_permanent_allowlist() -> set:
    """Load permanently allowed command patterns from config."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
        patterns = config.get("command_allowlist", [])
        return set(patterns) if patterns else set()
    except Exception:
        return set()


def _save_permanent_allowlist(patterns: set):
    """Save permanently allowed command patterns to config."""
    try:
        from hermes_cli.config import load_config, save_config
        config = load_config()
        config["command_allowlist"] = list(patterns)
        save_config(config)
    except Exception as e:
        logger.warning("Could not save allowlist: %s", e)


def _detect_dangerous_command(command: str) -> tuple:
    """
    Check if command matches any dangerous patterns.
    
    Returns:
        (is_dangerous, pattern_key, description) or (False, None, None)
    """
    import re
    command_lower = command.lower()
    
    for pattern, description in DANGEROUS_PATTERNS:
        if re.search(pattern, command_lower, re.IGNORECASE):
            # Use a simplified pattern key for caching (first word + key chars)
            pattern_key = pattern.split(r'\b')[1] if r'\b' in pattern else pattern[:20]
            return (True, pattern_key, description)
    
    return (False, None, None)


def _is_command_approved(pattern_key: str) -> bool:
    """Check if a pattern is approved (session or permanent)."""
    if pattern_key in _session_approved_patterns:
        return True
    
    permanent = _load_permanent_allowlist()
    if pattern_key in permanent:
        return True
    
    return False


def _prompt_dangerous_approval(command: str, description: str, timeout_seconds: int = 60) -> str:
    """
    Prompt user to approve a dangerous command (CLI only).
    
    If an _approval_callback is registered (by the CLI), delegates to it so the
    prompt integrates with prompt_toolkit's UI.  Otherwise falls back to the
    raw input() approach (works outside the TUI, e.g. tests).
    
    Returns: 'once', 'session', 'always', or 'deny'
    """
    import sys
    import threading
    
    # Use the registered callback when available (prompt_toolkit-compatible)
    if _approval_callback is not None:
        try:
            return _approval_callback(command, description)
        except Exception:
            return "deny"

    # Pause spinner if one is running
    os.environ["HERMES_SPINNER_PAUSE"] = "1"
    
    try:
        print()
        print(f"  ⚠️  DANGEROUS COMMAND: {description}")
        print(f"      {command[:80]}{'...' if len(command) > 80 else ''}")
        print()
        print(f"      [o]nce  |  [s]ession  |  [a]lways  |  [d]eny")
        print()
        sys.stdout.flush()
        
        result = {"choice": ""}
        
        def get_input():
            try:
                result["choice"] = input("      Choice [o/s/a/D]: ").strip().lower()
            except (EOFError, OSError):
                result["choice"] = ""
        
        thread = threading.Thread(target=get_input, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)
        
        if thread.is_alive():
            print("\n      ⏱ Timeout - denying command")
            return "deny"
        
        choice = result["choice"]
        
        if choice in ('o', 'once'):
            print("      ✓ Allowed once")
            return "once"
        elif choice in ('s', 'session'):
            print("      ✓ Allowed for this session")
            return "session"
        elif choice in ('a', 'always'):
            print("      ✓ Added to permanent allowlist")
            return "always"
        else:
            print("      ✗ Denied")
            return "deny"
            
    except (EOFError, KeyboardInterrupt):
        print("\n      ✗ Cancelled")
        return "deny"
    finally:
        if "HERMES_SPINNER_PAUSE" in os.environ:
            del os.environ["HERMES_SPINNER_PAUSE"]
        print()
        sys.stdout.flush()


def _check_dangerous_command(command: str, env_type: str) -> dict:
    """
    Check if command is dangerous and handle approval.
    
    Only applies to local/ssh backends in interactive contexts.
    
    Args:
        command: The command to check
        env_type: The terminal backend type
        
    Returns:
        {"approved": True/False, "message": str or None}
    """
    # Skip check for isolated environments (containers are disposable)
    if env_type in ("docker", "singularity", "modal"):
        return {"approved": True, "message": None}
    
    # Detect dangerous command
    is_dangerous, pattern_key, description = _detect_dangerous_command(command)
    
    if not is_dangerous:
        return {"approved": True, "message": None}
    
    # Check if already approved
    if _is_command_approved(pattern_key):
        return {"approved": True, "message": None}
    
    # Check context - only prompt in interactive modes
    is_cli = os.getenv("HERMES_INTERACTIVE")
    is_gateway = os.getenv("HERMES_GATEWAY_SESSION")
    
    if not is_cli and not is_gateway:
        # Programmatic use - allow (user opted into local backend)
        return {"approved": True, "message": None}
    
    if is_gateway or os.getenv("HERMES_EXEC_ASK"):
        # Messaging context - return approval_required so the gateway can
        # prompt the user interactively instead of just blocking
        global _last_pending_approval
        _last_pending_approval = {
            "command": command,
            "pattern_key": pattern_key,
            "description": description,
        }
        return {
            "approved": False,
            "pattern_key": pattern_key,
            "status": "approval_required",
            "command": command,
            "description": description,
            "message": f"⚠️ This command is potentially dangerous ({description}). Asking the user for approval..."
        }
    
    # CLI context - prompt user
    choice = _prompt_dangerous_approval(command, description)
    
    if choice == "deny":
        return {"approved": False, "message": "BLOCKED: User denied this potentially dangerous command. Do NOT retry this command - the user has explicitly rejected it."}
    
    # Handle approval
    if choice == "session":
        _session_approved_patterns.add(pattern_key)
    elif choice == "always":
        _session_approved_patterns.add(pattern_key)
        permanent = _load_permanent_allowlist()
        permanent.add(pattern_key)
        _save_permanent_allowlist(permanent)
    
    return {"approved": True, "message": None}


def _handle_sudo_failure(output: str, env_type: str) -> str:
    """
    Check for sudo failure and add helpful message for messaging contexts.
    
    Returns enhanced output if sudo failed in messaging context, else original.
    """
    is_gateway = os.getenv("HERMES_GATEWAY_SESSION")
    
    if not is_gateway:
        return output
    
    # Check for sudo failure indicators
    sudo_failures = [
        "sudo: a password is required",
        "sudo: no tty present",
        "sudo: a terminal is required",
    ]
    
    for failure in sudo_failures:
        if failure in output:
            return output + "\n\n💡 Tip: To enable sudo over messaging, add SUDO_PASSWORD to ~/.hermes/.env on the agent machine."
    
    return output


def _prompt_for_sudo_password(timeout_seconds: int = 45) -> str:
    """
    Prompt user for sudo password with timeout.
    
    Returns the password if entered, or empty string if:
    - User presses Enter without input (skip)
    - Timeout expires (45s default)
    - Any error occurs
    
    Only works in interactive mode (HERMES_INTERACTIVE=1).
    If a _sudo_password_callback is registered (by the CLI), delegates to it
    so the prompt integrates with prompt_toolkit's UI.  Otherwise reads
    directly from /dev/tty with echo disabled.
    """
    import sys
    import time as time_module
    
    # Use the registered callback when available (prompt_toolkit-compatible)
    if _sudo_password_callback is not None:
        try:
            return _sudo_password_callback() or ""
        except Exception:
            return ""

    result = {"password": None, "done": False}
    
    def read_password_thread():
        """Read password from /dev/tty with echo disabled."""
        tty_fd = None
        old_attrs = None
        try:
            import termios
            tty_fd = os.open("/dev/tty", os.O_RDONLY)
            old_attrs = termios.tcgetattr(tty_fd)
            new_attrs = termios.tcgetattr(tty_fd)
            new_attrs[3] = new_attrs[3] & ~termios.ECHO
            termios.tcsetattr(tty_fd, termios.TCSAFLUSH, new_attrs)
            chars = []
            while True:
                b = os.read(tty_fd, 1)
                if not b or b in (b"\n", b"\r"):
                    break
                chars.append(b)
            result["password"] = b"".join(chars).decode("utf-8", errors="replace")
        except (EOFError, KeyboardInterrupt, OSError):
            result["password"] = ""
        except Exception:
            result["password"] = ""
        finally:
            if tty_fd is not None and old_attrs is not None:
                try:
                    import termios as _termios
                    _termios.tcsetattr(tty_fd, _termios.TCSAFLUSH, old_attrs)
                except Exception:
                    pass
            if tty_fd is not None:
                try:
                    os.close(tty_fd)
                except Exception:
                    pass
            result["done"] = True
    
    try:
        os.environ["HERMES_SPINNER_PAUSE"] = "1"
        time_module.sleep(0.2)
        
        print()
        print("┌" + "─" * 58 + "┐")
        print("│  🔐 SUDO PASSWORD REQUIRED" + " " * 30 + "│")
        print("├" + "─" * 58 + "┤")
        print("│  Enter password below (input is hidden), or:            │")
        print("│    • Press Enter to skip (command fails gracefully)     │")
        print(f"│    • Wait {timeout_seconds}s to auto-skip" + " " * 27 + "│")
        print("└" + "─" * 58 + "┘")
        print()
        print("  Password (hidden): ", end="", flush=True)
        
        password_thread = threading.Thread(target=read_password_thread, daemon=True)
        password_thread.start()
        password_thread.join(timeout=timeout_seconds)
        
        if result["done"]:
            password = result["password"] or ""
            print()  # newline after hidden input
            if password:
                print("  ✓ Password received (cached for this session)")
            else:
                print("  ⏭ Skipped - continuing without sudo")
            print()
            sys.stdout.flush()
            return password
        else:
            print("\n  ⏱ Timeout - continuing without sudo")
            print("    (Press Enter to dismiss)")
            print()
            sys.stdout.flush()
            return ""
            
    except (EOFError, KeyboardInterrupt):
        print()
        print("  ⏭ Cancelled - continuing without sudo")
        print()
        sys.stdout.flush()
        return ""
    except Exception as e:
        print(f"\n  [sudo prompt error: {e}] - continuing without sudo\n")
        sys.stdout.flush()
        return ""
    finally:
        if "HERMES_SPINNER_PAUSE" in os.environ:
            del os.environ["HERMES_SPINNER_PAUSE"]


def _transform_sudo_command(command: str) -> str:
    """
    Transform sudo commands to use -S flag if SUDO_PASSWORD is available.
    
    This is a shared helper used by all execution environments to provide
    consistent sudo handling across local, SSH, and container environments.
    
    If SUDO_PASSWORD is set (via env, config, or interactive prompt):
      'sudo apt install curl' -> password piped via sudo -S
      
    If SUDO_PASSWORD is not set and in interactive mode (HERMES_INTERACTIVE=1):
      Prompts user for password with 45s timeout, caches for session.
      
    If SUDO_PASSWORD is not set and NOT interactive:
      Command runs as-is (fails gracefully with "sudo: a password is required").
    """
    global _cached_sudo_password
    import re
    
    # Check if command even contains sudo
    if not re.search(r'\bsudo\b', command):
        return command  # No sudo in command, return as-is
    
    # Try to get password from: env var -> session cache -> interactive prompt
    sudo_password = os.getenv("SUDO_PASSWORD", "") or _cached_sudo_password
    
    if not sudo_password:
        # No password configured - check if we're in interactive mode
        if os.getenv("HERMES_INTERACTIVE"):
            # Prompt user for password
            sudo_password = _prompt_for_sudo_password(timeout_seconds=45)
            if sudo_password:
                _cached_sudo_password = sudo_password  # Cache for session
    
    if not sudo_password:
        return command  # No password, let it fail gracefully
    
    def replace_sudo(match):
        # Replace 'sudo' with password-piped version
        # The -S flag makes sudo read password from stdin
        # The -p '' suppresses the password prompt
        return f"echo '{sudo_password}' | sudo -S -p ''"
    
    # Match 'sudo' at word boundaries (not 'visudo' or 'sudoers')
    # This handles: sudo, sudo -flag, etc.
    return re.sub(r'\bsudo\b', replace_sudo, command)


class _LocalEnvironment:
    """
    Local execution environment with sudo support and non-blocking stdin.
    
    Features:
    - Uses stdin=DEVNULL to prevent hanging on interactive prompts (sudo, etc.)
    - Optional SUDO_PASSWORD support: if set, transforms `sudo` commands to use `sudo -S`
    - Graceful failure: sudo commands fail fast with clear error if no password configured
    
    Environment variables:
    - SUDO_PASSWORD: If set, enables sudo commands by piping password via `sudo -S`
    """
    
    def __init__(self, cwd: str = "", timeout: int = 60, env: dict = None):
        self.cwd = cwd or os.getcwd()
        self.timeout = timeout
        self.env = env or {}
    
    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None,
                stdin_data: str | None = None) -> dict:
        """
        Execute a command locally with sudo support.
        
        Uses Popen + polling so the global interrupt event can kill the
        process early when the user sends a new message, instead of
        blocking for the full timeout.
        
        A background reader thread drains stdout continuously to prevent
        pipe buffer deadlocks. Without this, commands producing >64KB of
        output would block (Linux pipe buffer = 64KB) while the poll loop
        waits for the process to finish — a classic deadlock.
        
        Args:
            stdin_data: If provided, piped to the process's stdin. This
                        bypasses shell ARG_MAX limits for large content.
        """
        work_dir = cwd or self.cwd or os.getcwd()
        effective_timeout = timeout or self.timeout
        
        # Transform sudo commands if SUDO_PASSWORD is available
        exec_command = _transform_sudo_command(command)
        
        try:
            proc = subprocess.Popen(
                exec_command,
                shell=True,
                text=True,
                cwd=work_dir,
                env=os.environ | self.env,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
                # Start in a new process group so we can kill the whole tree
                preexec_fn=os.setsid,
            )
            
            # Pipe stdin_data in a background thread to avoid deadlock
            # (large writes can block if the pipe buffer fills before the
            # process drains it).
            if stdin_data is not None:
                def _write_stdin():
                    try:
                        proc.stdin.write(stdin_data)
                        proc.stdin.close()
                    except (BrokenPipeError, OSError):
                        pass
                stdin_writer = threading.Thread(target=_write_stdin, daemon=True)
                stdin_writer.start()
            
            # Drain stdout in a background thread to prevent pipe buffer
            # deadlocks. The OS pipe buffer is 64KB on Linux; if the child
            # writes more than that before anyone reads, it blocks forever.
            _output_chunks: list[str] = []
            def _drain_stdout():
                try:
                    for line in proc.stdout:
                        _output_chunks.append(line)
                except ValueError:
                    pass  # stdout closed during interrupt/timeout
                finally:
                    try:
                        proc.stdout.close()
                    except Exception:
                        pass
            
            reader = threading.Thread(target=_drain_stdout, daemon=True)
            reader.start()
            
            deadline = time.monotonic() + effective_timeout
            
            # Poll every 200ms so we notice interrupts quickly
            while proc.poll() is None:
                if _interrupt_event.is_set():
                    # User sent a new message — kill the process tree and return
                    # what we have so far
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        proc.kill()
                    reader.join(timeout=2)
                    output = "".join(_output_chunks)
                    return {
                        "output": output + "\n[Command interrupted — user sent a new message]",
                        "returncode": 130  # Standard interrupted exit code
                    }
                
                if time.monotonic() > deadline:
                    # Timeout — kill process tree
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        proc.kill()
                    reader.join(timeout=2)
                    return {"output": f"Command timed out after {effective_timeout}s", "returncode": 124}
                
                # Short sleep to avoid busy-waiting
                time.sleep(0.2)
            
            # Process finished — wait for reader to drain remaining output
            reader.join(timeout=5)
            return {"output": "".join(_output_chunks), "returncode": proc.returncode}
            
        except Exception as e:
            return {"output": f"Execution error: {str(e)}", "returncode": 1}
    
    def cleanup(self):
        """No cleanup needed for local environment."""
        pass
    
    def stop(self):
        """Alias for cleanup."""
        pass


class _SingularityEnvironment:
    """
    Persistent Singularity/Apptainer container environment.
    
    Uses `apptainer instance` to create a long-running container that persists
    state (files, installs, env changes) across all commands within a task.
    The model experiences this as a real Linux VM.
    
    Features:
    - Persistent filesystem: files created in one command are visible in the next
    - Package installs persist: pip/apt installs survive across tool calls
    - Full isolation: --containall gives PID, IPC, and environment isolation
    - Writable tmpfs overlay: full root filesystem is writable (RAM-backed)
    - Automatic SIF caching: docker:// images converted to SIF once, reused forever
    """
    
    def __init__(self, image: str, cwd: str = "/root", timeout: int = 60):
        self.cwd = cwd
        self.timeout = timeout
        
        # Use apptainer if available, otherwise singularity
        self.executable = "apptainer" if shutil.which("apptainer") else "singularity"
        
        # Get or build SIF from docker:// URL (fast if already cached)
        self.image = _get_or_build_sif(image, self.executable)
        
        # Create unique instance name (must be alphanumeric + underscores)
        self.instance_id = f"hermes_{uuid.uuid4().hex[:12]}"
        self._instance_started = False
        
        # Start the persistent instance
        self._start_instance()
    
    def _start_instance(self):
        """Start a persistent apptainer instance.
        
        The instance runs as a background process. All subsequent execute() calls
        run commands inside this same instance, so state persists across calls.
        """
        cmd = [
            self.executable, "instance", "start",
            "--writable-tmpfs",  # RAM-backed writable overlay on read-only SIF
            "--containall",      # Full isolation: PID, IPC, environment, filesystem
            str(self.image),
            self.instance_id,
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 min for instance startup
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to start instance: {result.stderr}")
            
            self._instance_started = True
            logger.info("Singularity instance %s started (persistent container)", self.instance_id)
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Instance start timed out")
    
    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None,
                stdin_data: str | None = None) -> dict:
        """Execute a command in the persistent Singularity instance.
        
        All commands run in the same container, so files, installs, and
        environment changes persist between calls.
        """
        if not self._instance_started:
            return {"output": "Instance not started", "returncode": -1}
        
        cmd = [self.executable, "exec"]
        
        # Set working directory
        work_dir = cwd or self.cwd
        cmd.extend(["--pwd", work_dir])
        
        # Connect to the running instance
        cmd.append(f"instance://{self.instance_id}")
        
        # Transform sudo commands if SUDO_PASSWORD is available
        exec_command = _transform_sudo_command(command)
        
        # Execute the command
        cmd.extend(["bash", "-c", exec_command])
        
        run_kwargs = {
            "text": True,
            "timeout": timeout or self.timeout,
            "encoding": "utf-8",
            "errors": "replace",
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
        }
        if stdin_data is not None:
            run_kwargs["input"] = stdin_data
        else:
            run_kwargs["stdin"] = subprocess.DEVNULL
        
        try:
            result = subprocess.run(cmd, **run_kwargs)
            return {"output": result.stdout, "returncode": result.returncode}
        except subprocess.TimeoutExpired:
            return {"output": f"Command timed out after {timeout or self.timeout}s", "returncode": 124}
    
    def cleanup(self):
        """Stop the persistent instance and clean up."""
        if self._instance_started:
            try:
                subprocess.run(
                    [self.executable, "instance", "stop", self.instance_id],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                logger.info("Singularity instance %s stopped", self.instance_id)
            except Exception as e:
                logger.warning("Failed to stop Singularity instance %s: %s", self.instance_id, e)
            self._instance_started = False
    
    def stop(self):
        """Alias for cleanup."""
        self.cleanup()
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except Exception:
            pass


class _SSHEnvironment:
    """
    SSH-based remote execution environment.
    
    Runs commands on a remote machine over SSH, keeping the agent code
    completely isolated from the execution environment. Uses SSH ControlMaster
    for connection persistence (faster subsequent commands).
    
    Security benefits:
    - Agent cannot modify its own code
    - Remote machine acts as a sandbox
    - Clear separation between agent and execution environment
    """
    
    def __init__(self, host: str, user: str, cwd: str = "/tmp", timeout: int = 60,
                 port: int = 22, key_path: str = ""):
        self.host = host
        self.user = user
        self.cwd = cwd
        self.timeout = timeout
        self.port = port
        self.key_path = key_path
        
        # Create control socket directory for connection persistence
        self.control_dir = Path(tempfile.gettempdir()) / "hermes-ssh"
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.control_socket = self.control_dir / f"{user}@{host}:{port}.sock"
        
        # Test connection and establish ControlMaster
        self._establish_connection()
    
    def _build_ssh_command(self, extra_args: list = None) -> list:
        """Build base SSH command with connection options."""
        cmd = ["ssh"]
        
        # Connection multiplexing for performance
        cmd.extend(["-o", f"ControlPath={self.control_socket}"])
        cmd.extend(["-o", "ControlMaster=auto"])
        cmd.extend(["-o", "ControlPersist=300"])  # Keep connection alive for 5 min
        
        # Standard options
        cmd.extend(["-o", "BatchMode=yes"])  # No password prompts
        cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])  # Accept new hosts
        cmd.extend(["-o", "ConnectTimeout=10"])
        
        # Port
        if self.port != 22:
            cmd.extend(["-p", str(self.port)])
        
        # Private key
        if self.key_path:
            cmd.extend(["-i", self.key_path])
        
        # Extra args (like -t for TTY)
        if extra_args:
            cmd.extend(extra_args)
        
        # Target
        cmd.append(f"{self.user}@{self.host}")
        
        return cmd
    
    def _establish_connection(self):
        """Test SSH connection and establish ControlMaster."""
        cmd = self._build_ssh_command()
        cmd.append("echo 'SSH connection established'")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15
            )
            if result.returncode != 0:
                error_msg = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(f"SSH connection failed: {error_msg}")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"SSH connection to {self.user}@{self.host} timed out")
    
    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None,
                stdin_data: str | None = None) -> dict:
        """Execute a command on the remote host via SSH."""
        work_dir = cwd or self.cwd
        effective_timeout = timeout or self.timeout
        
        # Transform sudo commands if SUDO_PASSWORD is available
        exec_command = _transform_sudo_command(command)
        
        # Wrap command to run in the correct directory
        wrapped_command = f'cd {work_dir} && {exec_command}'
        
        cmd = self._build_ssh_command()
        cmd.extend(["bash", "-c", wrapped_command])
        
        run_kwargs = {
            "text": True,
            "timeout": effective_timeout,
            "encoding": "utf-8",
            "errors": "replace",
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
        }
        if stdin_data is not None:
            run_kwargs["input"] = stdin_data
        else:
            run_kwargs["stdin"] = subprocess.DEVNULL
        
        try:
            result = subprocess.run(cmd, **run_kwargs)
            return {"output": result.stdout, "returncode": result.returncode}
        except subprocess.TimeoutExpired:
            return {"output": f"Command timed out after {effective_timeout}s", "returncode": 124}
        except Exception as e:
            return {"output": f"SSH execution error: {str(e)}", "returncode": 1}
    
    def cleanup(self):
        """Close the SSH ControlMaster connection."""
        if self.control_socket.exists():
            try:
                # Send exit command to ControlMaster
                cmd = ["ssh", "-o", f"ControlPath={self.control_socket}", "-O", "exit", 
                       f"{self.user}@{self.host}"]
                subprocess.run(cmd, capture_output=True, timeout=5)
            except (OSError, subprocess.SubprocessError):
                pass
            
            # Remove socket file
            try:
                self.control_socket.unlink()
            except OSError:
                pass
    
    def stop(self):
        """Alias for cleanup."""
        self.cleanup()
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except Exception:
            pass


class _DockerEnvironment:
    """
    Docker execution environment wrapper with sudo support and non-blocking stdin.
    
    Wraps mini-swe-agent's DockerEnvironment but adds:
    - stdin=DEVNULL to prevent hanging on interactive prompts
    - SUDO_PASSWORD support via _transform_sudo_command
    """
    
    def __init__(self, image: str, cwd: str = "/", timeout: int = 60):
        from minisweagent.environments.docker import DockerEnvironment
        self._inner = DockerEnvironment(image=image, cwd=cwd, timeout=timeout)
        self.cwd = cwd
        self.timeout = timeout
    
    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None,
                stdin_data: str | None = None) -> dict:
        """Execute a command in the Docker container with sudo support."""
        # Transform sudo commands if SUDO_PASSWORD is available
        exec_command = _transform_sudo_command(command)
        
        work_dir = cwd or self.cwd
        effective_timeout = timeout or self.timeout
        
        # Get container_id from inner environment
        assert self._inner.container_id, "Container not started"
        
        cmd = [self._inner.config.executable, "exec"]
        if stdin_data is not None:
            cmd.append("-i")  # Enable stdin piping into the container
        cmd.extend(["-w", work_dir])
        for key in self._inner.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self._inner.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self._inner.container_id, "bash", "-lc", exec_command])
        
        run_kwargs = {
            "text": True,
            "timeout": effective_timeout,
            "encoding": "utf-8",
            "errors": "replace",
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
        }
        if stdin_data is not None:
            run_kwargs["input"] = stdin_data
        else:
            run_kwargs["stdin"] = subprocess.DEVNULL
        
        try:
            result = subprocess.run(cmd, **run_kwargs)
            return {"output": result.stdout, "returncode": result.returncode}
        except subprocess.TimeoutExpired:
            return {"output": f"Command timed out after {effective_timeout}s", "returncode": 124}
    
    def cleanup(self):
        """Cleanup the Docker container."""
        self._inner.cleanup()
    
    def stop(self):
        """Alias for cleanup."""
        self.cleanup()
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except Exception:
            pass


class _ModalEnvironment:
    """
    Modal cloud execution environment wrapper with sudo support.
    
    Wraps mini-swe-agent's SwerexModalEnvironment but adds:
    - SUDO_PASSWORD support via _transform_sudo_command
    - Automatic async-safety patches (applied once, before first use)
    
    The patches replace SwerexModalEnvironment's asyncio.run() calls with a
    background thread approach, making it safe to use inside any event loop
    (e.g., Atropos). Applied here at the point of use rather than relying on
    import-time side effects, so ALL callers get the fix automatically.
    """
    
    # Class-level flag: patches only need to be applied once
    _patches_applied = False
    
    def __init__(self, image: str, cwd: str = "/root", timeout: int = 60):
        # Ensure async-safety patches are applied before creating any
        # SwerexModalEnvironment instance. This is the single authoritative
        # place -- no other module needs to call apply_patches() for Modal.
        if not _ModalEnvironment._patches_applied:
            try:
                from environments.patches import apply_patches
                apply_patches()
            except ImportError:
                pass  # patches module not available (standalone use)
            _ModalEnvironment._patches_applied = True
        
        from minisweagent.environments.extra.swerex_modal import SwerexModalEnvironment
        # Generous startup timeout: sandbox creation can take 30-60s for cold images,
        # and the SWE-ReX runtime needs another 10-30s to boot inside it.
        self._inner = SwerexModalEnvironment(
            image=image, cwd=cwd, timeout=timeout,
            startup_timeout=180.0,
            runtime_timeout=3600.0,
        )
        self.cwd = cwd
        self.timeout = timeout
    
    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None,
                stdin_data: str | None = None) -> dict:
        """Execute a command in Modal with sudo support.
        
        Modal uses HTTP transport (no execve), so there's no ARG_MAX limit.
        When stdin_data is provided, we embed it as a heredoc since there's
        no process-level stdin pipe to the cloud sandbox.
        """
        if stdin_data is not None:
            marker = f"HERMES_EOF_{uuid.uuid4().hex[:8]}"
            while marker in stdin_data:
                marker = f"HERMES_EOF_{uuid.uuid4().hex[:8]}"
            command = f"{command} << '{marker}'\n{stdin_data}\n{marker}"
        
        # Transform sudo commands if SUDO_PASSWORD is available
        exec_command = _transform_sudo_command(command)
        
        # Delegate to inner environment with transformed command
        return self._inner.execute(exec_command, cwd=cwd, timeout=timeout)
    
    def cleanup(self):
        """Cleanup the Modal deployment."""
        if hasattr(self._inner, 'stop'):
            self._inner.stop()
    
    def stop(self):
        """Stop the Modal deployment."""
        self.cleanup()
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except Exception:
            pass


# Tool description for LLM
TERMINAL_TOOL_DESCRIPTION = """Execute commands on a Linux environment. Filesystem persists between calls.

Background processes: Set background=true to get a session_id, then use the 'process' tool to poll/wait/kill/write.
Working directory: Use 'workdir' for per-command cwd.
PTY mode: Set pty=true for interactive CLI tools (Codex, Claude Code, Python REPL).

Do NOT use vim/nano/interactive tools without pty=true — they hang without a pseudo-terminal. Pipe git output to cat if it might page.
"""

# Global state for environment lifecycle management
_active_environments: Dict[str, Any] = {}
_task_workdirs: Dict[str, str] = {}  # Maps task_id to working directory
_last_activity: Dict[str, float] = {}
_env_lock = threading.Lock()
_creation_locks: Dict[str, threading.Lock] = {}  # Per-task locks for sandbox creation
_creation_locks_lock = threading.Lock()  # Protects _creation_locks dict itself
_cleanup_thread = None
_cleanup_running = False

# Per-task environment overrides registry.
# Allows environments (e.g., TerminalBench2Env) to specify a custom Docker/Modal
# image for a specific task_id BEFORE the agent loop starts. When the terminal or
# file tools create a new sandbox for that task_id, they check this registry first
# and fall back to the TERMINAL_MODAL_IMAGE (etc.) env var if no override is set.
#
# This is never exposed to the model -- only infrastructure code calls it.
# Thread-safe because each task_id is unique per rollout.
_task_env_overrides: Dict[str, Dict[str, Any]] = {}


def register_task_env_overrides(task_id: str, overrides: Dict[str, Any]):
    """
    Register environment overrides for a specific task/rollout.

    Called by Atropos environments before the agent loop to configure
    per-task sandbox settings (e.g., a custom Dockerfile for the Modal image).

    Supported override keys:
        - modal_image: str -- Path to Dockerfile or Docker Hub image name
        - docker_image: str -- Docker image name
        - cwd: str -- Working directory inside the sandbox

    Args:
        task_id: The rollout's unique task identifier
        overrides: Dict of config keys to override
    """
    _task_env_overrides[task_id] = overrides


def clear_task_env_overrides(task_id: str):
    """
    Clear environment overrides for a task after rollout completes.

    Called during cleanup to avoid stale entries accumulating.
    """
    _task_env_overrides.pop(task_id, None)

# Configuration from environment variables
def _get_env_config() -> Dict[str, Any]:
    """Get terminal environment configuration from environment variables."""
    # Default image with Python and Node.js for maximum compatibility
    default_image = "nikolaik/python-nodejs:python3.11-nodejs20"
    env_type = os.getenv("TERMINAL_ENV", "local")
    
    # Default cwd depends on backend:
    #   - local: host's current working directory
    #   - ssh: remote user's home (agent code is local, execution is remote)
    #   - docker: / inside the container
    #   - singularity/modal: /root (ephemeral cloud/container)
    if env_type in ("modal", "singularity"):
        default_cwd = "/root"
    elif env_type == "docker":
        default_cwd = "/"
    elif env_type == "ssh":
        default_cwd = "~"
    else:
        default_cwd = os.getcwd()
    
    # Read TERMINAL_CWD but sanity-check it for non-local backends.
    # If the CWD looks like a host-local path that can't exist inside a
    # container/sandbox, fall back to the backend's own default.  This
    # catches the case where cli.py (or .env) leaked the host's CWD.
    cwd = os.getenv("TERMINAL_CWD", default_cwd)
    if env_type in ("modal", "docker", "singularity", "ssh") and cwd:
        # Paths containing common host-only prefixes are clearly wrong
        # inside a container.  Also catch Windows-style paths (C:\...).
        host_prefixes = ("/Users/", "/home/", "C:\\", "C:/")
        if any(cwd.startswith(p) for p in host_prefixes) and cwd != default_cwd:
            logger.info("Ignoring TERMINAL_CWD=%r for %s backend "
                        "(host path won't exist in sandbox). Using %r instead.",
                        cwd, env_type, default_cwd)
            cwd = default_cwd

    return {
        "env_type": env_type,
        "docker_image": os.getenv("TERMINAL_DOCKER_IMAGE", default_image),
        "singularity_image": os.getenv("TERMINAL_SINGULARITY_IMAGE", f"docker://{default_image}"),
        "modal_image": os.getenv("TERMINAL_MODAL_IMAGE", default_image),
        "cwd": cwd,
        "timeout": int(os.getenv("TERMINAL_TIMEOUT", "60")),
        "lifetime_seconds": int(os.getenv("TERMINAL_LIFETIME_SECONDS", "300")),
        # SSH-specific config
        "ssh_host": os.getenv("TERMINAL_SSH_HOST", ""),
        "ssh_user": os.getenv("TERMINAL_SSH_USER", ""),
        "ssh_port": int(os.getenv("TERMINAL_SSH_PORT", "22")),
        "ssh_key": os.getenv("TERMINAL_SSH_KEY", ""),  # Path to private key (optional, uses ssh-agent if empty)
    }


def _create_environment(env_type: str, image: str, cwd: str, timeout: int, ssh_config: dict = None):
    """
    Create an execution environment from mini-swe-agent.
    
    Args:
        env_type: One of "local", "docker", "singularity", "modal", "ssh"
        image: Docker/Singularity/Modal image name (ignored for local/ssh)
        cwd: Working directory
        timeout: Default command timeout
        ssh_config: SSH connection config (for env_type="ssh")
        
    Returns:
        Environment instance with execute() method
    """
    if env_type == "local":
        # Use our custom LocalEnvironment with sudo support and non-blocking stdin
        return _LocalEnvironment(cwd=cwd, timeout=timeout)
    
    elif env_type == "docker":
        # Use custom Docker wrapper with sudo support and non-blocking stdin
        return _DockerEnvironment(image=image, cwd=cwd, timeout=timeout)
    
    elif env_type == "singularity":
        # Use custom Singularity environment with better space management
        return _SingularityEnvironment(image=image, cwd=cwd, timeout=timeout)
    
    elif env_type == "modal":
        # Use custom Modal wrapper with sudo support
        return _ModalEnvironment(image=image, cwd=cwd, timeout=timeout)
    
    elif env_type == "ssh":
        if not ssh_config or not ssh_config.get("host") or not ssh_config.get("user"):
            raise ValueError("SSH environment requires ssh_host and ssh_user to be configured")
        return _SSHEnvironment(
            host=ssh_config["host"],
            user=ssh_config["user"],
            port=ssh_config.get("port", 22),
            key_path=ssh_config.get("key", ""),
            cwd=cwd,
            timeout=timeout
        )
    
    else:
        raise ValueError(f"Unknown environment type: {env_type}. Use 'local', 'docker', 'singularity', 'modal', or 'ssh'")


def _cleanup_inactive_envs(lifetime_seconds: int = 300):
    """Clean up environments that have been inactive for longer than lifetime_seconds."""
    global _active_environments, _last_activity

    current_time = time.time()

    # Check the process registry -- skip cleanup for sandboxes with active
    # background processes (their _last_activity gets refreshed to keep them alive).
    try:
        from tools.process_registry import process_registry
        for task_id in list(_last_activity.keys()):
            if process_registry.has_active_processes(task_id):
                _last_activity[task_id] = current_time  # Keep sandbox alive
    except ImportError:
        pass

    # Phase 1: collect stale entries and remove them from tracking dicts while
    # holding the lock.  Do NOT call env.cleanup() inside the lock -- Modal and
    # Docker teardown can block for 10-15s, which would stall every concurrent
    # terminal/file tool call waiting on _env_lock.
    envs_to_stop = []  # list of (task_id, env) pairs

    with _env_lock:
        for task_id, last_time in list(_last_activity.items()):
            if current_time - last_time > lifetime_seconds:
                env = _active_environments.pop(task_id, None)
                _last_activity.pop(task_id, None)
                _task_workdirs.pop(task_id, None)
                if env is not None:
                    envs_to_stop.append((task_id, env))

        # Also purge per-task creation locks for cleaned-up tasks
        with _creation_locks_lock:
            for task_id, _ in envs_to_stop:
                _creation_locks.pop(task_id, None)

    # Phase 2: stop the actual sandboxes OUTSIDE the lock so other tool calls
    # are not blocked while Modal/Docker sandboxes shut down.
    for task_id, env in envs_to_stop:
        # Invalidate stale file_ops cache entry (Bug fix: prevents
        # ShellFileOperations from referencing a dead sandbox)
        try:
            from tools.file_tools import clear_file_ops_cache
            clear_file_ops_cache(task_id)
        except ImportError:
            pass

        try:
            if hasattr(env, 'cleanup'):
                env.cleanup()
            elif hasattr(env, 'stop'):
                env.stop()
            elif hasattr(env, 'terminate'):
                env.terminate()

            logger.info("Cleaned up inactive environment for task: %s", task_id)

        except Exception as e:
            error_str = str(e)
            if "404" in error_str or "not found" in error_str.lower():
                logger.info("Environment for task %s already cleaned up", task_id)
            else:
                logger.warning("Error cleaning up environment for task %s: %s", task_id, e)


def _cleanup_thread_worker():
    """Background thread worker that periodically cleans up inactive environments."""
    global _cleanup_running

    while _cleanup_running:
        try:
            config = _get_env_config()
            _cleanup_inactive_envs(config["lifetime_seconds"])
        except Exception as e:
            logger.warning("Error in cleanup thread: %s", e)

        for _ in range(60):
            if not _cleanup_running:
                break
            time.sleep(1)


def _start_cleanup_thread():
    """Start the background cleanup thread if not already running."""
    global _cleanup_thread, _cleanup_running

    with _env_lock:
        if _cleanup_thread is None or not _cleanup_thread.is_alive():
            _cleanup_running = True
            _cleanup_thread = threading.Thread(target=_cleanup_thread_worker, daemon=True)
            _cleanup_thread.start()


def _stop_cleanup_thread():
    """Stop the background cleanup thread."""
    global _cleanup_running
    _cleanup_running = False
    if _cleanup_thread is not None:
        _cleanup_thread.join(timeout=5)


def get_active_environments_info() -> Dict[str, Any]:
    """Get information about currently active environments."""
    info = {
        "count": len(_active_environments),
        "task_ids": list(_active_environments.keys()),
        "workdirs": dict(_task_workdirs),
    }
    
    # Calculate total disk usage
    total_size = 0
    for task_id in _active_environments.keys():
        # Check sandbox and workdir sizes
        scratch_dir = _get_scratch_dir()
        for pattern in [f"hermes-*{task_id[:8]}*"]:
            import glob
            for path in glob.glob(str(scratch_dir / "hermes-*")):
                try:
                    size = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
                    total_size += size
                except OSError:
                    pass
    
    info["total_disk_usage_mb"] = round(total_size / (1024 * 1024), 2)
    return info


def cleanup_all_environments():
    """Clean up ALL active environments. Use with caution."""
    global _active_environments, _last_activity, _task_workdirs
    
    task_ids = list(_active_environments.keys())
    cleaned = 0
    
    for task_id in task_ids:
        try:
            cleanup_vm(task_id)
            cleaned += 1
        except Exception as e:
            logger.error("Error cleaning %s: %s", task_id, e)
    
    # Also clean any orphaned directories
    scratch_dir = _get_scratch_dir()
    import glob
    for path in glob.glob(str(scratch_dir / "hermes-*")):
        try:
            shutil.rmtree(path, ignore_errors=True)
            logger.info("Removed orphaned: %s", path)
        except OSError:
            pass
    
    if cleaned > 0:
        logger.info("Cleaned %d environments", cleaned)
    return cleaned


def cleanup_vm(task_id: str):
    """Manually clean up a specific environment by task_id."""
    global _active_environments, _last_activity, _task_workdirs

    # Remove from tracking dicts while holding the lock, but defer the
    # actual (potentially slow) env.cleanup() call to outside the lock
    # so other tool calls aren't blocked.
    env = None
    with _env_lock:
        env = _active_environments.pop(task_id, None)
        _task_workdirs.pop(task_id, None)
        _last_activity.pop(task_id, None)

    # Clean up per-task creation lock
    with _creation_locks_lock:
        _creation_locks.pop(task_id, None)

    # Invalidate stale file_ops cache entry
    try:
        from tools.file_tools import clear_file_ops_cache
        clear_file_ops_cache(task_id)
    except ImportError:
        pass

    if env is None:
        return

    try:
        if hasattr(env, 'cleanup'):
            env.cleanup()
        elif hasattr(env, 'stop'):
            env.stop()
        elif hasattr(env, 'terminate'):
            env.terminate()

        logger.info("Manually cleaned up environment for task: %s", task_id)

    except Exception as e:
        error_str = str(e)
        if "404" in error_str or "not found" in error_str.lower():
            logger.info("Environment for task %s already cleaned up", task_id)
        else:
            logger.warning("Error cleaning up environment for task %s: %s", task_id, e)


def _atexit_cleanup():
    """Stop cleanup thread and shut down all remaining sandboxes on exit."""
    _stop_cleanup_thread()
    if _active_environments:
        count = len(_active_environments)
        logger.info("Shutting down %d remaining sandbox(es)...", count)
        cleanup_all_environments()

atexit.register(_atexit_cleanup)


def terminal_tool(
    command: str,
    background: bool = False,
    timeout: Optional[int] = None,
    task_id: Optional[str] = None,
    force: bool = False,
    workdir: Optional[str] = None,
    check_interval: Optional[int] = None,
    pty: bool = False,
) -> str:
    """
    Execute a command using mini-swe-agent's execution environments.

    Args:
        command: The command to execute
        background: Whether to run in background (default: False)
        timeout: Command timeout in seconds (default: from config)
        task_id: Unique identifier for environment isolation (optional)
        force: If True, skip dangerous command check (use after user confirms)
        workdir: Working directory for this command (optional, uses session cwd if not set)
        check_interval: Seconds between auto-checks for background processes (gateway only, min 30)
        pty: If True, use pseudo-terminal for interactive CLI tools (local backend only)

    Returns:
        str: JSON string with output, exit_code, and error fields

    Examples:
        # Execute a simple command
        >>> result = terminal_tool(command="ls -la /tmp")

        # Run a background task
        >>> result = terminal_tool(command="python server.py", background=True)

        # With custom timeout
        >>> result = terminal_tool(command="long_task.sh", timeout=300)
        
        # Force run after user confirmation
        # Note: force parameter is internal only, not exposed to model API
    """
    global _active_environments, _last_activity

    try:
        # Get configuration
        config = _get_env_config()
        env_type = config["env_type"]

        # Use task_id for environment isolation
        effective_task_id = task_id or "default"

        # Check per-task overrides (set by environments like TerminalBench2Env)
        # before falling back to global env var config
        overrides = _task_env_overrides.get(effective_task_id, {})
        
        # Select image based on env type, with per-task override support
        if env_type == "docker":
            image = overrides.get("docker_image") or config["docker_image"]
        elif env_type == "singularity":
            image = overrides.get("singularity_image") or config["singularity_image"]
        elif env_type == "modal":
            image = overrides.get("modal_image") or config["modal_image"]
        else:
            image = ""
        
        cwd = overrides.get("cwd") or config["cwd"]
        default_timeout = config["timeout"]
        effective_timeout = timeout or default_timeout

        # For local environment in batch mode, create a unique subdirectory per task
        # This prevents parallel tasks from overwriting each other's files
        # In CLI mode (HERMES_QUIET), use the cwd directly without subdirectories
        if env_type == "local" and not os.getenv("HERMES_QUIET"):
            with _env_lock:
                if effective_task_id not in _task_workdirs:
                    task_workdir = Path(cwd) / f"hermes-{effective_task_id}-{uuid.uuid4().hex[:8]}"
                    task_workdir.mkdir(parents=True, exist_ok=True)
                    _task_workdirs[effective_task_id] = str(task_workdir)
                cwd = _task_workdirs[effective_task_id]

        # Start cleanup thread
        _start_cleanup_thread()

        # Get or create environment.
        # Use a per-task creation lock so concurrent tool calls for the same
        # task_id wait for the first one to finish creating the sandbox,
        # instead of each creating their own (wasting Modal resources).
        with _env_lock:
            if effective_task_id in _active_environments:
                _last_activity[effective_task_id] = time.time()
                env = _active_environments[effective_task_id]
                needs_creation = False
            else:
                needs_creation = True

        if needs_creation:
            # Per-task lock: only one thread creates the sandbox, others wait
            with _creation_locks_lock:
                if effective_task_id not in _creation_locks:
                    _creation_locks[effective_task_id] = threading.Lock()
                task_lock = _creation_locks[effective_task_id]

            with task_lock:
                # Double-check after acquiring the per-task lock
                with _env_lock:
                    if effective_task_id in _active_environments:
                        _last_activity[effective_task_id] = time.time()
                        env = _active_environments[effective_task_id]
                        needs_creation = False

                if needs_creation:
                    if env_type == "singularity":
                        _check_disk_usage_warning()
                    logger.info("Creating new %s environment for task %s...", env_type, effective_task_id[:8])
                    try:
                        ssh_config = None
                        if env_type == "ssh":
                            ssh_config = {
                                "host": config.get("ssh_host", ""),
                                "user": config.get("ssh_user", ""),
                                "port": config.get("ssh_port", 22),
                                "key": config.get("ssh_key", ""),
                            }

                        new_env = _create_environment(
                            env_type=env_type,
                            image=image,
                            cwd=cwd,
                            timeout=effective_timeout,
                            ssh_config=ssh_config
                        )
                    except ImportError as e:
                        return json.dumps({
                            "output": "",
                            "exit_code": -1,
                            "error": f"Terminal tool disabled: mini-swe-agent not available ({e})",
                            "status": "disabled"
                        }, ensure_ascii=False)

                    with _env_lock:
                        _active_environments[effective_task_id] = new_env
                        _last_activity[effective_task_id] = time.time()
                        env = new_env
                    logger.info("%s environment ready for task %s", env_type, effective_task_id[:8])

        # Check for dangerous commands (only for local/ssh in interactive modes)
        # Skip check if force=True (user has confirmed they want to run it)
        if not force:
            approval = _check_dangerous_command(command, env_type)
            if not approval["approved"]:
                # Check if this is an approval_required (gateway ask mode)
                if approval.get("status") == "approval_required":
                    return json.dumps({
                        "output": "",
                        "exit_code": -1,
                        "error": approval.get("message", "Waiting for user approval"),
                        "status": "approval_required",
                        "command": approval.get("command", command),
                        "description": approval.get("description", "dangerous command"),
                        "pattern_key": approval.get("pattern_key", ""),
                    }, ensure_ascii=False)
                # Command was blocked - return informative message
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": approval.get("message", "Command denied - potentially dangerous operation"),
                    "status": "blocked"
                }, ensure_ascii=False)

        # Prepare command for execution
        if background:
            # Spawn a tracked background process via the process registry.
            # For local backends: uses subprocess.Popen with output buffering.
            # For non-local backends: runs inside the sandbox via env.execute().
            from tools.process_registry import process_registry

            session_key = os.getenv("HERMES_SESSION_KEY", "")
            effective_cwd = workdir or cwd
            try:
                if env_type == "local":
                    proc_session = process_registry.spawn_local(
                        command=command,
                        cwd=effective_cwd,
                        task_id=effective_task_id,
                        session_key=session_key,
                        env_vars=env.env if hasattr(env, 'env') else None,
                        use_pty=pty,
                    )
                else:
                    proc_session = process_registry.spawn_via_env(
                        env=env,
                        command=command,
                        cwd=effective_cwd,
                        task_id=effective_task_id,
                        session_key=session_key,
                    )

                result_data = {
                    "output": "Background process started",
                    "session_id": proc_session.id,
                    "pid": proc_session.pid,
                    "exit_code": 0,
                    "error": None,
                }

                # Transparent timeout clamping note
                max_timeout = effective_timeout
                if timeout and timeout > max_timeout:
                    result_data["timeout_note"] = (
                        f"Requested timeout {timeout}s was clamped to "
                        f"configured limit of {max_timeout}s"
                    )

                # Register check_interval watcher (gateway picks this up after agent run)
                if check_interval and background:
                    effective_interval = max(30, check_interval)
                    if check_interval < 30:
                        result_data["check_interval_note"] = (
                            f"Requested {check_interval}s raised to minimum 30s"
                        )
                    process_registry.pending_watchers.append({
                        "session_id": proc_session.id,
                        "check_interval": effective_interval,
                        "session_key": session_key,
                        "platform": os.getenv("HERMES_SESSION_PLATFORM", ""),
                        "chat_id": os.getenv("HERMES_SESSION_CHAT_ID", ""),
                    })

                return json.dumps(result_data, ensure_ascii=False)
            except Exception as e:
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": f"Failed to start background process: {str(e)}"
                }, ensure_ascii=False)
        else:
            # Run foreground command with retry logic
            max_retries = 3
            retry_count = 0
            result = None
            
            while retry_count <= max_retries:
                try:
                    execute_kwargs = {"timeout": effective_timeout}
                    if workdir:
                        execute_kwargs["cwd"] = workdir
                    result = env.execute(command, **execute_kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    if "timeout" in error_str:
                        return json.dumps({
                            "output": "",
                            "exit_code": 124,
                            "error": f"Command timed out after {effective_timeout} seconds"
                        }, ensure_ascii=False)
                    
                    # Retry on transient errors
                    if retry_count < max_retries:
                        retry_count += 1
                        wait_time = 2 ** retry_count
                        logger.warning("Execution error, retrying in %ds (attempt %d/%d) - Command: %s - Error: %s: %s - Task: %s, Backend: %s",
                                       wait_time, retry_count, max_retries, command[:200], type(e).__name__, e, effective_task_id, env_type)
                        time.sleep(wait_time)
                        continue
                    
                    logger.error("Execution failed after %d retries - Command: %s - Error: %s: %s - Task: %s, Backend: %s",
                                 max_retries, command[:200], type(e).__name__, e, effective_task_id, env_type)
                    return json.dumps({
                        "output": "",
                        "exit_code": -1,
                        "error": f"Command execution failed: {type(e).__name__}: {str(e)}"
                    }, ensure_ascii=False)
                
                # Got a result
                break
            
            # Extract output
            output = result.get("output", "")
            returncode = result.get("returncode", 0)
            
            # Add helpful message for sudo failures in messaging context
            output = _handle_sudo_failure(output, env_type)
            
            # Truncate output if too long
            MAX_OUTPUT_CHARS = 50000
            if len(output) > MAX_OUTPUT_CHARS:
                truncated_notice = f"\n\n... [OUTPUT TRUNCATED - showing last {MAX_OUTPUT_CHARS} chars of {len(output)} total] ..."
                output = truncated_notice + output[-MAX_OUTPUT_CHARS:]

            return json.dumps({
                "output": output.strip() if output else "",
                "exit_code": returncode,
                "error": None
            }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "output": "",
            "exit_code": -1,
            "error": f"Failed to execute command: {str(e)}",
            "status": "error"
        }, ensure_ascii=False)


def check_terminal_requirements() -> bool:
    """Check if all requirements for the terminal tool are met."""
    config = _get_env_config()
    env_type = config["env_type"]
    
    try:
        if env_type == "local":
            from minisweagent.environments.local import LocalEnvironment
            return True
        elif env_type == "docker":
            from minisweagent.environments.docker import DockerEnvironment
            # Check if docker is available
            import subprocess
            result = subprocess.run(["docker", "version"], capture_output=True, timeout=5)
            return result.returncode == 0
        elif env_type == "singularity":
            from minisweagent.environments.singularity import SingularityEnvironment
            # Check if singularity/apptainer is available
            import subprocess
            import shutil
            executable = shutil.which("apptainer") or shutil.which("singularity")
            if executable:
                result = subprocess.run([executable, "--version"], capture_output=True, timeout=5)
                return result.returncode == 0
            return False
        elif env_type == "modal":
            from minisweagent.environments.extra.swerex_modal import SwerexModalEnvironment
            # Check for modal token
            return os.getenv("MODAL_TOKEN_ID") is not None or Path.home().joinpath(".modal.toml").exists()
        else:
            return False
    except Exception as e:
        logger.error("Terminal requirements check failed: %s", e)
        return False


if __name__ == "__main__":
    # Simple test when run directly
    print("Terminal Tool Module (mini-swe-agent backend)")
    print("=" * 50)
    
    config = _get_env_config()
    print(f"\nCurrent Configuration:")
    print(f"  Environment type: {config['env_type']}")
    print(f"  Docker image: {config['docker_image']}")
    print(f"  Modal image: {config['modal_image']}")
    print(f"  Working directory: {config['cwd']}")
    print(f"  Default timeout: {config['timeout']}s")
    print(f"  Lifetime: {config['lifetime_seconds']}s")

    if not check_terminal_requirements():
        print("\n❌ Requirements not met. Please check the messages above.")
        exit(1)

    print("\n✅ All requirements met!")
    print("\nAvailable Tool:")
    print("  - terminal_tool: Execute commands using mini-swe-agent environments")

    print("\nUsage Examples:")
    print("  # Execute a command")
    print("  result = terminal_tool(command='ls -la')")
    print("  ")
    print("  # Run a background task")
    print("  result = terminal_tool(command='python server.py', background=True)")

    print("\nEnvironment Variables:")
    default_img = "nikolaik/python-nodejs:python3.11-nodejs20"
    print(f"  TERMINAL_ENV: {os.getenv('TERMINAL_ENV', 'local')} (local/docker/singularity/modal/ssh)")
    print(f"  TERMINAL_DOCKER_IMAGE: {os.getenv('TERMINAL_DOCKER_IMAGE', default_img)}")
    print(f"  TERMINAL_SINGULARITY_IMAGE: {os.getenv('TERMINAL_SINGULARITY_IMAGE', f'docker://{default_img}')}")
    print(f"  TERMINAL_MODAL_IMAGE: {os.getenv('TERMINAL_MODAL_IMAGE', default_img)}")
    print(f"  TERMINAL_CWD: {os.getenv('TERMINAL_CWD', os.getcwd())}")
    print(f"  TERMINAL_TIMEOUT: {os.getenv('TERMINAL_TIMEOUT', '60')}")
    print(f"  TERMINAL_LIFETIME_SECONDS: {os.getenv('TERMINAL_LIFETIME_SECONDS', '300')}")
