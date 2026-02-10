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
import os
import sys
import time
import threading
import atexit
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, ClassVar, List

try:
    import yaml
except ImportError:
    yaml = None

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
        if not os.getenv("HERMES_QUIET"):
            print(f"[Terminal] Using /scratch for sandboxes: {user_scratch}")
        return user_scratch
    
    # Fall back to /tmp
    if not os.getenv("HERMES_QUIET"):
        print("[Terminal] Warning: /scratch not available, using /tmp (limited space)")
    return Path(tempfile.gettempdir())


def _get_apptainer_cache_dir() -> Path:
    """Get the Apptainer cache directory for SIF images."""
    # Check for APPTAINER_CACHEDIR env var
    cache_dir = os.getenv("APPTAINER_CACHEDIR")
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path
    
    # Use scratch dir parent for cache (one level up from sandboxes)
    scratch = _get_scratch_dir()
    cache_path = scratch.parent / ".apptainer"
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
        
        print(f"[Terminal] Building SIF image (one-time setup)...")
        print(f"[Terminal]   Source: {image}")
        print(f"[Terminal]   Target: {sif_path}")
        
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
                print(f"[Terminal] ⚠️ SIF build failed, falling back to docker:// URL")
                print(f"[Terminal]   Error: {result.stderr[:500]}")
                return image
            
            print(f"[Terminal] ✅ SIF image built successfully")
            return str(sif_path)
            
        except subprocess.TimeoutExpired:
            print(f"[Terminal] ⚠️ SIF build timed out, falling back to docker:// URL")
            # Clean up partial file
            if sif_path.exists():
                sif_path.unlink()
            return image
        except Exception as e:
            print(f"[Terminal] ⚠️ SIF build error: {e}, falling back to docker:// URL")
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
                    except:
                        pass
        
        total_gb = total_bytes / (1024 ** 3)
        
        if total_gb > DISK_USAGE_WARNING_THRESHOLD_GB:
            print(f"⚠️  [Terminal] WARNING: Disk usage ({total_gb:.1f}GB) exceeds threshold ({DISK_USAGE_WARNING_THRESHOLD_GB}GB)")
            print(f"    Consider running cleanup_all_environments() or reducing parallel workers")
            return True
        
        return False
    except Exception as e:
        return False


# Session-cached sudo password (persists until CLI exits)
_cached_sudo_password: str = ""

# =============================================================================
# Dangerous Command Approval System
# =============================================================================

# Session-cached dangerous command approvals (pattern -> approved)
_session_approved_patterns: set = set()

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
        print(f"  ⚠️ Could not save allowlist: {e}")


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
    
    Returns: 'once', 'session', 'always', or 'deny'
    """
    import sys
    import threading
    
    # Pause spinner if one is running
    os.environ["HERMES_SPINNER_PAUSE"] = "1"
    
    try:
        # Use simple ASCII art for compatibility (no ANSI color codes)
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
            except:
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
    
    if is_gateway:
        # Messaging context - return informative denial, agent should ask user
        return {
            "approved": False,
            "pattern_key": pattern_key,
            "message": f"BLOCKED: This command is potentially dangerous ({description}). Tell the user and ask if they want to add this command pattern to their allowlist. They can do this via 'hermes config edit' or by running the command directly on their machine."
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
    Uses getpass for hidden input with threading for timeout support.
    """
    import getpass
    import sys
    import time as time_module
    
    # ANSI escape codes for terminal control
    CLEAR_LINE = "\033[2K"      # Clear entire line
    CURSOR_START = "\r"         # Move cursor to start of line
    
    # Result container for thread
    result = {"password": None, "done": False}
    
    def get_password_thread():
        """Thread function to get password with getpass (hidden input)."""
        try:
            result["password"] = getpass.getpass("  Password (hidden): ")
        except (EOFError, KeyboardInterrupt):
            result["password"] = ""
        except Exception:
            result["password"] = ""
        finally:
            result["done"] = True
    
    try:
        # Pause the spinner animation while prompting for password
        os.environ["HERMES_SPINNER_PAUSE"] = "1"
        time_module.sleep(0.2)  # Give spinner time to pause
        
        # Clear any spinner/animation on current line
        sys.stdout.write(CURSOR_START + CLEAR_LINE)
        sys.stdout.flush()
        
        # Print a clear visual break with empty lines for separation
        print("\n")  # Extra spacing
        print("┌" + "─" * 58 + "┐")
        print("│  🔐 SUDO PASSWORD REQUIRED" + " " * 30 + "│")
        print("├" + "─" * 58 + "┤")
        print("│  Enter password below (input is hidden), or:            │")
        print("│    • Press Enter to skip (command fails gracefully)     │")
        print(f"│    • Wait {timeout_seconds}s to auto-skip" + " " * 27 + "│")
        print("└" + "─" * 58 + "┘")
        print()
        sys.stdout.flush()
        
        # Start password input in a thread so we can timeout
        password_thread = threading.Thread(target=get_password_thread, daemon=True)
        password_thread.start()
        
        # Wait for either completion or timeout
        password_thread.join(timeout=timeout_seconds)
        
        if result["done"]:
            # Got input (or user pressed Enter/Ctrl+C)
            password = result["password"] or ""
            if password:
                print("  ✓ Password received (cached for this session)")
            else:
                print("  ⏭ Skipped - continuing without sudo")
            print()
            sys.stdout.flush()
            return password
        else:
            # Timeout - thread is still waiting for input
            print("\n  ⏱ Timeout - continuing without sudo")
            print("    (Press Enter to dismiss the password prompt)")
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
        # Always resume the spinner when done
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
    
    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict:
        """Execute a command locally with sudo support."""
        work_dir = cwd or self.cwd or os.getcwd()
        effective_timeout = timeout or self.timeout
        
        # Transform sudo commands if SUDO_PASSWORD is available
        exec_command = _transform_sudo_command(command)
        
        try:
            result = subprocess.run(
                exec_command,
                shell=True,
                text=True,
                cwd=work_dir,
                env=os.environ | self.env,
                timeout=effective_timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,  # Prevent hanging on interactive prompts
            )
            return {"output": result.stdout, "returncode": result.returncode}
        except subprocess.TimeoutExpired:
            return {"output": f"Command timed out after {effective_timeout}s", "returncode": 124}
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
    Custom Singularity/Apptainer environment with better space management.
    
    - Automatically builds/caches SIF images from docker:// URLs
    - Builds sandbox in /scratch (if available) or configurable location
    - Binds a large working directory into the container
    - Keeps container isolated from host filesystem
    """
    
    def __init__(self, image: str, cwd: str = "/workspace", timeout: int = 60):
        self.cwd = cwd
        self.timeout = timeout
        
        # Use apptainer if available, otherwise singularity
        self.executable = "apptainer" if shutil.which("apptainer") else "singularity"
        
        # Get or build SIF from docker:// URL (fast if already cached)
        self.image = _get_or_build_sif(image, self.executable)
        
        # Get scratch directory for sandbox
        self.scratch_dir = _get_scratch_dir()
        
        # Create unique sandbox directory
        self.sandbox_id = f"hermes-{uuid.uuid4().hex[:12]}"
        self.sandbox_dir = self.scratch_dir / self.sandbox_id
        
        # Create a working directory that will be bound into the container
        self.work_dir = self.scratch_dir / f"{self.sandbox_id}-work"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Build the sandbox
        self._build_sandbox()
    
    def _build_sandbox(self):
        """Build a writable sandbox from the container image (SIF or other)."""
        try:
            result = subprocess.run(
                [self.executable, "build", "--sandbox", str(self.sandbox_dir), self.image],
                capture_output=True,
                text=True,
                timeout=300  # 5 min timeout for building
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to build sandbox: {result.stderr}")
            
            # Create /workspace directory inside the sandbox for bind mounting
            workspace_in_sandbox = self.sandbox_dir / "workspace"
            workspace_in_sandbox.mkdir(parents=True, exist_ok=True)
            
        except subprocess.TimeoutExpired:
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)
            raise RuntimeError("Sandbox build timed out")
    
    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict:
        """Execute a command in the Singularity container."""
        cmd = [self.executable, "exec"]
        
        # Isolation flags - contain but allow network
        cmd.extend(["--contain", "--cleanenv"])
        
        # Bind the working directory into the container at /workspace
        # This gives the container access to a large writable space
        cmd.extend(["--bind", f"{self.work_dir}:/workspace"])
        
        # Also bind it to /tmp inside container for pip cache etc.
        cmd.extend(["--bind", f"{self.work_dir}:/tmp"])
        
        # Set working directory
        work_dir = cwd or self.cwd
        cmd.extend(["--pwd", work_dir])
        
        # Use writable sandbox
        cmd.extend(["--writable", str(self.sandbox_dir)])
        
        # Transform sudo commands if SUDO_PASSWORD is available
        exec_command = _transform_sudo_command(command)
        
        # Execute the command
        cmd.extend(["bash", "-c", exec_command])
        
        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=timeout or self.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,  # Prevent hanging on interactive prompts
            )
            return {"output": result.stdout, "returncode": result.returncode}
        except subprocess.TimeoutExpired:
            return {"output": f"Command timed out after {timeout or self.timeout}s", "returncode": 124}
    
    def cleanup(self):
        """Clean up sandbox and working directory."""
        shutil.rmtree(self.sandbox_dir, ignore_errors=True)
        shutil.rmtree(self.work_dir, ignore_errors=True)
    
    def stop(self):
        """Alias for cleanup."""
        self.cleanup()
    
    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup()


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
    
    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict:
        """Execute a command on the remote host via SSH."""
        work_dir = cwd or self.cwd
        effective_timeout = timeout or self.timeout
        
        # Transform sudo commands if SUDO_PASSWORD is available
        exec_command = _transform_sudo_command(command)
        
        # Wrap command to run in the correct directory
        # Use bash -c to handle complex commands properly
        wrapped_command = f'cd {work_dir} && {exec_command}'
        
        cmd = self._build_ssh_command()
        cmd.extend(["bash", "-c", wrapped_command])
        
        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=effective_timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,  # Prevent hanging on interactive prompts
            )
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
            except:
                pass
            
            # Remove socket file
            try:
                self.control_socket.unlink()
            except:
                pass
    
    def stop(self):
        """Alias for cleanup."""
        self.cleanup()
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except:
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
    
    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict:
        """Execute a command in the Docker container with sudo support."""
        # Transform sudo commands if SUDO_PASSWORD is available
        exec_command = _transform_sudo_command(command)
        
        work_dir = cwd or self.cwd
        effective_timeout = timeout or self.timeout
        
        # Get container_id from inner environment
        assert self._inner.container_id, "Container not started"
        
        cmd = [self._inner.config.executable, "exec", "-w", work_dir]
        for key in self._inner.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self._inner.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self._inner.container_id, "bash", "-lc", exec_command])
        
        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=effective_timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,  # Prevent hanging on interactive prompts
            )
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
        except:
            pass


            pass



@dataclass
class ModalProfile:
    """
    Configuration for a Modal sandbox profile.
    
    Each profile defines the container image, resources, and pool scaling behavior.
    Different profiles can be used for different workloads
    
    Secrets:
        secrets: List of Modal Secret names to inject into the sandbox.
                 These secrets must be created on Modal dashboard or via CLI.
        
        env_vars: Dict of environment variables to pass directly to sandbox.
                  Use for non-sensitive configuration.
                  Example: {"DEBUG": "1", "LOG_LEVEL": "info"}
        
        use_dotenv: loads local dotenv
    """
    name: str
    image: str = "python:3.11"
    gpu: Optional[str] = None           # None, "T4", "A10G", "A100", "H100"
    cpu: float = 1.0
    memory: int = 2048                  # MB
    min_pool: int = 1
    max_pool: int = 5
    idle_timeout: int = 120             # Modal server-side auto-cleanup (seconds)
    max_lifetime: int = 3600            # Max sandbox lifetime (seconds)
    scale_down_idle: int = 180          # Client-side scale down threshold (seconds)
    workdir: str = "/workspace"
    # Secrets and environment variables
    secrets: List[str] = field(default_factory=list)  # Modal Secret names
    env_vars: Dict[str, str] = field(default_factory=dict)  # Direct env vars
    use_dotenv: bool = False            # Load .env file and pass to sandbox
    
    @classmethod
    def from_env(cls, profile_name: str) -> "ModalProfile":
        """Load profile configuration from environment variables."""
        prefix = f"TERMINAL_MODAL_PROFILE_{profile_name}_"
        
        # Parse secrets list from comma-separated string
        secrets_str = os.getenv(f"{prefix}SECRETS", "")
        secrets = [s.strip() for s in secrets_str.split(",") if s.strip()]
        
        # Parse env_vars from KEY=VALUE pairs separated by semicolons
        env_vars_str = os.getenv(f"{prefix}ENV_VARS", "")
        env_vars = {}
        if env_vars_str:
            for pair in env_vars_str.split(";"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    env_vars[k.strip()] = v.strip()
        
        return cls(
            name=profile_name,
            image=os.getenv(f"{prefix}IMAGE", "python:3.11"),
            gpu=os.getenv(f"{prefix}GPU"),
            cpu=float(os.getenv(f"{prefix}CPU", "1.0")),
            memory=int(os.getenv(f"{prefix}MEMORY", "2048")),
            min_pool=int(os.getenv(f"{prefix}MIN_POOL", "1")),
            max_pool=int(os.getenv(f"{prefix}MAX_POOL", "5")),
            idle_timeout=int(os.getenv(f"{prefix}IDLE_TIMEOUT", "120")),
            max_lifetime=int(os.getenv(f"{prefix}MAX_LIFETIME", "3600")),
            scale_down_idle=int(os.getenv(f"{prefix}SCALE_DOWN_IDLE", "180")),
            workdir=os.getenv(f"{prefix}WORKDIR", "/workspace"),
            secrets=secrets,
            env_vars=env_vars,
            use_dotenv=os.getenv(f"{prefix}USE_DOTENV", "").lower() in ("true", "1", "yes"),
        )
    
    @classmethod
    def load_profiles(cls, config_file: Optional[str] = None) -> Dict[str, "ModalProfile"]:
        """
        Load all profiles from YAML file or environment variables.
        
        Priority:
        1. YAML file specified by config_file or TERMINAL_MODAL_PROFILES_FILE
        2. Environment variables with TERMINAL_MODAL_PROFILE_<name>_* pattern
        3. Default profile with basic settings
        """
        profiles = {}
        
        # Try YAML file first
        yaml_path = config_file or os.getenv("TERMINAL_MODAL_PROFILES_FILE", "modal_profiles.yaml")
        if Path(yaml_path).exists():
            try:
                with open(yaml_path) as f:
                    config = yaml.safe_load(f)
                for name, cfg in config.get("profiles", {}).items():
                    profiles[name] = cls(name=name, **cfg)
                if not os.getenv("HERMES_QUIET"):
                    print(f"[Modal] Loaded {len(profiles)} profiles from {yaml_path}")
                return profiles
            except Exception as e:
                if not os.getenv("HERMES_QUIET"):
                    print(f"[Modal] Warning: Failed to load {yaml_path}: {e}")
        
        # Check for environment variable profiles
        # Look for any env vars starting with TERMINAL_MODAL_PROFILE_
        profile_names = set()
        for key in os.environ:
            if key.startswith("TERMINAL_MODAL_PROFILE_") and "_IMAGE" in key:
                # Extract profile name: TERMINAL_MODAL_PROFILE_<name>_IMAGE
                parts = key.replace("TERMINAL_MODAL_PROFILE_", "").rsplit("_IMAGE", 1)
                if parts[0]:
                    profile_names.add(parts[0])
        
        for name in profile_names:
            profiles[name] = cls.from_env(name)
        
        # If no profiles found, create a default one
        if not profiles:
            default_name = os.getenv("TERMINAL_MODAL_DEFAULT_PROFILE", "default")
            profiles[default_name] = cls(
                name=default_name,
                image=os.getenv("TERMINAL_MODAL_IMAGE", "python:3.11"),
                min_pool=int(os.getenv("TERMINAL_MODAL_MIN_POOL", "1")),
                max_pool=int(os.getenv("TERMINAL_MODAL_MAX_POOL", "5")),
                idle_timeout=int(os.getenv("TERMINAL_MODAL_IDLE_TIMEOUT", "120")),
                max_lifetime=int(os.getenv("TERMINAL_MODAL_MAX_LIFETIME", "3600")),
                scale_down_idle=int(os.getenv("TERMINAL_MODAL_SCALE_DOWN_IDLE", "180")),
            )
        
        return profiles


class _ModalSandboxPool:
    """
    Auto-scaling pool of warm Modal sandboxes for a single profile.
    
    Features:
    - Named sandboxes for recovery after restart
    - Reactive scale-up when demand exceeds capacity
    - Background scale-down when sandboxes are idle
    - Server-side idle_timeout for orphan protection
    """
    
    def __init__(self, profile: ModalProfile, app_name: str):
        self.profile = profile
        self.app_name = app_name
        self._app = None
        self._modal_image = None
        self._pool: Dict[str, Any] = {}          # sandbox_name -> modal.Sandbox
        self._in_use: Dict[str, str] = {}        # task_id -> sandbox_name
        self._last_used: Dict[str, float] = {}   # sandbox_name -> timestamp
        self._lock = threading.Lock()
        self._running = True
        self._next_index = 0
        
        # Start scale-down monitor if min_pool > 0 (worth keeping warm)
        self._monitor_thread = None
        if profile.min_pool > 0 or profile.max_pool > 0:
            self._monitor_thread = threading.Thread(
                target=self._scale_down_monitor, 
                daemon=True,
                name=f"modal-pool-{profile.name}"
            )
            self._monitor_thread.start()
    
    def _get_sandbox_name(self, index: int) -> str:
        """Generate a unique sandbox name for this profile."""
        return f"hermes-{self.profile.name}-{index}"
    
    def _ensure_app(self):
        """Lazy initialization of Modal app and image."""
        if self._app is None:
            try:
                import modal
                self._app = modal.App.lookup(self.app_name, create_if_missing=True)
                self._modal_image = modal.Image.from_registry(self.profile.image)
            except ImportError:
                raise ImportError("Modal package not installed. Run: pip install modal")
    
    def _recover_or_create_sandbox(self, name: str) -> Any:
        """
        Try to recover an existing named sandbox, or create a new one.
        
        Uses Modal's named sandbox feature for recovery after Hermes restart.
        Supports Modal Secrets for secure credential injection.
        """
        import modal
        
        # Try to recover existing sandbox
        try:
            sb = modal.Sandbox.from_name(self.app_name, name)
            if sb.poll() is None:  # Still running
                # Health check - verify sandbox is responsive
                try:
                    sb.exec("echo", "ok", timeout=10)
                    if not os.getenv("HERMES_QUIET"):
                        print(f"[Modal] Recovered existing sandbox: {name}")
                    return sb
                except Exception:
                    # Sandbox is not healthy, will create new
                    pass
        except modal.exception.NotFoundError:
            pass
        except Exception as e:
            if not os.getenv("HERMES_QUIET"):
                print(f"[Modal] Could not recover sandbox {name}: {e}")
        
        # Build create kwargs based on profile
        create_kwargs = {
            "app": self._app,
            "name": name,
            "image": self._modal_image,
            "timeout": self.profile.max_lifetime,
            "idle_timeout": self.profile.idle_timeout,
            "workdir": self.profile.workdir,
        }
        
        # Add resource specs
        if self.profile.cpu != 1.0:
            create_kwargs["cpu"] = self.profile.cpu
        if self.profile.memory != 2048:
            create_kwargs["memory"] = self.profile.memory
        
        # Add GPU if specified
        if self.profile.gpu:
            create_kwargs["gpu"] = self.profile.gpu
        
        # Build secrets list
        secrets_list = []
        
        # Add named secrets from Modal dashboard/CLI
        for secret_name in self.profile.secrets:
            try:
                secrets_list.append(modal.Secret.from_name(secret_name))
                if not os.getenv("HERMES_QUIET"):
                    print(f"[Modal] Adding secret: {secret_name}")
            except Exception as e:
                if not os.getenv("HERMES_QUIET"):
                    print(f"[Modal] Warning: Could not load secret '{secret_name}': {e}")
        
        # Add direct environment variables
        if self.profile.env_vars:
            secrets_list.append(modal.Secret.from_dict(self.profile.env_vars))
        
        # Add .env file if requested
        if self.profile.use_dotenv:
            try:
                secrets_list.append(modal.Secret.from_dotenv())
                if not os.getenv("HERMES_QUIET"):
                    print(f"[Modal] Loading .env file into sandbox")
            except Exception as e:
                if not os.getenv("HERMES_QUIET"):
                    print(f"[Modal] Warning: Could not load .env file: {e}")
        
        # Add global secrets from environment variable
        global_secrets_str = os.getenv("TERMINAL_MODAL_SECRETS", "")
        if global_secrets_str:
            for secret_name in global_secrets_str.split(","):
                secret_name = secret_name.strip()
                if secret_name and secret_name not in self.profile.secrets:
                    try:
                        secrets_list.append(modal.Secret.from_name(secret_name))
                    except Exception as e:
                        if not os.getenv("HERMES_QUIET"):
                            print(f"[Modal] Warning: Could not load global secret '{secret_name}': {e}")
        
        if secrets_list:
            create_kwargs["secrets"] = secrets_list
        
        if not os.getenv("HERMES_QUIET"):
            gpu_str = f" with GPU={self.profile.gpu}" if self.profile.gpu else ""
            secrets_str = f" with {len(secrets_list)} secret(s)" if secrets_list else ""
            print(f"[Modal] Creating sandbox: {name}{gpu_str}{secrets_str}")
        
        return modal.Sandbox.create(**create_kwargs)
    
    def _find_available_slot(self) -> Optional[str]:
        """Find an available sandbox in the pool (not currently in use)."""
        in_use_names = set(self._in_use.values())
        for name in self._pool:
            if name not in in_use_names:
                # Verify sandbox is still running
                try:
                    if self._pool[name].poll() is None:
                        return name
                    else:
                        # Sandbox died, remove it
                        del self._pool[name]
                        self._last_used.pop(name, None)
                except:
                    pass
        return None
    
    def _current_size(self) -> int:
        """Get current pool size."""
        return len(self._pool)
    
    def acquire(self, task_id: str, timeout: float = 60.0) -> Any:
        """
        Acquire a sandbox for a task.
        
        - Returns existing sandbox if task already has one
        - Finds available sandbox in pool if any
        - Scales up if under max_pool and all busy
        - Waits if at max_pool and all busy
        """
        deadline = time.time() + timeout
        
        while True:
            with self._lock:
                # Task already has a sandbox?
                if task_id in self._in_use:
                    name = self._in_use[task_id]
                    self._last_used[name] = time.time()
                    return self._pool[name]
                
                self._ensure_app()
                
                # Find available slot in pool
                available = self._find_available_slot()
                if available:
                    self._in_use[task_id] = available
                    self._last_used[available] = time.time()
                    return self._pool[available]
                
                # Scale up if under max
                if self._current_size() < self.profile.max_pool:
                    name = self._get_sandbox_name(self._next_index)
                    self._next_index += 1
                    try:
                        sb = self._recover_or_create_sandbox(name)
                        self._pool[name] = sb
                        self._in_use[task_id] = name
                        self._last_used[name] = time.time()
                        return sb
                    except Exception as e:
                        if not os.getenv("HERMES_QUIET"):
                            print(f"[Modal] Failed to create sandbox: {e}")
                        raise
            
            # At capacity - wait and retry
            if time.time() > deadline:
                raise TimeoutError(
                    f"No Modal sandbox available for profile '{self.profile.name}' "
                    f"within {timeout}s (pool size: {self._current_size()}/{self.profile.max_pool})"
                )
            time.sleep(0.5)
    
    def release(self, task_id: str, terminate: bool = False):
        """
        Release a sandbox back to the pool.
        
        If terminate=False, sandbox stays warm for reuse.
        If terminate=True, sandbox is terminated immediately.
        """
        with self._lock:
            if task_id not in self._in_use:
                return
            
            name = self._in_use.pop(task_id)
            self._last_used[name] = time.time()
            
            if terminate:
                self._terminate_sandbox(name)
    
    def _terminate_sandbox(self, name: str, during_shutdown: bool = False):
        """Terminate and remove a sandbox from the pool."""
        if name in self._pool:
            try:
                self._pool[name].terminate()
                if not os.getenv("HERMES_QUIET"):
                    print(f"[Modal] Terminated sandbox: {name}")
            except Exception as e:
                if not during_shutdown and not os.getenv("HERMES_QUIET"):
                    print(f"[Modal] Error terminating {name}: {e}")
            del self._pool[name]
            self._last_used.pop(name, None)
    
    def _scale_down_monitor(self):
        """Background thread: terminate idle sandboxes above min_pool size."""
        while self._running:
            time.sleep(30)  # Check every 30 seconds
            
            with self._lock:
                if self._current_size() <= self.profile.min_pool:
                    continue
                
                now = time.time()
                in_use_names = set(self._in_use.values())
                
                # Find idle sandboxes to terminate
                to_terminate = []
                for name, last_used in list(self._last_used.items()):
                    if name in in_use_names:
                        continue
                    if now - last_used > self.profile.scale_down_idle:
                        # Don't go below min_pool
                        if self._current_size() - len(to_terminate) > self.profile.min_pool:
                            to_terminate.append(name)
                
                for name in to_terminate:
                    if not os.getenv("HERMES_QUIET"):
                        print(f"[Modal] Scaling down idle sandbox: {name}")
                    self._terminate_sandbox(name)
    
    def shutdown(self, during_shutdown: bool = False):
        """Stop monitor thread and terminate all sandboxes."""
        self._running = False
        with self._lock:
            for name in list(self._pool.keys()):
                self._terminate_sandbox(name, during_shutdown=during_shutdown)


class _ModalPoolManager:
    """
    Manages multiple sandbox pools, one per profile.
    
    Singleton pattern - shared across all _ModalSandboxEnvironment instances.
    Each profile has its own pool with independent scaling.
    """
    
    _instance: ClassVar[Optional["_ModalPoolManager"]] = None
    _init_lock: ClassVar[threading.Lock] = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> "_ModalPoolManager":
        """Get or create the singleton instance."""
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton (for testing)."""
        with cls._init_lock:
            if cls._instance is not None:
                cls._instance.shutdown()
                cls._instance = None
    
    def __init__(self):
        self.app_name = os.getenv("TERMINAL_MODAL_APP_NAME", "hermes-sandbox")
        self.profiles = ModalProfile.load_profiles()
        self.default_profile = os.getenv("TERMINAL_MODAL_DEFAULT_PROFILE", "default")
        
        # Fall back to first profile if default not found
        if self.default_profile not in self.profiles and self.profiles:
            self.default_profile = next(iter(self.profiles.keys()))
        
        self._pools: Dict[str, _ModalSandboxPool] = {}
        self._pools_lock = threading.Lock()
        
        if not os.getenv("HERMES_QUIET"):
            print(f"[Modal] Pool manager initialized with profiles: {list(self.profiles.keys())}")
            print(f"[Modal] Default profile: {self.default_profile}")
    
    def _get_pool(self, profile_name: str) -> _ModalSandboxPool:
        """Get or create a pool for a profile."""
        with self._pools_lock:
            if profile_name not in self._pools:
                if profile_name not in self.profiles:
                    available = list(self.profiles.keys())
                    raise ValueError(
                        f"Unknown Modal profile: '{profile_name}'. "
                        f"Available profiles: {available}"
                    )
                profile = self.profiles[profile_name]
                self._pools[profile_name] = _ModalSandboxPool(profile, self.app_name)
            return self._pools[profile_name]
    
    def acquire(self, task_id: str, profile: Optional[str] = None, timeout: float = 60.0) -> Any:
        """Acquire a sandbox from the appropriate profile's pool."""
        profile_name = profile or self.default_profile
        return self._get_pool(profile_name).acquire(task_id, timeout=timeout)
    
    def release(self, task_id: str, profile: Optional[str] = None, terminate: bool = False):
        """Release a sandbox back to its pool."""
        profile_name = profile or self.default_profile
        if profile_name in self._pools:
            self._pools[profile_name].release(task_id, terminate=terminate)
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all pools."""
        status = {}
        with self._pools_lock:
            for name, pool in self._pools.items():
                with pool._lock:
                    status[name] = {
                        "pool_size": pool._current_size(),
                        "in_use": len(pool._in_use),
                        "max_pool": pool.profile.max_pool,
                        "min_pool": pool.profile.min_pool,
                    }
        return status
    
    def shutdown(self, during_shutdown: bool = False):
        """Shutdown all pools."""
        with self._pools_lock:
            for pool in self._pools.values():
                pool.shutdown(during_shutdown=during_shutdown)
            self._pools.clear()


class _ModalSandboxEnvironment:
    """
    Modal Sandbox environment with profile-based pool management.
    
    Features:
    - Profile selection for heterogeneous workloads
    - Auto-scaling warm sandbox pool
    - Named sandbox recovery
    - SUDO_PASSWORD support
    """
    
    def __init__(
        self,
        image: str,                     # Used only if no profile config
        cwd: str = "/workspace",
        timeout: int = 60,
        task_id: str = "",
        profile: Optional[str] = None,  # Profile name (e.g., "pytorch-gpu")
    ):
        self.cwd = cwd
        self.timeout = timeout
        self.task_id = task_id or str(uuid.uuid4())
        self.profile = profile
        self._released = False
        
        # Acquire sandbox from pool
        manager = _ModalPoolManager.get_instance()
        self._sandbox = manager.acquire(self.task_id, profile=profile)
    
    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict:
        """Execute a command in the Modal sandbox."""
        # Transform sudo commands if SUDO_PASSWORD is available
        exec_command = _transform_sudo_command(command)
        work_dir = cwd or self.cwd
        
        try:
            # Run command via bash with proper working directory
            process = self._sandbox.exec(
                "bash", "-c", f"cd {work_dir} && {exec_command}",
                timeout=timeout or self.timeout
            )
            
            # Read output
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            process.wait()
            
            # Combine stdout and stderr
            output = stdout
            if stderr:
                output = output + stderr if output else stderr
            
            return {"output": output, "returncode": process.returncode}
            
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                return {"output": f"Command timed out after {timeout or self.timeout}s", "returncode": 124}
            return {"output": f"Modal execution error: {error_msg}", "returncode": 1}
    
    def cleanup(self):
        """Release sandbox back to pool (stays warm for reuse)."""
        if not self._released:
            self._released = True
            _ModalPoolManager.get_instance().release(
                self.task_id, 
                profile=self.profile, 
                terminate=False
            )
    
    def stop(self):
        """Terminate this sandbox explicitly."""
        if not self._released:
            self._released = True
            _ModalPoolManager.get_instance().release(
                self.task_id, 
                profile=self.profile, 
                terminate=True
            )
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except:
            pass


# =============================================================================
# Slot Pool Environment — routes through atropos/backends/ for multiplexed
# sandbox execution. Supports Modal, Nomad (Docker + Singularity/Apptainer).
#
# Usage: TERMINAL_ENV=slot_pool  TERMINAL_SLOT_BACKEND=modal
# =============================================================================

class _SlotPoolAsyncWorker:
    """Background thread with its own event loop for running async backend ops."""

    def __init__(self):
        self._loop = None
        self._thread = None

    def start(self):
        import asyncio as _aio
        self._loop = _aio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import asyncio as _aio
        _aio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro, timeout=300):
        """Run an async coroutine synchronously on the worker thread."""
        import asyncio as _aio
        if self._loop is None or self._thread is None:
            raise RuntimeError("SlotPoolAsyncWorker not started")
        future = _aio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def stop(self):
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)


class _SlotPoolManager:
    """
    Singleton manager for the slot-pool sandbox backend.

    Wraps atropos/backends/ (ModalToolBackend or NomadToolBackend) and provides
    synchronous acquire/execute/release operations via a background async worker.

    Config via environment variables:
        TERMINAL_SLOT_BACKEND     = modal | nomad  (default: modal)
        # Modal settings (reuses TERMINAL_MODAL_* vars):
        TERMINAL_MODAL_IMAGE      = python:3.11
        TERMINAL_MODAL_SLOTS      = 10
        TERMINAL_MODAL_MIN        = 1
        TERMINAL_MODAL_MAX        = 5
        # Nomad settings:
        TERMINAL_NOMAD_ADDRESS    = http://localhost:4646
        TERMINAL_NOMAD_DRIVER     = docker | singularity
        TERMINAL_NOMAD_IMAGE      = atropos-sandbox:local
    """

    _instance: Optional["_SlotPoolManager"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "_SlotPoolManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance._start()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        with cls._lock:
            if cls._instance is not None:
                cls._instance._stop()
                cls._instance = None

    def __init__(self):
        self._backend = None
        self._worker = _SlotPoolAsyncWorker()
        self._slots: Dict[str, Any] = {}  # task_id → Slot
        self._slot_lock = threading.Lock()
        self._started = False

    def _start(self):
        """Initialize the backend and async worker."""
        self._worker.start()

        backend_type = os.getenv("TERMINAL_SLOT_BACKEND", "modal").strip().lower()
        print(f"[SlotPool] Starting {backend_type} backend...")

        if backend_type == "modal":
            self._backend = self._create_modal_backend()
        elif backend_type == "nomad":
            self._backend = self._create_nomad_backend()
        else:
            raise ValueError(
                f"Unknown TERMINAL_SLOT_BACKEND: {backend_type}. Use 'modal' or 'nomad'."
            )

        self._worker.run(self._backend.start(), timeout=120)
        self._started = True
        print(f"[SlotPool] {backend_type} backend started")

    def _create_modal_backend(self):
        from atropos.backends.modal_backend import ModalSandboxConfig, ModalToolBackend

        config = ModalSandboxConfig(
            name="default",
            app_name=os.getenv("TERMINAL_SLOT_APP_NAME", "hermes-slot-pool"),
            image=os.getenv("TERMINAL_MODAL_IMAGE") or os.getenv("TERMINAL_DOCKER_IMAGE", "python:3.11"),
            gpu=os.getenv("TERMINAL_MODAL_GPU") or None,
            cpu=float(os.getenv("TERMINAL_MODAL_CPU", "1.0")),
            memory=int(os.getenv("TERMINAL_MODAL_MEMORY", "2048")),
            slots_per_sandbox=int(os.getenv("TERMINAL_MODAL_SLOTS", "10")),
            min_sandboxes=int(os.getenv("TERMINAL_MODAL_MIN", "1")),
            max_sandboxes=int(os.getenv("TERMINAL_MODAL_MAX", "5")),
            idle_timeout=int(os.getenv("TERMINAL_MODAL_IDLE_TIMEOUT", "120")),
            max_lifetime=int(os.getenv("TERMINAL_MODAL_MAX_LIFETIME", "3600")),
            acquire_timeout_s=float(os.getenv("TERMINAL_MODAL_ACQUIRE_TIMEOUT", "60.0")),
            execution_timeout_s=float(os.getenv("TERMINAL_MODAL_EXEC_TIMEOUT", "300.0")),
            workspace_base=os.getenv("TERMINAL_MODAL_WORKSPACE", "/data"),
        )
        return ModalToolBackend(config)

    def _create_nomad_backend(self):
        from atropos.backends.nomad_backend import NomadBackendConfig, NomadToolBackend

        config = NomadBackendConfig(
            nomad_address=os.getenv("TERMINAL_NOMAD_ADDRESS", "http://localhost:4646"),
            job_id=os.getenv("TERMINAL_NOMAD_JOB_ID", "hermes-slot-pool"),
            image=os.getenv("TERMINAL_NOMAD_IMAGE") or os.getenv("TERMINAL_DOCKER_IMAGE", "atropos-sandbox:local"),
            driver=os.getenv("TERMINAL_NOMAD_DRIVER", "docker"),
            slots_per_container=int(os.getenv("TERMINAL_NOMAD_SLOTS", "10")),
            min_containers=int(os.getenv("TERMINAL_NOMAD_MIN", "1")),
            max_containers=int(os.getenv("TERMINAL_NOMAD_MAX", "10")),
        )
        return NomadToolBackend(config)

    def _stop(self):
        """Shut down the backend and worker."""
        if self._started and self._backend:
            try:
                # Release all held slots
                with self._slot_lock:
                    for task_id, slot in list(self._slots.items()):
                        try:
                            self._worker.run(
                                self._backend.release(slot, reset_workspace=True),
                                timeout=10,
                            )
                        except Exception:
                            pass
                    self._slots.clear()

                self._worker.run(self._backend.stop(purge=False), timeout=30)
            except Exception as e:
                print(f"[SlotPool] Warning: shutdown error: {e}")
            finally:
                self._started = False

        self._worker.stop()
        print("[SlotPool] Backend stopped")

    def acquire(self, task_id: str, timeout: float = 60.0):
        """Acquire a slot for a task_id. Returns the Slot object."""
        with self._slot_lock:
            if task_id in self._slots:
                return self._slots[task_id]

        slot = self._worker.run(
            self._backend.acquire(task_id), timeout=timeout
        )

        with self._slot_lock:
            self._slots[task_id] = slot

        return slot

    def execute(self, task_id: str, command: str, cwd: str = "", timeout: float = 300.0) -> dict:
        """Execute a command in the task's slot. Returns {"output": ..., "returncode": ...}."""
        with self._slot_lock:
            slot = self._slots.get(task_id)
        if slot is None:
            return {"output": "Error: no slot acquired for this task", "returncode": 1}

        # Build command with cwd prefix if needed
        full_command = f"cd {cwd} && {command}" if cwd else command

        results = self._worker.run(
            self._backend.execute_batch(
                [(slot, "bash", {"command": full_command})],
                timeout_s=timeout,
            ),
            timeout=timeout + 30,  # Extra margin for network
        )

        r = results[0]
        output = r.output if r.success else (
            f"{r.output}\n{r.error}" if r.output else r.error
        )
        returncode = r.metadata.get("returncode", 0 if r.success else 1)
        return {"output": output, "returncode": returncode}

    def release(self, task_id: str, reset_workspace: bool = True):
        """Release a task's slot back to the pool."""
        with self._slot_lock:
            slot = self._slots.pop(task_id, None)
        if slot is None:
            return

        try:
            self._worker.run(
                self._backend.release(slot, reset_workspace=reset_workspace),
                timeout=30,
            )
        except Exception as e:
            print(f"[SlotPool] Warning: release failed for {task_id}: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get pool status."""
        if not self._started or not self._backend:
            return {"status": "not started"}
        return self._backend.get_status()


class _SlotPoolEnvironment:
    """
    Slot-pool based execution environment.

    Routes terminal commands through atropos/backends/ (Modal, Nomad/Docker,
    Nomad/Singularity) with N:M slot multiplexing. Multiple tasks share a
    smaller number of sandboxes via slot assignment.

    Usage:
        TERMINAL_ENV=slot_pool
        TERMINAL_SLOT_BACKEND=modal    # or nomad
        TERMINAL_MODAL_IMAGE=python:3.11
        TERMINAL_MODAL_SLOTS=10
    """

    def __init__(
        self,
        cwd: str = "/data",
        timeout: int = 300,
        task_id: str = "",
    ):
        self.cwd = cwd
        self.timeout = timeout
        self.task_id = task_id or str(uuid.uuid4())
        self._released = False

        # Acquire a slot from the pool
        manager = _SlotPoolManager.get_instance()
        manager.acquire(self.task_id, timeout=60.0)

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict:
        """Execute a command in the slot's workspace."""
        exec_command = _transform_sudo_command(command)
        work_dir = cwd or self.cwd

        try:
            return _SlotPoolManager.get_instance().execute(
                self.task_id,
                exec_command,
                cwd=work_dir,
                timeout=float(timeout or self.timeout),
            )
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                return {"output": f"Command timed out after {timeout or self.timeout}s", "returncode": 124}
            return {"output": f"SlotPool execution error: {error_msg}", "returncode": 1}

    def cleanup(self):
        """Release slot back to the pool (workspace reset for reuse)."""
        if not self._released:
            self._released = True
            _SlotPoolManager.get_instance().release(self.task_id, reset_workspace=True)

    def stop(self):
        """Same as cleanup for slot pool."""
        self.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except:
            pass


def _shutdown_slot_pool():
    """Shutdown the slot pool manager (called at process exit)."""
    try:
        _SlotPoolManager.reset_instance()
    except Exception:
        pass

# Register slot pool shutdown alongside modal pool shutdown
import atexit as _atexit_slot
_atexit_slot.register(_shutdown_slot_pool)


# Tool description for LLM
TERMINAL_TOOL_DESCRIPTION = """Execute commands on a secure Linux environment.

**Environment:**
- Isolated execution environment (local, Docker, or Modal cloud based on configuration)
- Filesystem persists between tool calls within the same task
- Internet access available

**Command Execution:**
- Simple commands: Just provide the 'command' parameter
- Background processes: Set 'background': True for servers/long-running tasks
- Command timeout: Optional 'timeout' parameter in seconds

**Examples:**
- Run command: `{"command": "ls -la"}`
- Background task: `{"command": "source venv/bin/activate && python server.py", "background": True}`
- With timeout: `{"command": "long_task.sh", "timeout": 300}`

**Best Practices:**
- Run servers/long processes in background
- Monitor disk usage for large tasks
- Install whatever tools you need with apt-get or pip
- Do not be afraid to run pip with --break-system-packages

**Things to avoid:**
- Do NOT use interactive tools such as tmux, vim, nano, python repl - you will get stuck.
- Even git sometimes becomes interactive if the output is large. If you're not sure, pipe to cat.
"""

# Global state for environment lifecycle management
_active_environments: Dict[str, Any] = {}
_task_workdirs: Dict[str, str] = {}  # Maps task_id to working directory
_last_activity: Dict[str, float] = {}
_env_lock = threading.Lock()
_cleanup_thread = None
_cleanup_running = False

# Configuration from environment variables
def _get_env_config() -> Dict[str, Any]:
    """Get terminal environment configuration from environment variables."""
    # Default image with Python and Node.js for maximum compatibility
    default_image = "nikolaik/python-nodejs:python3.11-nodejs20"
    env_type = os.getenv("TERMINAL_ENV", "local")
    
    # Default cwd depends on backend:
    #   - local/ssh: current working directory (CLI resolves "." before we get here)
    #   - docker/singularity: /tmp inside the container (singularity bind-mounts /scratch there)
    #   - modal: /root (ephemeral cloud container, full filesystem access)
    if env_type == "modal":
        default_cwd = "/root"
    elif env_type in ("docker", "singularity"):
        default_cwd = "/tmp"
    else:
        default_cwd = os.getcwd()
    
    return {
        "env_type": env_type,
        "docker_image": os.getenv("TERMINAL_DOCKER_IMAGE", default_image),
        "singularity_image": os.getenv("TERMINAL_SINGULARITY_IMAGE", f"docker://{default_image}"),
        "modal_image": os.getenv("TERMINAL_MODAL_IMAGE", default_image),
        "cwd": os.getenv("TERMINAL_CWD", default_cwd),
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
        # Use native Modal Sandbox with auto-scaling pool and profile support
        return _ModalSandboxEnvironment(
            image=image,
            cwd=cwd,
            timeout=timeout,
            task_id=task_id,
            profile=profile,
        )
    
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
    
    elif env_type == "slot_pool":
        # Multiplexed sandbox pool via atropos/backends/ (Modal, Nomad/Docker, Nomad/Singularity)
        # N:M slot multiplexing for high-throughput parallel execution
        workspace = os.getenv("TERMINAL_MODAL_WORKSPACE", "/data")
        return _SlotPoolEnvironment(
            cwd=cwd or workspace,
            timeout=timeout,
            task_id=task_id if 'task_id' in dir() else "",
        )
    
    else:
        raise ValueError(
            f"Unknown environment type: {env_type}. "
            "Use 'local', 'docker', 'singularity', 'modal', 'ssh', or 'slot_pool'"
        )


def _cleanup_inactive_envs(lifetime_seconds: int = 300):
    """Clean up environments that have been inactive for longer than lifetime_seconds."""
    global _active_environments, _last_activity

    current_time = time.time()
    tasks_to_cleanup = []

    with _env_lock:
        for task_id, last_time in list(_last_activity.items()):
            if current_time - last_time > lifetime_seconds:
                tasks_to_cleanup.append(task_id)

        for task_id in tasks_to_cleanup:
            try:
                if task_id in _active_environments:
                    env = _active_environments[task_id]
                    # Try various cleanup methods
                    if hasattr(env, 'cleanup'):
                        env.cleanup()
                    elif hasattr(env, 'stop'):
                        env.stop()
                    elif hasattr(env, 'terminate'):
                        env.terminate()

                    del _active_environments[task_id]
                    if not os.getenv("HERMES_QUIET"):
                        print(f"[Terminal Cleanup] Cleaned up inactive environment for task: {task_id}")

                if task_id in _last_activity:
                    del _last_activity[task_id]
                if task_id in _task_workdirs:
                    del _task_workdirs[task_id]

            except Exception as e:
                error_str = str(e)
                if not os.getenv("HERMES_QUIET"):
                    if "404" in error_str or "not found" in error_str.lower():
                        print(f"[Terminal Cleanup] Environment for task {task_id} already cleaned up")
                    else:
                        print(f"[Terminal Cleanup] Error cleaning up environment for task {task_id}: {e}")
                
                # Always remove from tracking dicts
                if task_id in _active_environments:
                    del _active_environments[task_id]
                if task_id in _last_activity:
                    del _last_activity[task_id]
                if task_id in _task_workdirs:
                    del _task_workdirs[task_id]


def _cleanup_thread_worker():
    """Background thread worker that periodically cleans up inactive environments."""
    global _cleanup_running

    while _cleanup_running:
        try:
            config = _get_env_config()
            _cleanup_inactive_envs(config["lifetime_seconds"])
        except Exception as e:
            if not os.getenv("HERMES_QUIET"):
                print(f"[Terminal Cleanup] Error in cleanup thread: {e}")

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
                except:
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
            print(f"[Terminal Cleanup] Error cleaning {task_id}: {e}")
    
    # Also clean any orphaned directories
    scratch_dir = _get_scratch_dir()
    import glob
    for path in glob.glob(str(scratch_dir / "hermes-*")):
        try:
            shutil.rmtree(path, ignore_errors=True)
            print(f"[Terminal Cleanup] Removed orphaned: {path}")
        except:
            pass
    
    print(f"[Terminal Cleanup] Cleaned {cleaned} environments")
    return cleaned


def cleanup_vm(task_id: str):
    """Manually clean up a specific environment by task_id."""
    global _active_environments, _last_activity, _task_workdirs

    with _env_lock:
        try:
            if task_id in _active_environments:
                env = _active_environments[task_id]
                if hasattr(env, 'cleanup'):
                    env.cleanup()
                elif hasattr(env, 'stop'):
                    env.stop()
                elif hasattr(env, 'terminate'):
                    env.terminate()

                del _active_environments[task_id]
                if not os.getenv("HERMES_QUIET"):
                    print(f"[Terminal Cleanup] Manually cleaned up environment for task: {task_id}")

            if task_id in _task_workdirs:
                del _task_workdirs[task_id]

            if task_id in _last_activity:
                del _last_activity[task_id]

        except Exception as e:
            if not os.getenv("HERMES_QUIET"):
                error_str = str(e)
                if "404" in error_str or "not found" in error_str.lower():
                    print(f"[Terminal Cleanup] Environment for task {task_id} already cleaned up")
                else:
                    print(f"[Terminal Cleanup] Error cleaning up environment for task {task_id}: {e}")


atexit.register(_stop_cleanup_thread)

def _shutdown_modal_pools():
    """Shutdown Modal pool manager on exit (silently, as interpreter is shutting down)."""
    try:
        if _ModalPoolManager._instance is not None:
            _ModalPoolManager._instance.shutdown(during_shutdown=True)
    except:
        pass  # Ignore all errors during interpreter shutdown

atexit.register(_shutdown_modal_pools)


def terminal_tool(
    command: str,
    background: bool = False,
    timeout: Optional[int] = None,
    task_id: Optional[str] = None,
    force: bool = False,
    profile: Optional[str] = None,
) -> str:
    """
    Execute a command using mini-swe-agent's execution environments.

    Args:
        command: The command to execute
        background: Whether to run in background (default: False)
        timeout: Command timeout in seconds (default: from config)
        task_id: Unique identifier for environment isolation (optional)
        force: If True, skip dangerous command check (use after user confirms)

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
        
        # Select image based on env type
        if env_type == "docker":
            image = config["docker_image"]
        elif env_type == "singularity":
            image = config["singularity_image"]
        elif env_type == "modal":
            image = config["modal_image"]
        else:
            image = ""
        
        cwd = config["cwd"]
        default_timeout = config["timeout"]
        effective_timeout = timeout or default_timeout

        # Use task_id for environment isolation
        effective_task_id = task_id or "default"

        # For local environment in batch mode, create a unique subdirectory per task
        # This prevents parallel tasks from overwriting each other's files
        # In CLI mode (HERMES_QUIET), use the cwd directly without subdirectories
        if env_type == "local" and not os.getenv("HERMES_QUIET"):
            import uuid
            with _env_lock:
                if effective_task_id not in _task_workdirs:
                    task_workdir = Path(cwd) / f"hermes-{effective_task_id}-{uuid.uuid4().hex[:8]}"
                    task_workdir.mkdir(parents=True, exist_ok=True)
                    _task_workdirs[effective_task_id] = str(task_workdir)
                cwd = _task_workdirs[effective_task_id]

        # Start cleanup thread
        _start_cleanup_thread()

        # Get or create environment
        # Check under lock, but create OUTSIDE lock so we don't block
        # other concurrent rollouts during slow Modal/Docker startup
        needs_creation = False
        with _env_lock:
            if effective_task_id not in _active_environments:
                needs_creation = True
            else:
                _last_activity[effective_task_id] = time.time()
                env = _active_environments[effective_task_id]

        if needs_creation:
            _check_disk_usage_warning()
            if not os.getenv("HERMES_QUIET"):
                print(f"[Terminal] Creating new {env_type} environment for task {effective_task_id[:8]}...", flush=True)
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

            # Store under lock (brief)
            with _env_lock:
                if effective_task_id not in _active_environments:
                    _active_environments[effective_task_id] = new_env
                else:
                    # Another thread created it while we were building -- clean up ours
                    try:
                        if hasattr(new_env, 'stop'):
                            new_env.stop()
                    except Exception:
                        pass

                _last_activity[effective_task_id] = time.time()
                env = _active_environments[effective_task_id]
                if not os.getenv("HERMES_QUIET"):
                    print(f"[Terminal] {env_type} environment ready for task {effective_task_id[:8]}", flush=True)

        # Check for dangerous commands (only for local/ssh in interactive modes)
        # Skip check if force=True (user has confirmed they want to run it)
        if not force:
            approval = _check_dangerous_command(command, env_type)
            if not approval["approved"]:
                # Command was blocked - return informative message
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": approval.get("message", "Command denied - potentially dangerous operation"),
                    "status": "blocked"
                }, ensure_ascii=False)

        # Prepare command for execution
        if background:
            # Run in background with nohup and redirect output
            exec_command = f"nohup {command} > /tmp/bg_output.log 2>&1 &"
            try:
                result = env.execute(exec_command, timeout=10)
                return json.dumps({
                    "output": "Background task started successfully",
                    "exit_code": 0,
                    "error": None
                }, ensure_ascii=False)
            except Exception as e:
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": f"Failed to start background task: {str(e)}"
                }, ensure_ascii=False)
        else:
            # Run foreground command with retry logic
            max_retries = 3
            retry_count = 0
            result = None
            
            while retry_count <= max_retries:
                try:
                    result = env.execute(command, timeout=effective_timeout)
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
                        print(f"⚠️  Terminal: execution error, retrying in {wait_time}s (attempt {retry_count}/{max_retries})")
                        print(f"   Command: {command[:200]}")
                        print(f"   Error: {type(e).__name__}: {e}")
                        print(f"   Task ID: {effective_task_id}, Backend: {env_type}")
                        time.sleep(wait_time)
                        continue
                    
                    print(f"❌ Terminal: execution failed after {max_retries} retries")
                    print(f"   Command: {command[:200]}")
                    print(f"   Error: {type(e).__name__}: {e}")
                    print(f"   Task ID: {effective_task_id}, Backend: {env_type}")
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
        print(f"Terminal requirements check failed: {e}")
        return False


if __name__ == "__main__":
    """Simple test when run directly."""
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
