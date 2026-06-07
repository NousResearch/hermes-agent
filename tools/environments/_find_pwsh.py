"""Find the PowerShell 7 (pwsh) executable on Windows.

Ported from the KIMI agent discovery logic.  Provides a cached,
multi-strategy lookup with silent auto-install fallback.
"""

from __future__ import annotations

import functools
import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_config_dir() -> Path:
    """Return the config directory for cached shell paths."""
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / "config"
    except Exception:
        return Path.home() / ".hermes" / "config"


@functools.lru_cache(maxsize=1)
def find_pwsh() -> str | None:
    """Find the system PowerShell executable.

    On Windows, prioritises ``pwsh`` (PowerShell 7+) over ``powershell``
    (Windows PowerShell 5.1).  Falls back to silent auto-install when
    pwsh is not found.

    Returns ``None`` on non-Windows or when all strategies fail.
    """
    import sys

    if sys.platform != "win32":
        return None

    config_dir = _get_config_dir()
    cache_file = config_dir / "pwsh.txt"

    # Strategy 0: read cached pwsh path
    if cache_file.is_file():
        cached_path = cache_file.read_text().strip()
        if cached_path and Path(cached_path).exists():
            logger.debug("find_pwsh: using cached path %s", cached_path)
            return cached_path
        # Stale cache -- remove it
        try:
            cache_file.unlink()
        except Exception:
            pass

    # Strategy 1: pwsh (PowerShell 7+) via PATH
    pwsh_path = shutil.which("pwsh")
    if pwsh_path:
        resolved = str(Path(pwsh_path).resolve())
        logger.debug("find_pwsh: found on PATH: %s", resolved)
        return resolved

    # Strategy 2: pwsh.exe via PATH
    pwsh_path = shutil.which("pwsh.exe")
    if pwsh_path:
        resolved = str(Path(pwsh_path).resolve())
        logger.debug("find_pwsh: found pwsh.exe on PATH: %s", resolved)
        return resolved

    # Strategy 3: where.exe command for pwsh
    try:
        r = subprocess.run(
            ["where.exe", "pwsh.exe"],
            capture_output=True, text=True, check=True,
        )
        first = r.stdout.strip().splitlines()[0].strip()
        if first:
            resolved = str(Path(first).resolve())
            logger.debug("find_pwsh: found via where.exe: %s", resolved)
            return resolved
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Strategy 4: Common paths fallback for pwsh
    candidates = [
        r"C:\Program Files\PowerShell\7\pwsh.exe",
        r"C:\Program Files (x86)\PowerShell\7\pwsh.exe",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            logger.debug("find_pwsh: found at common path: %s", candidate)
            return str(Path(candidate).resolve())

    # Also check LOCALAPPDATA portable install
    local_pwsh = (
        Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
        / "PowerShell" / "7" / "pwsh.exe"
    )
    if local_pwsh.exists():
        logger.debug("find_pwsh: found local portable: %s", local_pwsh)
        return str(local_pwsh.resolve())

    # Strategy 5: Auto-install fallback
    try:
        from tools.environments._install_pwsh import install_pwsh

        logger.info(
            "PowerShell 7 not found – attempting silent auto-install ..."
        )
        pwsh_path = install_pwsh()
        if pwsh_path and Path(pwsh_path).exists():
            # Persist to cache so subsequent calls skip detection + install
            config_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(pwsh_path, encoding="utf-8")
            # Update in-process PATH so subprocesses see it immediately
            bin_dir = str(Path(pwsh_path).parent)
            current_path = os.environ.get("PATH", "")
            if bin_dir not in current_path.split(os.pathsep):
                os.environ["PATH"] = bin_dir + os.pathsep + current_path
            # Invalidate shell-detection caches so the terminal tool
            # description reflects pwsh availability going forward.
            try:
                from tools.terminal_tool import _detect_shell_for_description
                _detect_shell_for_description.cache_clear()
            except Exception:
                pass
            logger.info("PowerShell 7 auto-installed at: %s", pwsh_path)
            return pwsh_path
    except Exception as exc:
        logger.debug("find_pwsh: auto-install failed: %s", exc)

    return None
