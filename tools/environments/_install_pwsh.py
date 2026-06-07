"""Auto-install PowerShell 7 (pwsh) silently on Windows.

Ported from the KIMI agent install logic. Provides a silent fallback
that downloads PowerShell 7 when the local terminal backend wants to
use pwsh but can't find it.

Strategy (in priority order):
1. WinGet (official Microsoft recommendation)
2. MSI direct download from GitHub latest release (silent msiexec install)
3. ZIP portable download from GitHub latest release (no admin required)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================
# Global configuration
# ============================================================
_DEFAULT_ZIP_INSTALL_DIR = Path(
    os.environ.get("LOCALAPPDATA", str(Path.home()))) / "PowerShell" / "7"
"""Default install directory for the ZIP portable extraction strategy."""

_GITHUB_API_URL = (
    "https://api.github.com/repos/PowerShell/PowerShell/releases/latest"
)
"""GitHub API endpoint for the latest PowerShell release."""

_FALLBACK_VERSION = "7.6.2"
"""Hardcoded fallback version if GitHub API query fails."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _is_windows() -> bool:
    return sys.platform == "win32"


def _run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and return the result (stdout/stderr captured as text)."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _pwsh_found(install_dir: str | None = None) -> bool:
    """Return ``True`` if ``pwsh.exe`` is available.

    When *install_dir* is given, checks that directory first
    (looking for ``pwsh.exe``).  Falls back to checking
    PATH when *install_dir* is ``None``.
    """
    if install_dir:
        return (Path(install_dir) / "pwsh.exe").exists()
    return shutil.which("pwsh") is not None or shutil.which("pwsh.exe") is not None


def _get_latest_release_info() -> dict | None:
    """Fetch the latest PowerShell release info from GitHub API.

    Returns the parsed JSON data on success, or ``None`` on failure.
    """
    try:
        logger.info("Querying GitHub API for latest PowerShell release ...")
        req = urllib.request.Request(
            _GITHUB_API_URL,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "hermes-installer",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("GitHub API query failed: %s", exc)
        return None


def _get_version_from_release(release_info: dict) -> str:
    """Extract the version string from a GitHub release info dict.

    Strips the leading 'v' from ``tag_name`` (e.g. ``v7.6.2`` → ``7.6.2``).
    """
    tag = release_info.get("tag_name", "")
    return tag.lstrip("v")


def _find_asset(release_info: dict, suffix: str) -> dict | None:
    """Find an asset whose name ends with *suffix* from the release assets list."""
    assets = release_info.get("assets", [])
    for asset in assets:
        name = asset.get("name", "")
        if name.endswith(suffix):
            return asset
    return None


def _ensure_in_user_path(dirpath: str) -> None:
    """Add *dirpath* to the current user's PATH environment variable (persistent)."""
    import winreg

    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Environment",
            0,
            winreg.KEY_READ | winreg.KEY_WRITE,
        )
    except FileNotFoundError:
        return
    try:
        path_val, _ = winreg.QueryValueEx(key, "Path")
    except FileNotFoundError:
        path_val = ""
    entries = [p.strip() for p in path_val.split(";") if p.strip()]
    if dirpath in entries:
        winreg.CloseKey(key)
        return
    entries.append(dirpath)
    new_path = ";".join(entries)
    winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)
    winreg.CloseKey(key)


def _download_file(url: str, dest: Path) -> None:
    """Download *url* to *dest*, with a progress indicator."""
    logger.info("Downloading %s ...", dest.name)

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            pct = min(100, int(block_num * block_size * 100 / total_size))
            sys.stdout.write(f"\r  {pct}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, str(dest), _report)
    sys.stdout.write("\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# strategy implementations
# ---------------------------------------------------------------------------

def _try_winget() -> bool:
    """Install PowerShell 7 via WinGet (preferred, official Microsoft channel)."""
    if not shutil.which("winget"):
        return False
    try:
        logger.info("Installing PowerShell 7 via WinGet ...")
        _run(
            [
                "winget",
                "install",
                "--id",
                "Microsoft.PowerShell",
                "--source",
                "winget",
                "--accept-package-agreements",
                "--accept-source-agreements",
                "--silent",
            ],
            timeout=600,
        )
        return _pwsh_found()
    except Exception as exc:
        logger.warning("WinGet install failed: %s", exc)
        return False


def _try_msi(install_dir: str | None = None) -> bool:
    """Download the latest PowerShell MSI installer and run it silently.

    The MSI is installed system-wide (requires admin privileges).
    """
    release_info = _get_latest_release_info()
    if release_info is None:
        logger.info("Falling back to hardcoded version for MSI download.")
        version = _FALLBACK_VERSION
        msi_url = (
            f"https://github.com/PowerShell/PowerShell/releases/download/"
            f"v{version}/PowerShell-{version}-win-x64.msi"
        )
    else:
        version = _get_version_from_release(release_info)
        asset = _find_asset(release_info, "-win-x64.msi")
        if asset is None:
            logger.warning("No MSI asset found in latest release.")
            return False
        msi_url = asset["browser_download_url"]

    msi_name = f"PowerShell-{version}-win-x64.msi"
    msi_path = Path(tempfile.gettempdir()) / msi_name

    # --- download ---
    try:
        _download_file(msi_url, msi_path)
    except Exception as exc:
        logger.warning("Download failed: %s", exc)
        return False

    # --- install ---
    try:
        logger.info("Running silent MSI installer ...")
        cmd = [
            "msiexec.exe",
            "/i",
            str(msi_path),
            "/quiet",
            "/norestart",
        ]
        if install_dir:
            cmd.append(f'INSTALLDIR="{install_dir}"')
        _run(cmd, timeout=600)
    except subprocess.TimeoutExpired:
        logger.warning("MSI installer timed out.")
    except Exception as exc:
        logger.warning("MSI installer error: %s", exc)
    finally:
        # --- clean up ---
        msi_path.unlink(missing_ok=True)

    return _pwsh_found()


def _try_zip(install_dir: str | None = None) -> bool:
    """Download the latest PowerShell ZIP archive and extract it.

    This strategy does NOT require admin privileges.
    """
    release_info = _get_latest_release_info()
    if release_info is None:
        logger.info("Falling back to hardcoded version for ZIP download.")
        version = _FALLBACK_VERSION
        zip_url = (
            f"https://github.com/PowerShell/PowerShell/releases/download/"
            f"v{version}/PowerShell-{version}-win-x64.zip"
        )
    else:
        version = _get_version_from_release(release_info)
        asset = _find_asset(release_info, "-win-x64.zip")
        if asset is None:
            logger.warning("No ZIP asset found in latest release.")
            return False
        zip_url = asset["browser_download_url"]

    zip_name = f"PowerShell-{version}-win-x64.zip"
    zip_path = Path(tempfile.gettempdir()) / zip_name

    target = Path(install_dir) if install_dir else _DEFAULT_ZIP_INSTALL_DIR
    target.mkdir(parents=True, exist_ok=True)

    # --- download ---
    try:
        _download_file(zip_url, zip_path)
    except Exception as exc:
        logger.warning("Download failed: %s", exc)
        return False

    # --- extract ---
    try:
        logger.info("Extracting to %s ...", target)
        shutil.unpack_archive(str(zip_path), str(target))
    except Exception as exc:
        logger.warning("Extraction error: %s", exc)
    finally:
        zip_path.unlink(missing_ok=True)

    # --- verify ---
    if _pwsh_found(str(target)):
        _ensure_in_user_path(str(target))
        return True
    return False


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def install_pwsh(
    install_dir: str | None = None,
    *,
    add_to_path: bool = True,
    timeout: int = 600,
) -> str | None:
    """Silently install PowerShell 7 for Windows.

    Tries, in order:
      1. WinGet (``winget install Microsoft.PowerShell --silent``).
      2. MSI direct download from GitHub latest release (requires admin).
      3. ZIP portable download from GitHub latest release (no admin).

    Parameters
    ----------
    install_dir:
        Target directory.  For the ZIP strategy defaults to
        ``%LOCALAPPDATA%\\PowerShell\\7``.  For MSI it influences the
        ``INSTALLDIR`` property.
    add_to_path:
        Whether to append the install directory to the user PATH.
    timeout:
        Seconds to wait for each install subprocess.

    Returns
    -------
    The absolute path to ``pwsh.exe`` on success, or ``None`` on failure.
    """
    if not _is_windows():
        logger.debug("install_pwsh: this script only supports Windows.")
        return None

    # Already installed?
    if _pwsh_found(install_dir):
        where = f"at {install_dir}" if install_dir else "on PATH"
        logger.info("PowerShell 7 is already installed %s.", where)
        pwsh_path = (
            str(Path(install_dir) / "pwsh.exe")
            if install_dir
            else shutil.which("pwsh") or shutil.which("pwsh.exe")
        )
        if pwsh_path and add_to_path:
            bin_dir = str(Path(pwsh_path).parent)
            _ensure_in_user_path(bin_dir)
        return pwsh_path

    strategies: list[tuple[str, object]] = [
        ("winget", _try_winget),
        ("MSI direct download", lambda: _try_msi(install_dir)),
        ("ZIP portable", lambda: _try_zip(install_dir)),
    ]

    for name, fn in strategies:
        logger.info("Trying PowerShell 7 install: %s ...", name)
        try:
            ok = fn()  # type: ignore[operator]
        except Exception as exc:
            logger.warning("  %s raised: %s", name, exc)
            ok = False
        if ok and _pwsh_found():
            logger.info("PowerShell 7 installed successfully via %s.", name)
            pwsh_path = shutil.which("pwsh") or shutil.which("pwsh.exe")
            # For ZIP strategy, pwsh may not be on PATH yet; check install_dir
            if not pwsh_path and install_dir:
                candidate = Path(install_dir) / "pwsh.exe"
                if candidate.exists():
                    pwsh_path = str(candidate)
            # Also check default ZIP path
            if not pwsh_path:
                candidate = _DEFAULT_ZIP_INSTALL_DIR / "pwsh.exe"
                if candidate.exists():
                    pwsh_path = str(candidate)
            if pwsh_path and add_to_path:
                bin_dir = str(Path(pwsh_path).parent)
                _ensure_in_user_path(bin_dir)
            return pwsh_path
        logger.info("  %s did not succeed.", name)

    logger.warning("All PowerShell 7 installation strategies failed.")
    return None
