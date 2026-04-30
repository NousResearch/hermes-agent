"""RTK binary lifecycle: discovery, download, cache, verify.

Hermes can transparently rewrite shell commands through RTK to reduce
token consumption by 60-90%. This module manages the RTK binary so users
don't need to install it manually.

Usage:
    from tools.rtk_manager import ensure_rtk
    rtk_path = ensure_rtk()
    if rtk_path:
        subprocess.run([str(rtk_path), "rewrite", command], ...)
"""

import logging
import os
import platform
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

RTK_VERSION = "v0.38.0"
RTK_CACHE_DIR = Path.home() / ".cache" / "hermes" / "bin"
RTK_DOWNLOAD_TIMEOUT = 30  # seconds


def _detect_target_triple() -> Optional[str]:
    """Return the Rust target triple for the current platform, or None."""
    machine = platform.machine().lower()
    system = platform.system().lower()

    # Normalize architecture names
    if machine in ("x86_64", "amd64", "x64"):
        arch = "x86_64"
    elif machine in ("aarch64", "arm64"):
        arch = "aarch64"
    else:
        logger.debug("Unsupported architecture: %s", machine)
        return None

    if system == "linux":
        # Use musl for maximum compatibility (static binary)
        return f"{arch}-unknown-linux-musl"
    elif system == "darwin":
        return f"{arch}-apple-darwin"
    elif system == "windows":
        return f"{arch}-pc-windows-msvc"
    else:
        logger.debug("Unsupported OS: %s", system)
        return None


def _download_url(target: str) -> str:
    """Build the GitHub Releases download URL for the given target triple."""
    base = "https://github.com/rtk-ai/rtk/releases/download"
    if "windows" in target:
        return f"{base}/{RTK_VERSION}/rtk-{target}.zip"
    return f"{base}/{RTK_VERSION}/rtk-{target}.tar.gz"


def _download_and_extract(target: str, dest: Path) -> bool:
    """Download RTK binary for *target* and extract it to *dest*."""
    url = _download_url(target)
    logger.info("Downloading RTK %s for %s...", RTK_VERSION, target)

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Hermes-Agent/RTK-Downloader",
            },
        )
        with urllib.request.urlopen(req, timeout=RTK_DOWNLOAD_TIMEOUT) as resp:
            data = resp.read()
    except Exception as exc:
        logger.warning("RTK download failed: %s", exc)
        return False

    RTK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_archive = RTK_CACHE_DIR / f"rtk-{target}.tmp"

    try:
        tmp_archive.write_bytes(data)

        if url.endswith(".zip"):
            with zipfile.ZipFile(tmp_archive, "r") as zf:
                zf.extract("rtk.exe", RTK_CACHE_DIR)
                extracted = RTK_CACHE_DIR / "rtk.exe"
        else:
            with tarfile.open(tmp_archive, "r:gz") as tf:
                tf.extract("rtk", RTK_CACHE_DIR)
                extracted = RTK_CACHE_DIR / "rtk"

        extracted.chmod(extracted.stat().st_mode | 0o111)
        shutil.move(str(extracted), str(dest))
        return True
    except Exception as exc:
        logger.warning("RTK extraction failed: %s", exc)
        return False
    finally:
        if tmp_archive.exists():
            tmp_archive.unlink()


def _verify_binary(path: Path) -> bool:
    """Run ``rtk --version`` to confirm the binary works."""
    try:
        result = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and "rtk" in result.stdout.lower()
    except Exception:
        return False


def ensure_rtk(auto_download: bool = True) -> Optional[Path]:
    """Return a usable RTK binary path, downloading if necessary.

    Resolution order:
        1. ``rtk`` on the user's PATH (e.g. Homebrew, manual install)
        2. Cached binary at ``~/.cache/hermes/bin/rtk``
        3. Download from GitHub Releases (if *auto_download* is True)

    Returns ``None`` if RTK cannot be found or downloaded.
    """
    # 1. Check PATH
    if path_str := shutil.which("rtk"):
        path = Path(path_str)
        if _verify_binary(path):
            logger.debug("Using system RTK: %s", path)
            return path

    # 2. Check cache
    cached = RTK_CACHE_DIR / "rtk"
    if cached.exists():
        if _verify_binary(cached):
            logger.debug("Using cached RTK: %s", cached)
            return cached
        else:
            logger.warning("Cached RTK is corrupt, removing: %s", cached)
            cached.unlink()

    # 3. Download
    if not auto_download:
        logger.debug("RTK auto-download disabled and binary not found.")
        return None

    target = _detect_target_triple()
    if target is None:
        logger.warning("Cannot determine platform for RTK download.")
        return None

    if _download_and_extract(target, cached) and _verify_binary(cached):
        logger.info("RTK downloaded and verified: %s", cached)
        return cached

    return None


def rewrite_command(command: str, auto_download: bool = True) -> Optional[str]:
    """Rewrite *command* through RTK and return the rewritten version.

    Returns ``None`` when RTK is unavailable or has no rewrite for this
    command (the caller should fall back to the original command).
    """
    rtk_path = ensure_rtk(auto_download=auto_download)
    if rtk_path is None:
        return None

    try:
        result = subprocess.run(
            [str(rtk_path), "rewrite", command],
            capture_output=True,
            text=True,
            timeout=2,
        )
        # RTK returns exit code 3 on successful rewrite, 1 when no rewrite
        # is available. We trust non-empty stdout as the source of truth.
        rewritten = result.stdout.strip()
        return rewritten if rewritten and rewritten != command else None
    except Exception:
        return None
