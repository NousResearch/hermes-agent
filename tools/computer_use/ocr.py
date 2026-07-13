"""Bounded, local OCR for computer-use screenshots."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional


_MAX_TEXT_CHARS = 12_000
_MAX_LINES = 250
_MAX_WORDS = 800
_LANGUAGE_RE = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")


def extract_local_ocr(
    image_b64: str,
    *,
    image_mime_type: Optional[str] = None,
    language: Optional[str] = None,
    timeout_seconds: float = 20.0,
) -> Dict[str, Any]:
    """Run the host OCR engine without sending the screenshot to a model."""
    if sys.platform != "win32":
        return {"available": False, "reason": "local OCR is currently implemented on Windows"}
    if not image_b64:
        return {"available": False, "reason": "capture did not include an image"}

    requested_language = (language or "").strip()
    if requested_language and not _LANGUAGE_RE.fullmatch(requested_language):
        return {"available": False, "reason": "invalid OCR language tag"}

    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if not powershell:
        return {"available": False, "reason": "Windows PowerShell is unavailable"}

    suffix = ".jpg" if image_mime_type == "image/jpeg" else ".png"
    temp_path: Optional[Path] = None
    try:
        raw = base64.b64decode(image_b64, validate=False)
        with tempfile.NamedTemporaryFile(prefix="hermes-cua-ocr-", suffix=suffix, delete=False) as fh:
            fh.write(raw)
            temp_path = Path(fh.name)

        script = Path(__file__).with_name("windows_ocr.ps1")
        command = [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-Path",
            str(temp_path),
        ]
        if requested_language:
            command.extend(["-Language", requested_language])

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1.0, min(float(timeout_seconds), 30.0)),
            creationflags=creationflags,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "OCR process failed").strip()
            return {"available": False, "reason": detail[:500]}

        # Windows PowerShell 5.1's ConvertTo-Json can leave OCR control
        # characters unescaped inside a JSON string. JSON forbids raw U+0000
        # through U+001F, so replace them before decoding. Structural
        # whitespace can safely become a space as well.
        json_text = re.sub(r"[\x00-\x1f]", " ", completed.stdout.lstrip("\ufeff"))
        parsed = json.loads(json_text)
        text = str(parsed.get("text") or "")[:_MAX_TEXT_CHARS]
        lines = [str(line)[:1000] for line in (parsed.get("lines") or [])[:_MAX_LINES]]
        words = []
        for item in (parsed.get("words") or [])[:_MAX_WORDS]:
            if not isinstance(item, dict):
                continue
            bounds = item.get("bounds")
            if not isinstance(bounds, list) or len(bounds) != 4:
                continue
            words.append({"text": str(item.get("text") or "")[:250], "bounds": bounds})
        return {
            "available": True,
            "method": "windows-media-ocr",
            "language": parsed.get("language"),
            "width": parsed.get("width"),
            "height": parsed.get("height"),
            "text": text,
            "lines": lines,
            "words": words,
            "confidence_available": False,
            "truncated": len(str(parsed.get("text") or "")) > len(text)
            or len(parsed.get("words") or []) > len(words),
        }
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": str(exc)[:500]}
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
