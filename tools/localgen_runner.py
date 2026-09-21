"""Local video generation runner for the ``/localgen`` slash command.

This module is the single execution seam for ``/localgen``. It is a thin
wrapper around the existing, already-working ``localgen.py`` backend script
(located under the Hermes profile's turbofit skill tree). Unlike
``generate_image_runner`` (which drives an in-process content-engine service),
``/localgen`` shells out to ``localgen.py`` and parses its JSON stdout. That is
intentional:

* The backend is a long-running GPU job (~3 min at 720p/49f on this rig), lives
  outside the repo, and is maintained by the Sirvir model-fleet agent. Wrapping
  it as a subprocess keeps the repo free of ComfyUI/GGUF/Wan machinery.
* ``localgen.py`` already does VRAM pre-flight, ComfyUI health checks, image
  upload, workflow injection, submission, polling, and artifact collection.
  We do not re-implement any of that here.

Public API (mirrors ``generate_image_runner`` for reviewer familiarity):

* ``run_localgen(*, model, prompt, image=None, seed=None, length=None,
  output_dir=None, timeout=None)`` -> dict
* ``render_localgen_command(*, model, prompt, image=None, seed=None,
  length=None, output_dir=None)`` -> str
* ``LOCALGEN_ERROR_MESSAGES`` -> human-facing message per ``error_code``

The dict returned by ``run_localgen`` mirrors ``localgen.py``'s JSON success
shape::

    {
        "status": "success",
        "prompt_id": str,
        "model": str,
        "elapsed_seconds": float,
        "outputs": [{"file": str, "node_id": str, "filename": str}],
    }

On failure we raise ``LocalGenError`` carrying ``error_code`` and ``detail``,
or return a ``{"status": "error", ...}`` dict when ``raise_on_error=False``.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "run_localgen",
    "render_localgen_command",
    "LocalGenError",
    "LOCALGEN_ERROR_MESSAGES",
]

# Where the backend lives. It is part of the *active* Hermes profile's
# turbofit skill (Sirvir's model-fleet tooling), not this repo. The path is
# resolved relative to HERMES_HOME so it tracks the profile that is actually
# running — we never hardcode a username.
def _default_backend() -> Path:
    """Return the path to localgen.py, resolving the Hermes profile layout.

    ``HERMES_HOME`` may point at either:
      * the Hermes home (``~/.hermes``) — then the backend is at
        ``<home>/profiles/sirvir/skills/turbofit/scripts/localgen.py``, or
      * a specific profile dir (``~/.hermes/profiles/sirvir``) — then it is at
        ``<profile>/skills/turbofit/scripts/localgen.py``.

    We probe both shapes (plus a couple of fallbacks) and return the first
    existing file, so the runner tracks whichever layout is actually running.
    """
    home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")).expanduser()
    candidates = [
        home / "profiles" / "sirvir" / "skills" / "turbofit" / "scripts" / "localgen.py",
        home / "skills" / "turbofit" / "scripts" / "localgen.py",
        Path.home() / ".hermes" / "profiles" / "sirvir" / "skills" / "turbofit" / "scripts" / "localgen.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fall back to the first candidate (best guess) so callers still get a path.
    return candidates[0]


# Plain-English messages keyed by localgen.py's stable ``error_code`` values.
# Keep these jargon-free: the user is a ComfyUI novice.
LOCALGEN_ERROR_MESSAGES: dict[str, str] = {
    "bad_args": "A model and a prompt are needed. Try: /localgen model=fast-video|prompt=your idea",
    "unknown_model": "That model isn't available. Use model=fast-video or model=animate.",
    "image_required": "That model needs a reference photo. Add image=/path/to/photo.png",
    "image_not_found": "I couldn't find that image file. Check the path.",
    "server_down": "Local generation is offline — ComfyUI isn't responding. Start it and retry.",
    "insufficient_vram": "The GPU is busy right now. Free it up or pass --force and try again shortly.",
    "submit_rejected": "The workflow was rejected by ComfyUI. This looks like a setup bug — report it.",
    "generation_failed": "Generation failed inside ComfyUI. Try again; if it persists, report it.",
    "timeout": "That took too long. Try a shorter clip (length=25) or lower steps.",
    "no_outputs": "It finished but produced no video. Report this.",
    "unknown": "Something went wrong. Report this.",
}


class LocalGenError(RuntimeError):
    """Raised by ``run_localgen`` when the backend reports failure.

    Attributes:
        error_code: the stable ``error_code`` from localgen.py (or "unknown").
        detail:     the backend's human-facing error string.
    """

    def __init__(self, error_code: str, detail: str):
        self.error_code = error_code
        self.detail = detail
        super().__init__(f"{error_code}: {detail}")


def _resolve_backend() -> str:
    """Return the path to localgen.py, honoring an override env var."""
    override = os.environ.get("LOCALGEN_BACKEND", "").strip()
    if override:
        return override
    return str(_default_backend())


def run_localgen(
    *,
    model: str,
    prompt: str,
    image: Optional[str] = None,
    seed: Optional[int] = None,
    length: Optional[int] = None,
    output_dir: Optional[str] = None,
    timeout: Optional[int] = None,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    """Run one local video generation job via ``localgen.py``.

    Returns the backend's JSON success dict. On failure, raises
    :class:`LocalGenError` (or returns the error dict when
    ``raise_on_error=False``).

    The call is synchronous; callers must run it in an executor so it never
    blocks the gateway event loop (see the gateway ``/localgen`` handler).
    """
    backend = _resolve_backend()
    if not Path(backend).exists():
        err = ("backend_missing", f"localgen backend not found at {backend}")
        if raise_on_error:
            raise LocalGenError(*err)
        return {"status": "error", "error_code": err[0], "error": err[1]}

    cmd: list[str] = [backend, "--model", model, "--prompt", prompt]
    if image:
        cmd += ["--image", str(image)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if length is not None:
        cmd += ["--length", str(length)]
    if output_dir:
        cmd += ["--output-dir", str(output_dir)]
    if timeout is not None:
        cmd += ["--timeout", str(timeout)]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=(timeout or 1800) + 120,
        )
    except subprocess.TimeoutExpired as exc:
        if raise_on_error:
            raise LocalGenError("timeout", "localgen exceeded the outer timeout") from exc
        return {"status": "error", "error_code": "timeout",
                "error": "localgen exceeded the outer timeout"}

    # localgen.py prints only the JSON result on stdout; progress goes to stderr.
    raw = (proc.stdout or "").strip()
    if proc.returncode != 0 or not raw:
        # Surface the backend's own detail when present, else stderr.
        try:
            payload = json.loads(raw) if raw else {}
            code = payload.get("error_code", "unknown")
            detail = payload.get("error", (proc.stderr or "").strip() or "no output")
        except json.JSONDecodeError:
            code = "unknown"
            detail = (proc.stderr or raw or "no output").strip()
        if raise_on_error:
            raise LocalGenError(str(code), str(detail))
        return {"status": "error", "error_code": str(code), "error": str(detail)}

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        if raise_on_error:
            raise LocalGenError("unknown", f"bad JSON from backend: {raw[:200]}") from exc
        return {"status": "error", "error_code": "unknown",
                "error": f"bad JSON from backend: {raw[:200]}"}
    return payload


def render_localgen_command(
    *,
    model: str,
    prompt: str,
    image: Optional[str] = None,
    seed: Optional[int] = None,
    length: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> str:
    """Render a copy-paste executable ``localgen.py`` command.

    Every value is shell-quoted so the line is safe to paste into a terminal at
    the Hermes home. This is what both the CLI and gateway show the user before
    approval, mirroring ``render_generate_image_command``.
    """
    import shlex

    backend = _resolve_backend()
    parts: list[str] = [
        "python3",
        shlex.quote(backend),
        "--model", shlex.quote(model),
        "--prompt", shlex.quote(prompt),
    ]
    if image:
        parts += ["--image", shlex.quote(str(image))]
    if seed is not None:
        parts += ["--seed", str(seed)]
    if length is not None:
        parts += ["--length", str(length)]
    if output_dir:
        parts += ["--output-dir", shlex.quote(str(output_dir))]
    return " ".join(parts)
