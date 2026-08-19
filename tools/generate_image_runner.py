"""Shared in-process runner for the /generate-image feature.

Both the REPL (``hermes_cli.cli_commands_mixin``) and the gateway
(``gateway.slash_commands``) call this helper to execute one private
native-Codex image job.  It reuses the existing content-engine image
execution service — ``prepare_image_request`` → ``stage_and_plan_image_job``
→ ``execute_staged_image_job`` — without shelling out to
``content_engine.py`` and without duplicating generation logic.

The helper is deliberately synchronous: the content-engine service is
synchronous, and the gateway wraps the call in ``run_in_executor`` so it
never blocks the event loop.  A single ``run_generate_image`` entry point
keeps the two surfaces in sync and gives tests one seam to mock or to drive
with an injected ``NativeImageProvider``.

Importing the content-engine modules requires ``content_engine/`` on
``sys.path``; ``_ensure_content_engine_on_path`` handles that lazily on first
call so the CLI/gateway never pay the import cost until /generate-image runs.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

# Project root (this file lives at <root>/tools/generate_image_runner.py).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONTENT_ENGINE_DIR = _PROJECT_ROOT / "content_engine"


def _ensure_content_engine_on_path() -> None:
    """Make ``content_engine/`` importable if it isn't already."""
    if str(_CONTENT_ENGINE_DIR) not in sys.path:
        sys.path.insert(0, str(_CONTENT_ENGINE_DIR))


def run_generate_image(
    *,
    prompt: str,
    style: str,
    backend: str = "codex",
    references: Optional[list[str]] = None,
    blend: Optional[list[str]] = None,
    stage_root: str,
    job_id: str,
    aspect_ratio: str = "landscape",
    provider: Any = None,
    provider_cache_root: Optional[str] = None,
) -> dict[str, Any]:
    """Run one private native-Codex image job in-process.

    Returns a dict mirroring the JSON shape ``content_engine.py
    generate-image`` prints::

        {
            "backend": {"provider": str, "model": str},
            "completion_path": str,
            "job_id": str,
            "output_path": str,
            "sha256": str,
        }

    ``provider`` and ``provider_cache_root`` are forwarded to
    ``execute_staged_image_job`` so tests can inject a fake
    ``NativeImageProvider`` and a temp cache root; production callers omit
    them and the content-engine service resolves its own configured provider.
    """
    _ensure_content_engine_on_path()

    # Imported lazily so the helper import is cheap and never breaks the
    # CLI/gateway at startup when content-engine deps are absent.
    from image_job_service import (  # type: ignore[import-not-found]
        execute_staged_image_job,
        stage_and_plan_image_job,
    )
    from image_jobs import prepare_image_request  # type: ignore[import-not-found]

    prepared = prepare_image_request(
        prompt=prompt,
        style=style,
        backend=backend,
        references=references or (),
        blend=blend or (),
    )
    # Fail fast on a non-codex backend *before* staging, mirroring
    # content_engine.py:467-468.  Local execution is disabled pending
    # manual quality acceptance, so we must not waste staging work (or
    # worse, proceed) for backend='local'.
    if prepared.backend != "codex":
        from image_job_service import ImageExecutionError  # type: ignore[import-not-found]
        raise ImageExecutionError(
            "Local execution is disabled pending manual quality acceptance"
        )
    staged = stage_and_plan_image_job(
        prepared,
        staging_root=Path(stage_root),
        job_id=job_id,
    )
    completed = execute_staged_image_job(
        prepared,
        staged,
        staging_root=Path(stage_root),
        job_id=job_id,
        provider=provider,
        provider_cache_root=Path(provider_cache_root) if provider_cache_root else None,
        aspect_ratio=aspect_ratio,
    )
    return {
        "backend": {"provider": completed.provider, "model": completed.model},
        "completion_path": str(completed.completion_path),
        "job_id": job_id,
        "output_path": str(completed.output_path),
        "sha256": completed.sha256,
    }



def render_generate_image_command(
    *,
    prompt: str,
    style: str,
    backend: str = "codex",
    references: Optional[list[str]] = None,
    blend: Optional[list[str]] = None,
    stage_root: str,
    job_id: str,
    aspect_ratio: str = "landscape",
) -> str:
    """Render the exact canonical ``content_engine.py generate-image`` command.

    The rendered line is copy-paste executable from the repository root:
    it begins with ``PYTHONPATH=content_engine python3
    content_engine/content_engine.py generate-image ...`` so a user can
    paste it directly into a shell at the repo root and it runs the same
    content-engine image pipeline the in-process runner drives.  Every
    argument value is shell-quoted via :func:`shlex.quote` so the line is
    safe to copy-paste into a terminal.  Reference URLs are emitted as
    repeated ``--reference`` flags (one per URL), matching the argparse
    ``action="append"`` contract in ``content_engine.py``.  Blend slugs are
    emitted as repeated ``--blend`` flags.

    The returned string is the *canonical* form both the REPL and the
    gateway show to the user before approval, so the user can verify
    exactly what will run.
    """
    # Render a copy-paste-executable command rooted at the repository root:
    # the env-assignment + interpreter + script path means a user can paste
    # the line directly into a shell at the repo root and it will run the
    # same content-engine image pipeline the in-process runner drives.
    parts: list[str] = [
        "PYTHONPATH=content_engine",
        "python3",
        "content_engine/content_engine.py",
        "generate-image",
        f"--prompt {shlex.quote(prompt)}",
        f"--style {shlex.quote(style)}",
        f"--backend {shlex.quote(backend)}",
    ]
    for ref in references or ():
        parts.append(f"--reference {shlex.quote(ref)}")
    for slug in blend or ():
        parts.append(f"--blend {shlex.quote(slug)}")
    parts.append(f"--stage-root {shlex.quote(stage_root)}")
    parts.append(f"--job-id {shlex.quote(job_id)}")
    parts.append(f"--aspect-ratio {shlex.quote(aspect_ratio)}")
    return " ".join(parts)


__all__ = ["run_generate_image", "render_generate_image_command"]
