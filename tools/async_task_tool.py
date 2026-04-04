"""
Async Task Tool -- Inter-Profile Fire-and-Forget Tasks

Launches a hermes sub-process using a different profile (e.g. 'researcher',
'dev') without blocking the caller's session. Returns immediately with a
task_id; the result is delivered to the originating chat when the subprocess
finishes (via AsyncTaskRegistry / _watch_async_tasks in gateway/run.py).

Usage (by AI model):
    async_task(profile="researcher", prompt="Find latest papers on ...", context="...")
"""

import logging
import os
import subprocess
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

ASYNC_TASK_TOOL_SCHEMA: Dict[str, Any] = {
    "name": "async_task",
    "description": (
        "Lancia un task asincrono su un profilo Hermes specifico (es. 'researcher', 'dev'). "
        "Ritorna immediatamente con un task_id. Il risultato viene recapitato automaticamente "
        "nella chat quando il task termina. NON blocca la sessione corrente."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "profile": {
                "type": "string",
                "description": "Nome del profilo hermes (es. 'researcher', 'dev')",
            },
            "prompt": {
                "type": "string",
                "description": "Il task da eseguire",
            },
            "context": {
                "type": "string",
                "description": "Contesto opzionale da passare al profilo",
            },
        },
        "required": ["profile", "prompt"],
    },
}


# ---------------------------------------------------------------------------
# Registry / source helpers
# ---------------------------------------------------------------------------

def _get_registry():
    from gateway.async_task_registry import async_task_registry
    return async_task_registry


def _build_source_from_env():
    """
    Reconstruct a SessionSource from the env vars set by the gateway
    (_set_session_env).  Returns None if the env vars are missing
    (e.g. when running in CLI mode where there is no delivery target).
    """
    platform_val = os.environ.get("HERMES_SESSION_PLATFORM")
    chat_id = os.environ.get("HERMES_SESSION_CHAT_ID")
    if not platform_val or not chat_id:
        return None
    try:
        from gateway.session import SessionSource
        from gateway.config import Platform
        return SessionSource(
            platform=Platform(platform_val),
            chat_id=chat_id,
            chat_name=os.environ.get("HERMES_SESSION_CHAT_NAME"),
            thread_id=os.environ.get("HERMES_SESSION_THREAD_ID") or None,
        )
    except Exception as exc:
        logger.warning("async_task: could not build source from env: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------

async def async_task(
    profile: str,
    prompt: str,
    context: Optional[str] = None,
) -> str:
    """
    Launch a fire-and-forget hermes sub-process on the given profile.

    Returns a confirmation string with the task_id.
    The actual result is delivered asynchronously via the registry.
    """
    registry = _get_registry()

    # Determine source for result delivery
    source = _build_source_from_env()
    if source is None:
        return (
            "❌ async_task: impossibile determinare la chat di origine. "
            "Questo tool deve essere eseguito all'interno di una sessione gateway Telegram/Discord/ecc."
        )

    # Validate profile exists
    hermes_home = os.path.expanduser("~/.hermes")
    profile_dir = os.path.join(hermes_home, "profiles", profile)
    if not os.path.isdir(profile_dir):
        return (
            f"❌ async_task: profilo '{profile}' non trovato. "
            f"Controlla ~/.hermes/profiles/{profile}/"
        )

    # Build the full prompt (with optional context)
    # Add MEDIA delivery instructions so file outputs are delivered correctly
    media_instructions = (
        "\n\n[ISTRUZIONI DELIVERY]: Se produci file (HTML, PDF, markdown, ecc.), "
        "includi nella tua risposta finale la riga `MEDIA:/path/assoluto/del/file` "
        "per ogni file da consegnare. Esempio: se salvi in /tmp/report.html, "
        "scrivi `MEDIA:/tmp/report.html` nella risposta. "
        "Il file verrà allegato automaticamente alla chat."
    )
    full_prompt = prompt + media_instructions
    if context and context.strip():
        full_prompt = f"{prompt}\n\nCONTEXT:\n{context}{media_instructions}"

    # Generate task ID
    task_id = f"async_{profile}_{uuid.uuid4().hex[:8]}"

    # Find hermes executable
    hermes_bin = _find_hermes_binary()
    if not hermes_bin:
        return (
            "❌ async_task: eseguibile 'hermes' non trovato in PATH. "
            "Assicurati che hermes sia installato e nel PATH."
        )

    # Launch subprocess non-blocking with Popen
    # --output-format json makes hermes emit a JSON object with final_response
    # so the watcher can extract only the clean answer, not the tool call log
    cmd = [hermes_bin, "-p", profile, "chat", "-Q", "-q", full_prompt, "--max-turns", "50", "--yolo"]
    env = os.environ.copy()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )
    except Exception as exc:
        logger.exception("async_task: failed to launch subprocess for profile=%s", profile)
        return f"❌ async_task: errore nel lancio del subprocess: {exc}"

    # Register in registry (async)
    await registry.register(
        task_id=task_id,
        profile=profile,
        prompt=prompt,
        source=source,
        process=process,
    )

    prompt_preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
    return (
        f"🚀 Async task avviato!\n"
        f"Task ID: {task_id}\n"
        f"Profile: {profile}\n"
        f'Prompt: "{prompt_preview}"\n\n'
        f"Il risultato arriverà in questa chat quando il task terminerà."
    )


def _find_hermes_binary() -> Optional[str]:
    """Locate the hermes binary in PATH or common locations."""
    import shutil
    candidate = shutil.which("hermes")
    if candidate:
        return candidate
    # Fallback: check common venv/install locations
    common = [
        "/root/.hermes/hermes-agent/venv/bin/hermes",
        "/usr/local/bin/hermes",
        os.path.expanduser("~/.local/bin/hermes"),
    ]
    for path in common:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def _check_requirements() -> bool:
    """async_task has no external requirements beyond hermes itself."""
    return True


from tools.registry import registry

registry.register(
    name="async_task",
    toolset="async",
    schema=ASYNC_TASK_TOOL_SCHEMA,
    handler=lambda args, **kw: async_task(
        profile=args.get("profile"),
        prompt=args.get("prompt"),
        context=args.get("context"),
    ),
    check_fn=_check_requirements,
    is_async=True,
    emoji="🚀",
)
