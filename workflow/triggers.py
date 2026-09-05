"""Register the world's start conditions against existing Hermes surfaces.

A trigger node is not a new HTTP stack. Cron expressions become cron jobs;
webhook specs become rows in ``webhook_subscriptions.json``. Both already
exist. This module is the sync so authoring a trigger on the canvas is
enough.
"""

from __future__ import annotations

import json
import logging
import secrets
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from utils import atomic_write_text

from workflow.store import load_documents, put_secret, secret_for, workflows_dir

logger = logging.getLogger(__name__)

ROUTE_PREFIX = "wf:"
_OWNED = "hermes_workflow"


def _scenario(doc: dict) -> dict:
    raw = doc.get("scenario")
    return raw if isinstance(raw, dict) else {"steps": [], "edges": []}


def _steps(doc: dict) -> list[dict]:
    steps = _scenario(doc).get("steps") or []
    return [s for s in steps if isinstance(s, dict)]


def _config(step: dict) -> dict:
    raw = step.get("config")
    return raw if isinstance(raw, dict) else {}


def trigger_of(doc: dict) -> dict | None:
    """The first trigger step — a workflow has one start condition."""
    for step in _steps(doc):
        kind = step.get("kind") or (step.get("def") or {}).get("kind")
        if kind != "trigger":
            continue
        on = _config(step).get("on") or {}
        return {"type": str(on.get("type") or "manual"), "spec": str(on.get("spec") or "").strip()}
    return None


def route_name(workflow_id: str) -> str:
    """Path segment. Once a hook is minted this is unguessable — that is the auth."""
    token = secret_for(workflow_id)
    if token:
        return f"wf-{token}"
    return f"{ROUTE_PREFIX}{workflow_id}"


def webhook_secret(workflow_id: str) -> str:
    existing = secret_for(workflow_id)
    if existing:
        return existing
    secret = secrets.token_hex(16)
    put_secret(workflow_id, secret)
    return secret


def hook_url(workflow_id: str) -> str:
    try:
        from hermes_cli.webhook import _get_webhook_base_url

        base = _get_webhook_base_url()
    except Exception:
        base = "http://localhost:8644"
    return f"{base}/webhooks/{route_name(workflow_id)}"


def hook_info(workflow_id: str) -> dict[str, str]:
    secret = webhook_secret(workflow_id)
    return {"route": route_name(workflow_id), "secret": secret, "url": hook_url(workflow_id)}


def ensure_webhook_platform() -> None:
    """Turn the existing webhook adapter on so the URL is a real listener."""
    try:
        from hermes_cli.config import write_platform_config_field
        from hermes_cli.webhook import _is_webhook_enabled

        if _is_webhook_enabled():
            return
        write_platform_config_field("webhook", "enabled", True)
    except Exception as exc:
        logger.debug("could not enable webhook platform: %s", exc)


def _subscriptions_path() -> Path:
    return get_hermes_home() / "webhook_subscriptions.json"


def _read_subscriptions() -> dict[str, Any]:
    path = _subscriptions_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def sync_webhook_routes(docs: list[dict] | None = None) -> dict[str, str]:
    """Write one dynamic route per webhook-triggered workflow. Leave user routes alone."""
    docs = docs if docs is not None else load_documents()["docs"]
    wanted: dict[str, dict] = {}
    secrets_out: dict[str, str] = {}
    for doc in docs:
        trigger = trigger_of(doc)
        if trigger is None or trigger["type"] != "webhook":
            continue
        wid = doc["id"]
        info = hook_info(wid)
        wanted[info["route"]] = {
            "secret": info["secret"],
            "workflow": wid,
            "prompt": "",
            "script": _write_hook_script(wid),
            _OWNED: True,
        }
        secrets_out[wid] = info
    existing = _read_subscriptions()
    kept = {k: v for k, v in existing.items() if not (isinstance(v, dict) and v.get(_OWNED))}
    merged = {**kept, **wanted}
    atomic_write_text(_subscriptions_path(), json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
    if wanted:
        ensure_webhook_platform()
    return secrets_out


def _owned_jobs() -> list[dict]:
    try:
        from cron.jobs import list_jobs
    except Exception:
        return []
    out = []
    for job in list_jobs(include_disabled=True):
        origin = job.get("origin") or {}
        if isinstance(origin, dict) and origin.get("kind") == "workflow":
            out.append(job)
    return out


def _script_path(workflow_id: str) -> Path:
    scripts = get_hermes_home() / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    return scripts / f"workflow_{workflow_id}.py"


def _write_hook_script(workflow_id: str) -> str:
    """A route script the existing webhook adapter already knows how to run.

    In-process ``start_run`` is the new adapter's path. This file is for a
    gateway that hasn't picked up that branch yet — it still executes ``script``.
    """
    path = get_hermes_home() / "scripts" / f"workflow_hook_{workflow_id}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "import json, sys\n"
        "from workflow.runner import start_from_trigger\n"
        "payload = json.loads(sys.stdin.read() or '{}')\n"
        f"start_from_trigger({workflow_id!r}, source='webhook', payload=payload)\n"
        "print('[SILENT]')\n",
        encoding="utf-8",
    )
    return path.name


def workflow_id_for_route(route_name: str, route_config: dict | None = None) -> str:
    if isinstance(route_config, dict):
        wid = str(route_config.get("workflow") or "").strip()
        if wid:
            return wid
    token = str(route_name)[3:] if str(route_name).startswith("wf-") else ""
    if not token:
        return ""
    from workflow.store import load_secrets

    for wid, secret in load_secrets().items():
        if secret == token:
            return wid
    return ""


def _write_tick_script(workflow_id: str) -> str:
    path = _script_path(workflow_id)
    path.write_text(
        "from workflow.runner import start_from_trigger\n"
        f"print(start_from_trigger({workflow_id!r}, source='cron'))\n",
        encoding="utf-8",
    )
    return path.name


def sync_cron_jobs(docs: list[dict] | None = None) -> list[str]:
    """Create or refresh a no-agent cron job for each cron trigger."""
    docs = docs if docs is not None else load_documents()["docs"]
    try:
        from cron.jobs import create_job, remove_job, update_job
    except Exception as exc:
        logger.debug("cron unavailable, skipping workflow cron sync: %s", exc)
        return []

    wanted: dict[str, str] = {}
    for doc in docs:
        trigger = trigger_of(doc)
        if trigger is None or trigger["type"] != "cron" or not trigger["spec"]:
            continue
        wanted[doc["id"]] = trigger["spec"]

    existing = { (job.get("origin") or {}).get("workflow_id"): job for job in _owned_jobs() }
    kept_ids = []
    for workflow_id, schedule in wanted.items():
        script = _write_tick_script(workflow_id)
        job = existing.get(workflow_id)
        if job is None:
            created = create_job(
                prompt="",
                schedule=schedule,
                name=f"workflow:{workflow_id}",
                script=script,
                no_agent=True,
                deliver="local",
                origin={"kind": "workflow", "workflow_id": workflow_id},
            )
            kept_ids.append(created["id"])
            continue
        updates: dict[str, Any] = {"script": script, "no_agent": True}
        if job.get("schedule_display") != schedule:
            updates["schedule"] = schedule
        update_job(job["id"], updates)
        kept_ids.append(job["id"])

    for workflow_id, job in existing.items():
        if workflow_id not in wanted:
            remove_job(job["id"])

    return kept_ids


def sync_triggers(docs: list[dict] | None = None) -> dict[str, Any]:
    docs = docs if docs is not None else load_documents()["docs"]
    return {
        "webhooks": sync_webhook_routes(docs),
        "cron": sync_cron_jobs(docs),
        "home": str(workflows_dir()),
    }
