"""Register the world's start conditions against existing Hermes surfaces.

A trigger node is not a new HTTP stack. Cron expressions become cron jobs; a webhook trigger becomes
a row in ``webhook_subscriptions.json``. Both already exist; this module is the sync, so authoring
a trigger on the canvas is enough. Both ends only publish a ``workflow.cmd.start`` event — the
reactor starts the run (``workflow/reactor.py``). Event triggers need no sync at all: the reactor
matches them against every event it drains.

A workflow may have any number of triggers. Each cron trigger is its own job; the webhook route
is one per workflow and starts it from its first webhook trigger.
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
from workflow.topology import config_of, kind_of, scenario_of, steps_of

logger = logging.getLogger(__name__)

ROUTE_PREFIX = "wf:"
_OWNED = "hermes_workflow"


def triggers_of(doc: dict) -> list[dict[str, str]]:
    """Every trigger step: ``{"nodeId", "type", "spec"}``."""
    out = []
    for step in steps_of(scenario_of(doc)):
        if kind_of(step) != "trigger":
            continue
        on = config_of(step).get("on") or {}
        out.append({"nodeId": step["id"], "type": str(on.get("type") or "manual"), "spec": str(on.get("spec") or "").strip()})
    return out


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
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def sync_webhook_routes(docs: list[dict] | None = None) -> dict[str, dict[str, str]]:
    """Write one dynamic route per webhook-triggered workflow. Leave user routes alone."""
    docs = docs if docs is not None else load_documents()["docs"]
    wanted: dict[str, dict] = {}
    secrets_out: dict[str, dict[str, str]] = {}
    for doc in docs:
        hooks = [t for t in triggers_of(doc) if t["type"] == "webhook"]
        if not hooks:
            continue
        wid = doc["id"]
        info = hook_info(wid)
        wanted[info["route"]] = {
            "secret": info["secret"], "workflow": wid, "trigger": hooks[0]["nodeId"], "prompt": "", _OWNED: True,
        }
        secrets_out[wid] = info
    existing = _read_subscriptions()
    kept = {k: v for k, v in existing.items() if not (isinstance(v, dict) and v.get(_OWNED))}
    atomic_write_text(_subscriptions_path(), json.dumps({**kept, **wanted}, indent=2, ensure_ascii=False) + "\n")
    if wanted:
        ensure_webhook_platform()
    return secrets_out


def is_workflow_capability_route(route: str, route_config: dict) -> bool:
    """A hook this store minted: the ownership marker, the workflow it names, and the route the
    store derives from that workflow's persisted secret all agree. Only such a route's URL is a
    credential; a route merely called ``wf-*`` is an ordinary HMAC route."""
    if not isinstance(route_config, dict) or route_config.get(_OWNED) is not True:
        return False
    wid = str(route_config.get("workflow") or "").strip()
    return bool(wid) and bool(secret_for(wid)) and route_name(wid) == route


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


def _write_tick_script(workflow_id: str, node_id: str) -> str:
    """A no-agent cron script: publish the start; the reactor runs it."""
    scripts = get_hermes_home() / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    path = scripts / f"workflow_{workflow_id}_{node_id}.py"
    atomic_write_text(
        path,
        "from workflow import events\n"
        f"events.publish(events.START, {{'workflowId': {workflow_id!r}, 'trigger': {node_id!r}, 'source': 'cron'}},"
        " source='cron')\n"
        "print('[SILENT]')\n",
    )
    return path.name


def sync_cron_jobs(docs: list[dict] | None = None) -> list[str]:
    """Create or refresh one no-agent cron job per cron trigger."""
    docs = docs if docs is not None else load_documents()["docs"]
    try:
        from cron.jobs import create_job, remove_job, update_job
    except Exception as exc:
        logger.debug("cron unavailable, skipping workflow cron sync: %s", exc)
        return []

    wanted: dict[tuple[str, str], str] = {}
    for doc in docs:
        for trig in triggers_of(doc):
            if trig["type"] == "cron" and trig["spec"]:
                wanted[(doc["id"], trig["nodeId"])] = trig["spec"]

    def key_of(job: dict) -> tuple[str, str]:
        origin = job.get("origin") or {}
        return str(origin.get("workflow_id") or ""), str(origin.get("node_id") or "")

    existing = {key_of(job): job for job in _owned_jobs()}
    kept_ids = []
    for (workflow_id, node_id), schedule in wanted.items():
        script = _write_tick_script(workflow_id, node_id)
        job = existing.get((workflow_id, node_id))
        if job is None:
            created = create_job(
                prompt="", schedule=schedule, name=f"workflow:{workflow_id}:{node_id}", script=script,
                no_agent=True, deliver="local",
                origin={"kind": "workflow", "workflow_id": workflow_id, "node_id": node_id},
            )
            kept_ids.append(created["id"])
            continue
        updates: dict[str, Any] = {"script": script, "no_agent": True}
        if job.get("schedule_display") != schedule:
            updates["schedule"] = schedule
        update_job(job["id"], updates)
        kept_ids.append(job["id"])
    for key, job in existing.items():
        if key not in wanted:
            remove_job(job["id"])
    return kept_ids


def sync_triggers(docs: list[dict] | None = None) -> dict[str, Any]:
    docs = docs if docs is not None else load_documents()["docs"]
    return {"webhooks": sync_webhook_routes(docs), "cron": sync_cron_jobs(docs), "home": str(workflows_dir())}
