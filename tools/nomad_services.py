"""Nomad-hosted services/dependencies for the cron interflow graph.

A dependency the agent runs on Nomad — like Docker's detached containers —
escapes the process-lease model: a raw_exec allocation has no tracked
``ProcessSession`` in Hermes, and the nomad agent may restart it out from
under any in-memory registry. But Nomad has a uniform control plane, so we
track it the runtime-native way. The job SELF-DECLARES its dataflow through
its ``meta`` block (which persists in the job spec across restarts and even
gateway restarts — the declaration lives with the durable object), and
liveness is "does the job have a RUNNING allocation". The allocation is the
lease, exactly like a tracked process or a Docker container.

Meta keys (``hermes_service`` + ``hermes_description`` required; rest optional;
Nomad meta keys reject dots, so the Docker label vocabulary maps
``hermes.service`` → ``hermes_service``):
  hermes_service       display name — its presence marks a tracked service
  hermes_description   markdown, shown in Portal's node detail card (REQUIRED)
  hermes_inputs        comma/space-separated ``scheme:value`` reads
  hermes_outputs       comma/space-separated ``scheme:value`` writes
  hermes_side_effects  comma/space-separated ``scheme:value`` terminal actions
  hermes_relationships JSON array of ``{predicate, object}`` topology facts
  hermes_source_files  comma/space-separated file paths — the code behind the
                       service, browsable in the graph like a cron's scripts

Example — an Honcho instance that reads Postgres and serves the memory API:

  job "honcho" {
    meta {
      hermes_service       = "Honcho Memory API"
      hermes_description   = "Dialectic memory server (#honcho). Reads sessions from Postgres and serves peer context over HTTP :8761."
      hermes_inputs        = "postgres:honcho.sessions"
      hermes_side_effects  = "https:127.0.0.1:8761"
    }
    ...
  }

CLI shapes consumed (read-only, tolerant of extra fields):
  nomad job status -json            → list of {"ID", "Type", "Status", ...}
  nomad job inspect <id> -json      → job spec with a "Meta" object
  nomad job allocs -json <id>       → list of {"ClientStatus", ...}
"""
import json
import logging
import re
import subprocess
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_SERVICE_META_KEY = "hermes_service"


def _split_refs(value: Any) -> List[str]:
    """Split a meta value's comma/space-separated ref list into tokens.

    Nomad meta values are flat strings, so ``hermes_inputs`` arrives as e.g.
    ``"postgres:a, postgres:b"``. ``scheme:value`` refs never contain a comma
    or whitespace, so splitting on those is lossless.
    """
    if not isinstance(value, str):
        return []
    return [tok for tok in re.split(r"[,\s]+", value.strip()) if tok]


def _parse_relationships(value: Any) -> Any:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError("hermes_relationships must be a JSON string")
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("hermes_relationships must contain valid JSON") from exc


def _parse_meta_to_declaration(
    job_id: str, meta: Optional[Dict[str, str]]
) -> Optional[Dict[str, Any]]:
    """Build a validated service dict from a job's ``hermes_*`` meta keys.

    Returns ``None`` when the job isn't a tracked hermes service, or when its
    declaration is invalid (missing description, out-of-vocabulary scheme) —
    one malformed job drops that service from the graph, never crashes the
    overlay. Reuses ``normalize_service_declaration`` so a Nomad service, a
    Docker service, and a process service are validated by the same rules.
    """
    meta = meta or {}
    name = meta.get(_SERVICE_META_KEY)
    if not name:
        return None
    from cron.jobs import normalize_service_declaration

    try:
        decl = normalize_service_declaration(
            name=name,
            description=meta.get("hermes_description"),
            inputs=_split_refs(meta.get("hermes_inputs")),
            outputs=_split_refs(meta.get("hermes_outputs")),
            side_effects=_split_refs(meta.get("hermes_side_effects")),
            relationships=_parse_relationships(meta.get("hermes_relationships")),
            source_files=_split_refs(meta.get("hermes_source_files")),
        )
    except ValueError as exc:
        logger.warning("nomad job %s has an invalid declaration: %s", job_id, exc)
        return None
    service = {
        # `nomad:` is not a dataflow scheme, so this id can never collide with a
        # resource ref (postgres:/wiki:/…), a cron id (bare hex), or a docker:
        # /proc_ id.
        "id": f"nomad:{job_id}",
        "label": decl["name"],
        "description": decl["description"],
        "inputs": decl["inputs"],
        "outputs": decl["outputs"],
        "side_effects": decl["side_effects"],
        "source_files": decl["source_files"],
    }
    if decl.get("relationships"):
        service["relationships"] = decl["relationships"]
    return service


def _default_runner(args: List[str]) -> str:
    """Run a read-only nomad command and return stdout (raises on failure)."""
    return subprocess.check_output(
        args, text=True, stderr=subprocess.DEVNULL, timeout=5
    )


def collect_nomad_services(
    runner: Callable[[List[str]], str] = _default_runner,
) -> List[Dict[str, Any]]:
    """Live Nomad jobs that self-declared a hermes service, as graph nodes.

    Best-effort: if the nomad CLI is missing, the agent/server is down, or
    anything errors, returns ``[]`` — a missing control plane must never sink
    the graph. Liveness requires at least one allocation with
    ``ClientStatus == "running"`` (a job in `running` desired state whose
    allocations all failed is NOT live — a stale node is worse than a missing
    one, since the graph is used to reason about what is actually running).
    Shape matches ``cron.jobs.build_cron_graph(services=...)``.

    ``runner`` is injectable so the meta→declaration logic is unit-testable
    without a live Nomad agent.
    """
    try:
        raw = runner(["nomad", "job", "status", "-json"])
        jobs = json.loads(raw.strip() or "[]")
    except Exception:
        logger.debug("nomad job status unavailable for service overlay", exc_info=True)
        return []
    if not isinstance(jobs, list):
        logger.debug("nomad job status -json returned non-list; skipping overlay")
        return []

    services: List[Dict[str, Any]] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        # Only long-running `service`-type jobs in the running state can be
        # live dependencies — batch/system jobs are not services here.
        if job.get("Type") != "service" or job.get("Status") != "running":
            continue
        job_id = job.get("ID") or job.get("Name")
        if not job_id:
            continue

        try:
            spec = json.loads(
                runner(["nomad", "job", "inspect", "-json", str(job_id)])
            )
        except Exception:
            logger.debug("nomad job inspect failed for %s", job_id, exc_info=True)
            continue
        meta = spec.get("Meta") if isinstance(spec, dict) else None
        decl = _parse_meta_to_declaration(str(job_id), meta)
        if decl is None:
            continue

        try:
            allocs = json.loads(
                runner(["nomad", "job", "allocs", "-json", str(job_id)])
            )
        except Exception:
            logger.debug("nomad job allocs failed for %s", job_id, exc_info=True)
            continue
        if not isinstance(allocs, list):
            continue
        if not any(
            isinstance(a, dict) and a.get("ClientStatus") == "running"
            for a in allocs
        ):
            continue  # desired-running but nothing actually serving — not live

        decl["health"] = {
            "status": "unknown",
            "probe": "nomad-allocation",
            "target": decl["id"],
            "checked_at": "",
            "latency_ms": 0,
            "message": "Nomad allocation running; application health unavailable",
        }
        services.append(decl)
    return services
