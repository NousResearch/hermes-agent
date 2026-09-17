"""Docker-hosted services/dependencies for the cron interflow graph.

A long-running dependency the agent starts with Docker — a Postgres, a Redis, a
dashboard container — escapes the process-lease model: ``docker run -d``
detaches, so there is no tracked ``ProcessSession`` to hang liveness on. But
Docker has a uniform control plane, so we track it the runtime-native way
instead. The container SELF-DECLARES its dataflow through ``hermes.*`` labels
(which persist on the container across detach, restart, and even a gateway
restart — the declaration lives with the durable object, not our memory), and
liveness is simply "does ``docker ps`` still list it". The container is the
lease, exactly like a tracked process is for a background service.

Labels (``hermes.service`` + ``hermes.description`` required; rest optional):
  hermes.service       display name — its presence marks a tracked service
  hermes.description   markdown, shown in Portal's node detail card (REQUIRED)
  hermes.inputs        comma/space-separated ``scheme:value`` reads
  hermes.outputs       comma/space-separated ``scheme:value`` writes
  hermes.side_effects  comma/space-separated ``scheme:value`` terminal actions
  hermes.relationships JSON array of ``{predicate, object}`` topology facts
  hermes.source_files  comma/space-separated file paths — the code behind the
                       service, browsable in the graph like a cron's scripts

Example — a dashboard that reads a table a cron writes converges with that cron
on the shared ``postgres:analytics.events`` node:

  docker run -d \\
    --label hermes.service="Analytics Dashboard" \\
    --label hermes.description="Renders analytics from the events table." \\
    --label hermes.inputs="postgres:analytics.events" \\
    my/dashboard
"""
import json
import logging
import re
import subprocess
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_SERVICE_LABEL = "hermes.service"


def _split_refs(value: Any) -> List[str]:
    """Split a label's comma/space-separated ref list into tokens.

    Docker labels are flat strings, so ``hermes.inputs`` arrives as e.g.
    ``"postgres:a, postgres:b"``. ``scheme:value`` refs never contain a comma or
    whitespace, so splitting on those is lossless.
    """
    if not isinstance(value, str):
        return []
    return [tok for tok in re.split(r"[,\s]+", value.strip()) if tok]


def _parse_relationships(value: Any) -> Any:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError("hermes.relationships must be a JSON string")
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("hermes.relationships must contain valid JSON") from exc


def _parse_labels_to_declaration(
    container_id: str, labels: Optional[Dict[str, str]]
) -> Optional[Dict[str, Any]]:
    """Build a validated service dict from a container's ``hermes.*`` labels.

    Returns ``None`` when the container isn't a tracked hermes service, or when
    its declaration is invalid (missing description, out-of-vocabulary scheme) —
    one malformed label set drops that container from the graph, never crashes
    the overlay. Reuses ``normalize_service_declaration`` so a Docker service and
    a process service are validated by exactly the same rules.
    """
    labels = labels or {}
    name = labels.get(_SERVICE_LABEL)
    if not name:
        return None
    from cron.jobs import normalize_service_declaration

    try:
        decl = normalize_service_declaration(
            name=name,
            description=labels.get("hermes.description"),
            inputs=_split_refs(labels.get("hermes.inputs")),
            outputs=_split_refs(labels.get("hermes.outputs")),
            side_effects=_split_refs(labels.get("hermes.side_effects")),
            relationships=_parse_relationships(labels.get("hermes.relationships")),
            source_files=_split_refs(labels.get("hermes.source_files")),
        )
    except ValueError as exc:
        logger.warning(
            "docker service %s has an invalid declaration: %s",
            container_id[:12],
            exc,
        )
        return None
    service = {
        # `docker:` is not a dataflow scheme, so this id can never collide with a
        # resource ref (postgres:/wiki:/…), a cron id (bare), or a proc_ id.
        "id": f"docker:{container_id[:12]}",
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
    """Run a read-only docker command and return stdout (raises on failure)."""
    return subprocess.check_output(
        args, text=True, stderr=subprocess.DEVNULL, timeout=5
    )


def collect_docker_services(
    runner: Callable[[List[str]], str] = _default_runner,
) -> List[Dict[str, Any]]:
    """Live Docker containers that self-declared a hermes service, as graph nodes.

    Best-effort: if Docker is not installed, the daemon is down, or anything
    errors, returns ``[]`` — a missing control plane must never sink the graph.
    Only RUNNING containers are listed (``docker ps``), so presence == liveness.
    Shape matches ``cron.jobs.build_cron_graph(services=...)``.

    ``runner`` is injectable so the label→declaration logic is unit-testable
    without a live Docker daemon.
    """
    try:
        out = runner([
            "docker", "ps", "--no-trunc",
            "--filter", f"label={_SERVICE_LABEL}",
            "--format", "{{.ID}}",
        ])
    except Exception:
        logger.debug("docker ps unavailable for service overlay", exc_info=True)
        return []

    services: List[Dict[str, Any]] = []
    for cid in (line.strip() for line in out.splitlines()):
        if not cid:
            continue
        try:
            raw = runner(
                ["docker", "inspect", "--format", "{{json .Config.Labels}}", cid]
            )
            labels = json.loads(raw.strip() or "null")
        except Exception:
            logger.debug("docker inspect failed for %s", cid[:12], exc_info=True)
            continue
        decl = _parse_labels_to_declaration(cid, labels)
        if decl is not None:
            try:
                native = runner([
                    "docker", "inspect", "--format",
                    "{{if .State.Health}}{{.State.Health.Status}}{{end}}", cid,
                ]).strip().lower()
            except Exception:
                native = ""
            status = native if native in {"healthy", "unhealthy", "starting"} else "unknown"
            decl["health"] = {
                "status": status,
                "probe": "docker-healthcheck",
                "target": decl["id"],
                "checked_at": "",
                "latency_ms": 0,
                "message": (
                    f"Docker healthcheck: {native}"
                    if native else "container has no Docker HEALTHCHECK"
                ),
            }
            services.append(decl)
    return services
