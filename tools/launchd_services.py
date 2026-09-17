"""launchd-hosted services/dependencies for the cron interflow graph.

A dependency the agent runs under launchd on macOS — a brew service
(``postgresql@17``, ``redis``), a ``LaunchAgent``, a ``launchd``-supervised
daemon — escapes every liveness provider we have: it is not a tracked Hermes
process, not a Docker container, not a Nomad allocation. Unlike Docker labels
and Nomad meta, a launchd plist has **no free-form label store** an agent or
user can hang a dataflow declaration on, so the declaration lives in a
**sidecar registry**: one JSON file per service under
``~/.hermes/services/launchd/<service-label>.json``, keyed by the launchd
service label it describes. The sidecar is the durable object (it survives
restarts and gateway restarts); liveness is probed from the OS on every read
(``launchctl print`` → ``state``), mirroring the process-registry's
probe-don't-trust-bookkeeping fix: a dead service must drop off the graph
even if its sidecar remains.

Sidecar shape (``name`` + ``description`` required; rest optional):

  {
    "label":       "ai.hermes.gateway",      # launchd service label (required,
                                             # must match the file name)
    "name":        "Hermes Gateway",         # display name (required)
    "description": "The Hermes gateway …",   # markdown detail card (required)
    "inputs":      ["file:~/.hermes/config.yaml"],
    "outputs":     [],
    "side_effects":["https:127.0.0.1:8787"],
    "source_files":["scripts/gateway.py"],     # code behind the service,
                                               # browsable like a cron's scripts
    "service_health": {"type": "http", "url": "http://127.0.0.1:8787/health"}
  }

One JSON object per file; unknown keys ignored; malformed files are dropped
with a warning, never crash the overlay.
"""
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tools.service_health import normalize_service_health, probe_service_health

logger = logging.getLogger(__name__)

# The sidecar registry directory. Kept under HERMES_HOME alongside cron/ and
# artifacts/ — both durable operator-owned state.
_REGISTRY_DIR = Path.home() / ".hermes" / "services" / "launchd"

# launchctl print's per-service "state" values that mean RUNNING. Not an
# exhaustive machine-state table: only the states a healthy running service
# reports in practice (run loop entered / waiting in a run loop spin).
_RUNNING_STATES = frozenset({"running"})


def _registry_dir(home: Optional[str] = None) -> Path:
    if home:
        return Path(home) / "services" / "launchd"
    return _REGISTRY_DIR


def _parse_sidecar_to_declaration(
    path: Path, doc: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Build a validated service dict from one sidecar JSON document.

    Returns ``None`` when the document isn't a dict, its ``label`` doesn't
    match the file stem, or the declaration is invalid — one malformed sidecar
    drops that service, never crashes the overlay. Reuses
    ``normalize_service_declaration`` so a launchd service is validated by the
    same rules as process/Docker/Nomad services.
    """
    if not isinstance(doc, dict):
        return None
    label = doc.get("label")
    if not isinstance(label, str) or not label.strip():
        logger.warning("launchd sidecar %s has no 'label'; dropped", path.name)
        return None
    label = label.strip()
    if label != path.stem:
        logger.warning(
            "launchd sidecar %s declares label %r (expected %r); dropped",
            path.name, label, path.stem,
        )
        return None
    from cron.jobs import normalize_service_declaration

    try:
        decl = normalize_service_declaration(
            name=doc.get("name") or label,
            description=doc.get("description"),
            inputs=doc.get("inputs"),
            outputs=doc.get("outputs"),
            side_effects=doc.get("side_effects"),
            relationships=doc.get("relationships"),
            source_files=doc.get("source_files"),
        )
        health_spec = normalize_service_health(doc.get("service_health"))
    except ValueError as exc:
        logger.warning("launchd sidecar %s invalid: %s", path.name, exc)
        return None
    service = {
        # `launchd:` is not a dataflow scheme, so this id can never collide
        # with a resource ref, a cron id, or a docker:/nomad:/proc_ id.
        "id": f"launchd:{label}",
        "label": decl["name"],
        "description": decl["description"],
        "inputs": decl["inputs"],
        "outputs": decl["outputs"],
        "side_effects": decl["side_effects"],
        "source_files": decl["source_files"],
    }
    if decl.get("relationships"):
        service["relationships"] = decl["relationships"]
    if health_spec is not None:
        service["_health_spec"] = health_spec
    return service


def _default_runner(args: List[str]) -> str:
    """Run a read-only launchctl command and return stdout (raises on failure)."""
    return subprocess.check_output(
        args, text=True, stderr=subprocess.DEVNULL, timeout=5
    )


def _launchd_label_running(
    label: str, runner: Callable[[List[str]], str]
) -> bool:
    """True when ``launchctl print`` reports the label in a running state.

    ``launchctl print gui/<uid>/<label>`` emits a key-dump; the first
    ``state =`` line is the service's run-loop state. Non-zero exit (service
    not loaded / not found) counts as not-running. Best-effort: any probe
    failure returns False — the service drops off the graph rather than
    lingering as a stale node.
    """
    try:
        out = runner(["launchctl", "print", f"gui/{_uid()}/{label}"])
    except Exception:
        return False
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("state ="):
            value = stripped.split("=", 1)[1].strip().split()[0] if "=" in stripped else ""
            return value in _RUNNING_STATES
    return False


def _uid() -> int:
    import os

    getuid = getattr(os, "getuid", None)
    if getuid is None:
        raise RuntimeError("launchd service discovery requires POSIX UID support")
    return int(getuid())


def collect_launchd_services(
    runner: Callable[[List[str]], str] = _default_runner,
    registry_dir: Optional[Path] = None,
    health_prober: Callable[[Dict[str, Any]], Dict[str, Any]] = probe_service_health,
) -> List[Dict[str, Any]]:
    """Registered launchd services that are live right now, as graph nodes.

    Best-effort: a missing/empty registry directory, an unavailable launchctl,
    or per-sidecar errors all degrade to fewer-or-no services — never an
    exception into the graph. Only services whose OS probe says running are
    returned; presence == liveness, exactly like ``docker ps``.
    Shape matches ``cron.jobs.build_cron_graph(services=...)``.

    ``runner`` (launchctl) and ``registry_dir`` are injectable so the
    sidecar→declaration + probe logic is unit-testable without a live launchd.
    """
    root = registry_dir or _registry_dir()
    try:
        sidecars = sorted(p for p in root.glob("*.json") if p.is_file())
    except Exception:
        logger.debug("launchd registry unreadable for service overlay", exc_info=True)
        return []

    services: List[Dict[str, Any]] = []
    for path in sidecars:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning(
                "launchd sidecar %s is not valid JSON; dropped", path.name
            )
            continue
        decl = _parse_sidecar_to_declaration(path, doc)
        if decl is None:
            continue
        health_spec = decl.pop("_health_spec", None)
        label = decl["id"].split(":", 1)[1]
        if not _launchd_label_running(label, runner):
            logger.debug(
                "launchd service %s not running; skipped from overlay", label
            )
            continue
        if health_spec is not None:
            try:
                decl["health"] = health_prober(health_spec)
            except Exception as exc:
                logger.warning(
                    "launchd service %s health probe failed: %s", label, exc
                )
                decl["health"] = {
                    "status": "unhealthy",
                    "probe": health_spec["type"],
                    "target": health_spec["url"],
                    "checked_at": datetime.now(timezone.utc).isoformat().replace(
                        "+00:00", "Z"
                    ),
                    "latency_ms": 0,
                    "message": f"{type(exc).__name__}: {exc}",
                }
        else:
            decl["health"] = {
                "status": "unknown",
                "probe": "launchctl",
                "target": decl["id"],
                "checked_at": "",
                "latency_ms": 0,
                "message": "launchd lease running; application health not configured",
            }
        services.append(decl)
    return services
