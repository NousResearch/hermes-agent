"""Dependency/security health snapshot tool.

Offline-safe, stdlib-only summary of Python and Node dependency manifests:
estimated package counts, lockfile freshness indicators, and any obvious
outdated markers already present in the files.

Success envelope
----------------
* ``python_dependencies_count``: int
* ``node_dependencies_count``: int
* ``lockfile_status``: mapping of manifest name -> metadata dict
* ``notices``: small list of plain-language findings, capped for readability

Failure envelope
----------------
Returns a zodern:relay-style failure with code ``dependency_health_error``
when the project root is not identifiable, no dependency manifests are
detectable, or a required manifest cannot be parsed.

Design constraints
------------------
* No new runtime dependencies: stdlib only + existing ``tools.validation_helper``.
* No network operations, no ``pip``/``npm`` calls.
* Reads local files only: ``uv.lock``, ``package-lock.json``,
  ``pyproject.toml``, and ``requirements*.txt``.
* ``check_fn`` returns ``False`` when no dependency manifests are detectable.
"""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from tools.validation_helper import relay_failure, relay_success

logger = logging.getLogger(__name__)

TOOL_NAME = "dependency_health"
FAILURE_CODE = "dependency_health_error"

MAX_NOTICES = 6
COUNT_CAP = 1000

try:
    import tomllib  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover - older runtime fallback
    tomllib = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
def _resolve_project_root() -> Path:
    return Path.cwd()


# ---------------------------------------------------------------------------
# Manifest detection gate
# ---------------------------------------------------------------------------
def _manifest_files(root: Path) -> List[Path]:
    """Return detectable dependency manifests under *root*."""
    return [
        root / "package.json",
        root / "pyproject.toml",
        *[Path(p) for p in glob.glob(str(root / "requirements*.txt"))],
    ]


def check_dependency_health_requirements() -> bool:
    """Return ``True`` when any dependency file exists for inspection."""
    root = _resolve_project_root()
    if not root.is_dir():
        return False
    if any(path.exists() for path in _manifest_files(root)):
        return True
    if (root / "uv.lock").exists() or (root / "package-lock.json").exists():
        return True
    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _count_uv_lock(root: Path) -> int:
    path = root / "uv.lock"
    if not path.exists():
        return 0
    data = _try_parse_toml(path)
    if not isinstance(data, dict):
        return 0
    count = 0
    for value in data.values():
        if isinstance(value, list):
            count += len(value)
        elif isinstance(value, dict):
            count += 1
    return min(count, COUNT_CAP)


def _count_package_json(root: Path) -> int:
    path = root / "package.json"
    if not path.exists():
        return 0
    data = _try_parse_json(path)
    if not isinstance(data, dict):
        return 0
    deps = data.get("dependencies") or {}
    dev = data.get("devDependencies") or {}
    return min(len(deps) + len(dev), COUNT_CAP)


def _count_package_lock(root: Path) -> int:
    path = root / "package-lock.json"
    if not path.exists():
        return 0
    data = _try_parse_json(path)
    if not isinstance(data, dict):
        return 0
    packages = data.get("packages")
    if isinstance(packages, dict):
        return min(len(packages), COUNT_CAP)
    deps = data.get("dependencies")
    if isinstance(deps, dict):
        return min(len(deps), COUNT_CAP)
    return 0


def _count_pyproject(root: Path) -> int:
    path = root / "pyproject.toml"
    if not path.exists() or tomllib is None:
        return 0
    text = _safe_read_text(path)
    if not text:
        return 0
    try:
        data = tomllib.loads(text)
    except Exception as exc:
        raise ValueError(f"pyproject.toml is malformed: {exc}") from exc
    project = data.get("project") or {}
    deps = project.get("dependencies") or []
    opt = project.get("optional-dependencies") or {}
    extra = sum(len(v) for v in opt.values() if isinstance(v, list))
    return min(len(deps) + extra, COUNT_CAP)


def _count_requirements(root: Path) -> tuple[List[str], int]:
    files = [Path(p) for p in glob.glob(str(root / "requirements*.txt"))]
    if not files:
        return [], 0
    total = 0
    for path in files:
        text = _safe_read_text(path)
        if not text:
            continue
        for line in text.splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                total += 1
    return [p.name for p in files], min(total, COUNT_CAP)


def _try_parse_json(path: Path) -> Any:
    text = _safe_read_text(path)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is malformed: {exc}") from exc


def _try_parse_toml(path: Path) -> Any:
    if tomllib is None:
        raise ValueError(f"{path.name} is a TOML file but no TOML parser is available")
    text = _safe_read_text(path)
    if text is None:
        return None
    try:
        return tomllib.loads(text)
    except Exception as exc:
        raise ValueError(f"{path.name} is malformed: {exc}") from exc


def _build_lockfile_status(root: Path) -> dict[str, Any]:
    status: dict[str, Any] = {}
    uv = root / "uv.lock"
    if uv.exists():
        entry: dict[str, Any] = {"present": True}
        proj = root / "pyproject.toml"
        if proj.exists():
            try:
                if uv.stat().st_mtime < proj.stat().st_mtime:
                    entry["freshness"] = "possibly_stale"
                    entry["detail"] = "uv.lock is older than pyproject.toml"
                else:
                    entry["freshness"] = "ok"
            except OSError:
                entry["freshness"] = "unknown"
        status[uv.name] = entry

    plock = root / "package-lock.json"
    if plock.exists():
        entry = {"present": True}
        pj = root / "package.json"
        if pj.exists():
            try:
                if plock.stat().st_mtime < pj.stat().st_mtime:
                    entry["freshness"] = "possibly_stale"
                    entry["detail"] = "package-lock.json is older than package.json"
                else:
                    entry["freshness"] = "ok"
            except OSError:
                entry["freshness"] = "unknown"
        status[plock.name] = entry

    req_files = [Path(p) for p in glob.glob(str(root / "requirements*.txt"))]
    if req_files:
        status.setdefault("requirements", {
            "present": True,
            "files": [p.name for p in sorted(req_files)],
            "note": "requirements files present alongside lockfiles",
        })

    pj = root / "package.json"
    if pj.exists():
        status.setdefault("package.json", {"present": True})

    return status


def _estimate_python_dependencies(root: Path) -> int:
    # Prefer declared direct deps.
    py = root / "pyproject.toml"
    if py.exists():
        py_count = _count_pyproject(root)
        if py_count:
            return py_count
    # requirements files
    _, req_count = _count_requirements(root)
    if req_count:
        return req_count
    return _count_uv_lock(root)


def _estimate_node_dependencies(root: Path) -> int:
    plock = root / "package-lock.json"
    if plock.exists():
        return _count_package_lock(root)
    pj = root / "package.json"
    if pj.exists():
        return _count_package_json(root)
    return 0


def _collect_obvious_outdated_markers(root: Path) -> List[str]:
    notices: List[str] = []
    req_files = [Path(p) for p in glob.glob(str(root / "requirements*.txt"))]
    for rfile in req_files:
        text = _safe_read_text(rfile)
        if not text:
            continue
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("-e git+") or s.startswith("-e hg+") or s.startswith("-e svn+"):
                notices.append(f"VCS dependency in {rfile.name}: {s[:80]}")
            elif s.startswith("-r "):
                notices.append(f"Include directive in {rfile.name}: {s[:80]}")
            else:
                dep_name = s.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].split("[")[0].split(";")[0].strip()
                if "==" not in s and ">=" not in s and "<=" not in s and "~=" not in s and dep_name:
                    notices.append(f"Possibly unpinned requirement in {rfile.name}: {s[:80]}")

    pj = root / "pyproject.toml"
    if pj.exists():
        text = _safe_read_text(pj) or ""
        if "git+" in text or "hg+" in text:
            notices.append("pyproject.toml contains VCS or URL dependency marker")

    if not (root / "uv.lock").exists() and (
        (root / "pyproject.toml").exists()
        or any(Path(p).exists() for p in glob.glob(str(root / "requirements*.txt")))
    ):
        notices.append("Python project manifest exists without uv.lock")

    if (root / "package-lock.json").exists() and not (root / "package.json").exists():
        notices.append("package-lock.json exists without package.json")

    return notices[:MAX_NOTICES]


# ---------------------------------------------------------------------------
# Public tool interface
# ---------------------------------------------------------------------------
def run() -> Dict[str, Any]:
    """Inspect local dependency manifests and return a success/failure envelope."""
    try:
        root = _resolve_project_root()
    except Exception as exc:
        logger.debug("project root resolution failed: %s", exc)
        return relay_failure(
            f"project root could not be resolved: {exc}",
            code=FAILURE_CODE,
            meta={"tool": TOOL_NAME},
        )

    if not root.is_dir():
        return relay_failure(
            "project root is not identifiable",
            code=FAILURE_CODE,
            meta={"tool": TOOL_NAME},
        )

    if not check_dependency_health_requirements():
        return relay_failure(
            "no dependency manifests are detectable",
            code=FAILURE_CODE,
            meta={"tool": TOOL_NAME},
        )

    try:
        py_count = _estimate_python_dependencies(root)
    except ValueError as exc:
        logger.debug("Python manifest parse failed: %s", exc)
        return relay_failure(str(exc), code=FAILURE_CODE, meta={"tool": TOOL_NAME})

    try:
        node_count = _estimate_node_dependencies(root)
    except ValueError as exc:
        logger.debug("Node manifest parse failed: %s", exc)
        return relay_failure(str(exc), code=FAILURE_CODE, meta={"tool": TOOL_NAME})

    lockfile_status = _build_lockfile_status(root)
    notices = _collect_obvious_outdated_markers(root)

    snapshot = {
        "python_dependencies_count": py_count,
        "node_dependencies_count": node_count,
        "lockfile_status": lockfile_status,
        "notices": notices,
    }
    return relay_success(snapshot)


DEP_HEALTH_SCHEMA: Dict[str, Any] = {
    "name": "dependency_health",
    "description": (
        "Return an offline-safe summary of dependency manifests: estimated Python "
        "and Node package counts, lockfile freshness indicators, and obvious outdated "
        "markers found in local files. No network calls are made."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


def register(registry):  # pragma: no cover - registration side-effect
    registry.register(
        name="dependency_health",
        toolset="ops",
        schema=DEP_HEALTH_SCHEMA,
        handler=run,
        check_fn=check_dependency_health_requirements,
        emoji="📦",
    )


__all__ = [
    "TOOL_NAME",
    "FAILURE_CODE",
    "check_dependency_health_requirements",
    "run",
    "register",
]
