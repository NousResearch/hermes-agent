"""Hermes-owned PYTHONPATH stripping for child processes. Launchers prepend the repo
root and the Hermes venv's site-packages so the backend can ``import tools``; leaked
into a child Python of a DIFFERENT version they load the backend's C extensions and
crash. Only entries proven Hermes-owned by *path provenance* are removed — never by a
cross-version heuristic. Module state (``_hermes_repo_root_aliases``, ``_in_venv``,
``_hermes_site_packages``) lives in ``tools.environments.local`` (via :func:`_state`)
so tests monkeypatching it there keep working."""

import logging
import os
import platform
import sys
from pathlib import Path

from tools.environments.local_env_policy import _ACTIVE_VENV_MARKER_VARS

_IS_WINDOWS = platform.system() == "Windows"

logger = logging.getLogger("tools.environments.local")


def _state():
    """Return the ``tools.environments.local`` module (owner of the caches)."""
    from tools.environments import local

    return local


def _same_path(left: Path, right: Path) -> bool:
    """Compare path spellings with host filesystem case semantics."""
    return [os.path.normcase(p) for p in left.parts] == [os.path.normcase(p) for p in right.parts]


def _build_hermes_repo_root_aliases(
    resolved_root: Path, lexical_root: Path, configured_home: Path,
) -> tuple[Path, ...]:
    """Exact repo-root spellings emitted by Hermes launchers. Mirrors
    ``gateway_windows._preserve_hermes_home_path`` (physical path under the resolved
    HERMES_HOME -> configured spelling) so a junction-backed install matches without
    treating arbitrary HERMES_HOME descendants as Hermes-owned. A repo-level junction
    (possibly cross-drive) is accepted only when a strict resolve proves
    <root>/<repo dirname> is the physical root (fail-closed)."""
    candidates = [resolved_root, lexical_root]
    # Profile re-home: with --profile the configured home is <root>/profiles/<name>
    # and the repo lives beside the profiles dir (as get_default_hermes_root() does).
    home_candidates = [configured_home]
    if configured_home.parent.name == "profiles":
        home_candidates.append(configured_home.parent.parent)
    for home in home_candidates:
        try:
            resolved_home = home.resolve()
            home_key = os.path.normcase(str(resolved_home))
            if os.path.commonpath([home_key, os.path.normcase(str(resolved_root))]) == home_key:
                candidates.append(home / os.path.relpath(str(resolved_root), str(resolved_home)))
        except (OSError, ValueError):
            pass
    # Repo-level junction recovery (commonpath raises across drives, so the
    # home-relative mapping above cannot express a cross-drive link).
    for home in home_candidates:
        repo_candidate = home / resolved_root.name
        try:
            if repo_candidate.resolve(strict=True) == resolved_root.resolve(strict=True):
                candidates.append(repo_candidate)
        except OSError:
            pass
    aliases: list[Path] = []
    for candidate in candidates:
        if not any(_same_path(candidate, existing) for existing in aliases):
            aliases.append(candidate)
    return tuple(aliases)


def _validated_runtime_venv(env: dict) -> Path | None:
    """Producer-owned runtime venv identified by VIRTUAL_ENV, or None. The variable
    alone is not provenance (users carry unrelated venvs): require the legacy Windows
    base-Python producer's exact ``<repo>/venv`` layout AND a real ``pyvenv.cfg``."""
    candidate = Path(env.get("VIRTUAL_ENV") or "")
    if not env.get("VIRTUAL_ENV") or not any(
            _same_path(candidate, root / "venv") for root in _state()._hermes_repo_root_aliases):
        return None
    try:
        return candidate if (candidate / "pyvenv.cfg").is_file() else None
    except OSError:
        return None


def _get_hermes_site_packages(env: dict) -> list[Path]:
    """Exact site-packages dirs owned by the Hermes runtime (cached):
    ``site.getsitepackages()`` with a ``sys.prefix`` fallback, plus a validated
    Windows base-interpreter launch's ``VIRTUAL_ENV/Lib/site-packages``."""
    local = _state()
    if local._hermes_site_packages is None:
        result: list[Path] = []
        if local._in_venv:
            try:
                import site
                result.extend(Path(sp) for sp in site.getsitepackages())
            except Exception:
                pass
            if not result:
                pyver = f"python{sys.version_info[0]}.{sys.version_info[1]}"
                result.append(Path(sys.prefix) / "Lib" / "site-packages" if _IS_WINDOWS
                              else Path(sys.prefix) / "lib" / pyver / "site-packages")
        local._hermes_site_packages = list(result)
    result = list(local._hermes_site_packages)

    runtime_venv = _validated_runtime_venv(env)
    if runtime_venv is not None:
        runtime_site_packages = runtime_venv / "Lib" / "site-packages"
        if not any(_same_path(runtime_site_packages, existing) for existing in result):
            result.append(runtime_site_packages)
    return result


def _strip_hermes_owned_pythonpath_and_runtime_markers(env: dict) -> None:
    """Strip Hermes-owned PYTHONPATH entries, then the runtime marker vars. Order is
    load-bearing: PYTHONPATH filtering runs BEFORE the markers go so a validated Windows
    base-interpreter launch (VIRTUAL_ENV -> <repo>/venv) can still prove ownership."""
    _strip_hermes_owned_pythonpath(env)
    for _marker in _ACTIVE_VENV_MARKER_VARS:
        env.pop(_marker, None)


def _strip_hermes_owned_pythonpath(env: dict) -> None:
    """Remove Hermes-owned PYTHONPATH entries: only exact matches of the repo root
    (any launcher spelling) and runtime site-packages — never descendants, which are
    user paths. Empty components (= cwd) and everything else are preserved.

    Everything else -- user libs, Nix plugin paths, a pythonX.Y/site-packages entry meant for a DIFFERENT
    child version -- is preserved byte-for-byte: ownership is decided by path provenance, never by a
    cross-version heuristic (#74817 follow-up).
    """
    pp = env.get("PYTHONPATH")
    if not pp:
        return
    owned_paths = [*_get_hermes_site_packages(env), *_state()._hermes_repo_root_aliases]
    entries = pp.split(os.pathsep)
    stripped = [e for e in entries if e and any(_same_path(Path(e), p) for p in owned_paths)]
    kept = [e for e in entries if e not in stripped]
    if kept:
        env["PYTHONPATH"] = os.pathsep.join(kept)
    else:
        env.pop("PYTHONPATH", None)
    if stripped:
        logger.debug("Stripped Hermes-owned entries from PYTHONPATH: %s", stripped)
