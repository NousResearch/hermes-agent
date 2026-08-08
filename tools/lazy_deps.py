"""
Lazy dependency installer for opt-in Hermes Agent backends.

Many Hermes features (Mistral TTS, ElevenLabs TTS, Honcho memory, Bedrock,
Slack, Matrix, etc.) need Python packages that not every user wants. Each
one installs at first use, for two reasons. One quarantined or yanked
release on PyPI must not fail the whole resolve and cost a fresh install
ten unrelated extras. And a user who talks to one provider must not pull
hundreds of packages that they never import.

Backends call :func:`ensure` at the
top of their first-import path. If the deps are missing, ``ensure`` checks
the ``security.allow_lazy_installs`` config flag (default true) and runs
a venv-scoped pip install. If the user has explicitly disabled lazy
installs, ``ensure`` raises :class:`FeatureUnavailable` with a clear
remediation hint pointing at ``hermes tools`` or the manual pip command.

Security model:

* **Venv-scoped by default.** Installs target ``sys.executable`` in the
  active venv. We never touch the system Python.
* **Sealed deployments.** The Docker image sets
  ``HERMES_DISABLE_LAZY_INSTALLS=1`` and makes ``/opt/hermes`` read-only.
  Hermes refuses every install there. The image contains each extra that
  works in a container. A lazy install in the image means that the image
  does not have a dependency that it must ship.

* **Durable-target mode.** ``HERMES_LAZY_INSTALL_TARGET`` sends installs to
  a writable directory instead of the venv. The published image sets it to
  ``/opt/data/lazy-packages``, for :func:`install_specs` only: a plugin's
  packages come from its manifest, so no build can bake them. Hermes
  appends the directory to the END of ``sys.path``. It never prepends the
  directory, and it never exports ``PYTHONPATH``. The site-packages of the
  agent thus wins each name collision, and a package installed this way
  can only ADD modules.
* **PyPI by package name only.** Specs may be ``"package>=1.0,<2"`` etc.
  We do NOT support ``--index-url`` overrides, ``git+https://``, file:
  paths, or any other input that could be hijacked by a malicious config.
* **Allowlist.** Only specs that appear in :data:`LAZY_DEPS` can be
  installed via this path. A typo in feature name doesn't get the user
  install-anything semantics.
* **Opt-out.** Setting ``security.allow_lazy_installs: false`` in
  ``config.yaml`` disables runtime installs in BOTH modes. Users in
  restricted networks or strict security postures can pin themselves to
  whatever was installed at setup time.
* **Offline detection.** If the install fails (offline, mirror down,
  PyPI 404 / quarantine), we surface the failure as
  :class:`FeatureUnavailable` with the actual pip stderr — no silent
  retries, no caching of bad state.

Adding a new backend:

1. Add the packages as an extra in pyproject.toml, then map the feature
   to that extra in :data:`LAZY_DEPS`.
2. At the top of the backend module's import path, call
   ``ensure("feature.name")`` inside a try/except that converts
   :class:`FeatureUnavailable` to a useful runtime error.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import shutil
import site
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_cli._subprocess_compat import windows_hide_flags

logger = logging.getLogger(__name__)


# =============================================================================
# Feature to extra map.
#
# Each key is a feature name with a dot ("namespace.backend"). Each value is
# the name of the ``[project.optional-dependencies]`` extra in pyproject.toml
# that holds the packages for that backend.
#
# pyproject.toml holds the specs. No other file holds them. Do not add a
# table of pins to this module. Such a table cannot read ``[tool.uv]
# override-dependencies``, so a backend that holds a security-pinned package
# below its patched version downgrades that package at first use.
# =============================================================================


LAZY_DEPS: dict[str, str] = {
    # ─── Inference providers ───────────────────────────────────────────────
    "provider.anthropic": "anthropic",
    "provider.bedrock": "bedrock",
    "provider.vertex": "vertex",
    "provider.azure_identity": "azure-identity",

    # ─── Web search backends ───────────────────────────────────────────────
    "search.exa": "exa",
    "search.firecrawl": "firecrawl",
    "search.parallel": "parallel-web",

    # ─── Monitoring ────────────────────────────────────────────────────────
    "export.otlp": "otlp",

    # ─── Speech to text ────────────────────────────────────────────────────
    # stt-whisper, not voice: this feature transcribes audio files, which
    # include voice notes that arrive over the network. It must not pull
    # the microphone stack in, and the Docker image bakes it.
    "stt.faster_whisper": "stt-whisper",
    "stt.mistral": "mistral",
    "stt.silk": "silk",

    # ─── Text to speech ────────────────────────────────────────────────────
    "tts.edge": "edge-tts",
    "tts.elevenlabs": "tts-premium",
    "tts.mistral": "mistral",

    # ─── Wake word engines ─────────────────────────────────────────────────
    "wake.openwakeword": "wake-openwakeword",
    "wake.openwakeword.tflite": "wake-tflite",
    "wake.sherpa": "wake-sherpa",
    "wake.porcupine": "wake-porcupine",

    # ─── Image generation backends ─────────────────────────────────────────
    "image.fal": "fal",

    # ─── Memory providers ──────────────────────────────────────────────────
    "memory.honcho": "honcho",
    "memory.hindsight": "hindsight",
    "memory.supermemory": "supermemory",
    "memory.mem0": "mem0",

    # ─── Messaging platforms ───────────────────────────────────────────────
    "platform.telegram": "telegram",
    "platform.discord": "discord",
    "platform.slack": "slack",
    "platform.matrix": "matrix",
    "platform.dingtalk": "dingtalk",
    "platform.feishu": "feishu",
    "platform.wecom_callback": "wecom",
    "platform.teams": "teams",

    # ─── Terminal backends ─────────────────────────────────────────────────
    "terminal.modal": "modal",
    "terminal.daytona": "daytona",
    "terminal.vercel": "vercel",

    # ─── Skills ────────────────────────────────────────────────────────────
    "skill.google_workspace": "google",
    "skill.youtube": "youtube",

    # ─── Tools ─────────────────────────────────────────────────────────────
    # [acp] has no entry here on purpose. The ACP entry point is a console
    # script, so its dependency must exist before the agent loop starts. It
    # ships in [all] instead, and an extra cannot be in both.
    "tool.dashboard": "web",
    "tool.computer_use": "computer-use",
    "tool.trace_upload": "trace-upload",
    "tool.doc_extract": "doc-extract",
}


# =============================================================================
# pyproject extra -> specs
# =============================================================================


def _project_root() -> Optional[Path]:
    """Return the root directory that holds pyproject.toml, or None.

    Hermes supports two install types. ``install.sh`` clones the repository,
    and the Docker image copies ``pyproject.toml`` and ``uv.lock`` to its
    WORKDIR. Each other layout, such as a copy in site-packages, has no
    project root. The extras table then comes from the dist metadata
    instead (see :func:`_metadata_optional_dependencies`).
    """
    root = Path(__file__).resolve().parent.parent
    return root if (root / "pyproject.toml").is_file() else None


@functools.lru_cache(maxsize=1)
def _pyproject() -> dict:
    """Parse pyproject.toml once, or return {} when it is not on disk.

    A Nix build puts the code in site-packages with no pyproject.toml beside
    it, so callers must handle an empty result.
    """
    root = _project_root()
    if root is None:
        return {}
    try:
        return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug("Could not read pyproject.toml: %s", e)
        return {}


def _optional_dependencies() -> dict[str, tuple[str, ...]]:
    """Return ``[project.optional-dependencies]``.

    pyproject.toml is the primary source. On a checkout it is ahead of the
    installed dist metadata. A wheel install, such as Nix, does not have
    the file on disk. There the same table comes from the dist metadata.
    """
    raw = _pyproject().get("project", {}).get("optional-dependencies", {}) or {}
    if raw:
        return {k: tuple(v) for k, v in raw.items()}
    return _metadata_optional_dependencies()


# Finds the ``extra == "name"`` clause that setuptools appends to the marker
# of each Requires-Dist line that belongs to an extra.
_EXTRA_CLAUSE = re.compile(r"""\bextra\s*==\s*["']([^"']+)["']""")


@lru_cache(maxsize=1)
def _metadata_optional_dependencies() -> dict[str, tuple[str, ...]]:
    """The extras table, read from the installed dist metadata.

    A wheel install has no pyproject.toml beside the code. The dist-info
    carries the same table: each spec of an extra becomes one
    ``Requires-Dist`` line, and its marker holds ``extra == "name"``. A
    pin's own marker is ANDed on, for example ``platform_system ==
    "Darwin" and extra == "wake-tflite"``. Remove the extra clause and
    keep the rest of the marker.

    Without this fallback, each lazy_deps entry point raised on a Nix
    install. ensure() raised even for a feature whose packages were baked
    through extraDependencyGroups, and that call must be a no-op.
    """
    try:
        from importlib.metadata import metadata

        md = metadata("hermes-agent")
    except Exception as e:
        logger.debug("Could not read hermes-agent dist metadata: %s", e)
        return {}
    table: dict[str, list[str]] = {}
    for raw in md.get_all("Requires-Dist") or []:
        base, sep, marker = raw.partition(";")
        if not sep:
            continue  # core dependency — not part of any extra
        m = _EXTRA_CLAUSE.search(marker)
        if not m:
            continue
        rest = (marker[: m.start()] + marker[m.end() :]).strip()
        rest = re.sub(r"^\s*and\s+|\s+and\s*$", "", rest).strip()
        spec = base.strip() + (f"; {rest}" if rest else "")
        table.setdefault(m.group(1), []).append(spec)
    return {k: tuple(v) for k, v in table.items()}


_SELF_REF = re.compile(r"^hermes[-_]agent\[([^\]]+)\]$", re.IGNORECASE)


def extra_specs(extra: str, _seen: Optional[frozenset] = None) -> tuple[str, ...]:
    """Return the specs for ``extra`` and expand each ``hermes-agent[...]``.

    An extra can contain other extras. ``[messaging]`` contains
    ``hermes-agent[telegram]``, ``hermes-agent[discord]`` and
    ``hermes-agent[slack]``. This function expands each such reference. If
    the references make a loop, or point to an extra that does not exist,
    the function returns nothing and does not repeat forever.

    A marker belongs on the pin inside the extra that holds it, not on the
    reference. _is_satisfied reads the marker, so a spec for another
    platform needs no install here.
    """
    seen = _seen or frozenset()
    if extra in seen:
        logger.debug("Cyclic extra reference at %r — stopping", extra)
        return ()
    table = _optional_dependencies()
    if extra not in table:
        return ()
    seen = seen | {extra}
    out: list[str] = []

    def _add(spec: str) -> None:
        if spec not in out:
            out.append(spec)

    for spec in table[extra]:
        m = _SELF_REF.match(spec)
        if m:
            for sub in m.group(1).split(","):
                for nested in extra_specs(sub.strip(), seen):
                    _add(nested)
        else:
            _add(spec)
    return tuple(out)


def _anchor_spec(extra: str, _seen: Optional[frozenset] = None) -> Optional[str]:
    """Return the spec that identifies ``extra``: its first direct pin.

    ``extra_specs`` expands ``hermes-agent[...]`` references in place, so
    its first element can be a shared helper from a composed extra —
    ``[voice]`` starts with ``hermes-agent[audio-io]``, and expansion puts
    ``sounddevice`` first. sounddevice is in every audio extra, so it
    identifies none of them. The pin that identifies an extra is the first
    one written directly in it (``faster-whisper`` for ``[voice]``).

    Only when an extra holds nothing but references (``[computer-use]`` is
    ``hermes-agent[mcp]`` alone) does this recurse into the first reference.
    """
    seen = _seen or frozenset()
    if extra in seen:
        return None
    table = _optional_dependencies()
    if extra not in table:
        return None
    seen = seen | {extra}

    refs: list[str] = []
    for spec in table[extra]:
        m = _SELF_REF.match(spec)
        if m:
            refs.extend(sub.strip() for sub in m.group(1).split(","))
        else:
            return spec
    for ref in refs:
        found = _anchor_spec(ref, seen)
        if found:
            return found
    return None


class FeatureUnavailable(RuntimeError):
    """A lazily-installable feature is missing and cannot be made available.

    Either the deps were never installed and the user has disabled lazy
    installs, or the install attempt failed.
    """

    def __init__(
        self,
        feature: str,
        missing: tuple[str, ...],
        reason: str,
        *,
        actionable: bool = True,
    ):
        self.feature = feature
        self.missing = missing
        self.reason = reason
        # Set this to False to remove the "install it yourself" footer. A
        # sealed Docker venv and a package-manager install are both
        # read-only, so the user cannot run the command. A command that
        # always fails is worse than no command.
        self.actionable = actionable
        super().__init__(self._format())

    def _format(self) -> str:
        base = f"Feature {self.feature!r} unavailable: {self.reason}"
        if not self.actionable or not self.missing:
            return base
        spec_list = " ".join(repr(s) for s in self.missing)
        return (
            f"{base}. "
            f"To enable manually: uv pip install {spec_list}  "
            f"(or: pip install {spec_list})."
        )


@dataclass(frozen=True)
class _InstallResult:
    success: bool
    stdout: str
    stderr: str


# =============================================================================
# Internals
# =============================================================================


# Environment variable that sends lazy installs to a writable directory on a
# durable volume instead of the agent venv. The published image sets it to
# /opt/data/lazy-packages. There ensure() still refuses (the image bakes
# every extra it can run), so the directory serves install_specs alone.
# This is an internal bridge variable, not configuration for the user. The
# control for the user is security.allow_lazy_installs in config.yaml. When
# the variable is empty, lazy installs go into the active venv.
_LAZY_TARGET_ENV = "HERMES_LAZY_INSTALL_TARGET"

# Name of the stamp file written into the target dir recording the Python
# X.Y + ABI it was populated for. If a container rebuild bumps the
# interpreter, compiled wheels (.so) in the durable store would be ABI-
# incompatible; we detect the mismatch and wipe the store so packages get
# re-resolved against the new interpreter rather than importing a stale .so.
_TARGET_STAMP_NAME = ".python-abi"


def _python_abi_tag() -> str:
    """A stable token identifying the running interpreter's ABI.

    Combines the X.Y version with the EXT_SUFFIX (which encodes the ABI
    tag and platform, e.g. ``cpython-313-x86_64-linux-gnu``). Two
    interpreters that can share compiled wheels produce the same token.
    """
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    ext = sysconfig.get_config_var("EXT_SUFFIX") or ""
    return f"{ver}:{ext}"


def _lazy_install_target() -> Optional[Path]:
    """Return the durable install-target dir, or None for venv-scoped mode.

    Returns a path only when :data:`_LAZY_TARGET_ENV` is set to a non-empty
    value. The directory is created on demand by :func:`_ensure_target_ready`.
    """
    raw = os.environ.get(_LAZY_TARGET_ENV, "").strip()
    if not raw:
        return None
    return Path(raw)


def _ensure_target_ready(target: Path) -> Optional[str]:
    """Create the target dir and validate its ABI stamp.

    If the stamp is missing it is written. If it is present but records a
    different interpreter ABI than the one now running (e.g. the container
    image was rebuilt onto a newer Python), the directory's contents are
    wiped and the stamp rewritten, so stale compiled wheels can't be
    imported against an incompatible interpreter.

    Returns ``None`` on success, or an error string if the directory can't
    be created / written (e.g. read-only mount, permission error).
    """
    want = _python_abi_tag()
    stamp = target / _TARGET_STAMP_NAME
    try:
        if target.exists():
            have = ""
            try:
                have = stamp.read_text(encoding="utf-8").strip()
            except (OSError, FileNotFoundError):
                have = ""
            if have and have != want:
                logger.info(
                    "Lazy install target %s was built for ABI %r but running "
                    "ABI is %r; wiping stale packages.",
                    target, have, want,
                )
                for child in target.iterdir():
                    if child.is_dir() and not child.is_symlink():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        try:
                            child.unlink()
                        except OSError:
                            pass
        target.mkdir(parents=True, exist_ok=True)
        stamp.write_text(want, encoding="utf-8")
    except OSError as e:
        return f"lazy install target {target} is not writable: {e}"
    return None


def _activate_target_on_syspath(target: Path) -> None:
    """Append the durable target to ``sys.path`` so its packages import.

    Appended to the END (never prepended) so the agent's own venv
    site-packages takes precedence on every name collision. Idempotent.
    Uses :func:`site.addsitedir` so ``.pth`` files (namespace packages,
    editable installs) inside the target are honoured, then enforces the
    append ordering — ``addsitedir`` would otherwise insert near the front.
    """
    target_str = str(target)
    # Snapshot existing entries so we can restore precedence afterwards.
    before = list(sys.path)
    if target_str not in before:
        site.addsitedir(target_str)
    # site.addsitedir may have inserted target (and any .pth-added dirs) at
    # the front. Move every newly-added entry to the end, preserving the
    # core venv's precedence. New entries are those not present `before`.
    new_entries = [p for p in sys.path if p not in before]
    if new_entries:
        sys.path[:] = [p for p in sys.path if p not in new_entries] + new_entries
    # importlib.metadata caches the path-based distribution finder; clear it
    # so a just-activated dir is visible to version() checks this process.
    try:
        import importlib
        importlib.invalidate_caches()
    except Exception:
        pass


def activate_durable_lazy_target() -> None:
    """Public: wire the durable lazy-install target onto ``sys.path``.

    Safe no-op when :data:`_LAZY_TARGET_ENV` is unset or the directory does
    not yet exist. Called once early in process startup (before backends
    import) so packages installed into the durable store on a previous run
    are importable on this run. Never raises.
    """
    target = _lazy_install_target()
    if target is None:
        return
    try:
        if target.exists():
            _activate_target_on_syspath(target)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Failed to activate durable lazy target %s: %s", target, e)


# One wording for the config kill switch. ensure() and install_specs both
# report it, and two spellings of the same cause read like two causes.
_CONFIG_DISABLED_REASON = "lazy installs disabled (security.allow_lazy_installs=false)"


def managed_install_reason(feature: str, extra: Optional[str] = None) -> str:
    """Return the message for an install that this deployment cannot run.

    Each caller reaches this when Hermes cannot install a package at run
    time. The remedy differs by deployment, so name the deployment and give
    the command or option that works there.

    ``extra`` is the pyproject extra that holds the packages, when the
    caller knows it. The NixOS remedy needs that name.

    Public on purpose: plugin setup flows (google_chat, honcho) report the
    same remedies when their own install paths cannot run.
    """
    # Check the package manager first. A managed install can also carry
    # HERMES_DISABLE_LAZY_INSTALLS, and the remedy for that user is the
    # package manager, not a bug report about a container image.
    #
    # get_managed_system() returns the string "NixOS" for each Nix install.
    # That value is an identifier, not a platform: `nix profile install` and
    # nix-darwin give the same value on a host that does not run NixOS. The
    # message below therefore says Nix, and gives both ways to set the
    # option.
    managed_by = _managed_system()
    if managed_by == "NixOS":
        target = f'"{extra}"' if extra else "the extra for this feature"
        return (
            "this build comes from Nix, and the /nix/store is read-only, so "
            f"Hermes cannot install packages at run time. Add {target} to "
            "extraDependencyGroups and rebuild. That option puts the extra "
            "into the sealed venv. On NixOS, set "
            "services.hermes-agent.extraDependencyGroups. Elsewhere, use "
            "pkgs.hermes-agent.override { extraDependencyGroups = [ ... ]; }. "
            "For a package that pyproject.toml does not declare, use "
            "extraPythonPackages instead."
        )
    if managed_by:
        return (
            f"this build comes from {managed_by}, so Hermes cannot install "
            f"packages at run time. Add the dependencies for {feature!r} "
            f"through {managed_by}."
        )

    if os.environ.get("HERMES_DISABLE_LAZY_INSTALLS") == "1":
        return (
            "runtime dependency installs are disabled in this deployment "
            "(HERMES_DISABLE_LAZY_INSTALLS=1). The container image contains "
            "each backend that it can run, so this is probably a bug in the "
            "image build. Please report it. Do not install the package into "
            "the container. /opt/hermes is read-only, and the next image "
            "update removes the change."
        )

    return _CONFIG_DISABLED_REASON


def _managed_system() -> str:
    """Return the name of the package manager that owns this install."""
    try:
        from hermes_cli.config import get_managed_system

        return get_managed_system() or ""
    except Exception:
        return ""


def _sealed_venv_reason() -> Optional[str]:
    """Return why a sealed deployment refuses an install, or None."""
    if os.environ.get("HERMES_DISABLE_LAZY_INSTALLS") != "1":
        return None
    return managed_install_reason("", None)


def _allow_lazy_installs() -> bool:
    """Return whether lazy installs are permitted in this environment.

    Hermes reads two controls, in this order:

    1. ``security.allow_lazy_installs: false`` in config.yaml. This is the
       control for the user, and it stops every install.
    2. ``HERMES_DISABLE_LAZY_INSTALLS=1``, which the Docker image sets. This
       control also stops every install. The image contains each extra that
       works in a container, so no correct install remains at run time.

    The default is True. If Hermes cannot read the config, it permits the
    install. A refusal locks the user out of a backend that the user owns,
    so the user must select the refusal.
    """
    # (1) Config kill switch wins in every mode.
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception:
        cfg = None
    if cfg is not None:
        sec = cfg.get("security") or {}
        if not bool(sec.get("allow_lazy_installs", True)):
            return False

    # (2) Sealed deployment. The image contains each extra that a container
    # can run, so a LAZY_DEPS feature never needs an install there.
    #
    # install_specs is different. Its specs come from a plugin manifest, and
    # a plugin outside this repository declares packages that pyproject.toml
    # does not hold, so the image cannot have baked them. Hindsight appends
    # `hindsight-all` at setup time for the same reason. Sealing those off
    # would stop a user installing a memory provider in the container at all.
    # HERMES_LAZY_INSTALL_TARGET names a writable directory on the data
    # volume for exactly that case.
    if os.environ.get("HERMES_DISABLE_LAZY_INSTALLS") == "1":
        return _lazy_install_target() is not None

    return True


def _unsupported_feature_reason(feature: str) -> Optional[str]:
    """Return why a lazy feature cannot work on this host, or ``None``.

    This is a platform capability gate, not a security policy gate. It keeps
    known-impossible installs out of both first-use lazy installation and the
    ``hermes update`` lazy-refresh pass.
    """
    if sys.platform == "win32" and feature == "platform.matrix":
        return (
            "unsupported on Windows: Matrix E2EE depends on python-olm, "
            "which has no Windows wheel and requires make + libolm to build "
            "from sdist. Run Hermes under WSL to use Matrix on Windows."
        )
    return None


def _parse_spec(spec: str):
    """Parse a PEP 508 spec, or return None when it is not usable.

    ``packaging`` is a core dependency, so use it. A regex over a spec has
    to re-handle the extras block, the version set and the environment
    marker, and getting the marker wrong makes a specifier unparseable
    ("==2.1.6; platform_system == 'Darwin'").

    Import it here, not at module scope. hermes_bootstrap imports this
    module during startup, before a broken venv has been repaired, and a
    missing package must not stop Hermes from starting.
    """
    try:
        from packaging.requirements import InvalidRequirement, Requirement
    except ImportError:  # pragma: no cover - packaging is a core dependency
        return None
    try:
        return Requirement(spec)
    except InvalidRequirement:
        return None


def _pkg_name_from_spec(spec: str) -> str:
    """Return the bare package name, or the input when it does not parse."""
    req = _parse_spec(spec)
    return req.name if req else spec


def _is_satisfied(spec: str) -> bool:
    """Is ``spec`` already met in this environment?

    Checks the version, not only presence, so `hermes update` carries a pin
    bump to a backend that a user installed at an older version.

    A spec whose marker is false for this host counts as met. There is
    nothing to install: ``ai-edge-litert`` is for macOS, and asking pip for
    it on Linux gets an error, not a package.

    ``SpecifierSet.contains`` covers the rest. An empty specifier accepts
    any version, and a version string it cannot read gives False, which
    reinstalls and repairs the entry.
    """
    req = _parse_spec(spec)
    if req is None:
        return True
    if req.marker is not None and not req.marker.evaluate():
        return True

    from importlib.metadata import version

    try:
        installed = version(req.name)
    except Exception:
        # PackageNotFoundError is the normal miss; anything else (broken
        # dist-info metadata) also means "not usable, reinstall".
        return False
    return req.specifier.contains(installed, prereleases=True)


def _is_present(spec: str) -> bool:
    """Is the package installed, at any version?

    :func:`active_features` uses this to find the backends that a user
    turned on. A moved pin must still count as active, so drop the version
    and ask only about the name.
    """
    return _is_satisfied(_pkg_name_from_spec(spec))


def _run(
    cmd: list[str],
    *,
    timeout: int,
    env: Optional[dict] = None,
    check: bool = False,
):
    """Run ``cmd`` and capture its output.

    One place for the flags each install command needs: capture the output,
    decode it without raising on a byte that does not fit the locale, give
    the child no stdin so a prompt cannot hang the agent, and hide the
    console window on Windows.

    Call ``subprocess.run`` through the module attribute. The tests replace
    that attribute to read the argv, so an imported ``run`` would bypass
    them.
    """
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
        check=check,
        stdin=subprocess.DEVNULL,
        creationflags=windows_hide_flags(),
    )


def _write_temp_requirements(lines, prefix: str) -> Optional[Path]:
    """Write ``lines`` to a temporary requirements file and return its path.

    Returns None for an empty list, and None when the write fails. Each
    caller treats None as "run the install without this file".
    """
    lines = list(lines)
    if not lines:
        return None
    try:
        fd, path = tempfile.mkstemp(prefix=prefix, suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return Path(path)
    except Exception as e:
        logger.debug("Could not write %s file: %s", prefix, e)
        return None


def _core_constraints_file() -> Optional[Path]:
    """Write a pip constraints file pinning every package already importable
    in the core environment to its installed version.

    Passed as ``--constraint`` for durable-target installs so the resolver
    pins shared transitive deps (httpx, pydantic, aiohttp, …) to the exact
    versions the core venv already ships, instead of pulling newer copies
    into the durable store. Two payoffs:

    * The durable store stays minimal — only genuinely-new packages land
      there; shared deps resolve to "already satisfied" against core.
    * A backend that *requires* a version conflicting with core fails loudly
      at install time (resolver conflict) rather than silently installing a
      shadowed copy that can never win on sys.path anyway.

    Returns the path to a temp constraints file, or None if enumeration
    failed (in which case the caller installs without constraints — still
    safe, just less tidy).
    """
    try:
        from importlib.metadata import distributions
    except ImportError:
        return None
    try:
        lines = []
        seen = set()
        for dist in distributions():
            name = dist.metadata["Name"] if dist.metadata else None
            ver = dist.version
            if not name or not ver:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"{name}=={ver}")
        return _write_temp_requirements(sorted(lines), "hermes-core-constraints-")
    except Exception as e:
        logger.debug("Could not build core constraints file: %s", e)
        return None


# Hermes applies these overrides to each lazy install. They repeat
# ``[tool.uv] override-dependencies`` in pyproject.toml.
#
# ``uv pip install`` and ``pip install`` do not read ``[tool.uv]``. Thus a
# transitive dependency can hold a security-pinned package below its patched
# version, and the first use of that backend downgrades the core venv.
#
# Example, measured with cryptography. The core venv has 50.0.0. The user
# enables DingTalk, which needs ``alibabacloud-dingtalk``, which needs
# ``alibabacloud-tea-openapi==0.4.5``, which holds ``cryptography<49``. The
# install gives::
#
#     + cryptography==48.0.1     # three open advisories, again
#
# A pin next to the specs does not correct this. The resolver obeys the pin
# and moves ``alibabacloud-tea-openapi`` back to 0.3.16, an sdist build from
# two years ago. A pin on both packages has no solution. Only an overrides
# file keeps the patched version and the correct backend version together,
# so Hermes gives the file to the uv tier below.
@lru_cache(maxsize=1)
def _security_overrides() -> tuple[str, ...]:
    """Return ``[tool.uv] override-dependencies`` from pyproject.toml.

    Read the list, instead of duplicating, to avoid drift.
    """
    raw = (
        _pyproject().get("tool", {}).get("uv", {}).get("override-dependencies", [])
        or []
    )
    return tuple(str(s) for s in raw)


def _security_overrides_file() -> Optional[Path]:
    """Write the overrides to a temporary file for ``--overrides``.

    Returns the path, or None when Hermes cannot write the file. The caller
    then installs without the overrides, and the downgrade that these
    prevent becomes possible again.
    """
    return _write_temp_requirements(
        _security_overrides(), "hermes-lazy-overrides-"
    )


def _pip_reassert_overrides(
    pip_cmd: list[str],
    target_args: list[str],
    *,
    timeout: int,
):
    """Install the overrides again with ``--no-deps`` after pip runs.

    pip has no ``--overrides`` option. A ``--constraint`` file does keep the
    pinned package, but pip then moves the backend back instead
    (alibabacloud-tea-openapi 0.4.5 to 0.3.16, an sdist from two years ago).
    A second pass with ``--no-deps`` prevents this. The pass changes only the
    overridden package and keeps each other package that pip resolved.

    Returns the failed ``CompletedProcess`` if the second pass gave an error.
    Returns None if the pass succeeded, or if there was no work. The caller
    then keeps its own result. This function reports a failure, because a
    downgraded security package is the fault that it must prevent.
    """
    overrides = _security_overrides()
    if not overrides:
        return None
    try:
        r = _run(
            pip_cmd + ["install", "--no-deps", *target_args, *overrides],
            timeout=timeout,
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("pip override re-assert failed to run: %s", e)
        return None
    if r.returncode != 0:
        logger.warning(
            "pip override re-assert failed (rc=%d); a security-pinned package "
            "may have been downgraded by this install: %s",
            r.returncode, (r.stderr or "").strip()[:400],
        )
        return r
    return None


def _uv_sync_extra(feature: str) -> Optional[_InstallResult]:
    """Install the extra of ``feature`` with ``uv sync``.

    Hermes tries ``uv sync`` first. It is the only installer that reads
    ``uv.lock`` and applies ``[tool.uv] override-dependencies``. It thus
    installs the versions that CI examined, and it applies the security
    overrides that ``uv pip`` and ``pip`` cannot read.

    Returns None in these conditions, and the caller then uses the pip
    tiers:

    * A durable install target is active. That mode installs to a different
      directory, so that it cannot change the sealed venv. ``uv sync``
      controls a full venv and has no equal to ``--target``.
    * There is no project root that holds ``uv.lock`` and ``pyproject.toml``.
    * uv is not available.
    * pyproject.toml does not declare the extra of the feature.

    The ``--inexact`` flag is necessary. A plain ``uv sync`` removes each
    package outside the extras that it syncs, and this removes every other
    backend that the user enabled. The ``--no-install-project`` flag stops
    uv from installing Hermes over an editable checkout.
    """
    if _lazy_install_target() is not None:
        return None
    root = _project_root()
    if root is None or not (root / "uv.lock").is_file():
        return None
    extra = LAZY_DEPS.get(feature)
    if extra is None or extra not in _optional_dependencies():
        return None

    try:
        from hermes_cli.managed_uv import resolve_uv

        uv_bin = resolve_uv() or shutil.which("uv")
    except Exception:
        uv_bin = shutil.which("uv")
    if not uv_bin:
        return None

    try:
        from tools.environments.local import hermes_subprocess_env

        env = hermes_subprocess_env(inherit_credentials=False)
    except Exception:
        env = dict(os.environ)
    # uv sync targets the project environment; point it at the running venv so
    # a lazy install lands where the agent will import from.
    env["UV_PROJECT_ENVIRONMENT"] = str(Path(sys.executable).parent.parent)
    # --locked needs [tool.uv] visible; UV_NO_CONFIG would drop exclude-newer.
    env.pop("UV_NO_CONFIG", None)

    cmd = [
        uv_bin, "sync",
        # uv finds the project from its own working directory. The agent
        # runs from the user's working directory, not from the install
        # tree, so name the project. Without this flag the sync fails in
        # the wrong directory and this tier never runs.
        "--project", str(root),
        "--extra", extra,
        "--inexact",
        "--locked",
        "--no-install-project",
        "--python", sys.executable,
    ]
    try:
        r = _run(cmd, timeout=600, env=env)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.debug("uv sync unavailable (%s) — falling back to pip ladder", e)
        return None
    if r.returncode == 0:
        logger.info("Installed extra [%s] for feature %r via uv sync", extra, feature)
        return _InstallResult(True, r.stdout or "", r.stderr or "")
    # A stale lockfile (--locked refuses) or any other sync failure falls back
    # rather than hard-failing: the pip ladder can still install the specs.
    logger.debug(
        "uv sync --extra %s failed (rc=%d), falling back: %s",
        extra, r.returncode, (r.stderr or "").strip()[:300],
    )
    return None


def _venv_pip_install(specs: tuple[str, ...], *, timeout: int = 300) -> _InstallResult:
    """Install ``specs`` using the uv → pip → ensurepip ladder.

    Two modes:

    * **Venv-scoped (default).** Installs into the active venv
      (``sys.executable``). Used on normal installs.
    * **Durable-target.** When :data:`_LAZY_TARGET_ENV` is set, installs into
      that directory via ``--target`` and constrains shared deps to the
      core venv's versions (see :func:`_core_constraints_file`). The target
      is append-only on ``sys.path`` so it can never shadow core. Used by
      the immutable Docker image to keep lazy installs off the sealed venv.

    Mirrors the strategy in ``hermes_cli.tools_config._pip_install`` but
    kept independent here so this module has no CLI dependency.
    """
    if not specs:
        return _InstallResult(True, "", "")

    target = _lazy_install_target()
    constraints: Optional[Path] = None

    if target is not None:
        err = _ensure_target_ready(target)
        if err:
            return _InstallResult(False, "", err)
        constraints = _core_constraints_file()

    overrides = _security_overrides_file()

    def _flag(name: str, value) -> list[str]:
        return [] if value is None else [name, str(value)]

    # --target tells both uv and pip to install into an arbitrary dir.
    target_args = _flag("--target", target)
    constraint_args = _flag("--constraint", constraints)
    # uv-only: pip has no --overrides. See _security_overrides().
    override_args = _flag("--overrides", overrides)

    try:
        venv_root = Path(sys.executable).parent.parent
        from tools.environments.local import hermes_subprocess_env
        uv_env = hermes_subprocess_env(inherit_credentials=False)
        uv_env["VIRTUAL_ENV"] = str(venv_root)

        # Tier 1: uv (preferred — fast, doesn't need pip in the venv)
        # Managed uv first: $HERMES_HOME/bin is never on PATH, so a bare
        # which() misses the uv Hermes installed and falls through to the
        # slower pip tier. Deliberately a lookup and not ensure_uv(): this runs
        # mid-turn to install an optional dependency, and downloading uv +
        # migrating the Python runtime as a side effect of that is a far bigger
        # action than the caller asked for. Tier 2 pip covers the no-uv case.
        try:
            from hermes_cli.managed_uv import resolve_uv

            uv_bin = resolve_uv() or shutil.which("uv")
        except Exception:
            uv_bin = shutil.which("uv")
        if uv_bin:
            try:
                r = _run(
                    [uv_bin, "pip", "install", *target_args,
                     *constraint_args, *override_args, *specs],
                    timeout=timeout, env=uv_env,
                )
                if r.returncode == 0:
                    if target is not None:
                        _activate_target_on_syspath(target)
                    return _InstallResult(True, r.stdout or "", r.stderr or "")
                logger.debug("uv pip install failed: %s", r.stderr)
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                logger.debug("uv invocation failed: %s", e)

        # Tier 2: python -m pip (with ensurepip bootstrap if needed)
        pip_cmd = [sys.executable, "-m", "pip"]
        try:
            probe = _run(pip_cmd + ["--version"], timeout=15)
            if probe.returncode != 0:
                raise FileNotFoundError("pip not in venv")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            try:
                _run(
                    [sys.executable, "-m", "ensurepip", "--upgrade",
                     "--default-pip"],
                    timeout=120, check=True,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                return _InstallResult(False, "",
                                      f"pip not available and ensurepip failed: {e}")

        try:
            r = _run(
                pip_cmd + ["install", *target_args, *constraint_args, *specs],
                timeout=timeout,
            )
            if r.returncode == 0:
                # pip has no --overrides, so a backend whose metadata caps a
                # security-pinned package below its floor has just downgraded
                # it. Re-assert the floor with --no-deps, which rewrites only
                # the overridden package and leaves the backend at the version
                # pip resolved. Measured on the DingTalk case: this yields
                # cryptography 50.0.0 AND alibabacloud-tea-openapi 0.4.5 —
                # identical to uv's --overrides. (`pip check` will report the
                # violated cap afterwards; that is what an override IS, and uv
                # produces the same end state without the diagnostic.)
                repair = _pip_reassert_overrides(pip_cmd, target_args, timeout=timeout)
                if repair is not None:
                    r = repair
            if r.returncode == 0 and target is not None:
                _activate_target_on_syspath(target)
            return _InstallResult(r.returncode == 0, r.stdout or "", r.stderr or "")
        except subprocess.TimeoutExpired as e:
            return _InstallResult(False, "", f"pip install timed out: {e}")
        except Exception as e:
            return _InstallResult(False, "", f"pip install failed: {e}")
    finally:
        for tmp in (constraints, overrides):
            if tmp is not None:
                try:
                    tmp.unlink()
                except OSError:
                    pass


# =============================================================================
# Public API
# =============================================================================


def feature_specs(feature: str) -> tuple[str, ...]:
    """Return the specs for ``feature``, read from its pyproject extra.

    Raises KeyError for an unknown feature, and FeatureUnavailable if the
    feature maps to an extra that pyproject doesn't define (a mapping typo, or
    a stripped install with no pyproject) — failing loudly beats installing
    nothing and reporting success.
    """
    extra = LAZY_DEPS[feature]
    specs = extra_specs(extra)
    if not specs:
        raise FeatureUnavailable(
            feature,
            (),
            f"feature {feature!r} maps to extra [{extra}], which resolved to no "
            f"packages. Either [{extra}] does not exist, or neither "
            f"pyproject.toml (root: {_project_root()!r}) nor the hermes-agent "
            f"dist metadata is readable here.",
        )
    return specs


def feature_missing(feature: str) -> tuple[str, ...]:
    """Return the subset of specs for ``feature`` not currently installed."""
    return tuple(s for s in feature_specs(feature) if not _is_satisfied(s))


def ensure(feature: str, *, prompt: bool = True) -> None:
    """Make sure all packages for ``feature`` are importable.

    If they're missing, attempts to install them in the active venv. Raises
    :class:`FeatureUnavailable` if the user has disabled lazy installs or
    if the install attempt fails.

    ``prompt``: when True (default) and stdin is a TTY, asks the user to
    confirm before installing. Non-interactive callers (gateway, cron,
    batch) get prompt=False and skip the confirmation — config flag is
    the gate in that case.
    """
    if feature not in LAZY_DEPS:
        raise FeatureUnavailable(
            feature, (), f"feature {feature!r} not in LAZY_DEPS allowlist"
        )

    missing = feature_missing(feature)
    if not missing:
        # The backend is in use with everything installed. Record the use,
        # so `hermes update` refreshes this feature when a pin moves.
        _record_feature_use(feature)
        return

    unsupported = _unsupported_feature_reason(feature)
    if unsupported:
        raise FeatureUnavailable(feature, missing, unsupported)

    # Package-manager installs (NixOS, and any other distro that ships Hermes
    # from a read-only store) cannot receive lazy pip installs: the venv's
    # site-packages lives in the store, so the uv -> pip -> ensurepip ladder
    # below burns ~15s bootstrapping ensurepip only to fail on a read-only
    # target. Fail fast with an actionable message instead.
    #
    # Skipped when a durable install target is configured: the container
    # deployment sets HERMES_MANAGED=true *and* HERMES_LAZY_INSTALL_TARGET
    # (a writable volume), where lazy installs legitimately work.
    #
    # The reason string starts with "unsupported " on purpose:
    # refresh_active_features classifies FeatureUnavailable by that prefix and
    # reports anything else as a hard failure rather than a skip.
    if _lazy_install_target() is None:
        managed_by = _managed_system()
        if managed_by:
            raise FeatureUnavailable(
                feature, missing,
                "unsupported on a managed install: "
                + managed_install_reason(feature, LAZY_DEPS.get(feature)),
                # The store is read-only. A `uv pip install` hint here
                # fails with EROFS.
                actionable=False,
            )


    # A sealed image contains each extra that a container can run, so a
    # LAZY_DEPS feature must never install here, even when a durable target
    # exists. That target is for install_specs, whose packages come from a
    # plugin manifest and cannot be in the image.
    sealed = _sealed_venv_reason()
    if sealed is not None:
        raise FeatureUnavailable(feature, missing, sealed, actionable=False)

    if not _allow_lazy_installs():
        raise FeatureUnavailable(feature, missing, _CONFIG_DISABLED_REASON)

    # Only show the interactive confirmation when we own a TTY and
    # prompt_toolkit isn't running.  A bare input() deadlocks when a
    # prompt_toolkit app owns the terminal because keystrokes route to
    # its event loop rather than stdin, so the prompt blocks forever.
    # Under the TUI we skip the prompt and proceed — lazy installs are
    # gated by security.allow_lazy_installs, so reaching here is
    # already user opt-in.
    _pt_active = False
    if "prompt_toolkit.application.current" in sys.modules:
        try:
            from prompt_toolkit.application.current import get_app_or_none
            _app = get_app_or_none()
            _pt_active = _app is not None and getattr(_app, "is_running", False)
        except Exception:
            _pt_active = False

    if prompt and not _pt_active and sys.stdin.isatty() and sys.stdout.isatty():
        spec_list = ", ".join(missing)
        try:
            answer = input(
                f"\nFeature {feature!r} requires: {spec_list}\n"
                f"Install into the active venv now? [Y/n] "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer and answer not in {"y", "yes"}:
            raise FeatureUnavailable(
                feature, missing, "user declined install at prompt"
            )

    logger.info("Lazy-installing %s for feature %r", " ".join(missing), feature)
    # Tier 0: `uv sync --extra <name>`, which resolves against uv.lock and
    # honours [tool.uv] override-dependencies. This is the only installer that
    # reproduces exactly what CI audited, so it is tried before the
    # pip-compatible ladder. Needs a project root + lockfile, and cannot serve
    # durable-target mode (it manages a venv wholesale, and the sealed-venv
    # image redirects installs to a separate dir on purpose).
    result = _uv_sync_extra(feature)
    if result is None:
        result = _venv_pip_install(missing)
    if not result.success:
        # Surface the actual pip error so the user can debug PyPI-side
        # issues (404 quarantine, network down, etc.).
        snippet = (result.stderr or result.stdout or "").strip()
        if snippet:
            # Clip to a readable size — pip can dump pages of resolution traces.
            snippet = snippet[-2000:]
        raise FeatureUnavailable(
            feature, missing,
            f"pip install failed: {snippet or 'no error output'}"
        )

    # Verify post-install. importlib.metadata caches per-process, so if we
    # just installed something the cache may not see it without a refresh.
    try:
        import importlib.metadata as _md
        if hasattr(_md, "_cache_clear"):
            _md._cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass

    still_missing = feature_missing(feature)
    if still_missing:
        raise FeatureUnavailable(
            feature, still_missing,
            "install reported success but packages still not importable "
            "(may require Python restart)"
        )

    logger.info("Lazy install complete for feature %r", feature)
    _record_feature_use(feature)


def is_available(feature: str) -> bool:
    """Return True if the feature's deps are already satisfied.

    Never raises. Callers use this in status displays and in registry
    ``check_fn``s, and an exception there kills the caller. When the specs
    are unreadable — no pyproject and no dist metadata — the answer is
    False, not an error.
    """
    if feature not in LAZY_DEPS:
        return False
    try:
        return not feature_missing(feature)
    except Exception as e:
        logger.debug("is_available(%r): specs unreadable: %s", feature, e)
        return False


def feature_install_command(feature: str, *, venv_pip: bool = False) -> Optional[str]:
    """Return the ``pip install`` command a user could run manually, or None.

    ``venv_pip=True`` targets the running interpreter's pip
    (``{sys.executable} -m pip install …``) — correct in every layout
    (default install, ``HERMES_HOME`` overrides, profile installs) and
    immune to Ubuntu 24.04's PEP 668 ``externally-managed-environment``
    failure that a bare/system ``pip install`` hint invites.  The default
    ``uv pip install`` form is kept for contexts that document uv usage.

    Never raises. The contract is Optional[str], and callers put the
    result into hint strings with no try/except.
    """
    if feature not in LAZY_DEPS:
        return None
    try:
        specs = feature_specs(feature)
    except Exception:
        return None
    joined = " ".join(repr(s) for s in specs)
    if venv_pip:
        return f"{sys.executable} -m pip install {joined}"
    return "uv pip install " + joined


@dataclass
class InstallSpecsResult:
    """Outcome of :func:`install_specs` for one batch of pip specs.

    ``ok``       — install succeeded (or nothing was missing).
    ``blocked``  — installs are gated off (config kill switch, sealed venv
                   without a durable target) or a spec failed validation;
                   nothing was executed. ``reason`` explains why.
    ``command``  — human-readable description of what ran (for UIs/logs).
    """
    ok: bool
    blocked: bool = False
    reason: str = ""
    command: str = ""
    stdout: str = ""
    stderr: str = ""


def install_specs(specs: list[str] | tuple[str, ...], *, timeout: int = 300) -> InstallSpecsResult:
    """Install arbitrary (validated) pip specs through the lazy-install pipeline.

    This is the environment-aware install path for callers whose package
    lists come from data (e.g. memory-provider plugin manifests declaring
    ``pip_dependencies``) rather than the static :data:`LAZY_DEPS` allowlist.
    It applies the exact same environment routing as :func:`ensure`:

    * **Venv-scoped by default** — installs into ``sys.executable``'s venv.
    * **Durable-target on immutable images** — when the deployment seals the
      agent venv (``HERMES_DISABLE_LAZY_INSTALLS=1``) and sets
      ``HERMES_LAZY_INSTALL_TARGET``, installs are redirected to the writable
      data-volume dir (``--target`` + core-venv constraints), then activated
      on ``sys.path`` so the packages import in this process immediately.
    * **Gated** — honors ``security.allow_lazy_installs`` and refuses to run
      when the venv is sealed with no durable target (never attempts a write
      to a read-only tree; reports *why* instead of surfacing EROFS/EACCES).

    Unlike :func:`ensure`, a package outside pyproject.toml is permitted.
    Hindsight appends ``hindsight-all`` at setup time, and a plugin outside
    this repository declares its own packages, so a list of permitted names
    cannot work here.

    This function does NOT check the shape of a spec, and a check would give
    nothing. The specs come from ``plugin.yaml``, and the same file holds
    ``external_dependencies[].install``, which
    hermes_cli/web_server.py runs through ``subprocess.run(shell=True)``. The
    plugin's ``__init__.py`` runs as well, at import. Anyone who can write
    that manifest already runs code as the user, so a pattern that rejects
    ``--index-url`` protects nothing.

    Never raises; inspect the returned :class:`InstallSpecsResult`.
    """
    cleaned = tuple(str(s).strip() for s in specs if str(s).strip())
    if not cleaned:
        return InstallSpecsResult(ok=True, command="")

    if not _allow_lazy_installs():
        reason = _sealed_venv_reason() or _CONFIG_DISABLED_REASON
        return InstallSpecsResult(ok=False, blocked=True, reason=reason)

    # The same managed-install guard as in ensure(). A package-manager
    # install (Nix) has its venv in a read-only store, so the pip ladder
    # below can only burn 15s and then fail with EROFS. Report the remedy
    # for the deployment instead. A durable target overrides this guard,
    # as it does in ensure(): the NixOS container module sets
    # HERMES_MANAGED=true and a writable target, and the install works
    # there.
    if _lazy_install_target() is None:
        managed_by = _managed_system()
        if managed_by:
            return InstallSpecsResult(
                ok=False, blocked=True,
                reason="unsupported on a managed install: "
                + managed_install_reason("install_specs", None),
            )

    target = _lazy_install_target()
    display = "uv pip install " + (
        f"--target {target} " if target is not None else ""
    ) + " ".join(cleaned)

    logger.info("Installing pip specs %s (target=%s)", " ".join(cleaned), target or "venv")
    try:
        result = _venv_pip_install(cleaned, timeout=timeout)
    except Exception as exc:
        logger.warning("install_specs failed unexpectedly: %s", exc)
        return InstallSpecsResult(
            ok=False, command=display, stderr=f"install failed: {exc}"
        )

    # Freshly-installed dists must be visible to importers and metadata
    # checks in this same process (dashboard rechecks availability inline).
    try:
        import importlib
        importlib.invalidate_caches()
        import importlib.metadata as _md
        if hasattr(_md, "_cache_clear"):
            _md._cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass

    return InstallSpecsResult(
        ok=result.success,
        command=display,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def active_features() -> list[str]:
    """Return the list of features the user has lazy-installed and still has.

    The primary signal is the record file (:func:`_record_feature_use`).
    ``ensure`` writes a feature's name there on every call, and each backend
    calls ``ensure`` at start, so the record names exactly the features in
    use. A package check cannot do that: the extras share packages
    (sounddevice is in every audio extra), so presence of a package does not
    say which feature the user enabled.

    A recorded feature still needs its anchor package installed to count.
    The record says "used at some point"; a user who uninstalled the
    packages since then must not get them back on ``hermes update``.

    An install that predates the record has an empty one, so its first
    ``hermes update`` refreshes nothing. That is fine: ``ensure`` runs at
    each backend's start, repairs a stale pin there, and records the
    feature, so the next update refreshes it.

    Used by ``hermes update`` to figure out which lazy backends need a
    refresh pass when pins move in pyproject.toml.
    """
    recorded = _read_feature_record()
    return [
        f for f in LAZY_DEPS if f in recorded and _feature_anchor_present(f)
    ]


def _feature_anchor_present(feature: str) -> bool:
    """Is the anchor package of ``feature`` installed, at any version?"""
    anchor = _anchor_spec(LAZY_DEPS[feature])
    return anchor is not None and _is_present(anchor)


# The record of the features that ensure() has served. One name per line
# in a JSON list, in $HERMES_HOME, so it survives a venv rebuild and, in a
# container, lives on the data volume.
_FEATURE_RECORD_NAME = "lazy-features.json"


def _feature_record_path() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / _FEATURE_RECORD_NAME


def _read_feature_record() -> set[str]:
    """Return the recorded feature names.

    A record that is absent or does not parse counts as empty, so a corrupt
    file heals on the next write instead of raising forever.
    """
    try:
        raw = json.loads(_feature_record_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    if not isinstance(raw, list):
        return set()
    return {str(f) for f in raw}


def _write_feature_record(features: set[str]) -> None:
    """Write the record. A failure only costs the record, so never raise."""
    try:
        path = _feature_record_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(sorted(features), indent=0) + "\n", encoding="utf-8"
        )
    except OSError as e:
        logger.debug("Could not write the lazy-feature record: %s", e)


def _record_feature_use(feature: str) -> None:
    """Add ``feature`` to the record of features that ensure() has served."""
    recorded = _read_feature_record()
    if feature not in recorded:
        _write_feature_record(recorded | {feature})


def refresh_active_features(*, prompt: bool = False) -> dict[str, str]:
    """Re-run ``ensure`` for every feature the user has previously activated.

    Returns a ``{feature: status}`` map where status is one of:
        ``"current"``  — pins already satisfied, no install run
        ``"refreshed"`` — pins were stale, reinstall succeeded
        ``"failed: <reason>"`` — install attempt failed; caller decides
                                  whether to surface it (we don't raise)
        ``"skipped: <reason>"`` — gated off (config flag, user decline)

    Intended for ``hermes update``. Never raises; lazy-install failures
    here must not block the rest of the update flow.
    """
    results: dict[str, str] = {}
    for feature in active_features():
        missing = feature_missing(feature)
        if not missing:
            results[feature] = "current"
            continue

        unsupported = _unsupported_feature_reason(feature)
        if unsupported:
            results[feature] = f"skipped: {unsupported}"
            continue

        try:
            ensure(feature, prompt=prompt)
            results[feature] = "refreshed"
        except FeatureUnavailable as e:
            # Distinguish "user opted out" or platform-incompatible features
            # from install failures so the update command can render the
            # right non-error message.
            if (
                "lazy installs disabled" in str(e)
                or "declined" in str(e)
                or e.reason.startswith("unsupported ")
            ):
                results[feature] = f"skipped: {e.reason}"
            else:
                results[feature] = f"failed: {e.reason}"
        except Exception as e:
            results[feature] = f"failed: {e}"
    return results


def ensure_and_bind(
    feature: str,
    importer: Callable[[], dict[str, Any]],
    target_globals: dict,
    *,
    prompt: bool = False,
) -> bool:
    """Ensure a feature is installed, then rebind names into the caller's globals.

    Combines :func:`ensure` with a post-install import step that rebinds
    module-level names.  This eliminates the error-prone pattern of manually
    listing every global that needs updating after lazy-install.

    ``importer`` is a zero-arg callable that returns a dict of
    ``{name: value}`` for all symbols the caller needs rebound.  It is called
    only after :func:`ensure` succeeds (or if the packages are already
    installed).

    Returns True on success, False if deps couldn't be installed or imported.

    Example usage in a platform adapter::

        def check_slack_requirements() -> bool:
            if SLACK_AVAILABLE:
                return True
            def _import():
                from slack_bolt.async_app import AsyncApp
                from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
                from slack_sdk.web.async_client import AsyncWebClient
                import aiohttp
                return {
                    "AsyncApp": AsyncApp,
                    "AsyncSocketModeHandler": AsyncSocketModeHandler,
                    "AsyncWebClient": AsyncWebClient,
                    "aiohttp": aiohttp,
                    "SLACK_AVAILABLE": True,
                }
            return ensure_and_bind("platform.slack", _import, globals(), prompt=False)
    """
    try:
        ensure(feature, prompt=prompt)
    except Exception:
        return False

    try:
        bindings = importer()
    except ImportError:
        return False

    target_globals.update(bindings)
    return True
