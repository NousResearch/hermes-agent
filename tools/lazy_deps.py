"""Lazy dependency installer for opt-in Hermes backends.

Backends call :func:`ensure(feature)` on first import; missing packages are installed into the
active venv (or the durable target) unless ``security.allow_lazy_installs: false``, in which
case :class:`FeatureUnavailable` carries a remediation hint. Security model: venv-scoped
(never system Python); durable-target mode (``HERMES_LAZY_INSTALL_TARGET``, sealed images)
APPENDS the target to ``sys.path`` so core site-packages wins every collision and a lazy
package can only add modules, never shadow core; PyPI-by-name specs only (``_spec_is_safe``);
``ensure`` accepts only the :data:`LAZY_DEPS` allowlist; failures surface pip's stderr, no retry.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import site
import subprocess
import sys
import sysconfig
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_cli._subprocess_compat import windows_hide_flags

logger = logging.getLogger(__name__)

# pip mutates one shared environment. Even different lazy features must not
# install concurrently, and same-feature contenders must recheck after the
# first install completes rather than replaying a stale `missing` snapshot.
_INSTALL_LOCK = threading.Lock()


# Allowlist: "namespace.backend" -> pip specs matching the pyproject extra. Pins are exact
# (security posture); bump here AND in pyproject. Shared patched floors (aiohttp==3.14.3,
# starlette==1.3.1) are literals in every feature: tests/test_packaging_metadata.py checks by AST.
LAZY_DEPS: dict[str, tuple[str, ...]] = {
    # ─── Inference providers ───────────────────────────────────────────────
    # Native Anthropic SDK (provider=anthropic; aggregators use the openai SDK).
    "provider.anthropic": ("anthropic==0.87.0",),  # CVE-2026-34450, CVE-2026-34452
    # Official Claude Agent SDK (provider=claude-agent-sdk, subscription-OAuth agent loop); bump with the
    # `claude-agent-sdk` extra in lockstep.
    "provider.claude_agent_sdk": ("claude-agent-sdk==0.2.144",),
    "provider.bedrock": ("boto3==1.42.89",),
    # Vertex OAuth2 token minting; google-auth is NOT in [all] on purpose.
    "provider.vertex": (
        "google-auth==2.55.1",
        "pyasn1==0.6.4",
    ),
    # Foundry Entra ID auth; only when model.auth_mode=entra_id.
    "provider.azure_identity": ("azure-identity==1.25.3",),

    # ─── Web search backends ───────────────────────────────────────────────
    "search.exa": ("exa-py==2.10.2",),
    "search.firecrawl": ("firecrawl-py==4.17.0",),
    "search.parallel": ("parallel-web==0.4.2",),

    # ─── Monitoring ─────────────────────────────────────────────────────────
    # OTLP export; tracks the `otlp` extra.
    "export.otlp": (
        "opentelemetry-sdk==1.39.1",
        "opentelemetry-exporter-otlp-proto-http==1.39.1",
    ),

    # ─── TTS providers ─────────────────────────────────────────────────────
    # mistralai: 2.4.6 was a malicious quarantined release — never pin below 2.4.7.
    # Voxtral STT + TTS share the SDK.
    "tts.mistral": ("mistralai==2.4.8",),
    "tts.edge": ("edge-tts==7.2.7",),
    "tts.elevenlabs": ("elevenlabs==1.59.0",),

    # ─── Speech-to-text providers ──────────────────────────────────────────
    "stt.mistral": ("mistralai==2.4.8",),
    "stt.faster_whisper": (
        "faster-whisper==1.2.1",
        "sounddevice==0.5.5",
        "numpy==2.4.3",
    ),
    # SILK voice-note decoding (WeChat/QQ); silk-v3 codec binding.
    "stt.silk": ("pilk==0.2.4",),

    # ─── Wake word ("Hey Hermes") engines (sync with the `wake` extra) ──────
    # openWakeWord's ONNX model scores ~0 on macOS ARM64, so macOS uses the tflite backend
    # (ai-edge-litert, bridged in tools/wake_word.py). Separate feature because specs cannot
    # carry PEP 508 markers (";" is rejected) — the caller applies the platform gate.
    "wake.openwakeword.tflite": (
        "ai-edge-litert==2.1.6",
    ),
    "wake.openwakeword": (
        "openwakeword==0.6.0",
        "onnxruntime==1.27.0",
        "sounddevice==0.5.5",
        "numpy==2.4.3",
    ),
    # Open-vocabulary keyword spotting. sentencepiece is needed by
    # sherpa_onnx.text2token but undeclared by sherpa-onnx.
    "wake.sherpa": (
        "sherpa-onnx==1.13.4",
        "sentencepiece==0.2.2",
        "sounddevice==0.5.5",
        "numpy==2.4.3",
    ),
    "wake.porcupine": (
        "pvporcupine==4.0.3",
        "sounddevice==0.5.5",
        "numpy==2.4.3",
    ),

    # ─── Image generation backends ─────────────────────────────────────────
    "image.fal": ("fal-client==0.13.1",),

    # ─── Memory providers ──────────────────────────────────────────────────
    "memory.honcho": ("honcho-ai==2.2.0",),
    "memory.hindsight": ("hindsight-client==0.6.1",),
    # Cloud memory SDKs MUST be allowlisted + ensure()'d at the import site, or they never
    # install on the sealed Docker image (durable-target only).
    "memory.supermemory": ("supermemory==3.50.0",),
    "memory.mem0": ("mem0ai==2.0.10",),

    # ─── Messaging platforms (lazy-installable on demand) ──────────────────
    "platform.telegram": ("python-telegram-bot[webhooks]==22.8",),
    # brotlicffi: aiohttp needs its 2-arg Decompressor for Discord CDN Brotli attachments
    # (google's 1-arg `Brotli` fails "Can not decode br"). aiohttp is only capped transitively
    # by these adapters, so pin the patched floor explicitly.
    # Without it, aiohttp falls back to google's `Brotli` package (1-arg API), and any .txt/.md/.doc
    # uploaded to the Discord gateway fails to decode at att.read() with "Can not decode content-encoding:
    # br" — see #12511 / #15744.
    "platform.discord": (
        "discord.py[voice]==2.7.1",
        "brotlicffi==1.2.0.1",
        "aiohttp==3.14.3",
    ),
    "platform.slack": (
        "slack-bolt==1.30.0",
        "slack-sdk==3.43.0",
        "aiohttp==3.14.3",
    ),
    "platform.matrix": (
        "mautrix[encryption]==0.21.1",
        "aiosqlite==0.22.1",
        "asyncpg==0.31.0",
        "aiohttp-socks==0.11.0",
        "aiohttp==3.14.3",
    ),
    "platform.dingtalk": (
        "dingtalk-stream==0.24.3",
        "alibabacloud-dingtalk==2.2.42",
        "qrcode==7.4.2",
    ),
    "platform.feishu": (
        "lark-oapi==1.6.8",
        "qrcode==7.4.2",
    ),
    # WeCom callback adapter parses untrusted XML POST bodies -> defusedxml.
    "platform.wecom_callback": ("defusedxml==0.7.1",),
    # Teams pulls a heavy tree (msal, dependency-injector); also the `teams` extra.
    "platform.teams": ("microsoft-teams-apps==2.0.13.4", "aiohttp==3.14.3"),

    # ─── Terminal backends ─────────────────────────────────────────────────
    "terminal.modal": ("modal==1.3.4",),
    "terminal.daytona": ("daytona==0.155.0",),
    "terminal.vercel": ("vercel==0.7.2",),

    # ─── Skills ────────────────────────────────────────────────────────────
    "skill.google_workspace": (
        "google-api-python-client==2.194.0",
        "google-auth==2.55.1",
        "google-auth-oauthlib==1.3.1",
        "google-auth-httplib2==0.3.1",
        # Explicit transitive pins: httplib2 <0.32 has a decompression-bomb DoS.
        "httplib2==0.32.0",
        "pyasn1==0.6.4",
    ),
    "skill.youtube": ("youtube-transcript-api==1.2.4",),

    # ─── Tools ─────────────────────────────────────────────────────────────
    # ACP adapter (VS Code / Zed / JetBrains)
    "tool.acp": ("agent-client-protocol==0.9.0",),
    "tool.dashboard": (
        "fastapi==0.133.1",
        "uvicorn[standard]==0.41.0",
        "starlette==1.3.1",
        "python-multipart==0.0.32",  # FastAPI UploadFile/Form streaming uploads
    ),
    # Pillow and firecrawl-anydoc are CORE deps; these entries self-heal lean/partial installs.
    # Call sites use prompt=False so read_file / vision never block on input() mid-session.
    # Vision image-resize recovery (Pillow). Pillow is now a CORE dependency (pyproject `dependencies`), so
    # this entry is a belt-and-suspenders fallback for stripped/source-build installs that somehow dropped
    # it. See #40490.
    "tool.vision": ("Pillow==12.3.0",),
    "tool.doc_extract": ("firecrawl-anydoc==0.2.4",),  # imports as `anydoc`; lockstep with pyproject
    # MCP client SDK for the cua-driver, so computer_use never dead-ends on `No module named 'mcp'`.
    "tool.computer_use": (
        "mcp==2.0.0",
        "httpx2==2.7.0",  # mcp 2.x HTTP stack — sync with pyproject [computer-use]
        "starlette==1.3.1",
    ),
    # huggingface-hub is SHARED with transformers (>=1.5.0,<2 via Hindsight) and marked active
    # on mere presence, so `hermes update` re-asserts this pin everywhere hub exists. MUST stay
    # inside transformers' window and match uv.lock (tests/test_project_metadata.py enforces).
    # HF Agent Trace Viewer upload (hermes trace upload / /upload-trace). huggingface-hub is a SHARED
    # dependency: transformers (pulled by sentence-transformers for local Hindsight embeddings) requires
    # >=1.5.0,<2, and faster-whisper/tokenizers depend on it transitively. Because active_features() marks a
    # feature active from mere package presence, the `hermes update` lazy-refresh pass re-asserts THIS pin
    # on every install where hub is present — so an exact pin below 1.5.0 force-downgrades the shared
    # package and breaks Hindsight startup (#60783). Policy: keep the exact pin (no ranges — security
    # posture), but it MUST stay inside transformers' accepted window and MUST match uv.lock so the whole
    # tree converges on ONE hub version (tests/test_project_metadata.py enforces both). When bumping: update
    # here AND `uv lock --upgrade-package huggingface-hub` in lockstep.
    "tool.trace_upload": ("huggingface-hub==1.24.0",),
}


# Spec validation: name[extras]specifier only — no URLs, paths, or shell metachars.
_NAME_RE = r"[A-Za-z0-9_][A-Za-z0-9_.\-]*"
_NAME_EXTRAS_RE = re.compile(rf"^{_NAME_RE}(?:\[[A-Za-z0-9_,\-]+\])?")
_SAFE_SPEC = re.compile(rf"^{_NAME_RE}(?:\[[A-Za-z0-9_,\-]+\])?(?:[<>=!~]=?[A-Za-z0-9_.\-+,*<>=!~]+)?$")


class FeatureUnavailable(RuntimeError):
    """A lazily-installable feature is missing and cannot be made available (installs disabled or failed)."""

    def __init__(self, feature: str, missing: tuple[str, ...], reason: str):
        self.feature, self.missing, self.reason = feature, missing, reason
        spec_list = " ".join(repr(s) for s in missing)
        super().__init__(
            f"Feature {feature!r} unavailable: {reason}. "
            f"To enable manually: uv pip install {spec_list}  (or: pip install {spec_list}).")


@dataclass(frozen=True)
class _InstallResult:
    success: bool
    stdout: str
    stderr: str


# Internal bridge var (set by the Docker image, not user config) redirecting lazy installs from the
# sealed venv to a writable durable volume.
_LAZY_TARGET_ENV = "HERMES_LAZY_INSTALL_TARGET"
# Stamp of the Python X.Y + ABI the target was populated for; a mismatch after an image rebuild
# wipes the store so stale .so files are never imported.
_TARGET_STAMP_NAME = ".python-abi"

_SUBPROCESS_KW = dict(capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)


def _python_abi_tag() -> str:
    """X.Y + EXT_SUFFIX (ABI tag + platform); interpreters sharing compiled wheels match."""
    return f"{sys.version_info.major}.{sys.version_info.minor}:{sysconfig.get_config_var('EXT_SUFFIX') or ''}"


def _lazy_install_target() -> Optional[Path]:
    """Durable install-target dir (:data:`_LAZY_TARGET_ENV`), or None for venv-scoped mode."""
    raw = os.environ.get(_LAZY_TARGET_ENV, "").strip()
    return Path(raw) if raw else None


def _ensure_target_ready(target: Path) -> Optional[str]:
    """Create the target dir and validate its ABI stamp; a different-ABI stamp wipes the contents
    first (stale .so must never import). None on success, else an error string."""
    want = _python_abi_tag()
    stamp = target / _TARGET_STAMP_NAME
    try:
        if target.exists():
            have = ""
            with contextlib.suppress(OSError):
                have = stamp.read_text(encoding="utf-8").strip()
            if have and have != want:
                logger.info("Lazy install target %s was built for ABI %r but running ABI is %r; wiping stale packages.", target, have, want)
                for child in target.iterdir():
                    if child.is_dir() and not child.is_symlink():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        with contextlib.suppress(OSError):
                            child.unlink()
        target.mkdir(parents=True, exist_ok=True)
        stamp.write_text(want, encoding="utf-8")
    except OSError as e:
        return f"lazy install target {target} is not writable: {e}"
    return None


def _activate_target_on_syspath(target: Path) -> None:
    """Append the durable target to ``sys.path`` (idempotent). ``site.addsitedir`` honours ``.pth``
    files but inserts near the front, so new entries are moved to the END — core venv site-packages must win collisions."""
    target_str = str(target)
    before = list(sys.path)
    if target_str not in before:
        site.addsitedir(target_str)
    new_entries = [p for p in sys.path if p not in before]
    if new_entries:
        sys.path[:] = [p for p in sys.path if p not in new_entries] + new_entries
    _invalidate_import_caches()


def _invalidate_import_caches() -> None:
    """Make just-installed dists visible to importers and importlib.metadata in this process."""
    with contextlib.suppress(Exception):
        import importlib
        importlib.invalidate_caches()
        import importlib.metadata as _md
        if hasattr(_md, "_cache_clear"):
            _md._cache_clear()  # type: ignore[attr-defined]


def activate_durable_lazy_target() -> None:
    """Wire the durable target onto ``sys.path`` at startup so packages installed on a previous run
    import on this one. No-op when unset or absent. Never raises."""
    target = _lazy_install_target()
    try:
        if target is not None and target.exists():
            _activate_target_on_syspath(target)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Failed to activate durable lazy target %s: %s", target, e)


def _allow_lazy_installs() -> bool:
    """Whether lazy installs are permitted: (1) ``security.allow_lazy_installs: false`` blocks in BOTH
    modes; (2) the sealed venv (``HERMES_DISABLE_LAZY_INSTALLS=1``) blocks only without a durable
    target to redirect into. Unreadable config fails OPEN — blocking is an explicit opt-in."""
    cfg = None
    with contextlib.suppress(Exception):
        from hermes_cli.config import load_config
        cfg = load_config()
    if cfg is not None and not bool((cfg.get("security") or {}).get("allow_lazy_installs", True)):
        return False
    if os.environ.get("HERMES_DISABLE_LAZY_INSTALLS") == "1":
        return _lazy_install_target() is not None
    return True


def _unsupported_feature_reason(feature: str) -> Optional[str]:
    """Platform capability gate (not policy): why a feature cannot work on this host, or None."""
    if sys.platform == "win32" and feature == "platform.matrix":
        return ("unsupported on Windows: Matrix E2EE depends on python-olm, which has no Windows wheel and "
                "requires make + libolm to build from sdist. Run Hermes under WSL to use Matrix on Windows.")
    return None


def _spec_is_safe(spec: str) -> bool:
    """Reject pip specs that contain URLs, paths, or shell metacharacters."""
    return bool(spec and len(spec) <= 200
                and not any(ch in spec for ch in (";", "|", "&", "`", "$", "\n", "\r", "\t", "\\"))
                and not spec.startswith(("-", "/", ".")) and "://" not in spec and "@" not in spec
                and _SAFE_SPEC.match(spec))


def _pkg_name_from_spec(spec: str) -> str:
    """``"mautrix[encryption]>=0.20"`` -> ``"mautrix"``."""
    m = re.match(rf"^({_NAME_RE})", spec)
    return m.group(1) if m else spec


def _specifier_from_spec(spec: str) -> str:
    """``"mautrix[encryption]>=0.20,<1"`` -> ``">=0.20,<1"``; ``""`` if unconstrained."""
    m = _NAME_EXTRAS_RE.match(spec)
    return spec[m.end():] if m else ""


def _installed_version(spec: str) -> Optional[str]:
    """Installed version of the spec's package, or None when absent."""
    try:
        from importlib.metadata import version

        return version(_pkg_name_from_spec(spec))
    except Exception:
        return None


def _is_satisfied(spec: str) -> bool:
    """Present AND inside the spec's version range, so ``hermes update`` propagates pin bumps to
    installed backends. Unparseable specs/versions or a missing ``packaging`` count as satisfied — err
    toward "don't churn"."""
    installed = _installed_version(spec)
    if installed is None:
        return False
    if not (spec_tail := _specifier_from_spec(spec)):
        return True
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        return Version(installed) in SpecifierSet(spec_tail)
    except Exception:
        return True


def _is_present(spec: str) -> bool:
    """Presence-only check (any version) — how :func:`active_features` detects activated backends."""
    return _installed_version(spec) is not None


def _core_constraints_file() -> Optional[Path]:
    """Temp ``--constraint`` file pinning every core-venv package to its installed version for
    durable-target installs: shared deps resolve as satisfied (store stays minimal) and a conflicting
    backend fails loudly instead of installing a shadowed copy that can never win on sys.path. None if
    enumeration failed (install unconstrained)."""
    try:
        import tempfile
        from importlib.metadata import distributions

        pins: dict[str, str] = {}
        for dist in distributions():
            name = dist.metadata["Name"] if dist.metadata else None
            if name and dist.version and name.lower() not in pins:
                pins[name.lower()] = f"{name}=={dist.version}"
        if not pins:
            return None
        fd, path = tempfile.mkstemp(prefix="hermes-core-constraints-", suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(pins.values())) + "\n")
        return Path(path)
    except Exception as e:
        logger.debug("Could not build core constraints file: %s", e)
        return None


def _installed_dist_roots(spec: str, target: Optional[Path]) -> set[Path]:
    """Package dirs a freshly installed *spec* owns, from the dist's file list (``python-telegram-bot``
    ships ``telegram``; some ship several)."""
    name = _pkg_name_from_spec(spec)
    roots: set[Path] = set()
    try:
        import importlib.metadata as _md

        dist = next(iter(_md.distributions(name=name, path=[str(target)])), None) if target is not None else _md.distribution(name)
        for entry in dist.files or () if dist is not None else ():
            top = entry.parts[0] if entry.parts else ""
            # Skip hidden entries, __pycache__ and metadata dirs (no importable code).
            if not top or top.startswith(".") or top == "__pycache__" or top.endswith((".dist-info", ".egg-info")):
                continue
            root = Path(dist.locate_file(top))
            if root.is_dir():
                roots.add(root)
    except Exception:
        return set()
    return roots


def _warm_installed_bytecode(specs: tuple[str, ...], target: Optional[Path]) -> None:
    """Byte-compile what was just installed: a fresh install writes no ``__pycache__``, so the next
    import (often a user request, ~2-10s for a big SDK, reading as a hang) would pay the compile. Pay
    it here while the caller already waits. Best-effort; never fails the install.

    A pip/uv install writes ``.py`` sources and no ``__pycache__`` — and an install of the *same* version
    still deletes the cache the old copy had. Whoever imports the package next pays the whole compile: for
    ``anthropic==0.87.0`` (541 modules) on cpython-3.12.13 that measured 2.2-2.7s cold against 0.7-1.0s
    warm, and 10.5s cold under concurrent load. That bill lands wherever the first import happens, and for a
    lazily-installed backend that is the foreground of a user request (#100461) — with nothing printed while
    it runs, so it reads as a hang. Worse, N per-profile daemons cold-starting together each pay it in full
    before any of them has written the cache.
    """
    if sys.dont_write_bytecode:
        return
    try:
        import compileall
    except Exception:  # pragma: no cover — stdlib, but never break an install
        return
    for spec in specs:
        try:
            for root in _installed_dist_roots(spec, target):
                try:
                    compileall.compile_dir(str(root), quiet=2, force=False, workers=1)
                except Exception as exc:
                    logger.debug("Bytecode warm skipped for %s: %s", root, exc)
        except Exception as exc:
            logger.debug("Bytecode warm skipped for %s: %s", spec, exc)


def _run_installer(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    # _SUBPROCESS_KW carries stdin=DEVNULL  # noqa: subprocess-stdin
    return subprocess.run(cmd, **_SUBPROCESS_KW, creationflags=windows_hide_flags(), **kw)


def _uv_binary() -> Optional[str]:
    """Managed uv first ($HERMES_HOME/bin is never on PATH), then PATH. A lookup, not ensure_uv():
    downloading uv mid-turn is more than the caller asked for; pip covers no-uv."""
    try:
        from hermes_cli.managed_uv import resolve_uv

        return resolve_uv() or shutil.which("uv")
    except Exception:
        return shutil.which("uv")


def _venv_pip_install(specs: tuple[str, ...], *, timeout: int = 300) -> _InstallResult:
    """Install ``specs`` via the uv -> pip -> ensurepip ladder, venv-scoped or into the durable
    ``--target`` (constrained to core versions) when :data:`_LAZY_TARGET_ENV` is set. Independent of
    ``hermes_cli.tools_config._pip_install`` (no CLI dependency)."""
    if not specs:
        return _InstallResult(True, "", "")
    target = _lazy_install_target()
    constraints: Optional[Path] = None
    extra_args: list[str] = []
    if target is not None:
        if err := _ensure_target_ready(target):
            return _InstallResult(False, "", err)
        constraints = _core_constraints_file()
        extra_args += ["--target", str(target)]
        if constraints is not None:
            extra_args += ["--constraint", str(constraints)]

    def _finish(r: subprocess.CompletedProcess) -> _InstallResult:
        if r.returncode == 0:
            if target is not None:
                _activate_target_on_syspath(target)
            _warm_installed_bytecode(specs, target)
        return _InstallResult(r.returncode == 0, r.stdout or "", r.stderr or "")

    try:
        from tools.environments.local import hermes_subprocess_env
        uv_env = hermes_subprocess_env(inherit_credentials=False)
        uv_env["VIRTUAL_ENV"] = str(Path(sys.executable).parent.parent)
        # Tier 1: uv. --compile-bytecode because uv writes no __pycache__ by default, so the first
        # import would recompile the backend AND its transitives (_warm_installed_bytecode is the
        # belt-and-braces pass for the spec's own roots on any tier).
        if uv_bin := _uv_binary():
            try:
                r = _run_installer([uv_bin, "pip", "install", "--compile-bytecode", *extra_args, *specs], timeout=timeout, env=uv_env)
                if r.returncode != 0:
                    logger.debug("uv pip install failed: %s", r.stderr)
                # A uv resolver failure is authoritative: falling through to pip would discard uv
                # policy (exclude-newer) and could install a quarantined release.
                return _finish(r)
            except subprocess.TimeoutExpired as e:
                logger.debug("uv invocation failed: %s", e)
                return _InstallResult(False, "", f"uv pip install timed out: {e}")
            except FileNotFoundError as e:  # uv vanished between lookup and spawn; it never evaluated the requirements
                logger.debug("uv invocation failed: %s", e)
        # Tier 2: python -m pip (ensurepip bootstrap if needed)
        pip_cmd = [sys.executable, "-m", "pip"]
        try:
            if _run_installer(pip_cmd + ["--version"], timeout=15).returncode != 0:
                raise FileNotFoundError("pip not in venv")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            try:
                _run_installer([sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"], timeout=120, check=True)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                return _InstallResult(False, "", f"pip not available and ensurepip failed: {e}")
        try:
            return _finish(_run_installer(pip_cmd + ["install", *extra_args, *specs], timeout=timeout))
        except subprocess.TimeoutExpired as e:
            return _InstallResult(False, "", f"pip install timed out: {e}")
        except Exception as e:
            return _InstallResult(False, "", f"pip install failed: {e}")
    finally:
        if constraints is not None:
            with contextlib.suppress(OSError):
                constraints.unlink()


def feature_missing(feature: str) -> tuple[str, ...]:
    """Return the subset of specs for ``feature`` not currently installed."""
    if feature not in LAZY_DEPS:
        raise KeyError(f"Unknown lazy feature: {feature!r}")
    return tuple(s for s in LAZY_DEPS[feature] if not _is_satisfied(s))


def _prompt_toolkit_active() -> bool:
    """A bare input() deadlocks while a prompt_toolkit app owns the terminal, so ensure() skips the
    confirmation under the TUI — reaching it is already gated by security.allow_lazy_installs."""
    if "prompt_toolkit.application.current" not in sys.modules:
        return False
    try:
        from prompt_toolkit.application.current import get_app_or_none
        return bool(getattr(get_app_or_none(), "is_running", False))
    except Exception:
        return False


def ensure(feature: str, *, prompt: bool = True) -> None:
    """Make every package for ``feature`` importable, installing if needed; raises
    :class:`FeatureUnavailable` when installs are disabled or fail. ``prompt``: confirm on a TTY first
    (non-interactive callers pass False and rely on the config gate)."""
    if feature not in LAZY_DEPS:
        raise FeatureUnavailable(feature, (), f"feature {feature!r} not in LAZY_DEPS allowlist")
    missing = feature_missing(feature)
    if not missing:
        return
    if unsupported := _unsupported_feature_reason(feature):
        raise FeatureUnavailable(feature, missing, unsupported)
    # Package-manager installs (NixOS etc.) have read-only site-packages: fail fast instead of burning
    # ~15s on ensurepip — unless a durable target is configured. The reason MUST start with
    # "unsupported ": _refresh_features classifies skips by that prefix.
    if _lazy_install_target() is None:
        managed_by = ""  # config unreadable — proceed with the install
        with contextlib.suppress(Exception):
            from hermes_cli.config import get_managed_system
            managed_by = get_managed_system()
        if managed_by:
            raise FeatureUnavailable(
                feature, missing,
                f"unsupported on {managed_by}-managed installs: this build's packages come from {managed_by}, "
                f"so Hermes cannot install them at runtime. Add the dependencies for {feature!r} via "
                f"{managed_by} (or run a pip/uv install of Hermes instead).")
    for spec in missing:  # belt and braces on top of the allowlist
        if not _spec_is_safe(spec):
            raise FeatureUnavailable(feature, missing, f"refusing to install unsafe spec {spec!r}")
    if not _allow_lazy_installs():
        raise FeatureUnavailable(feature, missing, "lazy installs disabled (security.allow_lazy_installs=false)")
    with _INSTALL_LOCK:
        # A contender that waited must not replay a stale `missing` snapshot: re-resolve under the lock so
        # only one pip process acts on a given missing dependency set.
        missing = feature_missing(feature)
        if not missing:
            return
        if prompt and not _prompt_toolkit_active() and sys.stdin.isatty() and sys.stdout.isatty():
            try:
                answer = input(f"\nFeature {feature!r} requires: {', '.join(missing)}\nInstall into the active venv now? [Y/n] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"
            if answer and answer not in {"y", "yes"}:
                raise FeatureUnavailable(feature, missing, "user declined install at prompt")
        logger.info("Lazy-installing %s for feature %r", " ".join(missing), feature)
        result = _venv_pip_install(missing)
        if not result.success:  # surface pip's own error (quarantine 404, network), tail-clipped
            snippet = (result.stderr or result.stdout or "").strip()[-2000:]
            raise FeatureUnavailable(feature, missing, f"pip install failed: {snippet or 'no error output'}")
        _invalidate_import_caches()
        if still_missing := feature_missing(feature):
            raise FeatureUnavailable(feature, still_missing, "install reported success but packages still not importable (may require Python restart)")
        logger.info("Lazy install complete for feature %r", feature)


def is_available(feature: str) -> bool:
    """Return True if the feature's deps are already satisfied."""
    return feature in LAZY_DEPS and not feature_missing(feature)


def feature_install_command(feature: str, *, venv_pip: bool = False) -> Optional[str]:
    """Manual install command for a feature, or None. ``venv_pip=True`` uses ``{sys.executable} -m pip``
    — immune to PEP 668 failures a bare ``pip install`` invites."""
    if feature not in LAZY_DEPS:
        return None
    joined = " ".join(repr(s) for s in LAZY_DEPS[feature])
    return f"{sys.executable} -m pip install {joined}" if venv_pip else "uv pip install " + joined


@dataclass
class InstallSpecsResult:
    """Outcome of :func:`install_specs` for one batch of pip specs. ``blocked`` means installs are gated
    off (config kill switch, sealed venv without a durable target) or a spec failed validation — nothing
    was executed, ``reason`` says why. ``command`` is the human-readable description of what ran."""
    ok: bool
    blocked: bool = False
    reason: str = ""
    command: str = ""
    stdout: str = ""
    stderr: str = ""


def install_specs(specs: list[str] | tuple[str, ...], *, timeout: int = 300) -> InstallSpecsResult:
    """Install data-driven pip specs (plugin manifest ``pip_dependencies``) with the same routing and
    gating as :func:`ensure`, but unknown packages are allowed — the caller owns manifest trust, this
    owns spec hygiene. Never raises; inspect the :class:`InstallSpecsResult`."""
    cleaned = tuple(str(s).strip() for s in specs if str(s).strip())
    if not cleaned:
        return InstallSpecsResult(ok=True, command="")
    for spec in cleaned:
        if not _spec_is_safe(spec):
            return InstallSpecsResult(ok=False, blocked=True, reason=f"refusing to install unsafe spec {spec!r}")
    target = _lazy_install_target()
    if not _allow_lazy_installs():
        sealed = os.environ.get("HERMES_DISABLE_LAZY_INSTALLS") == "1" and target is None
        reason = ("runtime installs are disabled on this deployment: the agent environment is immutable "
                  "and no writable install target is configured (HERMES_LAZY_INSTALL_TARGET)"
                  ) if sealed else "runtime installs disabled (security.allow_lazy_installs=false)"
        return InstallSpecsResult(ok=False, blocked=True, reason=reason)
    display = "uv pip install " + (f"--target {target} " if target is not None else "") + " ".join(cleaned)
    logger.info("Installing pip specs %s (target=%s)", " ".join(cleaned), target or "venv")
    try:
        with _INSTALL_LOCK:
            result = _venv_pip_install(cleaned, timeout=timeout)
    except Exception as exc:
        logger.warning("install_specs failed unexpectedly: %s", exc)
        return InstallSpecsResult(ok=False, command=display, stderr=f"install failed: {exc}")

    _invalidate_import_caches()  # dashboard rechecks availability inline
    return InstallSpecsResult(ok=result.success, command=display, stdout=result.stdout, stderr=result.stderr)


def active_features() -> list[str]:
    """Features whose ANCHOR package (first spec) is present at any version — shared helpers like
    asyncpg are deliberately not proof a backend was enabled. Drives ``hermes update``."""
    return [f for f, specs in LAZY_DEPS.items() if specs and _is_present(specs[0])]


def refresh_active_features(*, prompt: bool = False) -> dict[str, str]:
    """Re-run ``ensure`` for every active feature (``hermes update``); returns
    ``{feature: "current" | "refreshed" | "failed: <reason>" | "skipped: <reason>"}``. Never raises."""
    return _refresh_features(active_features(), prompt=prompt, restoring=False)


def restore_features(features: list[str]) -> dict[str, str]:
    """Restore features captured before a managed-runtime rebuild; opt-out -> "skipped"."""
    return _refresh_features(features, prompt=False, restoring=True)


def _refresh_features(features: list[str], *, prompt: bool, restoring: bool) -> dict[str, str]:
    """Refresh or restore a known set of allowlisted lazy features."""
    results: dict[str, str] = {}
    for feature in features:
        if feature not in LAZY_DEPS:
            continue
        if not feature_missing(feature):
            results[feature] = "current"
            continue
        if unsupported := _unsupported_feature_reason(feature):
            results[feature] = f"skipped: {unsupported}"
            continue
        try:
            ensure(feature, prompt=False if restoring else prompt)
            results[feature] = "restored" if restoring else "refreshed"
        except FeatureUnavailable as e:  # opt-outs and platform-incompatible features are skips, not failures
            skip = "lazy installs disabled" in str(e) or "declined" in str(e) or e.reason.startswith("unsupported ")
            results[feature] = f"skipped: {e.reason}" if skip else f"failed: {e.reason}"
        except Exception as e:
            results[feature] = f"failed: {e}"
    return results


def ensure_and_bind(feature: str, importer: Callable[[], dict[str, Any]], target_globals: dict, *, prompt: bool = False) -> bool:
    """:func:`ensure` the feature, then ``target_globals.update(importer())`` so module-level names are
    rebound after a lazy install (``importer`` returns ``{name: obj}`` and runs only after ensure
    succeeds). Returns False (and logs) if deps could not be installed or imported."""
    try:
        ensure(feature, prompt=prompt)
    except FeatureUnavailable as exc:
        logger.warning("%s", exc)
        return False
    except Exception as exc:
        logger.warning("Failed to ensure feature %r: %s", feature, exc)
        return False
    try:
        target_globals.update(importer())
    except ImportError as exc:
        logger.warning("Failed to import feature %r after install: %s", feature, exc)
        return False
    return True


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def feature_specs(feature: str) -> tuple[str, ...]:
    """Return the registered specs for a feature, or raise KeyError."""
    if feature not in LAZY_DEPS:
        raise KeyError(f"Unknown lazy feature: {feature!r}")
    return LAZY_DEPS[feature]
# ---- END PLUGIN-COMPAT ----
