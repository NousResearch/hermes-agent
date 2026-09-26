"""Optional local bridge to the pinned standalone Skill Admission Gate."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from tools.quarantine_approval_models import (
    APPROVAL_BINDING_SCHEMA, ApprovalBinding, evidence_bundle_digest,
)

_VALID_MODES = {"off", "shadow", "enforce"}
_VALID_DECISIONS = {"ALLOW_TO_LAB": 0, "QUARANTINE": 10, "BLOCK": 20}


@dataclass(frozen=True)
class ExternalAdmissionResult:
    mode: str = "off"
    executed: bool = False
    decision: str = "OFF"
    allow_continue: bool = True
    candidate_sha256: str = ""
    evidence_dir: str = ""
    approval_binding: ApprovalBinding | None = None
    error: str = ""

@dataclass(frozen=True)
class _Settings:
    mode: str
    python_executable: Path | None = None
    release_archive: Path | None = None
    release_sha256: str = ""
    evidence_root: Path | None = None
    timeout_seconds: float = 300.0
    scanner_timeout_seconds: float = 180.0


def _load_settings() -> _Settings:
    from hermes_cli.config import load_config

    root = load_config()
    skills = root.get("skills") if isinstance(root, dict) else None
    raw = skills.get("external_admission_gate") if isinstance(skills, dict) else None
    if not isinstance(raw, dict):
        return _Settings("off")
    mode = str(raw.get("mode", "off")).strip().lower()
    if mode == "off":
        return _Settings("off")
    return _Settings(
        mode=mode,
        python_executable=Path(str(raw.get("python_executable", ""))),
        release_archive=Path(str(raw.get("release_archive", ""))),
        release_sha256=str(raw.get("release_sha256", "")).strip().lower(),
        evidence_root=Path(str(raw.get("evidence_root", ""))),
        timeout_seconds=float(raw.get("timeout_seconds", 300)),
        scanner_timeout_seconds=float(raw.get("scanner_timeout_seconds", 180)),
    )


def _is_redirect(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _candidate_hash(root: Path) -> str:
    if not root.is_dir() or _is_redirect(root):
        raise ValueError("candidate root is missing or redirected")
    entries: list[tuple[str, Path]] = []
    for current, dirs, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in list(dirs):
            child = current_path / name
            if _is_redirect(child):
                raise ValueError(f"candidate contains redirected directory: {child}")
        for name in files:
            child = current_path / name
            if _is_redirect(child):
                raise ValueError(f"candidate contains redirected file: {child}")
            entries.append((child.relative_to(root).as_posix(), child))
    digest = hashlib.sha256()
    for rel, path in sorted(entries, key=lambda item: item[0]):
        digest.update(rel.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(bytes.fromhex(hashlib.sha256(path.read_bytes()).hexdigest()))
    return digest.hexdigest()


def candidate_hash_matches(candidate: Path, expected_sha256: str) -> bool:
    expected = str(expected_sha256 or "").strip().lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        return False
    try:
        return _candidate_hash(Path(candidate)).lower() == expected
    except (OSError, ValueError):
        return False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or ":" in name:
                raise ValueError(f"unsafe release member: {name}")
            mode = (info.external_attr >> 16) & 0o170000
            if stat.S_ISLNK(mode):
                raise ValueError(f"release contains symlink: {name}")
        zf.extractall(destination)


def _sanitized_env(package_root: Path) -> dict[str, str]:
    keep = ("SystemRoot", "WINDIR", "TEMP", "TMP", "PATH", "PATHEXT", "COMSPEC")
    env = {key: os.environ[key] for key in keep if key in os.environ}
    env["PYTHONPATH"] = str(package_root)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _error(mode: str, message: str, *, executed: bool = False,
           candidate_sha256: str = "", evidence_dir: str = "") -> ExternalAdmissionResult:
    return ExternalAdmissionResult(
        mode=mode, executed=executed, decision="ERROR",
        allow_continue=(mode == "shadow"), candidate_sha256=candidate_sha256,
        evidence_dir=evidence_dir, error=message,
    )


def _last_json_line(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("external gate produced no stdout JSON")
    parsed = json.loads(lines[-1])
    if not isinstance(parsed, dict):
        raise ValueError("external gate stdout JSON is not an object")
    return parsed


def _raw_reason_codes(evidence_root: Path, record: dict[str, Any]) -> tuple[str, ...]:
    codes = {
        str(item.get("rule_id", "")).strip()
        for item in record.get("findings", [])
        if isinstance(item, dict) and str(item.get("rule_id", "")).strip()
    }
    if bool(record.get("complete")):
        return tuple(sorted(codes))

    raw_path = Path(str(record.get("raw_report_path", "")))
    if not raw_path.is_file() or _is_redirect(raw_path):
        raise ValueError("incomplete scanner raw report missing or redirected")
    raw_resolved = raw_path.resolve()
    if not raw_resolved.is_relative_to(evidence_root):
        raise ValueError("incomplete scanner raw report outside evidence root")
    raw = json.loads(raw_resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("incomplete scanner raw report is not an object")

    for code in raw.get("reason_codes", []):
        if str(code).strip():
            codes.add(str(code).strip())

    completeness = raw.get("analysis_completeness")
    if isinstance(completeness, dict):
        for item in completeness.get("ledger_exceptions", []):
            if isinstance(item, dict):
                code = str(
                    item.get("reason_code") or item.get("reason") or item.get("code") or ""
                ).strip()
            else:
                code = str(item).strip()
            if code:
                codes.add(code)
    if not codes:
        raise ValueError("incomplete scanner missing machine-readable reason code")
    return tuple(sorted(codes))


def _build_approval_binding(
    evidence: dict[str, Any], evidence_root: Path, settings: _Settings,
    candidate_sha256: str, source_id: str, source_version: str | None,
) -> ApprovalBinding:
    policy_version = str(evidence.get("policy_version", "")).strip()
    if not policy_version:
        raise ValueError("quarantine evidence missing policy version")
    records = evidence.get("scanners")
    if not isinstance(records, list):
        raise ValueError("quarantine evidence missing scanner records")
    expected = {"hermes_skills_guard", "cisco_ai_defense", "nvidia_skillspector"}
    by_id: dict[str, dict[str, Any]] = {}

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("quarantine scanner record is not an object")
        scanner = str(record.get("scanner", "")).strip()
        if scanner not in expected or scanner in by_id:
            raise ValueError("quarantine scanner set is missing, duplicate, or unexpected")
        if record.get("execution_ok") is not True:
            raise ValueError(f"quarantine scanner execution not successful: {scanner}")
        if not isinstance(record.get("complete"), bool):
            raise ValueError(f"quarantine scanner completeness missing: {scanner}")
        version = str(record.get("version", "")).strip()
        if not version or version.lower() == "unknown":
            raise ValueError(f"quarantine scanner version missing: {scanner}")
        by_id[scanner] = record
    if set(by_id) != expected:
        raise ValueError("quarantine evidence does not contain exactly three scanners")

    completeness = tuple((name, bool(by_id[name]["complete"])) for name in sorted(expected))
    reason_codes = tuple(
        (name, _raw_reason_codes(evidence_root, by_id[name])) for name in sorted(expected)
    )
    return ApprovalBinding(
        schema_version=APPROVAL_BINDING_SCHEMA,
        candidate_sha256=candidate_sha256.lower(),
        source_id=source_id,
        source_version=source_version or "",
        gate_release_sha256=settings.release_sha256,

        policy_version=policy_version,
        hermes_guard_version=str(by_id["hermes_skills_guard"]["version"]),
        cisco_version=str(by_id["cisco_ai_defense"]["version"]),
        nvidia_version=str(by_id["nvidia_skillspector"]["version"]),
        scanner_completeness=completeness,
        scanner_reason_codes=reason_codes,
        evidence_digest=evidence_bundle_digest(evidence_root),
        decision="QUARANTINE",
    )


def run_external_admission(candidate: Path, source_id: str = "",
                           source_version: str | None = None) -> ExternalAdmissionResult:
    try:
        settings = _load_settings()
    except (OSError, TypeError, ValueError) as exc:
        try:
            from hermes_cli.config import load_config
            root = load_config()
            skills = root.get("skills") if isinstance(root, dict) else None
            raw = skills.get("external_admission_gate") if isinstance(skills, dict) else None
            mode = str(raw.get("mode", "off")).strip().lower() if isinstance(raw, dict) else "off"
        except Exception:
            mode = "off"
        return _error(mode, f"invalid external admission config: {exc}")
    if settings.mode == "off":
        return ExternalAdmissionResult()
    if settings.mode not in _VALID_MODES:
        return _error(settings.mode, f"invalid external admission mode: {settings.mode}")

    try:
        candidate = Path(candidate)
        before_hash = _candidate_hash(candidate)
        if settings.python_executable is None or not settings.python_executable.is_file():
            return _error(settings.mode, "python executable missing", candidate_sha256=before_hash)
        if settings.release_archive is None or not settings.release_archive.is_file():
            return _error(settings.mode, "release archive missing", candidate_sha256=before_hash)
        if not settings.release_sha256 or _file_sha256(settings.release_archive) != settings.release_sha256:
            return _error(settings.mode, "release archive SHA256 mismatch", candidate_sha256=before_hash)
        if settings.evidence_root is None:
            return _error(settings.mode, "evidence root missing", candidate_sha256=before_hash)
        settings.evidence_root.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as exc:
        return _error(settings.mode, str(exc))

    with tempfile.TemporaryDirectory(prefix="hermes-external-admission-") as td:
        extracted = Path(td)
        try:
            _safe_extract(settings.release_archive, extracted)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            return _error(settings.mode, f"release extraction failed: {exc}", candidate_sha256=before_hash)
        package_root = extracted / "src"
        scanner_config = extracted / "config" / "local_windows.json"
        if not package_root.is_dir() or not scanner_config.is_file():
            return _error(settings.mode, "release layout incomplete", candidate_sha256=before_hash)
        cmd = [
            str(settings.python_executable), "-m", "skill_admission_gate",
            "--skill", str(candidate), "--source-id", source_id or "hermes",
            "--config", str(scanner_config), "--evidence-root", str(settings.evidence_root),
            "--timeout-seconds", str(settings.scanner_timeout_seconds),
        ]
        if source_version:
            cmd.extend(["--source-version", source_version])
        try:
            completed = subprocess.run(
                cmd, shell=False, capture_output=True, text=True, encoding="utf-8", errors="replace",
                stdin=subprocess.DEVNULL, timeout=settings.timeout_seconds, check=False,
                env=_sanitized_env(package_root),
            )
        except subprocess.TimeoutExpired:
            return _error(settings.mode, "external admission gate timeout", executed=True,
                          candidate_sha256=before_hash)
        except OSError as exc:
            return _error(settings.mode, f"external admission launch failed: {exc}", executed=True,
                          candidate_sha256=before_hash)

    try:
        payload = _last_json_line(completed.stdout or "")
        decision = str(payload.get("decision", ""))
        candidate_sha256 = str(payload.get("candidate_sha256", ""))
        evidence_dir = str(payload.get("evidence_dir", ""))
        expected_code = _VALID_DECISIONS.get(decision)
        if expected_code is None or completed.returncode != expected_code:
            raise ValueError(f"decision/exit mismatch: {decision}/{completed.returncode}")
        if len(candidate_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in candidate_sha256):
            raise ValueError("invalid candidate SHA256 in external decision")
        if candidate_sha256.lower() != before_hash.lower():
            raise ValueError("candidate SHA256 does not match pre-scan content")
        after_hash = _candidate_hash(candidate)
        if after_hash.lower() != candidate_sha256.lower():
            raise ValueError("candidate hash drift after external scan")
        evidence_root = settings.evidence_root.resolve()
        evidence_path = Path(evidence_dir)
        if not evidence_path.is_dir() or _is_redirect(evidence_path):
            raise ValueError("external evidence directory invalid")
        evidence_resolved = evidence_path.resolve()
        if evidence_resolved == evidence_root or not evidence_resolved.is_relative_to(evidence_root):
            raise ValueError("external evidence directory outside configured root")
        decision_path = evidence_resolved / "decision.json"
        if not decision_path.is_file() or _is_redirect(decision_path):
            raise ValueError("external decision evidence missing or redirected")
        evidence = json.loads(decision_path.read_text(encoding="utf-8"))
        if not isinstance(evidence, dict):
            raise ValueError("external decision evidence is not an object")
        if str(evidence.get("decision", "")) != decision:
            raise ValueError("external evidence decision mismatch")
        if str(evidence.get("candidate_sha256", "")).lower() != candidate_sha256.lower():
            raise ValueError("external evidence hash mismatch")
        approval_binding = None
        if decision == "QUARANTINE":
            approval_binding = _build_approval_binding(
                evidence, evidence_resolved, settings, candidate_sha256,
                source_id, source_version,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _error(settings.mode, str(exc), executed=True,
                      candidate_sha256=before_hash,
                      evidence_dir=str(payload.get("evidence_dir", "")) if 'payload' in locals() else "")

    return ExternalAdmissionResult(
        mode=settings.mode,
        executed=True,
        decision=decision,
        allow_continue=(settings.mode == "shadow" or decision == "ALLOW_TO_LAB"),
        candidate_sha256=candidate_sha256.lower(),
        evidence_dir=evidence_dir,
        approval_binding=approval_binding,
        error="",
    )
