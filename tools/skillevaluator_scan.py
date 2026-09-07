#!/usr/bin/env python3
"""External second-opinion scanning for Hermes install surfaces.

The native Hermes guards remain the primary enforcement layer. SkillEvaluator
provides deterministic hygiene checks and SkillSpector provides static security
analysis without LLM calls. The default policy is report-only; operators may
enable blocking for unsuppressed high/critical SkillSpector findings. Scanner
failure remains fail-open unless ``fail_on_incomplete`` is explicitly enabled.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

SCANNER_BIN = "skillevaluator"
SKILLSPECTOR_BIN = "skillspector"
# Keyless deterministic hygiene checks. Security is run once, directly through
# SkillSpector, because SkillEvaluator v0.1.0 rejects valid SkillSpector v2.11
# metadata and otherwise causes an immediate duplicate scan.
TIER1_CHECKS = "pii,unicode,lint,license"
SCAN_TIMEOUT_SECONDS = 120

# pii_patterns.yaml categories indicating a possible REAL credential (not PII hygiene) — the only prompt-worthy ones.
SECRETS_CLASS_CHECKS = frozenset({"database_credentials", "hardcoded_secrets", "jwt_tokens", "webhook_urls",
                                  "aws_identifiers", "github_tokens", "private_keys"})


@dataclass
class Tier1Finding:
    check: str          # e.g. "emails", "database_credentials"
    validator: str      # e.g. "PII Scan"
    severity: str       # "critical" | "high" | "medium" | "low" | "info"
    message: str
    file: str = ""
    line: int = 0
    suggestion: str = ""
    scanner: str = "skillevaluator"

    @property
    def is_secrets_class(self) -> bool:
        return self.check in SECRETS_CLASS_CHECKS

    def location(self) -> str:
        return f"{self.file}:{self.line}" if self.file and self.line else self.file or "?"

    # Plugin dashboard compatibility: native plugin findings use these names.
    @property
    def pattern_id(self) -> str:
        return self.check

    @property
    def category(self) -> str:
        return self.validator

    @property
    def description(self) -> str:
        return self.message


@dataclass
class Tier1Report:
    available: bool                 # scanner ran and produced a report
    passed: bool = True
    findings: List[Tier1Finding] = field(default_factory=list)
    incomplete_checks: List[str] = field(default_factory=list)
    error: str = ""                 # why the scan is unavailable (debug only)
    scanner: str = "skillevaluator"
    risk_score: float = 0.0
    risk_severity: str = ""
    recommendation: str = ""
    analysis_complete: bool = True
    llm_used: bool = False
    suppressed_count: int = 0

    @property
    def advisory_findings(self) -> List[Tier1Finding]:
        return [f for f in self.findings if not f.is_secrets_class]

    @property
    def secrets_findings(self) -> List[Tier1Finding]:
        return [f for f in self.findings if f.is_secrets_class]


_FALSE_STRINGS = frozenset({"false", "0", "no", "off"})
_TRUE_STRINGS = frozenset({"true", "1", "yes", "on"})


def _config_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _FALSE_STRINGS:
            return False
        if normalized in _TRUE_STRINGS:
            return True
        return default
    return bool(value)


def tier1_advisory_enabled() -> bool:
    """``skills.tier1_advisory`` (default True; safe because the scan is a no-op without the binary)."""
    try:
        from hermes_cli.config import load_config
        skills_cfg = load_config().get("skills") or {}
        value = skills_cfg.get("tier1_advisory", True) if isinstance(skills_cfg, dict) else True
        return _config_bool(value, default=True)
    except Exception:
        return True


def _external_scanner_config() -> dict:
    try:
        from hermes_cli.config import cfg_get, load_config
        scanner = cfg_get(load_config(), "security", "external_scanner", default={})
        return scanner if isinstance(scanner, dict) else {}
    except Exception:
        return {}


def external_surface_enabled(surface: str) -> bool:
    """Whether direct external scanning is enabled for a non-skill surface."""
    key = {"plugins": "scan_plugins", "mcp": "scan_mcp"}.get(surface, "")
    if not key:
        return False
    return _config_bool(_external_scanner_config().get(key), default=False)


def should_allow_tier1(report: Tier1Report, config: dict | None = None) -> tuple[bool, str]:
    """Evaluate operator policy while keeping report-only as the safe default.

    ``block_high`` blocks only unsuppressed high/critical findings. SkillSpector
    applies the configured baseline before this point. Missing or incomplete
    scanners remain advisory unless ``fail_on_incomplete`` is explicitly true.
    """
    config = config if isinstance(config, dict) else _external_scanner_config()
    mode = str(config.get("mode") or "report").strip().lower()
    fail_on_incomplete = _config_bool(config.get("fail_on_incomplete"), default=False)
    if mode != "block_high":
        return True, "External scanner is report-only"
    if not report.available:
        return (False, "External scanner unavailable (fail-closed)") if fail_on_incomplete else (
            True, "External scanner unavailable; advisory policy continues")
    if report.incomplete_checks or not report.analysis_complete:
        if fail_on_incomplete:
            return False, "External scanner analysis incomplete (fail-closed)"
    blocking = [f for f in report.findings
                if f.scanner == "skillspector" and f.severity in {"high", "critical"}]
    if blocking:
        levels = ", ".join(sorted({f.severity for f in blocking}))
        return False, f"External scanner found {len(blocking)} high/critical finding(s): {levels}"
    return True, "No unsuppressed high/critical external findings"


def _parse_report(report: dict) -> Tier1Report:
    """Reduce a SkillEvaluator JSON report to install-relevant findings. Findings from ``status == "incomplete"``
    validators are kept (partial evidence is evidence) but excluded from the pass/fail signal."""
    findings: List[Tier1Finding] = []
    incomplete: List[str] = []
    failed = False
    for res in report.get("results", []) or []:
        validator = str(res.get("validator", "unknown"))
        if str(res.get("status", "")).lower() == "incomplete":
            incomplete.append(validator)
        else:
            failed = failed or not res.get("passed", True)
        findings.extend(Tier1Finding(
            check=str(f.get("check_name", "")), validator=validator, severity=str(f.get("severity", "info")).lower(),
            message=str(f.get("message", ""))[:200], file=str(f.get("file_path", "")),
            line=int(f.get("line_number") or 0), suggestion=str(f.get("suggestion", ""))[:200],
            scanner="skillspector" if validator.casefold() == "security scan" else "skillevaluator")
            for f in res.get("findings", []) or [] if isinstance(f, dict))
    return Tier1Report(available=True, passed=not failed and not findings, findings=findings,
                       incomplete_checks=incomplete)


def _parse_skillspector_report(report: dict) -> Tier1Report:
    """Parse SkillSpector's native JSON without normalizing its risk metadata.

    SkillEvaluator v0.1.0 rejects SkillSpector v2.11.0 reports where a clean
    static-only scan has ``risk_severity=low`` and ``recommendation=caution``.
    That consistency check must not erase the underlying static evidence, so
    this parser is the compatibility fallback.
    """
    risk = report.get("risk_assessment") or {}
    metadata = report.get("metadata") or report.get("scan_metadata") or {}
    completeness = report.get("analysis_completeness") or {}
    complete = bool(completeness.get("is_complete", metadata.get("analysis_complete", False)))
    findings: List[Tier1Finding] = []
    for issue in report.get("issues", []) or []:
        if not isinstance(issue, dict):
            continue
        location = issue.get("location") or {}
        if not isinstance(location, dict):
            location = {}
        title = str(issue.get("finding") or issue.get("title") or "")
        description = str(issue.get("explanation") or issue.get("description") or "")
        message = f"{title}: {description}".strip(": ")[:200]
        findings.append(Tier1Finding(
            check=str(issue.get("id") or issue.get("rule_id") or issue.get("check_name") or ""),
            validator=str(issue.get("category") or "SkillSpector security"),
            severity=str(issue.get("severity") or "info").lower(),
            message=message,
            file=str(location.get("path") or location.get("file") or issue.get("file_path") or ""),
            line=int(location.get("start_line") or location.get("line") or issue.get("line_number") or 0),
            suggestion=str(issue.get("remediation") or issue.get("recommendation") or issue.get("suggestion") or "")[:200],
            scanner="skillspector",
        ))
    return Tier1Report(
        available=True,
        passed=not findings,
        findings=findings,
        incomplete_checks=[] if complete else ["SkillSpector security"],
        scanner="skillspector",
        risk_score=float(risk.get("score", risk.get("risk_score")) or 0.0),
        risk_severity=str(risk.get("severity", risk.get("risk_severity")) or "").lower(),
        recommendation=str(risk.get("recommendation") or "").lower(),
        analysis_complete=complete,
        llm_used=bool(metadata.get("llm_used", False) or metadata.get("inference_usage")),
        suppressed_count=int(report.get("suppressed_count", metadata.get("suppressed_count")) or 0),
    )


def _configured_baseline(config: dict | None = None) -> str:
    """Return the operator-controlled SkillSpector baseline path, if configured."""
    config = config if isinstance(config, dict) else _external_scanner_config()
    return str(config.get("baseline", "")).strip()


def _run_skillspector(skill_dir: Path, timeout: int, *, baseline: str | None = None) -> Tier1Report:
    unavailable = lambda why: Tier1Report(available=False, error=why, scanner="skillspector")  # noqa: E731
    if shutil.which(SKILLSPECTOR_BIN) is None:
        return unavailable("SkillSpector not on PATH")
    with tempfile.TemporaryDirectory(prefix="skillspector-static-") as outdir:
        report_file = Path(outdir) / "report.json"
        cmd = [SKILLSPECTOR_BIN, "scan", str(skill_dir), "--no-llm", "--format", "json",
               "--output", str(report_file)]
        baseline = _configured_baseline() if baseline is None else baseline
        if baseline:
            cmd.extend(["--baseline", baseline, "--show-suppressed"])
        try:
            subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           stdin=subprocess.DEVNULL, timeout=timeout)
        except subprocess.TimeoutExpired:
            return unavailable(f"scan timed out after {timeout}s")
        except OSError as exc:
            return unavailable(f"scanner failed to launch: {exc}")
        if not report_file.is_file():
            return unavailable("scanner produced no JSON report")
        try:
            parsed = json.loads(report_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return unavailable(f"unparseable report: {exc}")
        return _parse_skillspector_report(parsed) if isinstance(parsed, dict) else unavailable(
            "unexpected report shape")


def _merge_security_report(base: Tier1Report, security: Tier1Report) -> Tier1Report:
    """Merge distinct hygiene and security results, deduplicating defensively."""
    incomplete = [name for name in base.incomplete_checks if name != "Security Scan"]
    incomplete.extend(name for name in security.incomplete_checks if name not in incomplete)
    if not security.available and "SkillSpector security" not in incomplete:
        incomplete.append("SkillSpector security")
    findings: List[Tier1Finding] = []
    seen = set()
    for finding in [*base.findings, *security.findings]:
        key = (finding.scanner, finding.check, finding.severity, finding.file,
               finding.line, finding.message)
        if key not in seen:
            findings.append(finding)
            seen.add(key)
    return Tier1Report(
        available=base.available or security.available,
        passed=base.passed and security.passed and not findings,
        findings=findings,
        incomplete_checks=incomplete,
        error="; ".join(part for part in (base.error, security.error) if part),
        scanner="skillevaluator+skillspector",
        risk_score=security.risk_score,
        risk_severity=security.risk_severity,
        recommendation=security.recommendation,
        analysis_complete=security.available and security.analysis_complete,
        llm_used=security.llm_used,
        suppressed_count=security.suppressed_count,
    )


def run_tier1_scan(skill_dir: Path, timeout: int = SCAN_TIMEOUT_SECONDS) -> Tier1Report:
    """Run distinct hygiene and static-security checks within one total budget."""
    unavailable = lambda why: Tier1Report(available=False, error=why)  # noqa: E731
    config = _external_scanner_config()
    budget = max(1, int(timeout))
    deadline = time.monotonic() + budget
    hygiene = unavailable("SkillEvaluator not on PATH")

    if shutil.which(SCANNER_BIN) is not None:
        # Reserve most of the total budget for the security scan, which is the
        # install-time enforcement signal. Hygiene checks are advisory.
        evaluator_timeout = max(1, min(30, budget // 3))
        with tempfile.TemporaryDirectory(prefix="se-tier1-") as outdir:
            try:
                subprocess.run(
                    [SCANNER_BIN, "validate", str(skill_dir), "--checks", TIER1_CHECKS,
                     "--no-dedup", "-r", "json", "-o", outdir],
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                    stdin=subprocess.DEVNULL, timeout=evaluator_timeout,
                )
            except subprocess.TimeoutExpired:
                hygiene = unavailable(f"SkillEvaluator timed out after {evaluator_timeout}s")
            except OSError as exc:
                hygiene = unavailable(f"SkillEvaluator failed to launch: {exc}")
            else:
                reports = sorted(Path(outdir).glob("skillevaluator-output-*.json"))
                if not reports:
                    hygiene = unavailable("SkillEvaluator produced no JSON report")
                else:
                    try:
                        parsed = json.loads(reports[-1].read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError) as exc:
                        hygiene = unavailable(f"unparseable SkillEvaluator report: {exc}")
                    else:
                        hygiene = _parse_report(parsed) if isinstance(parsed, dict) else unavailable(
                            "unexpected SkillEvaluator report shape")

    remaining = int(deadline - time.monotonic())
    security = (_run_skillspector(
        skill_dir, max(1, remaining), baseline=_configured_baseline(config),
    ) if remaining > 0 else Tier1Report(
        available=False, error=f"total scan budget exhausted after {budget}s",
        scanner="skillspector", analysis_complete=False,
    ))
    if hygiene.available:
        return _merge_security_report(hygiene, security)
    if security.available:
        return security
    return unavailable("; ".join(part for part in (hygiene.error, security.error) if part))


def evaluate_external_surface(path: Path, surface: str) -> tuple[Tier1Report, bool, str] | None:
    """Scan a plugin or MCP tree once and evaluate policy from one config snapshot."""
    key = {"plugins": "scan_plugins", "mcp": "scan_mcp"}.get(surface)
    config = _external_scanner_config()
    if not key or not _config_bool(config.get(key), default=False):
        return None
    try:
        timeout = max(1, int(config.get("scan_timeout_seconds", SCAN_TIMEOUT_SECONDS)))
    except (TypeError, ValueError):
        timeout = SCAN_TIMEOUT_SECONDS
    report = _run_skillspector(path, timeout, baseline=_configured_baseline(config))
    allowed, reason = should_allow_tier1(report, config=config)
    return report, allowed, reason


def format_tier1_report(report: Tier1Report, limit: int = 10) -> str:
    """Plain-text advisory summary for console display ("" when unavailable)."""
    if not report.available:
        return ""
    lines: List[str] = []
    if "skillspector" in report.scanner:
        lines.append(
            "SkillSpector static: "
            f"risk={report.risk_severity or 'unknown'} score={report.risk_score:g} "
            f"recommendation={report.recommendation or 'unknown'} "
            f"complete={'yes' if report.analysis_complete else 'no'} "
            f"suppressed={report.suppressed_count}"
        )
    if not report.findings:
        label = "External static scan" if report.scanner == "skillspector" else "SkillEvaluator Tier 1"
        lines.append(f"{label}: no findings from completed checks." if report.incomplete_checks
                     else f"{label}: no findings.")
    else:
        lines.append(f"SkillEvaluator Tier 1 (advisory): {len(report.findings)} finding(s) — informational, "
                     "verify before relying on this skill.")
        shown = report.secrets_findings + report.advisory_findings
        lines.extend(f"  [{'SECRETS' if f.is_secrets_class else f.severity.upper()}] {f.location()} — {f.message}"
                     for f in shown[:limit])
        if len(shown) > limit:
            lines.append(f"  … and {len(shown) - limit} more")
    if report.incomplete_checks:
        lines.append(f"  (not run: {', '.join(report.incomplete_checks)} — no opinion from these checks)")
    return "\n".join(lines)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Optional  # noqa: F401,E402

SCANNER_NAME = "skillevaluator-tier1"

def scanner_available() -> bool:
    return shutil.which(SCANNER_BIN) is not None
# ---- END PLUGIN-COMPAT ----
