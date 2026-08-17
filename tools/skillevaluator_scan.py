#!/usr/bin/env python3
"""Advisory NVIDIA SkillEvaluator Tier 1 scan for skill installs.

Runs alongside (never instead of) the built-in skills guard
(``tools/skills_guard.py``). The skills guard remains the enforcement
layer — trust levels, install policy, block verdicts. This module adds a
second, advisory opinion from NVIDIA's SkillEvaluator: deterministic,
keyless Tier 1 static checks (PII, unicode smuggling, script lint).

Design contract (deliberate):

- **Warn, don't block.** PII-class findings (emails, personal paths,
  connection-string placeholders) are shown to the user with file/line and
  the install continues. The upstream PII scanner has known false-positive
  classes (``git@github.com``, documentation example emails, ``op://``
  secret-manager references), so its findings are surfaced as information,
  never used to reject a skill outright.
- **Prompt only for secrets-class criticals.** Findings that look like a
  real leaked credential (private keys, cloud access keys, tokens,
  credentialed connection strings) get one confirmation beat in
  interactive installs. ``--force`` skips the prompt; non-interactive
  installs (TUI/agent, ``skip_confirm=True``) proceed with a loud warning
  rather than wedging on a prompt nobody can answer.
- **Never break installs.** Scanner missing from PATH, crashing, timing
  out, or emitting unparseable output all degrade to a no-op. The
  built-in guard has already run by the time this executes.

The scanner binary is optional::

    uv tool install --python 3.13 \
        "skillevaluator @ git+https://github.com/NVIDIA/SkillEvaluator.git"

Enable/disable via ``skills.tier1_advisory`` in config.yaml (default: on;
a no-op unless the binary is installed).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

SCANNER_BIN = "skillevaluator"
SCANNER_NAME = "skillevaluator-tier1"

# Keyless, deterministic Tier 1 checks. Schema/quality are excluded on
# purpose: they are hygiene signal for the index pipeline
# (scripts/scan_skills_index.py), not install-time signal — a missing
# author field should never make an install noisier.
#
# `security` invokes NVIDIA SkillSpector (a second optional binary,
# pinned separately: uv tool install
# "git+https://github.com/NVIDIA/SkillSpector.git") in its static-rules
# mode — still keyless, no LLM calls. When SkillSpector is absent or its
# report fails SkillEvaluator's internal consistency checks, the check
# reports status="incomplete" and is treated as "no opinion" here.
TIER1_CHECKS = "pii,unicode,lint,license,security"
SCAN_TIMEOUT_SECONDS = 120

# check_name values (from SkillEvaluator's pii_patterns.yaml categories)
# that indicate a possible REAL credential rather than personal-info
# hygiene. These are the only findings that earn a confirmation prompt.
SECRETS_CLASS_CHECKS = frozenset({
    "database_credentials",
    "hardcoded_secrets",
    "jwt_tokens",
    "webhook_urls",
    "aws_identifiers",
    "github_tokens",
    "private_keys",
})


@dataclass
class Tier1Finding:
    check: str          # e.g. "emails", "database_credentials"
    validator: str      # e.g. "PII Scan"
    severity: str       # "critical" | "high" | "medium" | "low" | "info"
    message: str
    file: str = ""
    line: int = 0
    suggestion: str = ""

    @property
    def is_secrets_class(self) -> bool:
        return self.check in SECRETS_CLASS_CHECKS

    def location(self) -> str:
        if self.file and self.line:
            return f"{self.file}:{self.line}"
        return self.file or "?"


@dataclass
class Tier1Report:
    available: bool                 # scanner ran and produced a report
    passed: bool = True
    findings: List[Tier1Finding] = field(default_factory=list)
    incomplete_checks: List[str] = field(default_factory=list)
    error: str = ""                 # why the scan is unavailable (debug only)

    @property
    def advisory_findings(self) -> List[Tier1Finding]:
        return [f for f in self.findings if not f.is_secrets_class]

    @property
    def secrets_findings(self) -> List[Tier1Finding]:
        return [f for f in self.findings if f.is_secrets_class]


def scanner_available() -> bool:
    return shutil.which(SCANNER_BIN) is not None


def tier1_advisory_enabled() -> bool:
    """Read skills.tier1_advisory from config (default True).

    On-by-default is safe: without the optional scanner binary on PATH
    the scan is a silent no-op, so fresh installs see no behavior change
    until a user opts in by installing SkillEvaluator.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        skills_cfg = cfg.get("skills") or {}
        if not isinstance(skills_cfg, dict):
            return True
        value = skills_cfg.get("tier1_advisory", True)
        if isinstance(value, str):
            return value.strip().lower() not in ("false", "0", "no", "off")
        return bool(value)
    except Exception:
        return True


def _parse_report(report: dict) -> Tier1Report:
    """Reduce a SkillEvaluator JSON report to install-relevant findings.

    A validator whose ``status`` is ``"incomplete"`` produced no usable
    evidence (e.g. SkillSpector missing, or its report failed
    SkillEvaluator's internal consistency checks) — its fail verdict
    carries no findings, so it is recorded as an incomplete check and
    excluded from the pass/fail signal rather than rendered as an
    unexplained failure.
    """
    findings: List[Tier1Finding] = []
    incomplete: List[str] = []
    any_complete_failed = False
    for res in report.get("results", []) or []:
        validator = str(res.get("validator", "unknown"))
        if str(res.get("status", "")).lower() == "incomplete":
            incomplete.append(validator)
            continue
        if not res.get("passed", True):
            any_complete_failed = True
        for f in res.get("findings", []) or []:
            if not isinstance(f, dict):
                continue
            findings.append(Tier1Finding(
                check=str(f.get("check_name", "")),
                validator=validator,
                severity=str(f.get("severity", "info")).lower(),
                message=str(f.get("message", ""))[:200],
                file=str(f.get("file_path", "")),
                line=int(f.get("line_number") or 0),
                suggestion=str(f.get("suggestion", ""))[:200],
            ))
    return Tier1Report(
        available=True,
        passed=not any_complete_failed and not findings,
        findings=findings,
        incomplete_checks=incomplete,
    )


def run_tier1_scan(skill_dir: Path, timeout: int = SCAN_TIMEOUT_SECONDS) -> Tier1Report:
    """Run SkillEvaluator Tier 1 over one skill directory.

    Returns a report with ``available=False`` (and no findings) on any
    failure — the caller treats that as "no advisory opinion", never as
    an error.
    """
    if not scanner_available():
        return Tier1Report(available=False, error="scanner not on PATH")
    with tempfile.TemporaryDirectory(prefix="se-tier1-") as outdir:
        try:
            subprocess.run(
                [SCANNER_BIN, "validate", str(skill_dir),
                 "--checks", TIER1_CHECKS, "--no-dedup",
                 "-r", "json", "-o", outdir],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return Tier1Report(available=False, error=f"scan timed out after {timeout}s")
        except OSError as exc:
            return Tier1Report(available=False, error=f"scanner failed to launch: {exc}")
        reports = sorted(Path(outdir).glob("skillevaluator-output-*.json"))
        if not reports:
            return Tier1Report(available=False, error="scanner produced no JSON report")
        try:
            parsed = json.loads(reports[-1].read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return Tier1Report(available=False, error=f"unparseable report: {exc}")
        if not isinstance(parsed, dict):
            return Tier1Report(available=False, error="unexpected report shape")
        return _parse_report(parsed)


def format_tier1_report(report: Tier1Report, limit: int = 10) -> str:
    """Plain-text advisory summary for console display."""
    if not report.available:
        return ""
    lines: List[str] = []
    if not report.findings:
        lines.append("SkillEvaluator Tier 1: no findings.")
    else:
        lines.append(
            f"SkillEvaluator Tier 1 (advisory): "
            f"{len(report.findings)} finding(s) — informational, verify before relying on this skill."
        )
        shown = report.secrets_findings + report.advisory_findings
        for f in shown[:limit]:
            tag = "SECRETS" if f.is_secrets_class else f.severity.upper()
            lines.append(f"  [{tag}] {f.location()} — {f.message}")
        if len(shown) > limit:
            lines.append(f"  … and {len(shown) - limit} more")
    if report.incomplete_checks:
        names = ", ".join(report.incomplete_checks)
        lines.append(f"  (not run: {names} — no opinion from these checks)")
    return "\n".join(lines)
