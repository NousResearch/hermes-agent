#!/usr/bin/env python3
"""Mechanical package preflight for externally sourced skills.

Every skill downloaded from a registry passes through this preflight before
installation. It validates package structure, path containment, symlinks,
file types, executable bits, and bounded file/count/size limits. Authored text
is never classified: wording and invisible Unicode are preserved for the
model to interpret.

The trust-aware install policy applies only to mechanical preflight findings.

Trust levels:
  - builtin:   Ships with Hermes. Mechanical findings are reported but allowed.
  - trusted:   Verified source repositories. Caution findings are allowed.
  - community: Everything else. Caution findings require --force.

Usage:
    from tools.skills_guard import scan_skill, should_allow_install, format_scan_report

    result = scan_skill(Path("skills/.hub/quarantine/some-skill"), source="community")
    allowed, reason = should_allow_install(result)
    if not allowed:
        print(format_scan_report(result))
"""

import fnmatch
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Hardcoded trust configuration
# ---------------------------------------------------------------------------

TRUSTED_REPOS = {
    "openai/skills",
    "anthropics/skills",
    "huggingface/skills",
    # NVIDIA-verified skills: each entry ships a signed `skill.oms.sig`
    # and a governance `skill-card.md` (sync pipeline drops anything
    # missing the signature or card). Catalog details:
    # https://github.com/NVIDIA/skills
    "NVIDIA/skills",
}

INSTALL_POLICY = {
    #                  safe      caution    dangerous
    "builtin":       ("allow",  "allow",   "allow"),
    "trusted":       ("allow",  "allow",   "block"),
    "community":     ("allow",  "block",   "block"),
    # Agent-created: "ask" for a mechanically dangerous package. This gate only runs when
    # skills.guard_agent_created is enabled (off by default) — see
    # tools/skill_manager_tool.py::_guard_agent_created_enabled.
    "agent-created": ("allow",  "allow",   "ask"),
}

VERDICT_INDEX = {"safe": 0, "caution": 1, "dangerous": 2}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    pattern_id: str
    severity: str       # "critical" | "high" | "medium" | "low"
    category: str       # mechanical category: "structural" | "traversal"
    file: str
    line: int
    match: str
    description: str


@dataclass
class ScanResult:
    skill_name: str
    source: str
    trust_level: str    # "builtin" | "trusted" | "community"
    verdict: str        # "safe" | "caution" | "dangerous"
    findings: List[Finding] = field(default_factory=list)
    scanned_at: str = ""
    summary: str = ""


# ---------------------------------------------------------------------------
# Mechanical package limits
# ---------------------------------------------------------------------------

# Structural limits for skill directories
MAX_FILE_COUNT = 50       # skills shouldn't have 50+ files
MAX_TOTAL_SIZE_KB = 1024  # 1MB total is suspicious for a skill
MAX_SINGLE_FILE_KB = 256  # individual file > 256KB is suspicious

# Known binary extensions that should NOT be in a skill
SUSPICIOUS_BINARY_EXTENSIONS = {
    '.exe', '.dll', '.so', '.dylib', '.bin', '.dat', '.com',
    '.msi', '.dmg', '.app', '.deb', '.rpm',
}

# ---------------------------------------------------------------------------
# Scanning functions
# ---------------------------------------------------------------------------

def scan_skill(skill_path: Path, source: str = "community") -> ScanResult:
    """Run the mechanical package preflight for a skill path.

    Authored text is not inspected or classified. Directory packages retain
    path/symlink/file-type/executable/count/size checks, while a standalone
    file receives the same per-file type/permission/size checks.
    """
    skill_name = skill_path.name
    trust_level = _resolve_trust_level(source)

    if skill_path.is_dir():
        ignore = _load_skill_ignore(skill_path)
        findings = _check_structure(skill_path, ignore=ignore)
    elif skill_path.is_file():
        findings = _check_single_file_structure(skill_path, skill_path.name)
    else:
        findings = []

    verdict = _determine_verdict(findings)
    summary = _build_summary(skill_name, source, trust_level, verdict, findings)
    return ScanResult(
        skill_name=skill_name,
        source=source,
        trust_level=trust_level,
        verdict=verdict,
        findings=findings,
        scanned_at=datetime.now(timezone.utc).isoformat(),
        summary=summary,
    )


def should_allow_install(result: ScanResult, force: bool = False) -> Tuple[bool, str]:
    """Determine whether a skill may be installed after mechanical preflight.

    Args:
        result: Mechanical preflight result from scan_skill().
        force: If True, override eligible trust-policy decisions.

    Returns:
        (allowed, reason) tuple
    """
    policy = INSTALL_POLICY.get(result.trust_level, INSTALL_POLICY["community"])
    vi = VERDICT_INDEX.get(result.verdict, 2)
    decision = policy[vi]

    if decision == "allow":
        return True, f"Allowed ({result.trust_level} source, {result.verdict} verdict)"

    if force and not (result.verdict == "dangerous" and result.trust_level in ("community", "trusted")):
        return True, (
            f"Force-installed despite {result.verdict} verdict "
            f"({len(result.findings)} findings)"
        )

    if decision == "ask":
        # Return None to signal "needs user confirmation"
        return None, (
            f"Requires confirmation ({result.trust_level} source + {result.verdict} verdict, "
            f"{len(result.findings)} findings)"
        )

    # Dangerous verdicts cannot be overridden by --force (community/trusted);
    # other blocks can.
    if result.verdict == "dangerous" and result.trust_level in ("community", "trusted"):
        return False, (
            f"Blocked ({result.trust_level} source + dangerous verdict, "
            f"{len(result.findings)} findings). --force does not override a dangerous verdict."
        )
    return False, (
        f"Blocked ({result.trust_level} source + {result.verdict} verdict, "
        f"{len(result.findings)} findings). Use --force to override."
    )


def format_scan_report(result: ScanResult) -> str:
    """Format a mechanical preflight result for CLI or chat display.

    Returns a compact multi-line report suitable for CLI or chat display.
    """
    lines = []

    verdict_display = result.verdict.upper()
    lines.append(
        f"Structural preflight: {result.skill_name} "
        f"({result.source}/{result.trust_level})  Verdict: {verdict_display}"
    )

    if result.findings:
        # Group and sort: critical first, then high, medium, low
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_findings = sorted(result.findings, key=lambda f: severity_order.get(f.severity, 4))

        for f in sorted_findings:
            sev = f.severity.upper().ljust(8)
            cat = f.category.ljust(14)
            loc = f"{f.file}:{f.line}".ljust(30)
            lines.append(f"  {sev} {cat} {loc} \"{f.match[:60]}\"")

        lines.append("")

    allowed, reason = should_allow_install(result)
    if allowed is True:
        status = "ALLOWED"
    elif allowed is None:
        status = "NEEDS CONFIRMATION"
    else:
        status = "BLOCKED"
    lines.append(f"Decision: {status} — {reason}")

    return "\n".join(lines)


def content_hash(skill_path: Path) -> str:
    """Compute a SHA-256 hash of all files in a skill directory for integrity tracking.

    File paths (relative to ``skill_path``) are mixed into the hash alongside
    file contents so that swapping the contents of two files in a skill
    changes the hash. This must stay symmetric with
    ``tools.skills_hub.bundle_content_hash`` — both functions need to
    produce the same digest for the same skill (one operates on disk,
    one on an in-memory bundle), so any change to the hash shape MUST
    land in both places at once.
    """
    h = hashlib.sha256()
    if skill_path.is_dir():
        for f in sorted(skill_path.rglob("*")):
            if f.is_file():
                try:
                    rel = f.relative_to(skill_path).as_posix()
                    h.update(rel.encode("utf-8"))
                    h.update(b"\x00")
                    h.update(f.read_bytes())
                except OSError:
                    continue
    elif skill_path.is_file():
        h.update(skill_path.read_bytes())
    return f"sha256:{h.hexdigest()[:16]}"


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------

def _check_single_file_structure(file_path: Path, rel_path: str) -> List[Finding]:
    """Return mechanical type, permission, and size findings for one file."""
    try:
        stat = file_path.stat()
    except OSError:
        return []

    findings: List[Finding] = []
    size = stat.st_size
    if size > MAX_SINGLE_FILE_KB * 1024:
        findings.append(Finding(
            pattern_id="oversized_file",
            severity="medium",
            category="structural",
            file=rel_path,
            line=0,
            match=f"{size // 1024}KB",
            description=(
                f"file is {size // 1024}KB (limit: {MAX_SINGLE_FILE_KB}KB)"
            ),
        ))

    ext = file_path.suffix.lower()
    if ext in SUSPICIOUS_BINARY_EXTENSIONS:
        findings.append(Finding(
            pattern_id="binary_file",
            severity="critical",
            category="structural",
            file=rel_path,
            line=0,
            match=f"binary: {ext}",
            description=f"binary/executable file ({ext}) should not be in a skill",
        ))

    if ext not in {'.sh', '.bash', '.py', '.rb', '.pl'} and stat.st_mode & 0o111:
        findings.append(Finding(
            pattern_id="unexpected_executable",
            severity="medium",
            category="structural",
            file=rel_path,
            line=0,
            match="executable bit set",
            description=(
                "file has executable permission but is not a recognized script type"
            ),
        ))
    return findings

def _check_structure(skill_dir: Path, ignore=None) -> List[Finding]:
    """
    Check the skill directory for structural anomalies:
    - Too many files
    - Suspiciously large total size
    - Binary/executable files that shouldn't be in a skill
    - Symlinks pointing outside the skill directory
    - Individual files that are too large

    Args:
        skill_dir: Path to the skill directory.
        ignore: Optional callable taking a relative posix path and returning
            True if the path should be excluded (e.g. from `.skillignore`).
            Ignored files are not counted toward the file count, total size,
            or any structural finding.
    """
    if ignore is None:
        ignore = lambda _rel: False  # noqa: E731

    findings = []
    file_count = 0
    total_size = 0

    for f in skill_dir.rglob("*"):
        if not f.is_file() and not f.is_symlink():
            continue

        rel = str(f.relative_to(skill_dir))
        if ignore(rel):
            continue
        file_count += 1

        # Symlink check — must resolve within the skill directory
        if f.is_symlink():
            try:
                resolved = f.resolve()
                if not resolved.is_relative_to(skill_dir.resolve()):
                    findings.append(Finding(
                        pattern_id="symlink_escape",
                        severity="critical",
                        category="traversal",
                        file=rel,
                        line=0,
                        match=f"symlink -> {resolved}",
                        description="symlink points outside the skill directory",
                    ))
            except OSError:
                findings.append(Finding(
                    pattern_id="broken_symlink",
                    severity="medium",
                    category="traversal",
                    file=rel,
                    line=0,
                    match="broken symlink",
                    description="broken or circular symlink",
                ))
            continue

        # Size tracking
        try:
            size = f.stat().st_size
            total_size += size
        except OSError:
            continue

        findings.extend(_check_single_file_structure(f, rel))

    # File count limit
    if file_count > MAX_FILE_COUNT:
        findings.append(Finding(
            pattern_id="too_many_files",
            severity="medium",
            category="structural",
            file="(directory)",
            line=0,
            match=f"{file_count} files",
            description=f"skill has {file_count} files (limit: {MAX_FILE_COUNT})",
        ))

    # Total size limit
    if total_size > MAX_TOTAL_SIZE_KB * 1024:
        findings.append(Finding(
            pattern_id="oversized_skill",
            severity="high",
            category="structural",
            file="(directory)",
            line=0,
            match=f"{total_size // 1024}KB total",
            description=f"skill is {total_size // 1024}KB total (limit: {MAX_TOTAL_SIZE_KB}KB)",
        ))

    return findings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Ignore-file names a skill may ship to exclude dev/docs artifacts from the
# mechanical package preflight. `.skillignore` is the Hermes-native name;
# `.clawhubignore` is honored for compatibility with skills published through
# ClawHub.
_SKILL_IGNORE_FILENAMES = (".skillignore", ".clawhubignore")

# Paths that are always excluded from mechanical package accounting, and
# SKILL.md which can never be excluded via the ignore file.
_ALWAYS_IGNORED_NAMES = set(_SKILL_IGNORE_FILENAMES)
_NEVER_IGNORABLE = {"SKILL.md"}


def _load_skill_ignore(skill_dir: Path):
    """Build a matcher from a skill's `.skillignore` / `.clawhubignore`.

    Returns a callable ``ignore(rel_posix_path) -> bool``. The matcher
    supports gitignore-style basics: blank lines and ``#`` comments are
    skipped, a trailing ``/`` marks a directory (matches that dir and
    everything under it), and ``*``/``?`` globs are honored via fnmatch on
    both the full relative path and each path segment. A leading ``/``
    anchors a pattern to the skill root. The ignore files themselves are
    always excluded; ``SKILL.md`` can never be excluded.
    """
    patterns: List[str] = []
    for name in _SKILL_IGNORE_FILENAMES:
        ig = skill_dir / name
        try:
            if ig.is_file():
                for raw in ig.read_text(encoding="utf-8").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    patterns.append(line)
        except (UnicodeDecodeError, OSError):
            continue

    def ignore(rel: str) -> bool:
        rel_posix = Path(rel).as_posix()
        base = rel_posix.split("/")[-1]

        if base in _NEVER_IGNORABLE:
            return False
        if base in _ALWAYS_IGNORED_NAMES:
            return True

        for pat in patterns:
            anchored = pat.startswith("/")
            p = pat.lstrip("/")
            is_dir = p.endswith("/")
            p = p.rstrip("/")
            if not p:
                continue

            if is_dir:
                # Directory pattern: match the dir itself or anything under it.
                if rel_posix == p or rel_posix.startswith(p + "/"):
                    return True
                if not anchored and ("/" + rel_posix + "/").find("/" + p + "/") != -1:
                    return True
                continue

            # File/glob pattern.
            if fnmatch.fnmatch(rel_posix, p):
                return True
            if not anchored:
                # Unanchored: also match the basename and any path segment.
                if fnmatch.fnmatch(base, p):
                    return True
                if "/" not in p and any(
                    fnmatch.fnmatch(seg, p) for seg in rel_posix.split("/")
                ):
                    return True
                # Match a prefix directory component (e.g. `docs` ignores
                # `docs/plans/x.md`).
                if rel_posix.startswith(p + "/"):
                    return True
        return False

    return ignore


def _resolve_trust_level(source: str) -> str:
    """Map a source identifier to a trust level."""
    prefix_aliases = (
        "skills-sh/",
        "skills.sh/",
        "skils-sh/",
        "skils.sh/",
    )
    normalized_source = source
    for prefix in prefix_aliases:
        if normalized_source.startswith(prefix):
            normalized_source = normalized_source[len(prefix):]
            break

    # Agent-created skills get their own permissive trust level
    if normalized_source == "agent-created":
        return "agent-created"
    # Official optional skills must be identified by source provenance, not by
    # user-controlled GitHub identifiers such as "official/<repo>".
    if normalized_source == "official":
        return "builtin"
    # Check if source matches any trusted repo exactly, or a skill path inside
    # that repo. Do not trust sibling repositories that merely share a prefix.
    for trusted in TRUSTED_REPOS:
        if normalized_source == trusted or normalized_source.startswith(f"{trusted}/"):
            return "trusted"
    return "community"


def _determine_verdict(findings: List[Finding]) -> str:
    """Determine the overall verdict from a list of findings."""
    if not findings:
        return "safe"

    has_critical = any(f.severity == "critical" for f in findings)
    has_high = any(f.severity == "high" for f in findings)

    if has_critical:
        return "dangerous"
    if has_high:
        return "caution"
    # medium/low findings alone are informational, not blocking
    return "safe"


def _build_summary(name: str, source: str, trust: str, verdict: str, findings: List[Finding]) -> str:
    """Build a one-line summary of the scan result."""
    if not findings:
        return f"{name}: mechanical preflight passed"

    categories = {f.category for f in findings}
    return f"{name}: {verdict} — {len(findings)} mechanical finding(s) in {', '.join(sorted(categories))}"
