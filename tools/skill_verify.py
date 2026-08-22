"""Optional per-skill outcome verification (Layer 1 of failure tracing).

A skill directory may declare a ``metadata.hermes.verify`` block in its
SKILL.md frontmatter pointing at a deterministic check Hermes runs after the
skill is used. Verdicts feed the per-skill outcome pipeline
(``agent.turn_outcome`` → ``skill_usage.bump_outcome``) instead of being
inferred from tool-call I/O. SKILL.md prose is the policy (S); the verify
block is how it's judged (V).

Design notes:
  - Opt-in only — see ``skill_usage.set_verify_enabled`` / ``is_verify_enabled``.
    Frontmatter is author-controlled (untrusted for hub/agent-authored skills);
    a skill may declare a capability but never grant itself permission to run.
  - Applicability before judgment: most turns don't touch what a given skill's
    verifier checks. ``applicability_check`` decides that first — SKIP is a
    third outcome alongside PASS/FAIL and never reaches ``bump_outcome()``
    (it isn't recorded at all).
    Recording a pass for an inapplicable check would be a fake success;
    recording nothing is correct.
  - Structured feedback over bare exit codes: a script that prints
    ``{"success": bool, "reason": str}`` gives the next step (curator review,
    ACSS Hypothesize) something to reason about. Falls back to exit-code-only.
  - Best-effort throughout, same as the rest of the usage-telemetry system:
    a broken or missing verify block never breaks the turn that triggered it.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from agent.skill_utils import parse_frontmatter, verify_block_declared
from hermes_cli._subprocess_compat import windows_hide_flags
from tools.skill_usage import is_curation_eligible, is_verify_enabled

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30
_MAX_TIMEOUT = 300  # a runaway verifier must not be able to hang a turn indefinitely

_INTERPRETERS = {
    ".py": [sys.executable],
    ".sh": ["sh"],  # POSIX only — .sh on Windows is rare enough to skip special-casing
}


@dataclass
class VerifySpec:
    skill_dir: Path
    run: str
    applicability_check: Optional[str]
    timeout_seconds: int


@dataclass
class VerifyOutcome:
    success: bool
    reason: str


def load_verify_spec(skill_dir: Path) -> Optional[VerifySpec]:
    """Read the ``metadata.hermes.verify`` block from a skill's own SKILL.md.

    Returns None when the skill has no SKILL.md, no verify block, or a
    malformed one. The SKILL.md frontmatter is re-read here deliberately —
    the prompt-builder snapshot cache stores a reduced projection that drops
    ``metadata.hermes`` and can be stale (manifest-mismatched); only reading
    the actual file guarantees ``run`` is current.
    """
    path = skill_dir / "SKILL.md"
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        frontmatter, _body = parse_frontmatter(content)
        if not verify_block_declared(frontmatter):
            logger.debug("No usable verify block in %s", path)
            return None
        meta = frontmatter.get("metadata")
        hermes = meta.get("hermes") if isinstance(meta, dict) else None
        verify_cfg = hermes.get("verify") if isinstance(hermes, dict) else None
        timeout = verify_cfg.get("timeout_seconds", _DEFAULT_TIMEOUT)
        try:
            timeout = min(int(timeout), _MAX_TIMEOUT)
        except (TypeError, ValueError):
            timeout = _DEFAULT_TIMEOUT
        return VerifySpec(
            skill_dir=skill_dir,
            run=str(verify_cfg["run"]),
            applicability_check=verify_cfg.get("applicability_check"),
            timeout_seconds=timeout,
        )
    except Exception as e:
        logger.debug("Failed to parse verify block for skill at %s: %s", path, e)
        return None


def _resolve_command(skill_dir: Path, relative_path: str) -> Optional[List[str]]:
    """Resolve a skill-relative script path to an executable command list.

    Refuses paths that escape the skill directory — a skill's verifier must
    live inside the skill it declares (a buggy or hostile author must not be
    able to point ``run`` at an arbitrary host script).
    """
    script = (skill_dir / relative_path).resolve()
    try:
        script.relative_to(skill_dir.resolve())
    except ValueError:
        logger.debug("Verify script escapes skill dir: %s", script)
        return None
    if not script.exists():
        return None
    interp = _INTERPRETERS.get(script.suffix)
    if interp is None:
        logger.debug(
            "Verify script suffix not allowed: %s (allowed: %s)",
            script.suffix or "<none>",
            ", ".join(sorted(_INTERPRETERS)),
        )
        return None
    return [*interp, str(script)]


def _run(cmd: List[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        stdin=subprocess.DEVNULL,
        creationflags=windows_hide_flags(),
    )


def _evaluate(result: subprocess.CompletedProcess) -> VerifyOutcome:
    """Interpret a verifier's output: structured JSON first, exit code second."""
    stdout = (result.stdout or "").strip()
    try:
        parsed = json.loads(stdout)
        if isinstance(parsed, dict) and "success" in parsed:
            return VerifyOutcome(
                success=bool(parsed["success"]),
                reason=str(parsed.get("reason", "")),
            )
    except (json.JSONDecodeError, TypeError):
        pass
    ok = result.returncode == 0
    tail = (result.stderr or stdout or "").strip()[-300:]
    return VerifyOutcome(success=ok, reason=tail or f"exit code {result.returncode}")


def run_verification(
    skill_name: str, skill_dir: Path, task_cwd: Path
) -> Optional[VerifyOutcome]:
    """Run skill_name's declared verifier against task_cwd.

    None means "don't record anything": not opted in, no verify block, not
    curation-eligible, the applicability probe said this turn isn't judgeable,
    or the check itself broke. SKIP is a third outcome alongside PASS/FAIL.
    """
    if not (is_curation_eligible(skill_name, skill_dir) and is_verify_enabled(skill_name)):
        return None
    spec = load_verify_spec(skill_dir)
    if spec is None:
        return None
    try:
        if spec.applicability_check:
            probe_cmd = _resolve_command(skill_dir, spec.applicability_check)
            if probe_cmd is None:
                return None
            probe_timeout = min(spec.timeout_seconds, 10)
            try:
                probe = _run(probe_cmd, cwd=task_cwd, timeout=probe_timeout)
            except subprocess.TimeoutExpired:
                # A slow probe is not a judgment: the turn was never judgeable,
                # so it must SKIP (None), never record a mechanical FAIL.
                logger.debug(
                    "applicability probe for %s timed out after %ss — skip",
                    skill_name,
                    probe_timeout,
                )
                return None
            if probe.returncode != 0:
                return None  # not applicable this turn — skip, don't judge

        cmd = _resolve_command(skill_dir, spec.run)
        if cmd is None:
            return None
        result = _run(cmd, cwd=task_cwd, timeout=spec.timeout_seconds)
        return _evaluate(result)
    except subprocess.TimeoutExpired:
        return VerifyOutcome(
            success=False, reason=f"verifier timed out after {spec.timeout_seconds}s"
        )
    except Exception as e:
        logger.debug(
            "skill_verify.run_verification(%s) failed: %s", skill_name, e, exc_info=True
        )
        return None
