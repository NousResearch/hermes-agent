"""Plain-language explanations for command approval prompts.

The approval guard's regex description explains *why the guard fired*, not what
running the command means for a person.  This module adds a conservative,
deterministic explanation without asking another model to judge the command.
It deliberately says "visible"/"detected" instead of promising safety: shell
commands and inline programs can compute behavior dynamically.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import shlex


_INLINE_SCRIPT_RE = re.compile(r"\b(?:python[23]?|node|ruby|perl)\s+-[^\s]*[ec](?:\s|$)", re.I)
_DELETE_RE = re.compile(
    r"(?:\brm\s+[^\n]*(?:-[^\s]*r[^\s]*f|-[^\s]*f[^\s]*r)|"
    r"\b(?:unlink|rmdir)\b|shutil\.rmtree\s*\(|\.unlink\s*\()",
    re.I,
)
_WRITE_RE = re.compile(
    r"(?:\.write(?:_text|_bytes)?\s*\(|\bwritelines?\s*\(|"
    r"open\s*\([^\n]{0,240},\s*['\"](?:w|a|x|r\+)|"
    r"\b(?:mv|cp|install)\s+|(?:^|\s)>{1,2}\s*\S)",
    re.I,
)
_READ_RE = re.compile(
    r"(?:\.read(?:_text|_bytes)?\s*\(|\bopen\s*\(|\bcat\s+|"
    r"zipfile|namelist\s*\(|json\.loads?\s*\()",
    re.I,
)
_NETWORK_RE = re.compile(
    r"(?:\b(?:curl|wget|ssh|scp|rsync)\b|https?://|"
    r"\b(?:requests|httpx|urllib|socket)\s*\.|urlopen\s*\()",
    re.I,
)
_OUTPUT_RE = re.compile(r"(?:\bprint\s*\(|\bcat\s+|\b(?:head|tail)\s+)", re.I)
_PRIVILEGED_RE = re.compile(r"(?:^|\s)sudo(?:\s|$)|\bchmod\s+777\b", re.I)
_DEPLOY_RE = re.compile(r"\b(?:deploy|publish|release|kubectl\s+apply|terraform\s+apply)\b", re.I)


@dataclass(frozen=True)
class ApprovalSummary:
    purpose: str
    effects: tuple[str, ...]
    risk_level: str
    risk_reason: str
    session_scope: str
    recommendation: str
    once_only: bool = False

    def to_markdown(self) -> str:
        """Render a compact explanation suitable for chat surfaces."""
        icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(
            self.risk_level, "🟡"
        )
        effects = "\n".join(f"- {effect}" for effect in self.effects)
        return (
            f"**What Hermes wants to do**\n{self.purpose}\n\n"
            f"**Access and effects**\n{effects}\n\n"
            f"**Risk: {icon} {self.risk_level.upper()}** — {self.risk_reason}\n\n"
            f"**Approval scope**\n{self.session_scope}\n\n"
            f"**Recommendation**\n{self.recommendation}"
        )


def _shell_paths(command: str) -> list[str]:
    """Extract path-like shell arguments without treating code literals as paths."""
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()

    paths: list[str] = []
    for token in tokens:
        candidate = token.strip()
        if not (candidate.startswith("/") or candidate.startswith("~/")):
            continue
        # Inline program bodies are one large token and may contain newlines.
        if "\n" in candidate or len(candidate) > 300:
            continue
        candidate = candidate.rstrip(",;)")
        if candidate not in paths:
            paths.append(candidate)
    return paths[:3]


def _session_scope(description: str, inline_script: bool) -> str:
    description_lower = (description or "").lower()
    if inline_script or "script execution via -e/-c" in description_lower:
        return (
            "“Allow for Session” would also pre-approve later inline scripts "
            "(Python, Node, Ruby, or Perl) that match this rule—not just this command."
        )
    if "recursive delete" in description_lower:
        return (
            "“Allow for Session” would also pre-approve later recursive-delete "
            "commands that match this rule—not just this path."
        )
    return (
        "“Allow for Session” pre-approves later commands matching this safety "
        "rule; it is broader than approving this exact command once."
    )


def summarize_command_approval(command: str, description: str = "dangerous command") -> ApprovalSummary:
    """Return a cautious, non-technical explanation of a guarded command."""
    command = str(command or "")
    lower = command.lower()
    inline_script = bool(_INLINE_SCRIPT_RE.search(command))
    deletes = bool(_DELETE_RE.search(command))
    writes = deletes or bool(_WRITE_RE.search(command))
    reads = bool(_READ_RE.search(command))
    network = bool(_NETWORK_RE.search(command))
    privileged = bool(_PRIVILEGED_RE.search(command))
    deploys = bool(_DEPLOY_RE.search(command))
    emits_output = bool(_OUTPUT_RE.search(command))
    zip_inspection = "zipfile" in lower and ("namelist(" in lower or ".read(" in lower)
    paths = _shell_paths(command)

    if zip_inspection:
        purpose = "Inspect a ZIP archive and print selected file names and contents."
    elif deletes:
        purpose = "Delete files or directories on the execution environment."
    elif deploys:
        purpose = "Run a deployment, publishing, or release operation."
    elif network:
        purpose = "Contact another computer or network service."
    elif inline_script:
        purpose = "Run an inline program with the same system access as Hermes."
    else:
        purpose = "Run a command on Hermes’ configured execution environment."

    effects: list[str] = []
    shown_paths = ", ".join(f"`{path}`" for path in paths)
    if deletes:
        target = f" ({shown_paths})" if shown_paths else ""
        effects.append(f"Can permanently delete data{target}; this may not be reversible.")
    elif writes:
        target = f" at {shown_paths}" if shown_paths else ""
        effects.append(f"Can create or change files{target}.")
    elif reads:
        target = shown_paths or "files referenced by the program"
        effects.append(f"Reads data from {target}.")
        effects.append("The visible command does not modify or delete files.")
    else:
        effects.append(
            "The concrete data access and side effects could not be determined from the visible command."
        )

    if reads and emits_output:
        effects.append(
            "Content written to command output can enter Hermes’ model context and session logs."
        )
    if network:
        effects.append("The visible command can communicate over the network.")
    elif inline_script:
        effects.append("No network operation is visible, but inline code has broad capabilities.")
    if privileged:
        effects.append("It requests elevated system privileges.")

    unknown = not any((reads, writes, network, deploys, privileged))
    high = deletes or privileged or deploys
    if high:
        risk_level = "high"
        risk_reason = "It can cause external, privileged, or hard-to-reverse changes."
    elif reads and emits_output:
        risk_level = "medium"
        risk_reason = "No destructive change is visible, but file contents may be exposed to the AI context."
    elif network or writes or inline_script or unknown:
        risk_level = "medium"
        risk_reason = "The operation has broad or incompletely determined effects."
    else:
        risk_level = "low"
        risk_reason = "Only limited read-only effects are visible."

    once_only = inline_script or high or unknown
    recommendation = (
        "Use “Allow Once” only if this purpose and data access are expected. "
        "Otherwise deny it and ask Hermes for a safer or more limited approach."
        if once_only or risk_level != "low"
        else "Allow once if the described read-only access is expected."
    )

    return ApprovalSummary(
        purpose=purpose,
        effects=tuple(effects),
        risk_level=risk_level,
        risk_reason=risk_reason,
        session_scope=_session_scope(description, inline_script),
        recommendation=recommendation,
        once_only=once_only,
    )
