"""Canonical-repository proposed→active skill lifecycle and scheduled review batches."""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any, Callable

from hermes_constants import get_hermes_home
from utils import atomic_write_text

RISKS = frozenset({"read-only", "write", "send", "credentials"})
NIGHTLY_SCHEDULE, DIGEST_SCHEDULE, MONTHLY_SCHEDULE = "0 1 * * *", "30 7 * * *", "0 8 1 * *"
RECONCILE_SCHEDULE = "*/15 * * * *"
LIFECYCLE_TIMEZONE = "America/Chicago"
_META = ".proposal.json"


def roots(home: Path | None = None) -> tuple[Path, Path]:
    base = Path(home) if home is not None else get_hermes_home()
    return base / "skills" / "active", base / "skills" / "proposed"


def canonical_repo_path(repo: Path | None = None) -> Path:
    """Resolve the source-of-truth repository; never silently use the installed profile."""
    raw = repo or os.environ.get("HERMES_SKILLS_REPO")
    if not raw:
        raise ValueError("canonical skill repo required (argument or HERMES_SKILLS_REPO)")
    path = Path(raw).expanduser().resolve()
    if not (path / ".git").exists():
        raise ValueError(f"canonical skill repo is not a git checkout: {path}")
    return path


def ensure_layout(home: Path | None = None) -> tuple[Path, Path]:
    active, proposed = roots(home)
    active.mkdir(parents=True, exist_ok=True); proposed.mkdir(parents=True, exist_ok=True)
    return active, proposed


def bootstrap_layout(home: Path | None = None) -> dict[str, Any]:
    """Atomically migrate legacy skill packages before enabling active-only discovery."""
    base = Path(home) if home is not None else get_hermes_home(); root = base / "skills"
    legacy = sorted({p.parent for p in root.glob("*/SKILL.md")}) if root.is_dir() else []
    active = root / "active"
    collisions = [p.name for p in legacy if (active / p.name).exists()]
    if collisions:
        raise FileExistsError("active-name-collision: " + ", ".join(collisions))
    active.mkdir(parents=True, exist_ok=True); (root / "proposed").mkdir(parents=True, exist_ok=True)
    moved = []
    for package in legacy:
        package.replace(active / package.name); moved.append(package.name)
    marker = root / ".lifecycle-enabled.json"
    atomic_write_text(marker, json.dumps({"version": 1, "migrated": moved}, sort_keys=True) + "\n", create_mode=0o600)
    return {"migrated": moved, "marker": str(marker)}


def proposal_metadata(skill_dir: Path) -> dict[str, Any]:
    try:
        value = json.loads((skill_dir / _META).read_text(encoding="utf-8")); return value if isinstance(value, dict) else {}
    except (OSError, ValueError): return {}


def create_proposal(name: str, content: str, *, session_id: str, risk: str, why: dict[str, Any],
                    home: Path | None = None, canonical_repo: Path | None = None, creator_identity: str | None = None,
                    creator_capabilities: set[str] | None = None) -> Path:
    if not session_id: raise ValueError("automatic skill creation requires a creating session_id")
    if risk not in RISKS: raise ValueError(f"risk must be one of: {', '.join(sorted(RISKS))}")
    required = ("need", "tools", "example_run", "eval")
    if not isinstance(why, dict) or any(not why.get(k) for k in required):
        raise ValueError("why requires non-empty need, tools, and example_run fields")
    if canonical_repo is not None:
        proposed = canonical_repo_path(canonical_repo) / "skills" / "proposed"
        proposed.mkdir(parents=True, exist_ok=True)
    else:
        _active, proposed = ensure_layout(home)
    target = proposed / name; target.mkdir(exist_ok=False)
    atomic_write_text(target / "SKILL.md", content, create_mode=0o644)
    tools = why["tools"] if isinstance(why["tools"], list) else [why["tools"]]
    atomic_write_text(target / "why.md", f"# Why {name}\n\n## Need\n{why['need']}\n\n## Tools\n" + "\n".join(f"- {x}" for x in tools) + f"\n\n## Example run\n{why['example_run']}\n\n## Risk\n{risk}\n", create_mode=0o644)
    atomic_write_text(target / "eval.json", json.dumps(why["eval"], indent=2, sort_keys=True) + "\n", create_mode=0o644)
    fixtures = why.get("fixtures") or {}
    for filename, fixture_content in fixtures.items():
        fixture = target / "fixtures" / str(filename)
        fixture.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(fixture, str(fixture_content), create_mode=0o644)
    meta = {"name": name, "creating_session_id": session_id, "creator_identity": creator_identity or session_id,
            "creator_capabilities": sorted(creator_capabilities or ()), "risk": risk,
            "created_at": datetime.now(UTC).isoformat(), "status": "proposed"}
    atomic_write_text(target / _META, json.dumps(meta, indent=2, sort_keys=True) + "\n", create_mode=0o600)
    return target


def proposal_for_session(name: str, session_id: str | None, home: Path | None = None,
                         canonical_repo: Path | None = None) -> Path | None:
    """Return a canonical or profile draft only to its exact creating session."""
    candidates: list[Path] = []
    repo = canonical_repo or os.environ.get("HERMES_SKILLS_REPO")
    if repo:
        candidates.append(canonical_repo_path(Path(repo)) / "skills" / "proposed" / name)
    candidates.append(roots(home)[1] / name)
    for candidate in candidates:
        meta = proposal_metadata(candidate)
        if candidate.is_dir() and session_id and meta.get("creating_session_id") == session_id:
            return candidate
    return None


_RESTRICTED_SESSIONS: dict[str, tuple[Path, ...]] = {}
_SAFE_TOOLS = frozenset({"skills_list", "skill_view", "read_file", "search_files", "web_search", "web_extract", "vision_analyze"})
_SECRET_NAME = re.compile(r"^(?:\.env(?:\..*)?|auth\.json|credentials?|keychain|secrets?|tokens?)$", re.IGNORECASE)
_SECRET_QUERY = re.compile(r"(?:\.env|auth(?:\\?\.)json|credentials?|keychain|secrets?|tokens?)", re.IGNORECASE)
def restrict_session(session_id: str | None, safe_root: Path | None = None) -> None:
    if session_id:
        safe_roots = _RESTRICTED_SESSIONS.get(session_id, ())
        if safe_root is not None:
            safe_roots = tuple(dict.fromkeys((*safe_roots, Path(safe_root).resolve())))
        _RESTRICTED_SESSIONS[session_id] = safe_roots


def _safe_local_access(tool_name: str, args: dict[str, Any], safe_roots: tuple[Path, ...]) -> bool:
    if tool_name == "vision_analyze":
        image_url = str(args.get("image_url", ""))
        if image_url.startswith(("http://", "https://", "data:")):
            return True
        raw_path = image_url
    elif tool_name in {"read_file", "search_files"}:
        raw_path = str(args.get("path", "."))
    else:
        return True
    path = Path(raw_path).expanduser().resolve()
    if any(_SECRET_NAME.fullmatch(part) for part in path.parts):
        return False
    if tool_name == "search_files" and _SECRET_QUERY.search(str(args.get("pattern", ""))):
        return False
    if not safe_roots:
        return False
    return any(path == root or path.is_relative_to(root) for root in safe_roots)


def tool_allowed(session_id: str | None, tool_name: str, args: dict[str, Any] | None = None) -> bool:
    if not session_id or session_id not in _RESTRICTED_SESSIONS:
        return True
    if tool_name not in _SAFE_TOOLS:
        return False
    return _safe_local_access(tool_name, args or {}, _RESTRICTED_SESSIONS[session_id])


@dataclass(frozen=True)
class ScanResult:
    passed: bool; findings: tuple[str, ...]; eval_passed: bool; evidence: dict[str, Any] | None = None

_PATTERNS = {
 "url-or-webhook": re.compile(r"https?://|webhook", re.IGNORECASE), "slash-command": re.compile(r"(?m)(?:^|\s)/(?:[a-z][\w-]{1,})\b"),
 "secret-path": re.compile(r"(?:\.env|auth\.json|credentials?|keychain|/secrets?\b)", re.IGNORECASE),
 "send-call": re.compile(r"\b(?:send|post_message|send_email|slack|imessage)\b", re.IGNORECASE),
 "write-call": re.compile(r"\b(?:write_file|patch|skill_manage|rm\s|unlink|rmtree)\b", re.IGNORECASE),
}
_B64 = re.compile(r"(?<![A-Za-z0-9+/])(?:[A-Za-z0-9+/]{4}){10,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?(?![A-Za-z0-9+/])")

def _encoded_payload(text: str) -> bool:
    for token in _B64.findall(text):
        try: decoded = base64.b64decode(token, validate=True)
        except (binascii.Error, ValueError): continue
        if decoded and (any(b < 9 or 13 < b < 32 for b in decoded) or re.search(rb"https?://|BEGIN [A-Z ]+KEY|token|password|webhook", decoded, re.IGNORECASE)):
            return True
    return False


def _run_eval(skill_dir: Path) -> tuple[bool, dict[str, Any]]:
    """Execute one declarative fixture-backed case with a read-only tool allowlist."""
    allowed = {"read_file"}
    try:
        case = json.loads((skill_dir / "eval.json").read_text(encoding="utf-8"))
        tool = case.get("tool")
        if tool not in allowed:
            return False, {"case": case, "passed": False, "error": "tool-not-allowed"}
        fixture_name = str(case.get("fixture") or "")
        fixture = (skill_dir / "fixtures" / fixture_name).resolve()
        fixture_root = (skill_dir / "fixtures").resolve()
        if not fixture.is_relative_to(fixture_root) or not fixture.is_file():
            return False, {"case": case, "passed": False, "error": "invalid-fixture"}
        value = fixture.read_text(encoding="utf-8")
        expected = str(case.get("expect_contains") or "")
        passed = bool(expected) and expected in value
        return passed, {"case": case, "passed": passed, "fixture_sha256": hashlib.sha256(value.encode()).hexdigest()}
    except Exception as exc:
        return False, {"case": "fixture-read", "passed": False, "error": type(exc).__name__}


def deterministic_scan(skill_dir: Path) -> ScanResult:
    findings: set[str] = set()
    for path in sorted(p for p in skill_dir.rglob("*") if p.is_file() and p.name != _META):
        try: text = path.read_text(encoding="utf-8")
        except UnicodeError: findings.add("non-utf8"); continue
        if any(unicodedata.category(ch) in {"Cf", "Cs"} for ch in text): findings.add("hidden-unicode")
        for label, pattern in _PATTERNS.items():
            if pattern.search(text): findings.add(label)
        if _encoded_payload(text): findings.add("encoded-payload")
    eval_passed, evidence = _run_eval(skill_dir)
    return ScanResult(not findings and eval_passed, tuple(sorted(findings)), eval_passed, evidence)


def _approval_conflict(meta: dict[str, Any], reviewer: dict[str, Any] | None) -> bool:
    if not reviewer: return False
    if reviewer.get("identity") == meta.get("creator_identity"): return True
    requested = set(meta.get("creator_capabilities") or ()) | ({meta.get("risk")} - {"read-only"})
    return bool(requested & set(reviewer.get("capabilities") or ()))


def review_proposals(home: Path | None = None, *, booth_approved: set[str] | None = None,
                     reviewer_session_id: str | None = None, reviewer: dict[str, Any] | None = None,
                     apply: bool = True) -> dict[str, Any]:
    active, proposed = ensure_layout(home); approved = booth_approved or set()
    if reviewer_session_id and reviewer is None: reviewer = {"identity": reviewer_session_id, "capabilities": []}
    outcome = {"merged": [], "waiting_approval": [], "rejected": []}
    for skill_dir in sorted(p for p in proposed.iterdir() if p.is_dir()):
        meta = proposal_metadata(skill_dir); scan = deterministic_scan(skill_dir); name, risk = skill_dir.name, meta.get("risk")
        evidence = {"scan": list(scan.findings), "eval": scan.evidence}
        if not scan.passed:
            outcome["rejected"].append({"name": name, "reasons": list(scan.findings) or ["eval-failed"], "decision": "rewrite-or-delete", "evidence": evidence}); continue
        if risk != "read-only" and (name not in approved or _approval_conflict(meta, reviewer)):
            reason = "reviewer access conflict" if _approval_conflict(meta, reviewer) else "Booth approval required for access widening"
            outcome["waiting_approval"].append({"name": name, "risk": risk, "reason": reason, "evidence": evidence}); continue
        destination = active / name
        if destination.exists(): outcome["rejected"].append({"name": name, "reasons": ["active-name-collision"], "decision": "rewrite", "evidence": evidence}); continue
        if apply:
            meta.update(status="active", promoted_at=datetime.now(UTC).isoformat())
            atomic_write_text(skill_dir / _META, json.dumps(meta, indent=2, sort_keys=True) + "\n", preserve_mode=True); skill_dir.replace(destination)
        outcome["merged"].append({"name": name, "risk": risk, "evidence": evidence})
    return outcome


def write_batch_report(outcome: dict[str, Any], home: Path | None = None) -> Path:
    base = Path(home) if home is not None else get_hermes_home(); d = base / "skills" / ".lifecycle"; d.mkdir(parents=True, exist_ok=True)
    path = d / "latest.json"; atomic_write_text(path, json.dumps(outcome, indent=2, sort_keys=True) + "\n", create_mode=0o600); return path

def morning_digest(home: Path | None = None) -> str:
    base = Path(home) if home is not None else get_hermes_home()
    try: data = json.loads((base / "skills" / ".lifecycle" / "latest.json").read_text())
    except (OSError, ValueError): data = {"merged": [], "waiting_approval": [], "rejected": []}
    lines = ["Skill lifecycle digest"]
    for key, title in (("merged","Merged"),("waiting_approval","Waiting approval"),("rejected","Rejected")):
        rows=data.get(key) or []; lines.append(f"{title}: {len(rows)}")
        for row in rows: lines.append(f"- {row.get('name')}: {row.get('reason') or ', '.join(row.get('reasons', []))}".rstrip(": "))
    return "\n".join(lines)

def archive_review_candidates(home: Path | None = None, *, now: datetime | None = None, idle_days: int = 60) -> list[str]:
    base=Path(home) if home is not None else get_hermes_home(); active,_=roots(base)
    try: usage=json.loads((base/"skills"/".usage.json").read_text())
    except (OSError,ValueError): usage={}
    cutoff=(now or datetime.now(UTC)).timestamp()-idle_days*86400; flagged=[]
    for skill in sorted(p.name for p in active.iterdir() if p.is_dir()) if active.is_dir() else []:
        stamp=(usage.get(skill) or {}).get("last_used_at") or (usage.get(skill) or {}).get("last_activity_at")
        try: ts=datetime.fromisoformat(str(stamp)).timestamp()
        except (TypeError,ValueError): ts=0
        if ts<cutoff: flagged.append(skill)
    return flagged

def freeze_inventory(home: Path | None = None) -> dict[str, Any]:
    root=(Path(home) if home is not None else get_hermes_home())/"skills"; rows=[]
    for md in sorted(root.rglob("SKILL.md")) if root.is_dir() else []:
        package=md.parent; files=[]
        for p in sorted(x for x in package.rglob("*") if x.is_file()): files.append({"path":str(p.relative_to(root)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
        rows.append({"path":str(md.relative_to(root)),"sha256":hashlib.sha256(md.read_bytes()).hexdigest(),"files":files})
    return {"skills_root":str(root),"count":len(rows),"file_count":sum(len(x["files"]) for x in rows),"skills":rows}


def lifecycle_job_specs(deliver: str = "origin") -> list[dict[str, Any]]:
    py=sys.executable; module="tools.skill_lifecycle"
    return [
      {"name":"skill-lifecycle-nightly","schedule":NIGHTLY_SCHEDULE,"timezone":LIFECYCLE_TIMEZONE,"script":f'{py} -m {module} nightly',"no_agent":True,"deliver":"local"},
      {"name":"skill-lifecycle-reconcile","schedule":RECONCILE_SCHEDULE,"timezone":LIFECYCLE_TIMEZONE,"script":f'{py} -m {module} reconcile',"no_agent":True,"deliver":"local"},
      {"name":"skill-lifecycle-digest","schedule":DIGEST_SCHEDULE,"timezone":LIFECYCLE_TIMEZONE,"script":f'{py} -m {module} digest',"no_agent":True,"deliver":deliver},
      {"name":"skill-lifecycle-archive-review","schedule":MONTHLY_SCHEDULE,"timezone":LIFECYCLE_TIMEZONE,"script":f'{py} -m {module} archive',"no_agent":True,"deliver":deliver},]

def install_lifecycle_jobs(home: Path | None = None, *, deliver: str = "origin") -> list[dict[str, Any]]:
    from cron.jobs import create_job, load_jobs, use_cron_store
    base=Path(home) if home is not None else get_hermes_home(); created=[]
    with use_cron_store(base):
        existing={j.get("name") for j in load_jobs()}
        for spec in lifecycle_job_specs(deliver):
            if spec["name"] not in existing: created.append(create_job(prompt="", **spec))
    return created


def bootstrap_lifecycle(home: Path | None = None, *, canonical_repo: Path | None = None,
                        migrate_profile: bool = False, deliver: str = "origin") -> dict[str, Any]:
    """Supported install/upgrade hook: configure source identity and idempotently arm jobs.

    Profile migration is opt-in so installing this feature never mutates an existing live
    ``~/.hermes/skills`` tree. Canonical active content is synced by an explicit deployment step.
    """
    base = Path(home) if home is not None else get_hermes_home()
    repo = Path(canonical_repo).resolve() if canonical_repo is not None else None
    marker = base / "skills" / ".canonical-source.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    if repo is not None:
        atomic_write_text(marker, json.dumps({"repo": str(repo), "active": "skills/active"}, sort_keys=True) + "\n", create_mode=0o600)
    migration = bootstrap_layout(base) if migrate_profile else None
    created = install_lifecycle_jobs(base, deliver=deliver)
    return {"jobs_created": len(created), "canonical_repo": str(repo) if repo else None, "migration": migration}

def _gh(args:list[str], cwd:Path, *, check:bool=True)->subprocess.CompletedProcess[str]:
    return subprocess.run(["gh",*args],cwd=cwd,text=True,capture_output=True,check=check)
def parse_booth_approvals(text:str)->set[str]:
    return set(re.findall(r"(?im)^\s*APPROVE-SKILL:\s*([a-z0-9._-]+)\s*$",text or ""))


def trusted_booth_approvals(repo: Path, pr: int, booth_logins: set[str]) -> set[str]:
    """Read approval text only from explicitly trusted GitHub identities."""
    return set(trusted_booth_approval_records(repo, pr, booth_logins))


def trusted_booth_approval_records(repo: Path, pr: int, booth_logins: set[str]) -> dict[str, set[str]]:
    """Map approved skills to trusted approving logins for conflict checks."""
    trusted = {login.casefold() for login in booth_logins if login}
    approved: dict[str, set[str]] = {}
    endpoints = ((f"repos/{{owner}}/{{repo}}/issues/{pr}/comments", False),
                 (f"repos/{{owner}}/{{repo}}/pulls/{pr}/reviews", True))
    for endpoint, is_review in endpoints:
        data = json.loads(_gh(["api", endpoint], repo).stdout)
        for item in data:
            login = str((item.get("user") or {}).get("login") or "").casefold()
            if login not in trusted:
                continue
            if is_review and item.get("state") != "APPROVED":
                continue
            for name in parse_booth_approvals(item.get("body") or ""):
                approved.setdefault(name, set()).add(login)
    return approved


def checks_pass_exact_head(repo:Path, pr:int, expected_head:str)->bool:
    view=json.loads(_gh(["pr","view",str(pr),"--json","headRefOid,statusCheckRollup"],repo).stdout)
    checks=view.get("statusCheckRollup") or []
    return view.get("headRefOid")==expected_head and bool(checks) and all((c.get("conclusion") or c.get("state")) in {"SUCCESS","NEUTRAL","SKIPPED"} for c in checks)


def apply_batch_transitions(repo: Path, outcome: dict[str, Any]) -> Path:
    """Materialize all deterministic proposal decisions in one canonical-repo branch."""
    proposed, active = repo / "skills" / "proposed", repo / "skills" / "active"
    active.mkdir(parents=True, exist_ok=True)
    for row in outcome.get("merged", []):
        source, destination = proposed / row["name"], active / row["name"]
        if not source.is_dir() or destination.exists():
            raise FileExistsError(f"invalid promotion: {row['name']}")
        meta = proposal_metadata(source)
        meta.update(status="active", promoted_at=datetime.now(UTC).isoformat())
        atomic_write_text(source / _META, json.dumps(meta, indent=2, sort_keys=True) + "\n", preserve_mode=True)
        source.replace(destination)
    for row in outcome.get("rejected", []):
        source = proposed / row["name"]
        if row.get("decision") == "delete" and source.exists():
            shutil.rmtree(source)
        elif source.is_dir():
            atomic_write_text(source / ".rejection.json", json.dumps(row, indent=2, sort_keys=True) + "\n", create_mode=0o644)
    evidence = repo / ".hermes" / "skill-lifecycle" / "latest.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(evidence, json.dumps(outcome, indent=2, sort_keys=True) + "\n", create_mode=0o644)
    return evidence


def reconcile_batch_pr(repo: Path, pr: int, expected_head: str, outcome: dict[str, Any], *,
                       booth_logins: set[str], implementation_pr: int = 148971) -> dict[str, Any]:
    """Merge an exact checked batch head, never the lifecycle implementation PR."""
    if pr == implementation_pr:
        return {"merged": False, "reason": "implementation-pr-excluded"}
    if not checks_pass_exact_head(repo, pr, expected_head):
        return {"merged": False, "reason": "exact-head-checks-not-passing"}
    promoted = outcome.get("merged") or []
    widening = {row["name"] for row in promoted if row.get("risk") != "read-only"}
    if widening:
        approvals = trusted_booth_approval_records(repo, pr, booth_logins)
        missing = widening - set(approvals)
        if missing:
            return {"merged": False, "reason": "trusted-booth-approval-required", "missing": sorted(missing)}
        for name in widening:
            meta = proposal_metadata(repo / "skills" / "active" / name)
            creator = str(meta.get("creator_identity") or "").casefold()
            if creator and creator in approvals[name]:
                return {"merged": False, "reason": "self-approval-conflict", "skill": name}
    _gh(["pr", "merge", str(pr), "--merge", "--match-head-commit", expected_head], repo)
    return {"merged": True, "pr": pr, "head": expected_head}


def reconcile_open_batches(repo: Path, *, booth_logins: set[str], implementation_pr: int = 148971) -> list[dict[str, Any]]:
    """Reconcile every open lifecycle batch using evidence fetched from that PR's exact head."""
    repo = canonical_repo_path(repo)
    rows = json.loads(_gh(["pr", "list", "--state", "open", "--search", "in:title lifecycle batch",
                           "--json", "number,headRefOid"], repo).stdout)
    results = []
    for row in rows:
        number, head = int(row["number"]), str(row["headRefOid"])
        if number == implementation_pr:
            continue
        payload = json.loads(_gh(["api", f"repos/{{owner}}/{{repo}}/contents/.hermes/skill-lifecycle/latest.json?ref={head}"], repo).stdout)
        evidence = json.loads(base64.b64decode(payload["content"]).decode("utf-8"))
        results.append(reconcile_batch_pr(repo, number, head, evidence, booth_logins=booth_logins,
                                          implementation_pr=implementation_pr))
    return results

def nightly_batch(repo: Path, home: Path | None = None, *, implementation_pr: int = 148971) -> dict[str,Any]:
    """Create one canonical-repo branch/PR containing source transitions and evidence."""
    repo = canonical_repo_path(repo)
    _gh(["auth","status"],repo); outcome=review_proposals(repo,apply=False)
    if not any(outcome.values()): write_batch_report(outcome,home); return outcome
    stamp=datetime.now(UTC).strftime("%Y%m%d"); branch=f"skill-lifecycle/{stamp}"
    _gh(["checkout","-B",branch],repo)
    evidence = apply_batch_transitions(repo, outcome)
    _gh(["add","skills/proposed","skills/active",str(evidence.relative_to(repo))],repo); _gh(["commit","-m",f"chore(skills): lifecycle batch {stamp}"],repo); _gh(["push","-u","fork",branch,"--force-with-lease"],repo)
    title=f"chore(skills): lifecycle batch {stamp}"
    cp=_gh(["pr","create","--title",title,"--body",f"Canonical lifecycle source transitions and evidence. Access widening requires verified Booth approval.\n\nImplementation PR #{implementation_pr} is excluded.","--base","main","--head",branch],repo)
    outcome["pr_url"]=cp.stdout.strip(); write_batch_report(outcome,home); return outcome

def main(argv:list[str]|None=None)->int:
    action=(argv or sys.argv[1:] or [""])[0]
    if action=="install": print(json.dumps(bootstrap_lifecycle(canonical_repo=Path.cwd()),indent=2)); return 0
    if action=="digest": print(morning_digest()); return 0
    if action=="archive": print(json.dumps({"archive_review":archive_review_candidates()})); return 0
    if action=="nightly": print(json.dumps(nightly_batch(canonical_repo_path()),indent=2)); return 0
    if action=="reconcile":
        logins={x.strip() for x in os.environ.get("HERMES_BOOTH_GITHUB_LOGINS","").split(",") if x.strip()}
        if not logins: raise SystemExit("HERMES_BOOTH_GITHUB_LOGINS is required")
        print(json.dumps(reconcile_open_batches(canonical_repo_path(),booth_logins=logins),indent=2)); return 0
    raise SystemExit("usage: python -m tools.skill_lifecycle {install|nightly|reconcile|digest|archive}")
if __name__=="__main__": raise SystemExit(main())
