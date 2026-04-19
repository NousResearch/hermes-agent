"""
Hermes ↔ apmzoom skill bridge.

Auto-discovers all skills under ~/.hermes/skills/apmzoom/{read,write}/<name>/
and registers each as a hermes tool (callable from the LLM via OpenAI
function-calling).  Handler reads the skill's metadata to know:

  - api_method (GET / POST / PUT / DELETE)
  - api_url or base_url + endpoint
  - auth_type:
        none      → no auth header
        bearer    → Authorization: Bearer <merchant access_token>
        apm_sign  → AWS-gateway style:
                    v=7.0.1, p=1, t=<unix>, lang=zh-cn,
                    sign=MD5(<auth_sign_salt>).toUpperCase(),
                    authcode=HH <merchant access_token>

Per-request merchant identity is threaded via a ContextVar set by
api_server.py before the agent loop runs.  The handler reads
~/.hermes/merchants/<merchant_id>/credentials.json on each call so that
token refreshes (via merchant-onboard.py) are picked up live with no
hermes restart.

This makes the LLM a true APM agent: it sees `apm_*` tools in its
function-calling schema, picks the right one, and hermes executes the
HTTP call locally — preserving merchant isolation, computing signatures
worker-side-style, and returning the raw upstream JSON for the LLM to
summarize.
"""

import hashlib
import json
import logging
import os
import re
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as _urlerror
from urllib import parse as _urlparse
from urllib import request as _urlrequest

from tools.registry import registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-request merchant context
# ---------------------------------------------------------------------------
#
# Set by gateway/platforms/api_server.py at the start of each chat completion
# request when the X-Hermes-Merchant-Id header is present.  Tool handlers
# read it to look up the right merchant's APM credentials.
# Default empty string = "no merchant context" (handler will still try but
# auth-required skills will return an error the LLM can recover from).
_merchant_id_ctx: ContextVar[str] = ContextVar("apmzoom_merchant_id", default="")

# Per-request active workflow (X-Hermes-Active-Skill).  Used by api_server to
# prune the apm_* tool list to just the skills the workflow declares in its
# frontmatter `metadata.apmzoom.skills_used`, keeping the prompt lean.
_active_skill_ctx: ContextVar[str] = ContextVar("apmzoom_active_skill", default="")


def set_merchant_id(merchant_id: str) -> None:
    """Called from api_server when a merchant request lands.

    Safe to call with empty string to clear (no-op merchant_mode).
    """
    _merchant_id_ctx.set(merchant_id or "")


def current_merchant_id() -> str:
    return _merchant_id_ctx.get()


def set_active_skill(active_skill: str) -> None:
    """Propagate the X-Hermes-Active-Skill header for downstream filters."""
    _active_skill_ctx.set(active_skill or "")


def current_active_skill() -> str:
    return _active_skill_ctx.get()


def resolve_workflow_skills_used(active_skill: str) -> Optional[set]:
    """Parse `metadata.apmzoom.skills_used` out of a workflow's SKILL.md.

    active_skill can be a full path fragment like `apmzoom-workflows/<name>` or
    bare skill folder name; we look under ~/.hermes/skills/<active_skill>/
    SKILL.md and also ~/.hermes/skills/apmzoom-workflows/<name>/SKILL.md.

    Returns:
      - A set of skill names (without the `apm_` prefix) if the workflow
        declares `skills_used`.
      - None if the SKILL.md is missing, not a workflow, or doesn't specify
        `skills_used` (caller should fall back to the global allowlist).
    """
    if not active_skill:
        return None
    home = _hermes_home()
    candidates = [
        home / "skills" / active_skill / "SKILL.md",
        home / "skills" / "apmzoom-workflows" / active_skill / "SKILL.md",
    ]
    skill_md: Optional[Path] = None
    for c in candidates:
        if c.exists() and c.is_file():
            skill_md = c
            break
    if skill_md is None:
        return None
    try:
        content = skill_md.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    if not content.startswith("---"):
        return None
    parts = content.split("---", 2)
    if len(parts) < 3:
        return None
    fm = parts[1]
    # Locate the `skills_used:` key (may be at any indentation under metadata)
    m = re.search(r'(?m)^\s*skills_used:\s*$((?:\n\s+-\s*.*)+)', fm)
    if not m:
        return None
    block = m.group(1)
    # Each item: `      - "name"` or `      - name`
    items = re.findall(r'-\s*"?([A-Za-z0-9_.-]+)"?\s*$', block, re.MULTILINE)
    return set(items) if items else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hermes_home() -> Path:
    """Resolve hermes home, respecting profile overrides."""
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path.home() / ".hermes"


def _md5_upper(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest().upper()


def _read_merchant_creds(merchant_id: str) -> Dict[str, Any]:
    """Read fresh credentials.json each call (token refreshes pick up live)."""
    if not merchant_id:
        return {}
    p = _hermes_home() / "merchants" / merchant_id / "credentials.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("[apmzoom] failed to read %s: %s", p, e)
        return {}


def _resolve_apm_token(creds: Dict[str, Any]) -> str:
    """Pick the right token for AWS-gateway calls.

    Worker JWTs (`access_token` at top level when source=worker.apm_login)
    only auth against worker.  AWS-gateway endpoints need the raw IDS
    access_token, which apm_login returns nested under `apm_token`.

    Fallback chain: apm_token.access_token > access_token.
    """
    apm = creds.get("apm_token") or {}
    return apm.get("access_token") or creds.get("access_token") or ""


# ---------------------------------------------------------------------------
# SKILL.md frontmatter parsing
# ---------------------------------------------------------------------------
#
# We don't depend on pyyaml — minimal regex parser handles the keys we need.

_FM_KEY_RE = re.compile(r'^\s{0,4}([a-z_]+):\s*"?([^"\n]*)"?\s*$', re.MULTILINE)


def _parse_skill_md(skill_md_path: Path) -> Dict[str, Any]:
    """Extract the apmzoom-relevant fields from the SKILL.md frontmatter.

    Returns a flat dict with keys: api_method, api_url, endpoint, base_url,
    auth_type, auth_sign_salt, permission_level.  Missing keys default to "".
    """
    out = {
        "api_method": "POST", "api_url": "", "endpoint": "", "base_url": "",
        "auth_type": "none", "auth_sign_salt": "", "permission_level": "read",
    }
    try:
        content = skill_md_path.read_text(encoding="utf-8")
    except Exception:
        return out
    if not content.startswith("---"):
        return out
    parts = content.split("---", 2)
    if len(parts) < 3:
        return out
    fm = parts[1]
    for m in _FM_KEY_RE.finditer(fm):
        k, v = m.group(1), m.group(2).strip()
        if k in out:
            out[k] = v
    return out


# Quick Reference table row: `| Method | `POST` |` → capture `POST`
_QR_ROW_RE = re.compile(
    r"\|\s*(Method|Endpoint|Base URL|Name|Display Name|Category|Permission)\s*"
    r"\|\s*`?([^`|\n]+?)`?\s*\|",
    re.IGNORECASE,
)
# MD5 salt literal inside MD5(...).  Salt is conventionally a standalone string
# literal; login endpoints have it at the END of a concatenation
# (e.g. MD5(account + login_pwd + 'ggfgffgfggf')), read endpoints have it as
# the sole argument (e.g. MD5('jsm6y$dh3hjsb')).  We grab the last single- or
# double-quoted string inside each MD5(...) call.
_MD5_CALL_RE = re.compile(r"MD5\(([^)]*)\)", re.IGNORECASE)
_QUOTED_LIT_RE = re.compile(r"""['"]([^'"]+)['"]""")


def _extract_salt(content: str) -> str:
    for call in _MD5_CALL_RE.finditer(content):
        lits = _QUOTED_LIT_RE.findall(call.group(1))
        if lits:
            return lits[-1]
    return ""


def _parse_openclaw_endpoint_md(path: Path) -> Dict[str, Any]:
    """Parse an openclaw-imports endpoint file (e.g. apmzoom-gds/gds_m_storegoodslist.md).

    These docs store the HTTP spec in a Quick Reference markdown table and
    the MD5 signing salt inline in the prose.  Shape mirrors
    `_parse_skill_md` so downstream code is unchanged.
    """
    out = {
        "api_method": "POST", "api_url": "", "endpoint": "", "base_url": "",
        "auth_type": "none", "auth_sign_salt": "", "permission_level": "read",
        "display_name": "", "category": "",
    }
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return out

    # Frontmatter: permission_level
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            for m in _FM_KEY_RE.finditer(parts[1]):
                k, v = m.group(1), m.group(2).strip()
                if k == "permission_level":
                    out["permission_level"] = v

    # Quick Reference table
    for m in _QR_ROW_RE.finditer(content):
        key = m.group(1).strip().lower()
        val = m.group(2).strip()
        if key == "method":
            out["api_method"] = val.upper()
        elif key == "endpoint":
            out["endpoint"] = val
        elif key == "base url":
            out["base_url"] = val
        elif key == "display name":
            out["display_name"] = val
        elif key == "category":
            out["category"] = val
        elif key == "permission":
            out["permission_level"] = val.lower()

    out["auth_sign_salt"] = _extract_salt(content)

    # Infer auth_type:
    #   - AWS execute-api → apm_sign (with authcode header if token available;
    #     login endpoints naturally have no token yet and skip authcode)
    #   - worker.apmzoom.ai + explicit "no auth" hint → none
    #   - worker.apmzoom.ai otherwise → bearer
    base = out["base_url"].lower()
    no_auth_hints = ("인증 불필요", "인증이 필요하지 않", "authcode 헤더가 필요하지 않",
                     "공개 인터페이스", "no authentication")
    has_no_auth_hint = any(h in content for h in no_auth_hints)
    if "execute-api" in base or "amazonaws.com" in base:
        out["auth_type"] = "apm_sign"
    elif "worker.apmzoom.ai" in base:
        out["auth_type"] = "none" if has_no_auth_hint else "bearer"
    # else: default "none"
    return out


# ---------------------------------------------------------------------------
# HTTP execution
# ---------------------------------------------------------------------------

def _execute_skill(skill_name: str, params: Optional[Dict[str, Any]] = None,
                   merchant_id_override: str = "") -> str:
    """Call one apmzoom skill via HTTP.  Returns JSON string for LLM consumption."""
    params = params or {}
    home = _hermes_home()

    # Locate skill. Two layouts supported:
    #   1) Legacy: ~/.hermes/skills/apmzoom/{read,write}/<name>/SKILL.md
    #   2) openclaw-imports: ~/.hermes/skills/openclaw-imports/apmzoom-*/<name>.md
    skill_dir: Optional[Path] = None
    openclaw_md: Optional[Path] = None
    for bucket in ("read", "write"):
        candidate = home / "skills" / "apmzoom" / bucket / skill_name
        if (candidate / "SKILL.md").exists():
            skill_dir = candidate
            break
    if skill_dir is None:
        oc_base = home / "skills" / "openclaw-imports"
        if oc_base.is_dir():
            for group_dir in oc_base.glob("apmzoom-*"):
                cand = group_dir / f"{skill_name}.md"
                if cand.is_file():
                    openclaw_md = cand
                    break
    if skill_dir is None and openclaw_md is None:
        return json.dumps({
            "error": "unknown_skill",
            "message": f"No skill named '{skill_name}' under apmzoom/ or openclaw-imports/apmzoom-*/.",
        }, ensure_ascii=False)

    if openclaw_md is not None:
        meta = _parse_openclaw_endpoint_md(openclaw_md)
    else:
        meta = _parse_skill_md(skill_dir / "SKILL.md")
    api_method = (meta["api_method"] or "POST").upper()
    final_url = meta["api_url"] or (meta["base_url"] + meta["endpoint"])
    if not final_url:
        return json.dumps({
            "error": "skill_misconfigured",
            "message": f"Skill '{skill_name}' has no api_url or base_url+endpoint.",
        }, ensure_ascii=False)

    # Substitute {{key}} placeholders in URL with params (e.g. /skills/{{id}})
    def _sub(match):
        key = match.group(1)
        return str(params.pop(key, match.group(0)))
    final_url = re.sub(r"\{\{(\w+)\}\}", _sub, final_url)

    # Auth + headers
    merchant_id = merchant_id_override or current_merchant_id()
    creds = _read_merchant_creds(merchant_id)
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "hermes-apmzoom-bridge/1.0",
    }
    auth_type = (meta["auth_type"] or "none").lower()
    if auth_type == "apm_sign":
        if not meta["auth_sign_salt"]:
            return json.dumps({"error": "missing_sign_salt", "skill": skill_name},
                              ensure_ascii=False)
        token = _resolve_apm_token(creds)
        headers.update({
            "v": "7.0.1", "p": "1", "t": str(int(time.time())), "lang": "zh-cn",
            "sign": _md5_upper(meta["auth_sign_salt"]),
        })
        if token:
            headers["authcode"] = f"HH {token}"
        else:
            logger.warning("[apmzoom] %s requires apm_sign but no merchant token (mid=%r)",
                           skill_name, merchant_id)
    elif auth_type == "bearer":
        # Worker-style JWT (top-level access_token) for non-AWS endpoints
        token = creds.get("access_token") or _resolve_apm_token(creds)
        if token:
            headers["Authorization"] = f"Bearer {token}"

    # Request body / query
    body_bytes: Optional[bytes] = None
    request_url = final_url
    if api_method == "GET":
        if params:
            request_url = final_url + ("&" if "?" in final_url else "?") + _urlparse.urlencode(params)
    else:
        body_bytes = json.dumps(params, ensure_ascii=False).encode("utf-8")

    # Fetch
    req = _urlrequest.Request(request_url, data=body_bytes, headers=headers, method=api_method)
    try:
        with _urlrequest.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            # Truncate huge responses to keep the LLM context budget sane.
            if len(raw) > 8000:
                raw = raw[:8000] + "\n…[truncated for context budget]"
            return raw
    except _urlerror.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        return json.dumps({
            "error": "http_error", "status": e.code, "skill": skill_name,
            "url": request_url, "body": body,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "error": "request_failed", "skill": skill_name, "message": str(e)[:300],
        }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Auto-registration
# ---------------------------------------------------------------------------

def _build_tool_schema(skill_name: str, display_name: str, bucket: str,
                       category: str = "") -> Dict[str, Any]:
    """OpenAI function-calling schema for one apmzoom skill.

    We accept any object as `params` (LLM provides whatever the skill needs);
    the skill's own SKILL.md has the parameter spec which the LLM can be
    asked to load via skill_view if it needs more detail.  This keeps the
    schema small (important for local 7-14B models with finite context).
    """
    desc = f"[apmzoom · {bucket}] {display_name or skill_name}"
    if category:
        desc += f" ({category})"
    desc += ". Pass parameters in `params` per the skill's SKILL.md."
    return {
        "name": f"apm_{skill_name}",
        "description": desc[:500],
        "parameters": {
            "type": "object",
            "properties": {
                "params": {
                    "type": "object",
                    "description": "Skill parameters as a JSON object. See the skill's SKILL.md for the exact field spec.",
                },
            },
            "required": ["params"],
        },
    }


def _make_handler(skill_name: str):
    """Closure capturing skill_name so each tool handler knows which skill to call."""
    def _handler(args: Dict[str, Any], **kwargs) -> str:
        params = args.get("params") if isinstance(args, dict) else {}
        if not isinstance(params, dict):
            return json.dumps({"error": "bad_params", "message": "`params` must be an object"},
                              ensure_ascii=False)
        return _execute_skill(skill_name, params)
    return _handler


#
# Tool-budget control
# -------------------
#
# Registering all 117 apmzoom skills as tools blows past local-model context
# windows (each schema ~70 tokens × 117 ≈ 8.2K just in tool definitions).
# Two strategies, both honored:
#
#   1. APMZOOM_TOOL_ALLOWLIST env var: comma-separated skill names.
#      Set on mini for production: "products_search_by_text_lite,vision_analyze,..."
#      Empty/unset = enable the curated DEFAULT_ALLOWLIST below (~20 high-value
#      tools covering search + onboarding + the upload-publish workflow).
#
#   2. Per-request narrowing via X-Hermes-Active-Skill (TODO: api_server filter).
#      The workflow's frontmatter `metadata.skills_used` already declares which
#      apm_* tools that scenario needs; api_server can later prune to that subset.
#
# Override anytime by adding to ~/.hermes/.env:
#   APMZOOM_TOOL_ALLOWLIST=products_search_by_text_lite,upload_image,vision_analyze
#

DEFAULT_ALLOWLIST = {
    # Search (商家最常用)
    "products_search_by_text_lite",
    "products_search_by_image_lite",
    # Upload + vision pipeline
    "presign_upload",
    "upload_image",
    "vision_analyze",
    # Auth refresh + identity
    "auth_me",
    "auth_refresh",
    # Goods management (write — 上架/改价/库存)
    "gds_m_storegoodslist",
    "gds_m_addgoods",
    "gds_m_editgoodsprice",
    "gds_m_editgoodsstock",
    "gds_m_editgoodssell",
    "gds_m_goodsclasslist",
    "gds_m_goodsmakeaddresslist",
    # Chat infra (会话续连)
    "list_chat_conversations",
    "get_chat_conversation",
}


def _allowlist() -> set:
    raw = os.environ.get("APMZOOM_TOOL_ALLOWLIST", "").strip()
    if not raw:
        return DEFAULT_ALLOWLIST
    return {s.strip() for s in raw.split(",") if s.strip()}


def _scan_and_register() -> Tuple[int, int]:
    """Register every ALLOWED apmzoom endpoint as a hermes tool.

    Walks two layouts: legacy ~/.hermes/skills/apmzoom/{read,write}/<name>/ and
    the openclaw-imports ~/.hermes/skills/openclaw-imports/apmzoom-*/<name>.md.

    Returns (registered_count, skipped_count).
    """
    home = _hermes_home()
    legacy_base = home / "skills" / "apmzoom"
    openclaw_base = home / "skills" / "openclaw-imports"
    if not legacy_base.is_dir() and not openclaw_base.is_dir():
        logger.info("[apmzoom] no apmzoom/ or openclaw-imports/ — bridge not registering anything")
        return 0, 0

    allowlist = _allowlist()
    logger.info("[apmzoom] tool allowlist size: %d (set APMZOOM_TOOL_ALLOWLIST to override)",
                len(allowlist))

    registered = 0
    skipped = 0
    seen: set = set()  # skill names already registered, avoid double-register across layouts

    # Layout 2: openclaw-imports/apmzoom-*/<name>.md
    if openclaw_base.is_dir():
        for group_dir in sorted(openclaw_base.glob("apmzoom-*")):
            if not group_dir.is_dir():
                continue
            for md_path in sorted(group_dir.glob("*.md")):
                name = md_path.stem
                if name in ("SKILL", "README", "INSTALL", "PUBLISH_CLAWHUB", "DESCRIPTION"):
                    continue
                if name in seen or name not in allowlist:
                    skipped += 1
                    continue
                try:
                    meta = _parse_openclaw_endpoint_md(md_path)
                    if not meta.get("base_url") or not meta.get("endpoint"):
                        skipped += 1
                        continue
                    perm = (meta.get("permission_level") or "read").lower()
                    bucket = "write" if perm == "write" else "read"
                    display_name = meta.get("display_name") or name
                    category = meta.get("category") or group_dir.name.replace("apmzoom-", "")
                    schema = _build_tool_schema(name, display_name, bucket, category)
                    handler = _make_handler(name)
                    registry.register(
                        name=schema["name"],
                        toolset=f"apmzoom_{bucket}",
                        schema=schema,
                        handler=handler,
                    )
                    seen.add(name)
                    registered += 1
                except Exception as e:
                    logger.warning("[apmzoom] failed to register openclaw %s: %s", name, e)
                    skipped += 1

    if not legacy_base.is_dir():
        logger.info("[apmzoom] registered %d skills (%d skipped)", registered, skipped)
        return registered, skipped

    base = legacy_base
    for bucket in ("read", "write"):
        bdir = base / bucket
        if not bdir.is_dir():
            continue
        for skill_dir in sorted(bdir.iterdir()):
            if not skill_dir.is_dir():
                continue
            name = skill_dir.name
            if name in seen or name not in allowlist:
                skipped += 1
                continue
            meta_p = skill_dir / "_meta.json"
            skill_p = skill_dir / "SKILL.md"
            if not (meta_p.exists() and skill_p.exists()):
                skipped += 1
                continue
            try:
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                # Pull display_name + category from SKILL.md frontmatter for nicer schema
                fm_content = skill_p.read_text(encoding="utf-8")
                disp_match = re.search(r'display_name:\s*"([^"]*)"', fm_content)
                cat_match = re.search(r'category:\s*"([^"]*)"', fm_content)
                display_name = disp_match.group(1) if disp_match else name
                category = cat_match.group(1) if cat_match else ""

                schema = _build_tool_schema(name, display_name, bucket, category)
                handler = _make_handler(name)
                # Toolset: split into apmzoom_read / apmzoom_write so callers can
                # enable only the safe subset (read) for low-risk merchant chats.
                toolset = f"apmzoom_{bucket}"
                registry.register(
                    name=schema["name"],
                    toolset=toolset,
                    schema=schema,
                    handler=handler,
                )
                seen.add(name)
                registered += 1
            except Exception as e:
                logger.warning("[apmzoom] failed to register %s: %s", name, e)
                skipped += 1

    logger.info("[apmzoom] registered %d skills (%d skipped)", registered, skipped)
    return registered, skipped


# Run at import time so the registry sees us before AIAgent enumerates tools.
_scan_and_register()
