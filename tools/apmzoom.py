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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as _urlerror
from urllib import parse as _urlparse
from urllib import request as _urlrequest

from tools.registry import registry

logger = logging.getLogger(__name__)


def _attach_apmzoom_log_file() -> None:
    """Route apmzoom logger to ~/.hermes/logs/apmzoom.log.

    Tool handlers run in a ThreadPoolExecutor that doesn't inherit hermes'
    main log config, so `logger.info(...)` calls from inside a handler
    would otherwise vanish.  We attach a dedicated FileHandler once at
    import time so every tool invocation leaves a trail (→ params, ←
    result) that's easy to grep when a merchant reports a weird answer.
    """
    try:
        home = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
        logs_dir = home / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / "apmzoom.log"
        # Avoid duplicate handlers on module re-import
        if any(getattr(h, "_apmzoom_file", False) for h in logger.handlers):
            return
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh._apmzoom_file = True  # marker for idempotence
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
        # Don't double-print to parent (agent.log) — keep apmzoom.log as
        # the single source of truth for this bridge.
        logger.propagate = False
    except Exception:
        # Never let logging setup break tool registration.
        pass


_attach_apmzoom_log_file()

# ---------------------------------------------------------------------------
# Per-request merchant context
# ---------------------------------------------------------------------------
#
# Set by gateway/platforms/api_server.py at the start of each chat completion
# request when the X-Hermes-Merchant-Id header is present.  Tool handlers
# read it to look up the right merchant's APM credentials.
#
# Originally ContextVar, but hermes dispatches tool handlers via
# ThreadPoolExecutor (run_agent.py:844) and ContextVars do NOT propagate to
# pool workers by default — mid kept resolving to "" inside handlers even
# after api_server set it, so auth headers were never attached.  Plain
# module globals work for the Mac mini single-process deployment where this
# lives; requests are effectively serial.  If we ever need true per-request
# isolation under parallel load, wrap executor.submit in
# `contextvars.copy_context().run(...)` in run_agent and revert to ContextVar.
_current_merchant_id: str = ""
_current_active_skill: str = ""

# Sticky per-skill method override.  Populated on first successful fallback
# (POST → GET) so subsequent calls skip the wasted 405 roundtrip.  In-process
# only — reset on gateway restart, which is fine for this deployment.
_METHOD_CACHE: Dict[str, str] = {}


def set_merchant_id(merchant_id: str) -> None:
    """Called from api_server when a merchant request lands.

    Safe to call with empty string to clear (no-op merchant_mode).
    """
    global _current_merchant_id
    _current_merchant_id = merchant_id or ""


def current_merchant_id() -> str:
    return _current_merchant_id


def set_active_skill(active_skill: str) -> None:
    """Propagate the X-Hermes-Active-Skill header for downstream filters."""
    global _current_active_skill
    _current_active_skill = active_skill or ""


def current_active_skill() -> str:
    return _current_active_skill


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


# Parameter-hint block headers seen across openclaw docs (zh/ko/en).
# We pull the paragraph that follows any of these and truncate for the
# tool schema, so the LLM sees field names without having to call skill_view.
_PARAM_HEADER_RE = re.compile(
    r"""【
        (?:
            파라미터 | 요청\s*본문 | 필드\s*설명 | 호출\s*흐름 | 호출\s*절차 |
            参数 | 请求体 | 请求\s*本体 | 字段说明 | 调用\s*流程 | 调用\s*顺序 |
            Parameters | Request\s*Body | Fields | Call\s*Flow | Procedure
        )
        】""",
    re.IGNORECASE | re.VERBOSE,
)


def _extract_param_hint(content: str, max_chars: int = 320) -> str:
    """Pull the parameter section from an openclaw doc body.

    Strategy: find the first 【파라미터/参数/Parameters/…】 header, then
    take up to `max_chars` chars until the next 【…】 block or the Quick
    Reference table, whichever comes first.  Inline code fences (``` or
    single backticks) are flattened to keep the schema readable.
    """
    m = _PARAM_HEADER_RE.search(content)
    if not m:
        return ""
    tail = content[m.end():]
    # Stop at the next 【…】 header or the Quick Reference heading.
    stops = []
    next_header = re.search(r"【[^】]+】", tail)
    if next_header:
        stops.append(next_header.start())
    qr = re.search(r"##\s*Quick Reference", tail)
    if qr:
        stops.append(qr.start())
    cut = min(stops) if stops else len(tail)
    snippet = tail[:cut].strip()
    # Flatten code fences + collapse blank lines.
    snippet = re.sub(r"```[a-z]*\n?|```", "", snippet)
    snippet = re.sub(r"\n{3,}", "\n\n", snippet)
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rstrip() + "…"
    return snippet


def _parse_openclaw_endpoint_md(path: Path) -> Dict[str, Any]:
    """Parse an openclaw-imports endpoint file (e.g. apmzoom-gds/gds_m_storegoodslist.md).

    These docs store the HTTP spec in a Quick Reference markdown table and
    the MD5 signing salt inline in the prose.  Shape mirrors
    `_parse_skill_md` so downstream code is unchanged.
    """
    out = {
        "api_method": "POST", "api_url": "", "endpoint": "", "base_url": "",
        "auth_type": "none", "auth_sign_salt": "", "permission_level": "read",
        "display_name": "", "category": "", "param_hint": "",
    }
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return out
    out["param_hint"] = _extract_param_hint(content)

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

    # Upstream worker registers api_method="POST" for ~23 read-type endpoints
    # (gds_m_*list/info, gds_u_*, auth_me, ids_captcha_img, …) that actually
    # only accept GET.  Rather than edit each SKILL.md (skill-sync-worker.py
    # would overwrite), auto-fallback here: cache the first successful method
    # per skill to avoid paying 405 cost on every call.
    effective_method = _METHOD_CACHE.get(skill_name, api_method)

    def _do_fetch(method: str):
        """Returns (raw, http_status, url).  raw is text on success, None on HTTP error."""
        body_bytes: Optional[bytes] = None
        req_url = final_url
        if method == "GET":
            if params:
                req_url = final_url + ("&" if "?" in final_url else "?") + _urlparse.urlencode(params)
        else:
            body_bytes = json.dumps(params, ensure_ascii=False).encode("utf-8")
        req = _urlrequest.Request(req_url, data=body_bytes, headers=headers, method=method)
        try:
            with _urlrequest.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace"), 200, req_url
        except _urlerror.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            return body, e.code, req_url
        except Exception as e:
            return str(e)[:300], 0, req_url

    raw, status, request_url = _do_fetch(effective_method)

    # 405 + POST → retry with GET (and remember)
    if status == 405 and effective_method == "POST":
        logger.warning("[apmzoom] %s: POST→405, retrying with GET", skill_name)
        raw, status, request_url = _do_fetch("GET")
        if status == 200:
            _METHOD_CACHE[skill_name] = "GET"
            logger.info("[apmzoom] %s: cached method=GET (worker metadata override)", skill_name)

    if status == 200:
        if len(raw) > 8000:
            raw = raw[:8000] + "\n…[truncated for context budget]"
        return raw
    if status == 0:
        return json.dumps({
            "error": "request_failed", "skill": skill_name, "message": raw,
        }, ensure_ascii=False)
    return json.dumps({
        "error": "http_error", "status": status, "skill": skill_name,
        "url": request_url, "body": raw,
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Auto-registration
# ---------------------------------------------------------------------------

def _build_tool_schema(skill_name: str, display_name: str, bucket: str,
                       category: str = "", param_hint: str = "") -> Dict[str, Any]:
    """OpenAI function-calling schema for one apmzoom skill.

    We accept any object as `params` (LLM provides whatever the skill needs).
    When a `param_hint` is supplied (extracted from the endpoint doc's
    【파라미터】/【参数】 block), we inline it into the `params` property
    description so the LLM can pick field names without calling skill_view.
    This is the difference between "LLM guesses mark/page_size correctly on
    first try" and "LLM hallucinates merchant_id param and 400s".
    """
    desc = f"[apmzoom · {bucket}] {display_name or skill_name}"
    if category:
        desc += f" ({category})"
    desc += "."
    params_desc = (
        f"Skill parameters as a JSON object.\n\nField spec from the skill doc:\n{param_hint}"
        if param_hint
        else "Skill parameters as a JSON object. See the skill's SKILL.md for the exact field spec."
    )
    # Cap the params description so one bloated doc can't blow the tool
    # budget — 420 chars ≈ ~110 tokens, fits comfortably with 20 tools in 8K.
    if len(params_desc) > 420:
        params_desc = params_desc[:420].rstrip() + "…"
    return {
        "name": f"apm_{skill_name}",
        "description": desc[:500],
        "parameters": {
            "type": "object",
            "properties": {
                "params": {
                    "type": "object",
                    "description": params_desc,
                },
            },
            "required": ["params"],
        },
    }


def _make_handler(skill_name: str):
    """Closure capturing skill_name so each tool handler knows which skill to call."""
    def _handler(args: Dict[str, Any], **kwargs) -> str:
        # Accept both {"params": {...}} (schema-compliant) and a bare object
        # (local 7B models occasionally skip the wrapper).  Be permissive on
        # input, strict on output.
        if isinstance(args, dict) and "params" in args and isinstance(args["params"], dict):
            params = args["params"]
        elif isinstance(args, dict):
            params = args
        else:
            return json.dumps({"error": "bad_params", "message": "arguments must be an object"},
                              ensure_ascii=False)
        logger.info("[apmzoom] → %s mid=%s params=%s",
                    skill_name, current_merchant_id() or "(none)",
                    json.dumps(params, ensure_ascii=False)[:300])
        out = _execute_skill(skill_name, params)
        logger.info("[apmzoom] ← %s result=%s", skill_name, out[:300])
        return out
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


# Authentication-sensitive tools the LLM should NEVER call autonomously.
# Calling a login endpoint from the agent loop is both useless (the merchant
# already logged in via the frontend before landing on the chat) and
# dangerous (a misrouted user message like "帮我登录" could make the LLM
# POST credentials it doesn't have, or replay stale creds).  Out of the
# allowlist by default.  Opt in with APMZOOM_ALLOW_LOGIN_TOOLS=1 for
# debugging / integration testing only.
_LOGIN_TOOLS = {
    "ids_m_login_account", "ids_m_login_email", "ids_m_login_tel",
    "ids_u_login_account", "ids_u_login_email", "ids_u_login_tel",
    "ids_u_login_to_ce",   "ids_suppliers_login",
    "ids_admin_login",     "ids_admin_app_tool_login", "ids_admin_desk_tool_login",
    "ids_send_tel_code",   "ids_send_tel_code_r",
    "ids_send_email_code", "ids_send_email_code_r",
}


def _allowlist() -> set:
    raw = os.environ.get("APMZOOM_TOOL_ALLOWLIST", "").strip()
    base = {s.strip() for s in raw.split(",") if s.strip()} if raw else set(DEFAULT_ALLOWLIST)
    if os.environ.get("APMZOOM_ALLOW_LOGIN_TOOLS", "").lower() not in ("1", "true", "yes"):
        base -= _LOGIN_TOOLS
    return base


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
                    param_hint = meta.get("param_hint", "")
                    schema = _build_tool_schema(name, display_name, bucket, category, param_hint)
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
