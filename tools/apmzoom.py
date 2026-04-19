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
    Also captures `parameters` as raw Python-repr string (if present) for
    downstream JSON-schema generation + pre-flight validation.
    """
    out = {
        "api_method": "POST", "api_url": "", "endpoint": "", "base_url": "",
        "auth_type": "none", "auth_sign_salt": "", "permission_level": "read",
        "parameters_raw": "",
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
    # The `parameters:` value is a one-line Python-repr dict that the regex
    # above doesn't capture (mixed quotes).  Grab it with a dedicated pattern.
    m = re.search(
        r'^\s{0,4}parameters:\s*"(.+?)"\s*$', fm, re.MULTILINE | re.DOTALL,
    )
    if m:
        out["parameters_raw"] = m.group(1)
    return out


# Business-critical fields that worker metadata flags as `required: False`
# by mistake.  Example: gds_m_editgoodsprice registers goods_id/sale_price
# as optional but obviously requires both — the API 400s otherwise.  We
# force these required ONLY for write-type skills (names containing
# edit/add/del/update/upload/create — see _is_write_skill) where missing
# business id = data loss risk.  Read/list/search skills keep the registered
# requireds as-is because their call patterns legitimately vary (e.g.
# gds_m_storegoodslist supports either goods_id *or* keyword, not both).
_WRITE_FORCE_REQUIRED = {
    "goods_id", "goods_ids", "sku_id", "class_id", "goods_class_cascade_id",
    "make_address_id", "msg_id",
    "sale_price", "discount_price", "discount", "stock_count",
    "is_sell",
    "image_url", "img",
}


def _is_write_skill(skill_name: str) -> bool:
    """Heuristic: name contains a write verb → force-require business fields."""
    n = skill_name.lower()
    return any(v in n for v in (
        "edit", "add", "del", "update", "upload", "create", "publish",
    ))

# When a required field is missing from LLM-supplied params, return this hint
# so the LLM knows which upstream skill can fetch it.  Covers the 80% case;
# obscure fields fall through to a generic "see SKILL.md" message.
_FIELD_SOURCE_HINTS = {
    "goods_id":        "先调 apm_gds_m_storegoodslist 按关键字/编号搜到 goods_id",
    "goods_ids":       "先调 apm_gds_m_storegoodslist 列出,让商家选 id 数组",
    "sku_id":          "先调 apm_gds_m_goodseditskuinfo?goods_id=X 拿 sku 列表",
    "class_id":        "先调 apm_gds_m_goodsclasslist 列出分类让商家选",
    "goods_class_cascade_id": "先调 apm_gds_m_goodsclasslist 列出分类让商家选(级联 id)",
    "make_address_id": "先调 apm_gds_m_goodsmakeaddresslist 列出产地让商家选",
    "store_id":        "查自家用 identity.md 里的 merchant_uuid;查别家先搜到对方 uuid",
    "msg_id":          "先调 apm_pms_m_pushmsglist 列出消息,拿到具体 msg_id",
    "ver":             "版本号(乐观锁)— 先调同系列的 list/info 读最新 ver 再传",
    "image_url":       "先调 apm_presign_upload + apm_upload_image 拿 CDN URL",
    "img":             "先调 apm_presign_upload + apm_upload_image 拿 CDN URL",
}


_JSON_TYPE_MAP = {
    "integer": "integer", "int": "integer", "long": "integer",
    "number": "number", "float": "number", "double": "number",
    "string": "string", "str": "string", "text": "string",
    "boolean": "boolean", "bool": "boolean",
    "array": "array", "list": "array",
    "object": "object", "dict": "object", "map": "object",
}


def _extract_param_schema(parameters_raw: str, skill_name: str = "") -> Dict[str, Any]:
    """Parse the `parameters:` Python-repr blob into a JSON-schema-ish dict.

    Returns {"properties": {...}, "required": [...]} suitable for inlining
    under the `params` property of a tool schema.  Filters out `headers`
    (bridge-managed).  Applies the _ALWAYS_REQUIRED heuristic to correct
    worker metadata bugs where obvious business fields are left optional.
    """
    if not parameters_raw:
        return {"properties": {}, "required": []}
    import ast
    try:
        # Worker stores it as Python repr with escaped quotes — round-trip
        # through unicode_escape to decode \'s then literal_eval.
        spec = ast.literal_eval(parameters_raw.encode().decode("unicode_escape"))
    except Exception as e:
        logger.debug("[apmzoom] param spec parse failed: %s", e)
        return {"properties": {}, "required": []}
    if not isinstance(spec, dict):
        return {"properties": {}, "required": []}

    force_set = _WRITE_FORCE_REQUIRED if _is_write_skill(skill_name) else set()

    properties: Dict[str, Any] = {}
    required: list = []
    for section in ("body", "query", "path"):
        for f in spec.get(section) or []:
            if not isinstance(f, dict):
                continue
            name = f.get("name", "").strip()
            if not name:
                continue
            properties[name] = {
                "type": _JSON_TYPE_MAP.get((f.get("type") or "").lower(), "string"),
                "description": (f.get("description") or "")[:120],
            }
            if f.get("required") or name in force_set:
                if name not in required:
                    required.append(name)
    return {"properties": properties, "required": required}


def _maybe_autofetch_ver(skill_name: str, skill_meta: Dict[str, Any],
                         params: Dict[str, Any]) -> None:
    """Inject a fresh `ver` into params if the skill needs one and it's missing.

    Looks up the latest ver via `gds_m_storegoodslist?goods_id=X`.  Only fires
    when:
      - skill_name is a write-type (editgoods*, delgoods, …) that declared
        `ver` in the required set
      - params has a non-empty `goods_id`
      - params doesn't already carry a ver

    Silent no-op on any failure — the subsequent validation step will catch
    the still-missing ver and hint the LLM to fetch it manually.  Mutates
    `params` in place.
    """
    if not _is_write_skill(skill_name):
        return
    required = _extract_param_schema(
        skill_meta.get("parameters_raw", ""), skill_name
    ).get("required") or []
    if "ver" not in required:
        return
    if params.get("ver"):
        return
    goods_id = params.get("goods_id")
    if not goods_id:
        return
    # Use goodseditinfo for per-goods lookup: storegoodslist's goods_id
    # param is actually a cursor-pagination marker, not a filter.  editinfo
    # is the authoritative "give me one goods" endpoint and returns ver too.
    try:
        raw = _execute_skill("gds_m_goodseditinfo", {"goods_id": int(goods_id)})
        d = json.loads(raw)
        r = d.get("result")
        if not r:
            return
        obj = r[0] if isinstance(r, list) and r else r
        if not isinstance(obj, dict):
            return
        ver = obj.get("ver")
        if ver is None:
            return
        params["ver"] = ver
        logger.info(
            "[apmzoom] %s: auto-injected ver=%s for goods_id=%s (OCC)",
            skill_name, ver, goods_id,
        )
    except Exception as e:
        logger.debug("[apmzoom] ver auto-fetch failed for %s: %s", skill_name, e)


# Per-skill parameter defaults that the backend expects present even though
# SKILL.md marks them optional.  Seen in production:
#   gds_m_addgoods drops to 400 ("起购数量最小为1") when least_buy_num is
#   absent, despite description saying "default 1".  Bridge injects these
#   defaults when the caller doesn't — one place to fix upstream metadata
#   drift without editing every workflow that calls the skill.
_PARAM_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "gds_m_addgoods": {
        "least_buy_num": 1,   # backend rejects 0 even though "默认1" in doc
        "limit_buy_num": 0,   # 0 = no cap, safe default
        "is_sell": 1,         # 1 = listed on the shop
        "discount_percent": 1,  # 1 = no discount
        "currency_type": 1,   # 1 = KRW for apmzoom merchants
    },
}


def _maybe_enrich_addgoods_response(skill_name: str, raw_response: str) -> str:
    """After gds_m_addgoods succeeds, the backend only returns
    ``{"code":100,"message":"发布成功"}`` — no goods_id.  LLM then tries
    to guess one (seen: grabs timestamp from image URL) and calls
    goodseditinfo with a hallucinated id, gets '暂无数据', and retries
    addgoods → duplicate products get created.

    Bridge does the lookup itself: after a successful addgoods, fetch
    the top item from storegoodslist (mark=1) and splice its goods_id
    into the response.  LLM sees ``result.goods_id`` and doesn't need
    to guess.
    """
    if skill_name != "gds_m_addgoods":
        return raw_response
    try:
        d = json.loads(raw_response)
    except Exception:
        return raw_response
    if d.get("code") != 100:
        return raw_response
    if (d.get("result") or {}).get("goods_id"):
        return raw_response  # backend already included it (unlikely but future-proof)
    try:
        lookup = _execute_skill(
            "gds_m_storegoodslist",
            {"page_size": 1, "mark": 1},
            _internal=True,
        )
        lookup_d = json.loads(lookup)
        newest = (lookup_d.get("result") or [None])[0]
        if not newest or not newest.get("goods_id"):
            return raw_response
        # Splice into the original response so the LLM sees it
        existing = d.get("result")
        enriched = {
            "goods_id": newest.get("goods_id"),
            "goods_name": newest.get("goods_name"),
            "sale_price": newest.get("sale_price"),
            "stock_count": newest.get("stock_count"),
            "ver": newest.get("ver"),
        }
        if isinstance(existing, dict):
            enriched = {**existing, **enriched}
        d["result"] = enriched
        logger.info(
            "[apmzoom] gds_m_addgoods: enriched response with goods_id=%s (%s)",
            enriched["goods_id"], enriched.get("goods_name"),
        )
        return json.dumps(d, ensure_ascii=False)
    except Exception as e:
        logger.debug("[apmzoom] addgoods response enrichment failed: %s", e)
        return raw_response


def _build_class_cascade(goodsclasslist_result, target_leaf_id):
    """Walk a goodsclasslist tree and build the '1-6-7' cascade string
    for a given leaf class_id.  Returns empty string if not found."""
    def walk(nodes, path):
        for n in nodes:
            if not isinstance(n, dict):
                continue
            cid = n.get("goods_class_id")
            cur = path + [cid]
            if cid == target_leaf_id:
                return cur
            children = n.get("ls_child") or []
            if children:
                found = walk(children, cur)
                if found:
                    return found
        return None

    path = walk(goodsclasslist_result or [], [])
    return "-".join(str(x) for x in path) if path else ""


def _maybe_build_cascade_id(skill_name: str, params: Dict[str, Any]) -> None:
    """If addgoods payload has `goods_class_cascade_id` that doesn't look
    like a cascade ('1-6-7') — e.g. the LLM passed a single leaf id '7' or
    it's missing entirely while `class_id` is provided — fetch the tree
    once and upgrade the value to a real cascade.  One place for the
    bridge to rescue LLMs that struggle with the cascade format."""
    if skill_name != "gds_m_addgoods":
        return

    cascade = params.get("goods_class_cascade_id")
    class_id = params.get("class_id") or params.get("goods_class_id")

    def looks_cascade(v):
        return isinstance(v, str) and v.count("-") >= 1 and all(
            p.isdigit() for p in v.split("-")
        )

    # Already a proper cascade — nothing to do
    if looks_cascade(cascade):
        return

    # Pick the best source leaf id: prefer numeric cascade value, then
    # explicit class_id / goods_class_id fallback.
    target_leaf = None
    if cascade is not None:
        try:
            target_leaf = int(str(cascade).strip())
        except Exception:
            target_leaf = None
    if target_leaf is None and class_id is not None:
        try:
            target_leaf = int(class_id)
        except Exception:
            target_leaf = None

    if target_leaf is None:
        return  # nothing to rebuild from

    try:
        # Use internal=True to skip the 8KB response truncation — the class
        # tree is ~9-12 KB and would otherwise land invalid JSON here.
        raw = _execute_skill("gds_m_goodsclasslist", {}, _internal=True)
        d = json.loads(raw)
        tree = d.get("result") or []
        built = _build_class_cascade(tree, target_leaf)
        if built:
            params["goods_class_cascade_id"] = built
            # Drop the now-redundant single id so backend doesn't get confused
            params.pop("class_id", None)
            params.pop("goods_class_id", None)
            logger.info(
                "[apmzoom] %s: rebuilt goods_class_cascade_id '%s' → '%s' (from leaf=%s)",
                skill_name, cascade, built, target_leaf,
            )
    except Exception as e:
        logger.debug("[apmzoom] cascade_id rebuild failed: %s", e)


def _apply_param_defaults(skill_name: str, params: Dict[str, Any]) -> None:
    """Mutate `params` to fill in bridge-side defaults for well-known
    skill fields whose upstream defaults don't actually fire.  No-op for
    skills not in _PARAM_DEFAULTS."""
    defaults = _PARAM_DEFAULTS.get(skill_name)
    if not defaults:
        return
    for key, val in defaults.items():
        if key not in params or params.get(key) in (None, "", 0) and key != "limit_buy_num":
            # Note: 0 is a valid value for limit_buy_num (meaning unlimited),
            # so don't override if explicitly set to 0.
            params[key] = val


def _validate_params(skill_name: str, skill_meta: Dict[str, Any],
                     params: Dict[str, Any]) -> Optional[str]:
    """Check LLM-supplied params against the skill's schema before HTTP.

    Returns None if valid, else a JSON string error payload the LLM sees in
    place of the HTTP response.  Error payload includes `missing` list and
    `hint` mapping each missing field to the upstream skill that can fetch
    it, so the LLM can recover in-loop without needing an extra turn.
    """
    param_schema = _extract_param_schema(skill_meta.get("parameters_raw", ""), skill_name)
    required = param_schema.get("required") or []
    if not required:
        return None  # no schema info → can't validate, let HTTP be authoritative

    missing = [f for f in required if f not in params or params[f] in (None, "", [])]
    if not missing:
        return None

    hints = {f: _FIELD_SOURCE_HINTS.get(f, f"见 {skill_name} 的 SKILL.md 参数章节")
             for f in missing}
    return json.dumps({
        "error": "missing_required_params",
        "skill": skill_name,
        "missing": missing,
        "hint": hints,
        "message": f"缺少必填参数: {', '.join(missing)}。" +
                   "先补齐再重调此工具。",
    }, ensure_ascii=False)


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
                   merchant_id_override: str = "", _internal: bool = False) -> str:
    """Call one apmzoom skill via HTTP.  Returns JSON string for LLM consumption.

    _internal=True skips the 8KB response truncation so bridge-internal
    callers (cascade_id rebuild, etc.) can parse the full response.
    """
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

    # OCC `ver` auto-injection for write skills.  Most gds_m_edit* endpoints
    # declare `ver` required and reject stale values with HTTP 409.  Rather
    # than make the LLM call storegoodslist → read ver → carry it to the
    # edit call (2 round trips + context bloat), bridge fetches it itself
    # when goods_id is present and ver isn't.  Runs *before* validation so
    # the injected ver counts toward the required set.
    _maybe_autofetch_ver(skill_name, meta, params)
    _apply_param_defaults(skill_name, params)
    _maybe_build_cascade_id(skill_name, params)

    # Pre-flight param validation (short-circuit before HTTP).  For skills
    # whose frontmatter declares a `parameters:` spec, ensure required
    # fields are present — missing ones come back as a structured error
    # with a `hint` pointing at the upstream skill that can fetch them.
    # Local LLMs use the hint to bounce to the correct prereq skill
    # in-loop rather than guessing or 400-ing.
    err = _validate_params(skill_name, meta, params)
    if err is not None:
        logger.info("[apmzoom] ✗ %s: param validation failed, returning hint", skill_name)
        return err

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
        # Post-call enrichment hooks (bridge splices in data the LLM
        # would otherwise have to hunt for — e.g. addgoods lacks
        # goods_id in its response and the LLM was hallucinating one).
        if not _internal:
            raw = _maybe_enrich_addgoods_response(skill_name, raw)
        if len(raw) > 8000 and not _internal:
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
                       category: str = "", param_hint: str = "",
                       param_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """OpenAI function-calling schema for one apmzoom skill.

    When `param_schema` (from `_extract_param_schema`) is provided, we inline
    its `properties` + `required` fields under `params` so the LLM sees the
    exact JSON shape in its tool list — no need to call `skill_view`.  Local
    7-14B models pick the right field names ~3x more reliably this way.
    Falls back to freeform object + inline `param_hint` when no schema is
    available (openclaw-imports layout).
    """
    desc = f"[apmzoom · {bucket}] {display_name or skill_name}"
    if category:
        desc += f" ({category})"
    desc += "."

    if param_schema and param_schema.get("properties"):
        # Strongly-typed params: emit full properties + required list
        params_property = {
            "type": "object",
            "description": f"{skill_name} parameters.",
            "properties": param_schema["properties"],
        }
        if param_schema.get("required"):
            params_property["required"] = param_schema["required"]
    else:
        params_desc = (
            f"Skill parameters as a JSON object.\n\nField spec from the skill doc:\n{param_hint}"
            if param_hint
            else "Skill parameters as a JSON object. See the skill's SKILL.md for the exact field spec."
        )
        if len(params_desc) > 420:
            params_desc = params_desc[:420].rstrip() + "…"
        params_property = {
            "type": "object",
            "description": params_desc,
        }
    return {
        "name": f"apm_{skill_name}",
        "description": desc[:500],
        "parameters": {
            "type": "object",
            "properties": {"params": params_property},
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


def _collect_workflow_skills() -> set:
    """Walk ~/.hermes/skills/apmzoom-workflows/*/SKILL.md and union all
    `skills_used` entries.  Makes the allowlist self-adaptive: authors only
    need to declare the skills they use in the workflow frontmatter, and
    the bridge auto-registers them.  No more hand-editing DEFAULT_ALLOWLIST
    when a new workflow references a previously-unregistered skill.
    """
    out: set = set()
    base = _hermes_home() / "skills" / "apmzoom-workflows"
    if not base.is_dir():
        return out
    for wf_dir in base.iterdir():
        if not wf_dir.is_dir():
            continue
        md = wf_dir / "SKILL.md"
        if not md.is_file():
            continue
        try:
            content = md.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        # Extract the `skills_used:` YAML block from frontmatter.
        m = re.search(r'(?m)^\s*skills_used:\s*$((?:\n\s+-\s*.*)+)', content)
        if not m:
            continue
        for item in re.findall(r'-\s*"?([A-Za-z0-9_.-]+)"?\s*$', m.group(1), re.MULTILINE):
            # Skip ui_* (they're registered separately by tools/apmzoom_ui.py).
            if not item.startswith("ui_"):
                out.add(item)
    return out


def _allowlist() -> set:
    raw = os.environ.get("APMZOOM_TOOL_ALLOWLIST", "").strip()
    base = (
        {s.strip() for s in raw.split(",") if s.strip()}
        if raw
        else set(DEFAULT_ALLOWLIST) | _collect_workflow_skills()
    )
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

                # Parse SKILL.md frontmatter for the `parameters:` Python-repr
                # blob, then build a JSON schema so the tool definition shows
                # the LLM exact field names + required markers instead of a
                # generic `object`.  Falls back to freeform if missing.
                full_meta = _parse_skill_md(skill_p)
                param_schema = _extract_param_schema(full_meta.get("parameters_raw", ""), name)

                schema = _build_tool_schema(
                    name, display_name, bucket, category,
                    param_schema=param_schema,
                )
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
