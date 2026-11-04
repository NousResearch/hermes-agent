#!/usr/bin/env python3
"""
Contact Resolver Tool — ranked contact lookup from the Google Contacts Sheet.

Three-signal algorithm:
  Signal 1 (name)    — exact / partial / fuzzy across first_name(A), last_name(C),
                        nickname(I), org(K), alias(CE), voice_misspellings(CN).
                        Uses the noun_resolver in-memory index when available;
                        falls back to a direct sheet scan if the index isn't ready.
  Signal 2 (context) — boosts candidates whose project(CA), land(CB), topic(CC)
                        associations or conversation_history(CG) mention the context.
  Signal 3 (compound)— handles "FirstName CompanyAbbr" queries by matching
                        the first token on Col A and the rest on Col K (org).

Returns ranked candidates with full contact details (phones, emails, org) ready
for the calling skill to present — no further sheet reads needed by the model.

Registered as: contact_resolver
Toolset: general
"""

import json
import logging
import os
import re
import unicodedata

logger = logging.getLogger(__name__)

SHEET_ID     = "1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g"
CONTACTS_TAB = "NDR DRAAS Google contacts.csv"

# Contacts column indices (0-based).  A=0, B=1, ..., Z=25, AA=26, ..., CA=78 ...
COL_FIRST = 0    # A  first_name
COL_LAST  = 2    # C  last_name
COL_NICK  = 8    # I  nickname / addressed-as
COL_ORG   = 10   # K  organization
COL_PROJ  = 78   # CA project_association
COL_LAND  = 79   # CB land_association
COL_TOPIC = 80   # CC topic_association
COL_ADDR_AS = 82   # CE addressed_as
COL_HIST  = 84   # CG conversation_history
COL_MISSP = 91   # CN voice_misspellings
COL_SCORE = 92   # CO contact_score

# Offset when reading the CA:CO subrange (CA is position 0 of that range)
_CA_OFFSET = 78   # CA index in full row


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(text or ""))
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_str).strip().lower()


def _cell(row: list, idx: int) -> str:
    try:
        return str(row[idx]).strip() if idx < len(row) else ""
    except (IndexError, TypeError):
        return ""


def _sanitize_phone_value(raw: str) -> str:
    """Strip non-digit characters from a raw sheet phone value.

    Preserves a leading '+' (ISD/country-code indicator) if present; strips
    everything else (spaces, dashes, parens, sheet formatting artifacts).
    Keeps phone numbers returned to the LLM clean so downstream tools
    (e.g. whatsapp_link) never see spaces/dashes that could be mistranscribed.
    """
    raw = (raw or "").strip()
    has_plus = raw.startswith("+")
    digits = re.sub(r"\D", "", raw)
    return f"+{digits}" if has_plus and digits else digits


def _fuzzy(q: str, c: str) -> float:
    try:
        from rapidfuzz import fuzz
        from rapidfuzz.distance import JaroWinkler
        return max(fuzz.token_sort_ratio(q, c) / 100.0,
                   JaroWinkler.normalized_similarity(q, c))
    except ImportError:
        sq, sc = set(q), set(c)
        return len(sq & sc) / max(len(sq | sc), 1) if (sq or sc) else 1.0


def _field_score(query_norm: str, field_norm: str, base: float) -> float:
    """Score query against a single field.  Returns 0 if no reasonable match."""
    if not field_norm:
        return 0.0
    if query_norm == field_norm:
        return base
    if field_norm.startswith(query_norm) or query_norm.startswith(field_norm):
        return base * 0.90
    q_toks = query_norm.split()
    f_toks = set(field_norm.split())
    if q_toks and all(any(t in ft for ft in f_toks) for t in q_toks):
        return base * 0.85
    f = _fuzzy(query_norm, field_norm)
    return base * f if f >= 0.75 else 0.0


# Per-user OAuth via the gws-vault daemon -- the ONLY Google Workspace auth
# mechanism in this system (there is no service account / domain-wide
# delegation anywhere; see hermes-data/SOUL.md, AGENTS.md). Static
# per-account credential files (DRAAS_CRED_FILE etc.) were a dead,
# pre-vault-migration path and have been removed -- those files no longer
# exist on the host.
def _build_svc(account_email: str | None = None):
    """Return a Google Sheets API service for the contacts registry.

    account_email: optional override -- an email address (e.g.
    'user@example.com') or short label ('draas'/'ahfl'/'gmail') identifying
    which Google account's vault token to use. If omitted, defaults to the
    current session user's own configured gws_service (from their profile
    in tools._user_registry), same as noun_resolver.py.
    """
    from tools import gws_auth
    from gateway.session_context import get_gws_identity_env

    # get_gws_identity_env() covers both live-chat sessions and cron jobs
    # (cron deliberately clears HERMES_SESSION_USER_ID and stores the real
    # owner in HERMES_CRON_JOB_OWNER_ID instead -- a raw get_session_env()
    # read here would wrongly treat every cron-triggered call as "no user").
    session_uid = get_gws_identity_env()
    if not session_uid:
        raise RuntimeError(
            "Cannot identify session user for Google Sheets access "
            "(no live session or cron job owner context available)."
        )

    if account_email:
        from tools.gws_account_resolver_tool import _resolve_one
        resolved = _resolve_one(account_email)
        if "error" in resolved:
            raise RuntimeError(resolved["error"])
        service_name = resolved["service_name"]
        # Security: verify the requested account_email belongs to the session
        # user. The vault's read-path (session_uid == user_id) already prevents
        # cross-user token access, but this explicit check makes the policy clear
        # and prevents future refactoring errors.
        resolved_email = resolved.get("email") or account_email
        if not gws_auth.verify_email_ownership(resolved_email, session_uid):
            raise RuntimeError(
                f"Account email '{resolved_email}' is not associated with "
                f"your session user. You can only use accounts linked to "
                f"your own user profile."
            )
    else:
        from tools._user_registry import get_user_config
        cfg = get_user_config(session_uid) or {}
        service_name = cfg.get("gws_service")
        if not service_name:
            raise RuntimeError(
                f"User {session_uid!r} has no gws_service configured in their "
                "profile. An admin must set 'gws_service', or pass account_email "
                "explicitly (call gws_resolve_account to see valid accounts)."
            )

    return gws_auth.build_service(
        "sheets", "v4", service_name=service_name
    )


def _read_range(svc, range_str: str) -> list:
    resp = svc.spreadsheets().values().get(
        spreadsheetId=SHEET_ID, range=range_str
    ).execute()
    return resp.get("values", [])


# ── Signal 1 — name matching ───────────────────────────────────────────────────

def _s1_from_index(query_norm: str) -> dict:
    """
    Iterate the noun_resolver index collecting every *contacts* entry that
    has a matching variant key.  Returns {row_int: {canonical, score, reason}}.
    Collects ALL rows, not just the top-1 per key, so ambiguous first names
    (e.g. "bhuvanesh" matching 4 different contacts) all appear.
    """
    try:
        from tools.noun_resolver import get_resolver
        resolver = get_resolver()
        if not resolver._index_built:
            return {}

        results: dict = {}
        with resolver._index_lock:
            for key, entry in resolver._index.items():
                if entry.get("sheet") != "contacts":
                    continue
                row = entry.get("row")
                if not isinstance(row, int) or row < 2:
                    continue

                score = _field_score(query_norm, key, 100.0)
                if score <= 0:
                    continue

                # Keep the best score seen for this row
                if row not in results or results[row]["score"] < score:
                    results[row] = {
                        "canonical": entry.get("canonical", ""),
                        "score": score,
                        "reason": f"index key '{key}'",
                    }
        return results
    except Exception as e:
        logger.warning("S1 index lookup failed: %s", e)
        return {}


def _s1_from_sheet(svc, query_norm: str) -> dict:
    """
    Fallback: scan the contacts sheet directly (name columns only: A:CE).
    Used when the index isn't ready.  Python matching is deterministic.
    """
    results: dict = {}
    try:
        rows = _read_range(svc, f"'{CONTACTS_TAB}'!A:CE")
    except Exception as e:
        logger.warning("S1 sheet fallback failed: %s", e)
        return results

    for row_num, row in enumerate(rows[1:], start=2):
        first = _cell(row, COL_FIRST)
        last  = _cell(row, COL_LAST)
        nick  = _cell(row, COL_NICK)
        org   = _cell(row, COL_ORG)
        addr_as_raw = _cell(row, COL_ADDR_AS)
        missp_raw = _cell(row, min(COL_MISSP, len(row) - 1))

        full = f"{first} {last}".strip()

        # Build all searchable fields with their confidence weights
        fields = [
            (_normalize(full),  100.0),
            (_normalize(first), 100.0),
            (_normalize(last),   90.0),
            (_normalize(nick),   92.0),
        ]
        for a in re.split(r"[|,;]", addr_as_raw):
            a = a.strip()
            if a:
                fields.append((_normalize(a), 95.0))
        for m in re.split(r"[|,;]", missp_raw):
            m = m.strip()
            if m:
                fields.append((_normalize(m), 88.0))

        best, best_field = 0.0, ""
        for f_norm, base in fields:
            s = _field_score(query_norm, f_norm, base)
            if s > best:
                best, best_field = s, f_norm

        if best >= 60.0:
            results[row_num] = {
                "canonical": full or alias_raw or first,
                "score": best,
                "reason": f"sheet field '{best_field}'",
                "first_name": first,
                "last_name": last,
                "org": org,
            }
    return results


# ── Signal 2 — context disambiguation ─────────────────────────────────────────

def _s2_boost(svc, candidates: dict, ctx_norm: str) -> None:
    """Mutate candidates in-place: read CA:CO for each row and boost by context."""
    if not ctx_norm:
        return

    for row_num, cand in candidates.items():
        try:
            resp = svc.spreadsheets().values().get(
                spreadsheetId=SHEET_ID,
                range=f"'{CONTACTS_TAB}'!CA{row_num}:CO{row_num}",
            ).execute()
            d = (resp.get("values", [[]])[0]) if resp.get("values") else []

            # Positions within CA:CO subrange
            proj    = _normalize(_cell(d, 0))   # CA
            land    = _normalize(_cell(d, 1))   # CB
            topic   = _normalize(_cell(d, 2))   # CC
            history = _normalize(_cell(d, 6))   # CG (84-78=6)
            score_s = _cell(d, 14)              # CO (92-78=14)

            boost = 0
            reason = ""

            if proj and ctx_norm in proj:
                boost += 30; reason = "project_association"
            elif proj and _fuzzy(ctx_norm, proj) >= 0.70:
                boost += int(30 * _fuzzy(ctx_norm, proj)); reason = "project_association~"

            if land and ctx_norm in land:
                boost += 15; reason = reason or "land_association"
            if topic and ctx_norm in topic:
                boost += 10; reason = reason or "topic_association"

            # History mentions
            if history:
                hits = history.count(ctx_norm)
                if hits:
                    boost += min(hits * 5, 15)
                    reason = reason or "conversation_history"

            # Contact score tiebreaker
            try:
                boost += min(int(float(score_s or 0)), 5)
            except (ValueError, TypeError):
                pass

            if boost:
                cand["score"] += boost
                cand["context_boost"] = boost
                cand["context_match"] = reason

        except Exception as e:
            logger.debug("S2 boost row %d: %s", row_num, e)


# ── Signal 3 — compound name (FirstName + CompanyAbbr) ────────────────────────

def _is_company_token(t: str) -> bool:
    s = re.sub(r"[^a-zA-Z0-9]", "", t)
    return bool(s) and ((s.isupper() and len(s) <= 6) or bool(re.search(r"\d", s)))


def _s3_compound(svc, query: str, query_norm: str) -> dict:
    """
    Handle queries like 'Priya TruBld' where second token is a company abbreviation.
    Matches Col A (first) + Col K (org).
    """
    tokens = query.split()
    if len(tokens) < 2:
        return {}

    rest = " ".join(tokens[1:])
    if not _is_company_token(tokens[-1]) and not _is_company_token(rest):
        return {}

    first_norm = _normalize(tokens[0])
    rest_norm  = _normalize(rest)

    try:
        rows = _read_range(svc, f"'{CONTACTS_TAB}'!A:K")
    except Exception as e:
        logger.warning("S3 sheet read failed: %s", e)
        return {}

    results: dict = {}
    for row_num, row in enumerate(rows[1:], start=2):
        first = _cell(row, COL_FIRST)
        last  = _cell(row, COL_LAST)
        org   = _cell(row, COL_ORG)
        if not first or not org:
            continue

        fs = _field_score(first_norm, _normalize(first), 100.0)
        if fs < 70:
            continue

        org_n = _normalize(org)
        if rest_norm in org_n:
            org_s = 80.0
        elif org_n in rest_norm:
            org_s = 75.0
        else:
            f = _fuzzy(rest_norm, org_n)
            org_s = f * 70.0 if f >= 0.65 else 0.0

        if org_s >= 50:
            combined = fs * 0.6 + org_s * 0.4
            full = f"{first} {last}".strip()
            results[row_num] = {
                "canonical": full,
                "score": combined,
                "reason": f"first_name+org('{org}')",
                "first_name": first,
                "last_name": last,
                "org": org,
            }
    return results


# ── Hydrate: add phones + emails to top candidates ─────────────────────────────

def _hydrate(svc, row_num: int, cand: dict, header: list) -> dict:
    """Read full row and extract phones/emails from header-matched columns."""
    cand["row"] = row_num
    try:
        resp = svc.spreadsheets().values().get(
            spreadsheetId=SHEET_ID,
            range=f"'{CONTACTS_TAB}'!A{row_num}:CO{row_num}",
        ).execute()
        row = (resp.get("values", [[]])[0]) if resp.get("values") else []

        phones, emails = [], []
        for i, h in enumerate(header):
            if i >= len(row):
                break
            val = str(row[i]).strip()
            if not val:
                continue
            hl = h.lower()
            if "phone" in hl and "type" not in hl:
                phones.append({"type": h, "value": _sanitize_phone_value(val)})
            elif "e-mail" in hl and "type" not in hl:
                emails.append({"type": h, "value": val})

        cand["phones"] = phones
        cand["emails"] = emails

        # Fill name/org if not already set by Signal 1/3
        if not cand.get("first_name"):
            cand["first_name"] = _cell(row, COL_FIRST)
        if not cand.get("last_name"):
            cand["last_name"] = _cell(row, COL_LAST)
        if not cand.get("org"):
            cand["org"] = _cell(row, COL_ORG)
        if not cand.get("canonical"):
            f = cand.get("first_name", "")
            l = cand.get("last_name", "")
            cand["canonical"] = f"{f} {l}".strip()

        cand["nickname"] = _cell(row, COL_NICK)
        cand["addressed_as"] = _cell(row, COL_ADDR_AS)
        cand["voice_misspellings"] = _cell(row, COL_MISSP)

    except Exception as e:
        logger.warning("Hydrate row %d failed: %s", row_num, e)
        cand.setdefault("phones", [])
        cand.setdefault("emails", [])
    return cand


# ── Main handler ───────────────────────────────────────────────────────────────

def _handle_contact_resolver(args: dict, **_) -> str:
    query         = (args.get("query") or "").strip()
    context       = (args.get("context") or "").strip()
    account_email = (args.get("account_email") or "").strip()

    if not query:
        return json.dumps({"error": "query is required"})

    query_norm  = _normalize(query)
    ctx_norm    = _normalize(context)

    try:
        svc = _build_svc(account_email or None)
    except Exception as e:
        return json.dumps({"error": f"Sheets connection failed: {e}"})

    # ── Signal 1 ──────────────────────────────────────────────────────────
    candidates = _s1_from_index(query_norm)
    if not candidates:
        candidates = _s1_from_sheet(svc, query_norm)

    # ── Signal 3 — compound name ──────────────────────────────────────────
    compound = _s3_compound(svc, query, query_norm)
    for row_num, c in compound.items():
        if row_num not in candidates or candidates[row_num]["score"] < c["score"]:
            candidates[row_num] = c

    if not candidates:
        return json.dumps({
            "candidates": [],
            "auto_selected": False,
            "total_matches": 0,
            "query": query,
            "context": context,
            "message": (
                f"No contacts found matching '{query}'. "
                "Check spelling, try an alias, or add as a voice misspelling via noun_learner."
            ),
        })

    # ── Signal 2 — context boost (top 10 only to limit API calls) ─────────
    if ctx_norm:
        top10_rows = sorted(candidates, key=lambda r: -candidates[r]["score"])[:10]
        subset = {r: candidates[r] for r in top10_rows}
        _s2_boost(svc, subset, ctx_norm)
        candidates.update(subset)

    # ── Sort and hydrate top 5 ─────────────────────────────────────────────
    ranked = sorted(candidates.items(), key=lambda x: -x[1]["score"])

    try:
        hdr_resp = svc.spreadsheets().values().get(
            spreadsheetId=SHEET_ID,
            range=f"'{CONTACTS_TAB}'!A1:CO1",
        ).execute()
        header = (hdr_resp.get("values", [[]])[0]) if hdr_resp.get("values") else []
    except Exception:
        header = []

    top = []
    for row_num, cand in ranked[:5]:
        top.append(_hydrate(svc, row_num, dict(cand), header))

    # ── Auto-select logic ──────────────────────────────────────────────────
    auto_selected = False
    if top:
        best_score = top[0]["score"]
        if best_score >= 90:
            if len(top) == 1:
                auto_selected = True
            elif best_score - top[1]["score"] >= 20:
                auto_selected = True

    return json.dumps({
        "candidates": top,
        "auto_selected": auto_selected,
        "total_matches": len(candidates),
        "query": query,
        "context": context,
        "message": (
            f"Found {len(top)} contact(s) matching '{query}'."
            + (" Best match selected — please confirm." if auto_selected
               else " Multiple matches — please choose one.")
            if len(top) > 1 or not auto_selected
            else f"Found {len(top)} contact(s) matching '{query}'. Best match selected — please confirm."
        ),
    }, indent=2)


# ── Schema ─────────────────────────────────────────────────────────────────────

_SCHEMA = {
    "name": "contact_resolver",
    "description": (
        "Resolve a contact name to ranked candidates from the Google Contacts Sheet. "
        "ALWAYS use this tool instead of manually reading the contacts sheet. "
        "Handles: exact names, partial names, voice transcription errors, aliases, "
        "and 'FirstName Company' compound queries (e.g. 'Priya TruBld'). "
        "Provide 'context' (project or entity name) to rank by association — contacts "
        "linked to that project score higher, resolving ambiguous first names. "
        "Returns top matches with phones, emails, org, nickname, addressed_as "
        "(how the user addresses this contact in salutations), and voice_misspellings "
        "— no further sheet reads needed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The contact name to search for. Examples: "
                    "'Bhuvanesh' (first name), 'Bhuvanesh Krishnan' (full name), "
                    "'RO' (alias), 'narsem raju' (voice error), 'Priya TruBld' (first+company)."
                ),
            },
            "context": {
                "type": "string",
                "description": (
                    "Optional: the project, entity, or land being discussed in this message. "
                    "Contacts associated with this context are ranked higher. "
                    "Example: 'Riverstone Farms' or 'Ranka Amber'. "
                    "Critical for resolving ambiguous first-name-only queries."
                ),
            },
            "account_email": {
                "type": "string",
                "description": (
                    "Optional: override which Google account's token to use "
                    "-- an email address (e.g. 'user@example.com') or short label "
                    "('draas'/'ahfl'/'gmail'). Defaults to the current session "
                    "user's own configured account."
                ),
            },
        },
        "required": ["query"],
    },
}


# ── Availability check ─────────────────────────────────────────────────────────

def _check_available() -> bool:
    try:
        import googleapiclient  # noqa: F401
        from google.oauth2.credentials import Credentials  # noqa: F401
        return True
    except ImportError:
        return False


# ── Registration ───────────────────────────────────────────────────────────────

from tools.registry import registry  # noqa: E402

registry.register(
    name="contact_resolver",
    toolset="general",
    schema=_SCHEMA,
    handler=_handle_contact_resolver,
    check_fn=_check_available,
    requires_env=[],
    is_async=False,
    description=_SCHEMA["description"],
    emoji="🔍",
)
