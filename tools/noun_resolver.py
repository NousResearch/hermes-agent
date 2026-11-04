#!/usr/bin/env python3
"""
Contextual Noun Resolver — 3-tier cache for voice message noun correction.

Architecture:
  Tier 1 — Thin name index (ALL contacts + all projects/entities/land/topics in memory)
            Only stores: normalized_variant → {row, canonical, type, sheet}
            ~500KB total, rebuilt on startup + after learner writes.

  Tier 2 — Hot record cache (top 50 contacts by contact_score, full records)
            Refreshed every 5 minutes in background thread.

  Tier 3 — Live sheet fetch (name resolved via Tier 1 but not in Tier 2)
            Single targeted API call for that row only.

Resolution pipeline:
  1. Tokenize input text into candidate noun phrases (1–3 word windows)
  2. Look up each candidate in Tier 1 index (exact → fuzzy → phonetic)
  3. Score matches; auto-replace if confidence ≥ 0.92, flag if 0.75–0.91, skip if <0.75
  4. Use session context (last 5 messages) to disambiguate ties

5 sheets: contacts, projects, land_proposals, entities, topics

Spreadsheet: https://docs.google.com/spreadsheets/d/1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g
"""

import json
import logging
import os
import re
import threading
import time
import unicodedata
from typing import Any

logger = logging.getLogger(__name__)

# ── Sheet config ───────────────────────────────────────────────────────────────

_SHEET_ID_FALLBACK = ""  # must come from user config; no global default

# Contacts sheet column indices (0-based)
COL_FIRST_NAME        = 0   # A
COL_MIDDLE_NAME       = 1   # B
COL_LAST_NAME         = 2   # C
COL_NICKNAME          = 8   # I
COL_ORG_NAME          = 10  # K
COL_ADDR_AS           = 82  # CE
COL_PEOPLE_ASSOC      = 83  # CF
COL_CONV_HISTORY      = 84  # CG  (old index before new cols)
COL_VOICE_MISSPELL    = 91  # CN  (new)
COL_CONTACT_SCORE     = 92  # CO  (new)

# Non-contacts sheets: all have canonical_name(A), nicknames/aliases(B), voice_misspellings(C)
COL_CANONICAL   = 0
COL_ALIASES     = 1   # B — maps to the "nicknames" column in non-contacts sheets
COL_MISSPELL    = 2

SHEET_CONFIGS = {
    "contacts": {
        "range":    "NDR DRAAS Google contacts.csv!A:CO",
        "name_col": None,       # special: built from first+last
        "alias_col": COL_ADDR_AS,
        "misspell_col": COL_VOICE_MISSPELL,
        "score_col": COL_CONTACT_SCORE,
    },
    "projects": {
        "range":    "projects!A:I",
        "name_col": COL_CANONICAL,
        "alias_col": COL_ALIASES,
        "misspell_col": COL_MISSPELL,
        "score_col": None,
    },
    "land_proposals": {
        "range":    "land_proposals!A:J",
        "name_col": COL_CANONICAL,
        "alias_col": COL_ALIASES,
        "misspell_col": COL_MISSPELL,
        "score_col": None,
    },
    "entities": {
        "range":    "entities!A:H",
        "name_col": COL_CANONICAL,
        "alias_col": COL_ALIASES,
        "misspell_col": COL_MISSPELL,
        "score_col": None,
    },
    "topics": {
        "range":    "topics!A:J",
        "name_col": COL_CANONICAL,
        "alias_col": COL_ALIASES,
        "misspell_col": COL_MISSPELL,
        "score_col": None,
    },
}

# Confidence thresholds
THRESH_AUTO    = 0.92   # auto-replace silently
THRESH_MENTION = 0.75   # replace but tell user
THRESH_SEARCH  = 0.65   # lower bar for agent search() — phonetic hits score ~0.70–0.85

# Hot cache size
HOT_CACHE_SIZE = 50

# Refresh interval for hot cache (seconds)
HOT_REFRESH_INTERVAL = 300

# ── Module-level singleton ─────────────────────────────────────────────────────

_resolver: "NounResolver | None" = None
_resolver_lock = threading.Lock()


def get_resolver() -> "NounResolver":
    global _resolver
    with _resolver_lock:
        if _resolver is None:
            _resolver = NounResolver()
            _resolver.start()
        return _resolver


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    nfkd = unicodedata.normalize("NFKD", text or "")
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_str).strip().lower()


def _get_session_user_cfg() -> dict:
    """Return the user config dict for the current session user, or {} if unknown."""
    try:
        from gateway.session_context import get_gws_identity_env
        from tools._user_registry import get_user_config
        session_uid = get_gws_identity_env()
        return get_user_config(session_uid) or {}
    except Exception:
        return {}


# Single-tenant interim default -- the only Hermes user with this registry
# sheet configured today. Used ONLY to establish ambient session context
# (via the sanctioned set_session_vars()/clear_session_vars() mechanism --
# never passed directly to gws_auth, which refuses caller-supplied ids by
# design) for background work that has no session at all: the Tier-2 hot
# cache refresh thread, and the bare thread force_refresh() spawns (plain
# threading.Thread does not inherit contextvars from its parent, so even a
# cron-triggered rebuild loses HERMES_CRON_JOB_OWNER_ID once it hits a new
# thread -- see _run_with_identity_fallback below).
# TODO(multi-tenant): replace with a Postgres-backed per-user registry and
# propagate each user's own owner id into their rebuild instead of this
# single default.
_DEFAULT_OWNER_USER_ID = "ndr-7449813913"


def _run_with_identity_fallback(fn, *, inherited_context=None):
    """Run ``fn()`` with session identity guaranteed to be set.

    If real session/cron-owner context is already present (either ambiently,
    or copied in via ``inherited_context``), it is left untouched. Only when
    there is truly no identity anywhere does this establish the single-tenant
    default via the same ``set_session_vars`` mechanism a live chat uses --
    never by passing an id directly to gws_auth.
    """
    import contextvars
    from gateway.session_context import (
        get_gws_identity_env,
        set_session_vars,
        clear_session_vars,
    )

    def _runner():
        if get_gws_identity_env():
            fn()
            return
        tokens = set_session_vars(user_id=_DEFAULT_OWNER_USER_ID)
        try:
            fn()
        finally:
            clear_session_vars(tokens)

    if inherited_context is not None:
        inherited_context.run(_runner)
    else:
        _runner()


def _get_sheet_id() -> str:
    """Return the contacts spreadsheet ID for the current session user."""
    cfg = _get_session_user_cfg()
    sheet_id = cfg.get("contacts_sheet_id", "")
    if not sheet_id:
        uid = cfg.get("draas_user_id", "unknown")
        raise RuntimeError(
            f"User {uid!r} has no contacts_sheet_id configured in their profile"
        )
    return sheet_id


def _build_service():
    """Return a Google Sheets API service for the current session user.

    Delegates to ``gws_auth.build_service`` so the token is keyed by the
    canonical vault user_id (resolved from the session id) and loaded as a
    refresh-capable Credentials object.  No credentials pass through the LLM.
    """
    from tools import gws_auth
    from gateway.session_context import get_gws_identity_env

    cfg = _get_session_user_cfg()
    gws_service = cfg.get("gws_service", "")
    # get_gws_identity_env() covers live sessions + cron-owner fallback
    # (this was previously a raw get_session_env() read, which missed the
    # cron-owner case the same way _get_session_user_cfg() used to).
    session_uid = get_gws_identity_env()

    if not session_uid:
        raise RuntimeError(
            "Cannot identify session user for Google Sheets access. "
            "No live session or cron job owner context available."
        )
    if not gws_service:
        raise RuntimeError(
            f"User {session_uid!r} has no gws_service configured in their profile. "
            "An admin must add the 'gws_service' field and ensure the user has authorized."
        )

    return gws_auth.build_service("sheets", "v4", service_name=gws_service)


def _fuzzy_score(query: str, candidate: str) -> float:
    """Return similarity 0–1. Uses both token_sort_ratio and Jaro-Winkler (max of both)."""
    try:
        from rapidfuzz import fuzz
        from rapidfuzz.distance import JaroWinkler
        token_score = fuzz.token_sort_ratio(query, candidate) / 100.0
        jw_score = JaroWinkler.normalized_similarity(query, candidate)
        return max(token_score, jw_score)
    except ImportError:
        # Fallback: character overlap ratio
        q, c = set(query), set(candidate)
        if not q and not c:
            return 1.0
        return len(q & c) / max(len(q | c), 1)


def _phonetic(text: str) -> str:
    """
    Return Soundex code for the first token of text.

    Soundex produces 4-char fixed-width codes that align well across voice
    transcription errors (RAGOO=RAGHU=R200, NARSEM=NARASIMHA=N625).
    Falls back to first-4-consonants if jellyfish is missing.
    """
    if not text or len(text) < 2:
        return ""
    first_word = text.split()[0]
    try:
        import jellyfish
        return jellyfish.soundex(first_word)
    except ImportError:
        t = re.sub(r"[aeiou]", "", first_word.lower())
        return re.sub(r"(.)\1+", r"\1", t)[:4]


# Associated-contacts column index per non-contacts sheet (0-based)
_ASSOC_COL = {
    "projects":       3,   # D
    "entities":       4,   # E
    "land_proposals": 5,   # F
    "topics":         5,   # F
}


# ── NounResolver class ─────────────────────────────────────────────────────────

class NounResolver:
    """
    Thread-safe noun resolver with 3-tier cache.

    Usage:
        resolver = get_resolver()
        result = resolver.resolve("meeting with drast about ranka amba project")
        # result.corrected_text, result.substitutions, result.needs_confirmation
    """

    def __init__(self):
        # Tier 1: normalized_variant → {canonical, type, sheet, row}
        self._index: dict[str, dict] = {}
        self._index_lock = threading.RLock()
        self._index_built = False

        # Phonetic index: metaphone_code → [(variant, entry), ...]
        self._phonetic_index: dict[str, list] = {}

        # Tier 2: canonical_name → full row dict (top HOT_CACHE_SIZE by score)
        self._hot_cache: dict[str, dict] = {}
        self._hot_lock = threading.RLock()

        # Background refresh thread
        self._stop_event = threading.Event()
        self._refresh_thread: threading.Thread | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self):
        """Build initial index and start background refresh."""
        try:
            _run_with_identity_fallback(self._rebuild_index)
            _run_with_identity_fallback(self._refresh_hot_cache)
        except Exception as e:
            logger.warning(f"NounResolver: initial load failed ({e}) — will retry in background")

        self._refresh_thread = threading.Thread(
            target=self._background_refresh, daemon=True, name="noun-resolver-refresh"
        )
        self._refresh_thread.start()
        logger.info(f"NounResolver started: {len(self._index)} index entries, {len(self._hot_cache)} hot records")

    def stop(self):
        self._stop_event.set()

    def force_refresh(self):
        """Call this after the learner writes new data to the sheet."""
        import contextvars
        ctx = contextvars.copy_context()
        threading.Thread(
            target=lambda: _run_with_identity_fallback(self._rebuild_index, inherited_context=ctx),
            daemon=True,
        ).start()

    # ── Index building ─────────────────────────────────────────────────────────

    def _rebuild_index(self):
        """Rebuild the full thin name index from all 5 sheets."""
        try:
            svc = _build_service()
            new_index: dict[str, dict] = {}
            new_phonetic: dict[str, list] = {}

            for sheet_name, cfg in SHEET_CONFIGS.items():
                try:
                    resp = svc.spreadsheets().values().get(
                        spreadsheetId=_get_sheet_id(),
                        range=cfg["range"],
                    ).execute()
                    rows = resp.get("values", [])
                    if len(rows) < 2:
                        continue

                    for row_idx, row in enumerate(rows[1:], start=2):  # skip header
                        canonical = self._extract_canonical(sheet_name, cfg, row)
                        if not canonical:
                            continue

                        # Extract associated_contacts for non-contact sheets
                        assoc_col = _ASSOC_COL.get(sheet_name, -1)
                        assoc = row[assoc_col].strip() if assoc_col >= 0 and assoc_col < len(row) else ""

                        entries = self._extract_variants(sheet_name, cfg, row, canonical)
                        for variant, confidence in entries:
                            key = _normalize(variant)
                            if not key:
                                continue
                            entry = {
                                "canonical": canonical,
                                "type": sheet_name,
                                "sheet": sheet_name,
                                "row": row_idx,
                                "confidence": confidence,
                                "assoc": assoc,
                            }
                            # Main index: keep highest-confidence entry per key
                            existing = new_index.get(key)
                            if existing is None or existing["confidence"] < confidence:
                                new_index[key] = entry
                            # Phonetic index: accumulate all entries per code
                            ph = _phonetic(key)
                            if ph:
                                if ph not in new_phonetic:
                                    new_phonetic[ph] = []
                                new_phonetic[ph].append((key, entry))

                except Exception as e:
                    logger.warning(f"NounResolver: failed loading sheet '{sheet_name}': {e}")

            with self._index_lock:
                self._index = new_index
                self._phonetic_index = new_phonetic
                self._index_built = True

            logger.info(f"NounResolver: index rebuilt with {len(new_index)} entries, {len(new_phonetic)} phonetic codes")

        except Exception as e:
            logger.error(f"NounResolver: _rebuild_index failed: {e}")

    def _extract_canonical(self, sheet_name: str, cfg: dict, row: list) -> str:
        """Extract the canonical name for a row."""
        def _get(idx):
            return row[idx].strip() if idx < len(row) else ""

        if sheet_name == "contacts":
            first = _get(COL_FIRST_NAME)
            last  = _get(COL_LAST_NAME)
            nick  = _get(COL_NICKNAME)
            org   = _get(COL_ORG_NAME)
            name  = f"{first} {last}".strip()
            return name or nick or org
        else:
            return _get(cfg["name_col"])

    def _extract_variants(self, sheet_name: str, cfg: dict, row: list, canonical: str) -> list[tuple[str, float]]:
        """Return list of (variant_text, confidence) tuples for this row."""
        def _get(idx):
            return row[idx].strip() if idx < len(row) else ""

        variants = [(canonical, 1.0)]

        # Nickname / first name alone for contacts
        if sheet_name == "contacts":
            first = _get(COL_FIRST_NAME)
            nick  = _get(COL_NICKNAME)
            if first:
                variants.append((first, 0.85))
            if nick:
                variants.append((nick, 0.95))

        # Aliases (pipe or comma separated)
        alias_raw = _get(cfg["alias_col"])
        if alias_raw:
            for a in re.split(r"[|,;]", alias_raw):
                a = a.strip()
                if a:
                    variants.append((a, 1.0))

        # Voice misspellings (pipe or comma separated)
        misspell_raw = _get(cfg["misspell_col"])
        if misspell_raw:
            for m in re.split(r"[|,;]", misspell_raw):
                m = m.strip()
                if m:
                    variants.append((m, 0.95))

        return variants

    # ── Hot cache ──────────────────────────────────────────────────────────────

    def _refresh_hot_cache(self):
        """Fetch top HOT_CACHE_SIZE contacts by contact_score."""
        try:
            svc = _build_service()
            resp = svc.spreadsheets().values().get(
                spreadsheetId=_get_sheet_id(),
                range="NDR DRAAS Google contacts.csv!A:CO",
            ).execute()
            rows = resp.get("values", [])
            if len(rows) < 2:
                return

            # Score each row
            scored = []
            for row_idx, row in enumerate(rows[1:], start=2):
                score_raw = row[COL_CONTACT_SCORE] if COL_CONTACT_SCORE < len(row) else ""
                try:
                    score = float(score_raw) if score_raw else 0.0
                except ValueError:
                    score = 0.0
                if score > 0:
                    first = row[COL_FIRST_NAME].strip() if COL_FIRST_NAME < len(row) else ""
                    last  = row[COL_LAST_NAME].strip() if COL_LAST_NAME < len(row) else ""
                    name  = f"{first} {last}".strip()
                    if name:
                        scored.append((score, name, row_idx, row))

            scored.sort(reverse=True)
            hot = {}
            for _, name, row_idx, row in scored[:HOT_CACHE_SIZE]:
                hot[name] = {"row": row_idx, "data": row, "type": "contacts"}

            with self._hot_lock:
                self._hot_cache = hot

        except Exception as e:
            logger.warning(f"NounResolver: hot cache refresh failed: {e}")

    # ── Background thread ──────────────────────────────────────────────────────

    def _background_refresh(self):
        while not self._stop_event.wait(HOT_REFRESH_INTERVAL):
            _run_with_identity_fallback(self._refresh_hot_cache)

    # ── Resolution ─────────────────────────────────────────────────────────────

    def resolve(
        self,
        text: str,
        session_context: list[str] | None = None,
        optimistic_mode: bool = False,
    ) -> "ResolveResult":
        """
        Resolve nouns in text.

        Args:
            text: The text to resolve
            session_context: Recent message history for disambiguation
            optimistic_mode: If True, apply ALL matches (high + mid confidence) to proceed
                           without blocking on ambiguity. Default: False (block on ambiguity)

        Returns ResolveResult with:
          .corrected_text   — text with substitutions applied
          .substitutions    — list of {original, canonical, type, confidence}
          .needs_confirmation — list of ambiguous matches (empty if optimistic_mode=True)
        """
        if not self._index_built or not text:
            return ResolveResult(text, [], [])

        substitutions = []
        needs_confirmation = []
        corrected = text

        # Generate candidate phrases (1, 2, 3 word windows)
        candidates = self._candidate_phrases(text)

        # Track which character spans have been resolved (avoid double-substitution)
        resolved_spans: list[tuple[int, int]] = []

        for phrase, start, end in candidates:
            # Skip if overlaps an already-resolved span
            if any(s <= start < e or s < end <= e for s, e in resolved_spans):
                continue

            match = self._lookup(phrase, session_context)
            if not match:
                continue

            conf = match["confidence"]
            canonical = match["canonical"]
            norm_phrase = _normalize(phrase)
            norm_canonical = _normalize(canonical)

            # Skip if phrase already matches canonical (no correction needed)
            if norm_phrase == norm_canonical:
                continue

            # ENHANCED: Confidence thresholds based on mode
            auto_threshold = THRESH_AUTO if not optimistic_mode else THRESH_MENTION
            mention_threshold = THRESH_MENTION if not optimistic_mode else 0.0

            if conf >= auto_threshold:
                # Apply correction to text (auto-replace)
                corrected = corrected[:start] + canonical + corrected[end:]
                # Adjust subsequent positions
                delta = len(canonical) - len(phrase)
                resolved_spans = [(s + delta if s > start else s, e + delta if e > start else e)
                                  for s, e in resolved_spans]
                resolved_spans.append((start, start + len(canonical)))
                substitutions.append({
                    "original": phrase, "canonical": canonical,
                    "type": match["type"], "confidence": round(conf, 3),
                    "row": match.get("row"),
                })
            elif conf >= mention_threshold and not optimistic_mode:
                # Only flag for confirmation if NOT in optimistic mode
                needs_confirmation.append({
                    "original": phrase,
                    "candidates": [{"canonical": canonical, "type": match["type"], "confidence": round(conf, 3)}],
                })

        return ResolveResult(corrected, substitutions, needs_confirmation)

    def _candidate_phrases(self, text: str) -> list[tuple[str, int, int]]:
        """Return (phrase, start, end) for all 1–3 word windows."""
        words = list(re.finditer(r"\S+", text))
        result = []
        for n in (3, 2, 1):
            for i in range(len(words) - n + 1):
                group = words[i:i + n]
                start = group[0].start()
                end   = group[-1].end()
                phrase = text[start:end]
                result.append((phrase, start, end))
        return result

    def _lookup(self, phrase: str, context: list[str] | None = None) -> dict | None:
        """Look up a phrase in the index. Returns best match dict or None."""
        key = _normalize(phrase)
        if not key or len(key) < 2:
            return None

        with self._index_lock:
            # Exact match
            if key in self._index:
                return self._index[key]

            # Fuzzy match — scan index keys with sufficient prefix overlap
            best_score = 0.0
            best_entry = None
            for idx_key, entry in self._index.items():
                # Quick pre-filter: first char must match to avoid full scan cost
                if not idx_key or idx_key[0] != key[0]:
                    continue
                score = _fuzzy_score(key, idx_key) * entry["confidence"]
                if score > best_score:
                    best_score = score
                    best_entry = entry

            # Phonetic fallback — catches voice errors where Soundex codes match
            # (e.g. "ragoo"→"Raghu" R200=R200, "narsem"→"Narasimha" N625=N625)
            if best_score < THRESH_MENTION:
                ph = _phonetic(key)
                if ph and ph in self._phonetic_index:
                    for ph_variant, ph_entry in self._phonetic_index[ph]:
                        # Compare against both full variant and its first token
                        first_tok = ph_variant.split()[0]
                        sim = max(_fuzzy_score(key, ph_variant), _fuzzy_score(key, first_tok))
                        ph_score = sim * 0.90 * ph_entry["confidence"]
                        if ph_score > best_score:
                            best_score = ph_score
                            best_entry = ph_entry

            if best_entry and best_score >= THRESH_MENTION:
                # Session context boost: if canonical appears in recent messages, +0.03
                if context and best_score >= THRESH_MENTION:
                    _ctx_words = set()
                    for _msg in context:
                        _words = _normalize(_msg).split()
                        _ctx_words.update(_words)
                        _ctx_words.update(" ".join(_words[i:i+2]) for i in range(len(_words) - 1))
                        _ctx_words.update(" ".join(_words[i:i+3]) for i in range(len(_words) - 2))
                    if _normalize(best_entry.get("canonical", "")) in _ctx_words:
                        best_score = min(1.0, best_score + 0.03)

                result = dict(best_entry)
                result["confidence"] = best_score
                return result

        return None

    def search(
        self,
        query: str,
        limit: int = 10,
        types: list[str] | None = None,
        context: list[str] | None = None,
    ) -> list[dict]:
        """
        Search for entities by name. Returns top-N ranked matches across all sheets.

        Unlike resolve() (which corrects text in-place), search() is for agent-driven
        explicit lookups. Splits multi-word queries into 1–3 word windows and searches
        each using exact → fuzzy → phonetic matching.

        Args:
            query:   Name or phrase (e.g. "ragoo", "amber project", "narsem raju")
            limit:   Max results to return (default 10)
            types:   Filter by sheet: contacts, projects, entities, land_proposals, topics
            context: Recent message history — matching canonicals get +0.05 score boost

        Returns:
            List of {type, canonical, row, score, match_type, [associated_contacts]}
            sorted by score descending.
        """
        if not self._index_built:
            return []

        norm_query = _normalize(query)
        if not norm_query:
            return []

        # Generate 1–3 word windows (longest first to prefer multi-word matches)
        words = norm_query.split()
        phrases: list[str] = []
        seen_phrases: set[str] = set()
        for n in (3, 2, 1):
            for i in range(len(words) - n + 1):
                ph = " ".join(words[i:i + n])
                if ph not in seen_phrases and len(ph) >= 2:
                    phrases.append(ph)
                    seen_phrases.add(ph)

        best: dict[str, dict] = {}  # canonical → best hit

        with self._index_lock:
            for phrase in phrases:
                ph_code = _phonetic(phrase)

                # ── Exact + fuzzy scan ──────────────────────────────────────
                for idx_key, entry in self._index.items():
                    if types and entry["type"] not in types:
                        continue

                    if idx_key == phrase:
                        score, mtype = entry["confidence"], "exact"
                    elif idx_key and idx_key[0] == phrase[0]:
                        s = _fuzzy_score(phrase, idx_key) * entry["confidence"]
                        if s < THRESH_SEARCH:
                            continue
                        score, mtype = s, "fuzzy"
                    else:
                        continue

                    canonical = entry["canonical"]
                    existing = best.get(canonical)
                    if existing is None or existing["score"] < score:
                        best[canonical] = {
                            "type":       entry["type"],
                            "canonical":  canonical,
                            "row":        entry["row"],
                            "score":      round(score, 3),
                            "match_type": mtype,
                            "_assoc":     entry.get("assoc", ""),
                        }

                # ── Phonetic scan (Soundex bucket) — voice errors, spelling variants ──
                if ph_code and ph_code in self._phonetic_index:
                    for ph_variant, ph_entry in self._phonetic_index[ph_code]:
                        if types and ph_entry["type"] not in types:
                            continue
                        # Compare against both full variant and its first token
                        first_tok = ph_variant.split()[0]
                        sim = max(_fuzzy_score(phrase, ph_variant), _fuzzy_score(phrase, first_tok))
                        ph_score = sim * 0.90
                        if ph_score < THRESH_SEARCH:
                            continue
                        canonical = ph_entry["canonical"]
                        existing = best.get(canonical)
                        if existing is None or existing["score"] < ph_score:
                            best[canonical] = {
                                "type":       ph_entry["type"],
                                "canonical":  canonical,
                                "row":        ph_entry["row"],
                                "score":      round(ph_score, 3),
                                "match_type": "phonetic",
                                "_assoc":     ph_entry.get("assoc", ""),
                            }

        # ── Context boost (+0.05 for canonicals mentioned in recent messages) ──
        if context:
            ctx_words: set[str] = set()
            for msg in context:
                ws = _normalize(msg).split()
                ctx_words.update(ws)
                ctx_words.update(" ".join(ws[i:i + 2]) for i in range(len(ws) - 1))
            for hit in best.values():
                if _normalize(hit["canonical"]) in ctx_words:
                    hit["score"] = min(1.0, hit["score"] + 0.05)

        # ── Sort, format, and attach associated_contacts ─────────────────────
        ranked = sorted(best.values(), key=lambda x: -x["score"])[:limit]
        output = []
        for hit in ranked:
            result = {k: v for k, v in hit.items() if not k.startswith("_")}
            assoc_raw = hit.get("_assoc", "")
            if assoc_raw and hit["type"] in ("projects", "entities", "land_proposals", "topics"):
                result["associated_contacts"] = [
                    c.strip() for c in re.split(r"[|,;]", assoc_raw) if c.strip()
                ]
            output.append(result)

        return output

    def get_full_record(self, canonical: str, sheet: str, row: int) -> dict | None:
        """
        Get full record. Checks hot cache first, falls back to live API call.
        """
        # Tier 2: hot cache
        with self._hot_lock:
            if canonical in self._hot_cache:
                return self._hot_cache[canonical]

        # Tier 3: live fetch
        try:
            cfg = SHEET_CONFIGS.get(sheet)
            if not cfg:
                return None
            svc = _build_service()
            col_end = cfg["range"].split("!")[-1].split(":")[1]
            resp = svc.spreadsheets().values().get(
                spreadsheetId=_get_sheet_id(),
                range=f"'{sheet}'!A{row}:{col_end}{row}",
            ).execute()
            rows = resp.get("values", [])
            if rows:
                return {"row": row, "data": rows[0], "type": sheet}
        except Exception as e:
            logger.warning(f"NounResolver: live fetch failed for row {row} in {sheet}: {e}")
        return None

    def increment_contact_score(self, row: int, amount: int = 1):
        """Increment contact_score for a contacts row. Batched externally."""
        try:
            svc = _build_service()
            col = "CO"  # contact_score column
            resp = svc.spreadsheets().values().get(
                spreadsheetId=_get_sheet_id(),
                range=f"'NDR DRAAS Google contacts.csv'!{col}{row}",
            ).execute()
            current = 0
            vals = resp.get("values", [])
            if vals and vals[0]:
                try:
                    current = int(vals[0][0])
                except ValueError:
                    pass
            svc.spreadsheets().values().update(
                spreadsheetId=_get_sheet_id(),
                range=f"'NDR DRAAS Google contacts.csv'!{col}{row}",
                valueInputOption="RAW",
                body={"values": [[str(current + amount)]]},
            ).execute()
        except Exception as e:
            logger.warning(f"NounResolver: increment_contact_score failed: {e}")


# ── Result type ────────────────────────────────────────────────────────────────

class ResolveResult:
    def __init__(self, corrected_text: str, substitutions: list, needs_confirmation: list):
        self.corrected_text    = corrected_text
        self.substitutions     = substitutions       # auto-replaced
        self.needs_confirmation = needs_confirmation  # ambiguous, needs user input

    def has_changes(self) -> bool:
        return bool(self.substitutions or self.needs_confirmation)

    def summary(self) -> str:
        """Human-readable summary of changes made."""
        lines = []
        for s in self.substitutions:
            lines.append(f"'{s['original']}' → '{s['canonical']}' ({s['type']}, {s['confidence']:.0%} confident)")
        return "\n".join(lines) if lines else ""

    def to_dict(self) -> dict:
        return {
            "corrected_text":     self.corrected_text,
            "substitutions":      self.substitutions,
            "needs_confirmation": self.needs_confirmation,
        }
