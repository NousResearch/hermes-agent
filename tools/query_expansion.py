"""Bounded lexical query expansion for conversational FTS5 searches.

Multi-word FTS5 queries AND every word, so a conversational recall prompt
("what did we decide about the postgres pool") only matches messages that
also carry its filler words. The store's OR-relaxed retry covers the case
where the strict query finds NOTHING; this module covers the thin case,
where one off-topic message happens to contain every word and the real
session (which lacks "decide" or "we") never surfaces.

It extracts the meaningful keywords so the caller can run ONE bounded
supplemental ``kw1 OR kw2 ...`` probe whose rows rank below every strict
row. Constraints:
- At most ``MAX_EXPANSION_TERMS`` keywords: a long prompt cannot fan out
  into unbounded sqlite work.
- Explicit FTS5 syntax (quoted phrases, OR/NOT/AND, prefix wildcards) is
  the caller being precise and is never supplemented.
- When every token is already a keyword the probe would only dilute the
  strict AND result, so ``expansion_is_noop`` skips it.
"""

from __future__ import annotations

import re

# Cap on extracted keywords (bounds the probe's sqlite work).
MAX_EXPANSION_TERMS = 6

# Minimum keyword length (codepoints). Single characters are noise for
# unicode61 tokenization; CJK terms are exempted below since a 2-char
# CJK token is a full word.
_MIN_KEYWORD_LEN = 2

_CJK_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]"
)

# Explicit FTS5 query syntax: quoted phrase, boolean operators (upper
# case, FTS5's requirement), or a prefix wildcard. Presence of any of
# these means the user is writing a precise query — leave it alone.
_FTS_OPERATOR_RE = re.compile(r'"|\*|\b(?:OR|NOT|AND)\b')

# Filler vocabulary that carries no lexical search value. Grouped by
# function; deliberately conservative — a false keep costs one extra OR
# term, a false drop can hide the only meaningful term in the query.
_STOP_WORDS = frozenset(
    # articles / determiners
    "a an the this that these those some any each every".split()
    # pronouns
    + "i me my mine we us our you your yours he him his she her it its "
      "they them their there".split()
    # auxiliaries / common verbs
    + "is are was were be been being am have has had having do does did "
      "done will would could should shall can may might must".split()
    # prepositions / conjunctions
    + "in on at to for of with by from about into through during before not no "
      "after between under over and or but if then because so as while "
      "when where".split()
    # question / request words
    + "what which who whom whose how why please help find show get tell "
      "give look search remember recall".split()
    # vague time references — useless for lexical match
    + "yesterday today tomorrow earlier later recently ago just now once "
      "last previous".split()
    # vague nouns
    + "thing things stuff something anything everything nothing one ones "
      "way time".split()
    # conversational glue
    + "we're i'm it's that's did we you know like really actually kind "
      "sort mentioned discussed talked said told chat conversation "
      "session talking".split()
)

_TOKEN_RE = re.compile(r"[^\s]+")
_NON_WORD_RE = re.compile(r"\W")
# Strip leading/trailing punctuation from a token but keep interior
# hyphens/dots/underscores (identifiers like chat-send, app.config).
_EDGE_PUNCT_RE = re.compile(r"^[^\w\u0080-\uffff]+|[^\w\u0080-\uffff]+$")


def has_fts_operators(query: str) -> bool:
    """Return True when *query* uses explicit FTS5 syntax.

    Such queries express user intent precisely and must never be
    rewritten or supplemented by lexical expansion.
    """
    return bool(_FTS_OPERATOR_RE.search(query or ""))


def extract_keywords(query: str, max_terms: int = MAX_EXPANSION_TERMS) -> list[str]:
    """Extract up to *max_terms* meaningful keywords from a query.

    Tokenizes on whitespace, strips edge punctuation, drops stop words
    and too-short tokens (CJK tokens are exempt from the length floor —
    two CJK characters form a full word). Order of first appearance is
    preserved; duplicates are removed case-insensitively.
    """
    if not query:
        return []
    keywords: list[str] = []
    seen = set()
    for match in _TOKEN_RE.finditer(query):
        token = _EDGE_PUNCT_RE.sub("", match.group(0))
        if not token:
            continue
        lowered = token.lower()
        if lowered in _STOP_WORDS:
            continue
        if len(token) < _MIN_KEYWORD_LEN and not _CJK_RE.search(token):
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        keywords.append(token)
        if len(keywords) >= max_terms:
            break
    return keywords


def expansion_is_noop(query: str, keywords: list[str]) -> bool:
    """Return True when expansion did not narrow the query.

    If every token of the original query survived extraction, the OR
    probe covers the same terms as the strict AND query — it can only
    weaken relevance, never add meaningful recall beyond what a broader
    ranking would surface.
    """
    original_tokens = set()
    for match in _TOKEN_RE.finditer(query or ""):
        token = _EDGE_PUNCT_RE.sub("", match.group(0))
        if token:
            original_tokens.add(token.lower())
    return original_tokens == {k.lower() for k in keywords}


def build_expansion_query(keywords: list[str]) -> str:
    """Join keywords into a single FTS5 OR probe query. A keyword carrying punctuation
    (``HOME/.config/hermes``) is quoted so FTS5 reads it as a phrase, not as syntax: the
    sanitizer alone leaves fragments such as a bare ``.config`` that FTS5 rejects. CJK
    keywords stay bare because the CJK route re-quotes its own tokens."""
    return " OR ".join(f'"{k}"' if _NON_WORD_RE.search(k) and not _CJK_RE.search(k) else k for k in keywords)
