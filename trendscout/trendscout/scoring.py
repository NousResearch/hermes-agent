#!/usr/bin/env python3
"""
Emergence scoring for trendscout.

Two independent signals:
  1. Post engagement velocity — score/comments growth per hour since posting.
     Surfaces young posts with steep slopes (already-trending-but-fresh).
  2. Term frequency acceleration — noun phrases extracted from post titles
     and bodies, tracked daily. A term whose frequency is low in absolute
     terms but whose day-over-day frequency *change* is itself increasing
     (positive 2nd derivative) is in the "pre-emergence zone".
"""

import re
from datetime import datetime, timedelta, timezone

# Lazy import — raised with clear error if spacy extra not installed
def _get_spacy():
    try:
        import spacy
        return spacy
    except ImportError as e:
        raise ImportError(
            "spacy is required for term extraction. Install with: "
            "pip install trendscout[spacy]  OR  pip install spacy>=3.7\n"
            "Then download the model: python -m spacy download en_core_web_sm"
        ) from e

_NLP = None

# Generic / low-signal phrases that survive noun-chunk extraction but carry
# no topical meaning on their own.
STOPWORDS_PHRASES = {
    'it', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they',
    'what', 'who', 'something', 'someone', 'anyone', 'everyone', 'nothing',
    'today', 'yesterday', 'tomorrow', 'people', 'thing', 'things', 'way', 'lot',
}


def _nlp():
    global _NLP
    if _NLP is None:
        spacy = _get_spacy()
        _NLP = spacy.load('en_core_web_sm', disable=['ner', 'lemmatizer'])
    return _NLP


def extract_terms(text: str) -> list[str]:
    """Extract normalized noun-phrase terms from a block of text."""
    if not text:
        return []
    # spaCy chokes on huge selftexts and we only need the gist.
    doc = _nlp()(text[:5000])
    terms = []
    for chunk in doc.noun_chunks:
        term = re.sub(r'\s+', ' ', chunk.text.strip().lower())
        term = term.strip('"\'.,!?()[]')
        if not term or term in STOPWORDS_PHRASES:
            continue
        if len(term) < 3 or len(term) > 60:
            continue
        if not re.search(r'[a-z]', term):
            continue
        terms.append(term)
    return terms


# ── Term frequency / velocity ───────────────────────────────────────────────

def record_post_terms(conn, posts: list[dict], date: str):
    """Record extracted terms for each post. Idempotent via INSERT OR IGNORE.
    
    Each (post_id, term, date) combination is stored once. This replaces the
    old incrementing counter approach with a recomputed aggregate.
    """
    for post in posts:
        post_id = post['id']
        text = f"{post.get('title', '')}. {post.get('selftext', '') or ''}"
        terms = set(extract_terms(text))
        for term in terms:
            conn.execute(
                'INSERT OR IGNORE INTO post_terms (post_id, term, date) VALUES (?,?,?)',
                (post_id, term, date)
            )


def record_social_terms(conn, terms: list[str], date: str):
    """Record social trending terms (hashtags, etc.) for a given date.
    
    Idempotent via INSERT OR IGNORE — replaces update_term_frequency_raw.
    """
    for term in terms:
        if term:
            conn.execute(
                'INSERT OR IGNORE INTO social_terms (date, term) VALUES (?,?)',
                (date, term)
            )


def recompute_term_frequency(conn, date: str):
    """Recompute term_frequency as an aggregate from post_terms + social_terms.
    
    This is idempotent: deletes today's rows and rebuilds from source tables.
    Call this after record_post_terms and record_social_terms.
    """
    # Delete existing rows for this date
    conn.execute('DELETE FROM term_frequency WHERE date=?', (date,))
    
    # Recompute from both sources
    conn.execute("""
        INSERT INTO term_frequency (date, term, frequency)
        SELECT ?, term, COUNT(*) FROM (
            SELECT term FROM post_terms WHERE date=?
            UNION ALL
            SELECT term FROM social_terms WHERE date=?
        )
        GROUP BY term
    """, (date, date, date))


def compute_term_velocity(conn, date: str, config: dict):
    """
    Compute velocity (1st derivative) and acceleration (2nd derivative) of
    term frequency for `date`, relative to the prior two days. Flags terms
    whose acceleration exceeds min_acceleration while still below
    max_absolute_frequency (the pre-emergence zone).
    """
    terms_cfg = config['terms']
    min_accel = terms_cfg['min_acceleration']
    max_abs_freq = terms_cfg['max_absolute_frequency']

    d0 = datetime.strptime(date, '%Y-%m-%d')
    d1 = (d0 - timedelta(days=1)).strftime('%Y-%m-%d')
    d2 = (d0 - timedelta(days=2)).strftime('%Y-%m-%d')

    today_freq = {r['term']: r['frequency'] for r in
                   conn.execute('SELECT term, frequency FROM term_frequency WHERE date=?', (date,))}
    yest_freq = {r['term']: r['frequency'] for r in
                  conn.execute('SELECT term, frequency FROM term_frequency WHERE date=?', (d1,))}
    yest_velocity = {r['term']: r['velocity'] for r in
                      conn.execute('SELECT term, velocity FROM term_velocity WHERE date=?', (d1,))
                      if r['velocity'] is not None}

    for term, freq in today_freq.items():
        prev_freq = yest_freq.get(term, 0)
        velocity = freq - prev_freq

        prev_velocity = yest_velocity.get(term)
        acceleration = (velocity - prev_velocity) if prev_velocity is not None else None

        flagged = (
            acceleration is not None
            and acceleration > min_accel
            and freq <= max_abs_freq
        )

        conn.execute("""
            INSERT INTO term_velocity (date, term, frequency, velocity, acceleration, flagged)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(date, term) DO UPDATE SET
                frequency = excluded.frequency,
                velocity = excluded.velocity,
                acceleration = excluded.acceleration,
                flagged = excluded.flagged
        """, (date, term, freq, velocity, acceleration, int(flagged)))

    # silence unused var warning for d2 (kept for clarity of the 3-day window this implies)
    _ = d2


def top_emerging_terms(conn, date: str, n: int) -> list[dict]:
    rows = conn.execute("""
        SELECT term, frequency, velocity, acceleration
        FROM term_velocity
        WHERE date=? AND flagged=1
        ORDER BY acceleration DESC
        LIMIT ?
    """, (date, n)).fetchall()
    return [dict(r) for r in rows]


# ── Post engagement velocity ────────────────────────────────────────────────

def post_engagement_velocity(post: dict, now_dt: datetime = None) -> dict:
    """Score/comment growth rate per hour since the post was created."""
    now_dt = now_dt or datetime.now(timezone.utc)
    created = post.get('created_utc')
    if not created:
        return {'age_hours': None, 'score_velocity': None, 'comment_velocity': None}

    created_dt = datetime.fromtimestamp(created, tz=timezone.utc)
    age_hours = max((now_dt - created_dt).total_seconds() / 3600, 0.01)

    return {
        'age_hours': round(age_hours, 2),
        'score_velocity': round(post.get('score', 0) / age_hours, 2),
        'comment_velocity': round(post.get('num_comments', 0) / age_hours, 2),
    }


def top_velocity_posts(conn, n: int, now_dt: datetime = None) -> list[dict]:
    """Posts with the steepest score/comment growth, weighted toward young posts."""
    now_dt = now_dt or datetime.now(timezone.utc)
    rows = conn.execute("""
        SELECT p.id, p.source, p.subreddit, p.title, p.permalink, p.created_utc,
               s.score, s.num_comments
        FROM posts p
        JOIN post_snapshots s ON s.id = (
            SELECT id FROM post_snapshots WHERE post_id = p.id ORDER BY observed_at DESC LIMIT 1
        )
    """).fetchall()

    scored = []
    for row in rows:
        post = dict(row)
        velocity = post_engagement_velocity(post, now_dt)
        post.update(velocity)
        if velocity['score_velocity'] is not None:
            scored.append(post)

    scored.sort(key=lambda p: p['score_velocity'] + p['comment_velocity'], reverse=True)
    return scored[:n]
