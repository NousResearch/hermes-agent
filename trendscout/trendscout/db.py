#!/usr/bin/env python3
"""
SQLite storage for trendscout.

Tables:
  posts              - one row per ingested post/item, deduped by id across runs
  post_snapshots     - score/comment-count observations over time, used for velocity
  term_frequency     - daily frequency count per extracted term/phrase
  term_velocity      - daily velocity (1st derivative) and acceleration (2nd derivative)
                       per term, for flagging pre-emergence terms
  clusters           - semantic clusters tracked in ChromaDB, with metadata
  cluster_size_history - daily member-count snapshots per cluster, for growth tracking
  runs               - one row per pipeline run, for digest stats / debugging
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path.home() / '.hermes' / 'trendscout' / 'trendscout.db'

SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    id              TEXT PRIMARY KEY,
    source          TEXT NOT NULL,
    subreddit       TEXT,
    title           TEXT NOT NULL,
    selftext        TEXT,
    author          TEXT,
    permalink       TEXT NOT NULL,
    url             TEXT,
    created_utc     REAL,
    first_seen_at   TEXT NOT NULL,
    last_seen_at    TEXT NOT NULL,
    embedded        INTEGER DEFAULT 0,
    cluster_id      TEXT
);
CREATE INDEX IF NOT EXISTS idx_posts_cluster_id ON posts(cluster_id);

CREATE TABLE IF NOT EXISTS post_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id         TEXT NOT NULL REFERENCES posts(id),
    observed_at     TEXT NOT NULL,
    score           INTEGER NOT NULL,
    num_comments    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_post_snapshots_post_id ON post_snapshots(post_id);

CREATE TABLE IF NOT EXISTS term_frequency (
    date            TEXT NOT NULL,
    term            TEXT NOT NULL,
    frequency       INTEGER NOT NULL,
    PRIMARY KEY (date, term)
);

CREATE TABLE IF NOT EXISTS term_velocity (
    date            TEXT NOT NULL,
    term            TEXT NOT NULL,
    frequency       INTEGER NOT NULL,
    velocity        REAL,
    acceleration    REAL,
    flagged         INTEGER DEFAULT 0,
    PRIMARY KEY (date, term)
);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id      TEXT PRIMARY KEY,
    label           TEXT,
    first_seen_date TEXT NOT NULL,
    last_seen_date  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cluster_size_history (
    cluster_id      TEXT NOT NULL REFERENCES clusters(cluster_id),
    date            TEXT NOT NULL,
    member_count    INTEGER NOT NULL,
    PRIMARY KEY (cluster_id, date)
);

CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at          TEXT NOT NULL,
    posts_ingested  INTEGER DEFAULT 0,
    posts_new       INTEGER DEFAULT 0,
    sources         TEXT
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path=None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    conn.executescript(SCHEMA)
    return conn


def upsert_post(conn: sqlite3.Connection, post: dict, now: str = None) -> bool:
    """Insert or refresh a post, and record a score/comment snapshot.

    Returns True if this is a newly-seen post, False if it already existed.
    """
    now = now or now_iso()
    row = conn.execute('SELECT id FROM posts WHERE id=?', (post['id'],)).fetchone()

    is_new = row is None
    if is_new:
        conn.execute("""
            INSERT INTO posts (id, source, subreddit, title, selftext, author,
                permalink, url, created_utc, first_seen_at, last_seen_at, embedded)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,0)
        """, (
            post['id'], post['source'], post.get('subreddit'), post['title'],
            post.get('selftext', ''), post.get('author'), post['permalink'],
            post.get('url'), post.get('created_utc'), now, now,
        ))
    else:
        conn.execute('UPDATE posts SET last_seen_at=? WHERE id=?', (now, post['id']))

    conn.execute(
        'INSERT INTO post_snapshots (post_id, observed_at, score, num_comments) VALUES (?,?,?,?)',
        (post['id'], now, post.get('score', 0), post.get('num_comments', 0)),
    )
    return is_new


def record_run(conn: sqlite3.Connection, posts_ingested: int, posts_new: int, sources: list[str], now: str = None):
    import json
    conn.execute(
        'INSERT INTO runs (run_at, posts_ingested, posts_new, sources) VALUES (?,?,?,?)',
        (now or now_iso(), posts_ingested, posts_new, json.dumps(sorted(sources))),
    )
