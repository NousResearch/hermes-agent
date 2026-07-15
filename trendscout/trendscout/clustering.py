#!/usr/bin/env python3
"""
Semantic clustering for trendscout, backed by ChromaDB.

Approach: incremental nearest-neighbour clustering. Each new post is embedded
(title + selftext) and compared against existing embeddings. If its nearest
neighbour already belongs to a cluster and is within SIMILARITY_THRESHOLD,
the new post joins that cluster; otherwise it seeds a new one.

This is intentionally simple (no global re-clustering) so that cluster_ids
are stable across runs and cluster_size_history accumulates meaningfully.
ChromaDB's default embedding function (all-MiniLM-L6-v2, local/offline) uses
squared-L2 distance: near-duplicate text scores ~0.0-0.7, related-topic text
~0.7-1.3, unrelated text >1.5 (observed empirically).
"""

import uuid

import chromadb

from . import db

SIMILARITY_THRESHOLD = 1.0


def get_collection(config: dict):
    client = chromadb.PersistentClient(path=config['paths']['chroma'])
    return client.get_or_create_collection(name=config['clusters']['collection_name'])


def _document_for(post: dict) -> str:
    return f"{post['title']}. {post.get('selftext', '') or ''}".strip()[:2000]


def embed_new_posts(conn, collection, batch_size: int = 100) -> list[str]:
    """Embed any posts not yet in the ChromaDB collection. Returns their ids."""
    rows = conn.execute("SELECT id, title, selftext FROM posts WHERE embedded=0").fetchall()
    if not rows:
        return []

    new_ids = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        ids = [r['id'] for r in batch]
        docs = [_document_for(dict(r)) for r in batch]
        collection.add(ids=ids, documents=docs, metadatas=[{'cluster_id': ''} for _ in batch])
        new_ids.extend(ids)

    conn.executemany('UPDATE posts SET embedded=1 WHERE id=?', [(i,) for i in new_ids])
    return new_ids


def assign_clusters(conn, collection, new_ids: list[str], date: str) -> dict[str, set[str]]:
    """
    Assign each newly-embedded post to an existing or new cluster.
    Returns {cluster_id: set(post_ids touched this run)} for size-history updates.
    """
    touched: dict[str, set[str]] = {}

    for post_id in new_ids:
        got = collection.get(ids=[post_id], include=['embeddings'])
        if not got['ids']:
            continue
        embedding = got['embeddings'][0]

        result = collection.query(
            query_embeddings=[embedding], n_results=6,
            include=['distances', 'metadatas'],
        )

        best_cluster, best_distance = None, None
        for nid, dist, meta in zip(result['ids'][0], result['distances'][0], result['metadatas'][0]):
            if nid == post_id:
                continue
            cid = (meta or {}).get('cluster_id')
            if cid and (best_distance is None or dist < best_distance):
                best_cluster, best_distance = cid, dist

        if best_cluster and best_distance is not None and best_distance <= SIMILARITY_THRESHOLD:
            cluster_id = best_cluster
        else:
            cluster_id = uuid.uuid4().hex[:10]
            conn.execute("""
                INSERT OR IGNORE INTO clusters (cluster_id, label, first_seen_date, last_seen_date)
                VALUES (?, NULL, ?, ?)
            """, (cluster_id, date, date))

        conn.execute('UPDATE posts SET cluster_id=? WHERE id=?', (cluster_id, post_id))
        conn.execute('UPDATE clusters SET last_seen_date=? WHERE cluster_id=?', (date, cluster_id))
        collection.update(ids=[post_id], metadatas=[{'cluster_id': cluster_id}])
        touched.setdefault(cluster_id, set()).add(post_id)

    return touched


def update_cluster_size_history(conn, cluster_ids: set[str], date: str):
    """Record today's total member count for each touched cluster."""
    for cluster_id in cluster_ids:
        member_count = conn.execute(
            'SELECT COUNT(*) FROM posts WHERE cluster_id=?', (cluster_id,)
        ).fetchone()[0]
        conn.execute("""
            INSERT INTO cluster_size_history (cluster_id, date, member_count)
            VALUES (?,?,?)
            ON CONFLICT(cluster_id, date) DO UPDATE SET member_count=excluded.member_count
        """, (cluster_id, date, member_count))


def growing_clusters(conn, date: str, config: dict, n: int) -> list[dict]:
    """
    Clusters first seen within growth_window_days, currently at or above
    new_cluster_min_size, ordered by member_count descending.
    """
    from datetime import datetime, timedelta

    window_days = config['clusters']['growth_window_days']
    min_size = config['clusters']['new_cluster_min_size']
    cutoff = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=window_days)).strftime('%Y-%m-%d')

    rows = conn.execute("""
        SELECT c.cluster_id, c.first_seen_date, c.last_seen_date,
               COUNT(p.id) AS member_count
        FROM clusters c
        JOIN posts p ON p.cluster_id = c.cluster_id
        WHERE c.first_seen_date >= ?
        GROUP BY c.cluster_id
        HAVING member_count >= ?
        ORDER BY member_count DESC
        LIMIT ?
    """, (cutoff, min_size, n)).fetchall()

    clusters = []
    for row in rows:
        cluster = dict(row)
        examples = conn.execute("""
            SELECT title, permalink, subreddit, source FROM posts
            WHERE cluster_id=? ORDER BY first_seen_at DESC LIMIT 3
        """, (cluster['cluster_id'],)).fetchall()
        cluster['examples'] = [dict(e) for e in examples]
        clusters.append(cluster)

    return clusters
