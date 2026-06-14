#!/usr/bin/env python3
"""
trendscout entrypoint — ingest, score, cluster, and build the daily digest.

Pipeline:
  1. Ingest Reddit posts (config['reddit']['enabled']).
  2. Update term-frequency table from today's posts; compute velocity/acceleration.
  3. Embed new posts into ChromaDB and assign to semantic clusters.
  4. Build the Telegram digest text and print everything as JSON for the
     calling agent/cron job to relay.

Last stdout line: {"wakeAgent": true/false}
"""

import json
from datetime import datetime, timezone

from trendscout import clustering
from trendscout import config as cfg
from trendscout import db
from trendscout import digest as digest_mod
from trendscout import firecrawl_ingest
from trendscout import reddit_ingest
from trendscout import scoring
from trendscout import social_trends


def main():
    config = cfg.load_config()
    conn = db.connect(config['paths']['db'])
    now = db.now_iso()
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    sources_used = []
    posts = []

    if config['reddit']['enabled']:
        subreddits = cfg.load_subreddits(config)
        posts.extend(reddit_ingest.fetch_all(subreddits))
        sources_used.append('reddit')

    if config['firecrawl']['enabled']:
        urls = cfg.load_urls(config)
        if urls:
            posts.extend(firecrawl_ingest.fetch_all(urls))
            sources_used.append('firecrawl')

    new_count = 0
    for post in posts:
        if db.upsert_post(conn, post, now=now):
            new_count += 1
    conn.commit()

    # Social trending topics (X/Twitter via trends24.in, TikTok Creative
    # Center, etc.) — informational digest section + raw term-frequency
    # signal, not stored as posts.
    trending = {}
    st_config = config.get('social_trends', {})
    if st_config.get('enabled'):
        trending = social_trends.fetch_all(st_config)
        if trending:
            sources_used.append('social_trends')
            raw_terms = [
                social_trends.normalize_term(term)
                for terms in trending.values() for term in terms
            ]
            scoring.update_term_frequency_raw(conn, raw_terms, today)
            conn.commit()

    # Term frequency / acceleration
    scoring.update_term_frequency(conn, posts, today)
    conn.commit()
    scoring.compute_term_velocity(conn, today, config)
    conn.commit()
    top_terms = scoring.top_emerging_terms(conn, today, config['digest']['top_terms'])

    # Semantic clustering
    collection = clustering.get_collection(config)
    new_ids = clustering.embed_new_posts(conn, collection)
    touched = clustering.assign_clusters(conn, collection, new_ids, today)
    clustering.update_cluster_size_history(conn, set(touched.keys()), today)
    conn.commit()
    new_clusters = clustering.growing_clusters(conn, today, config, config['digest']['top_clusters'])

    # Engagement velocity
    velocity_posts = scoring.top_velocity_posts(conn, config['digest']['top_clusters'])

    flagged_terms = conn.execute(
        'SELECT COUNT(*) FROM term_velocity WHERE date=? AND flagged=1', (today,)
    ).fetchone()[0]
    terms_tracked = conn.execute(
        'SELECT COUNT(*) FROM term_frequency WHERE date=?', (today,)
    ).fetchone()[0]

    stats = {
        'posts_ingested': len(posts),
        'posts_new': new_count,
        'terms_tracked': terms_tracked,
        'flagged_terms': flagged_terms,
        'new_embeddings': len(new_ids),
        'clusters_touched': len(touched),
    }

    db.record_run(conn, posts_ingested=len(posts), posts_new=new_count, sources=sources_used, now=now)
    conn.commit()
    conn.close()

    digest_text = digest_mod.build_digest(today, top_terms, new_clusters, velocity_posts, stats, trending)

    output = {
        'run_at': now,
        'date': today,
        'stats': stats,
        'top_terms': top_terms,
        'new_clusters': new_clusters,
        'velocity_posts': velocity_posts,
        'trending': trending,
        'digest_text': digest_text,
        'sources': sources_used,
        'db_file': config['paths']['db'],
    }

    print(json.dumps(output, indent=2))
    print(json.dumps({'wakeAgent': bool(top_terms or new_clusters)}))


if __name__ == '__main__':
    main()
