#!/usr/bin/env python3
"""Build the Telegram digest text for trendscout's daily run."""


def _source_label(post: dict) -> str:
    if post.get('subreddit'):
        return f"r/{post['subreddit']}"
    return post.get('source', 'web')


def _fmt_term(term: dict) -> str:
    return (f"  - \"{term['term']}\" — freq {term['frequency']}, "
            f"velocity {term['velocity']:+.0f}, accel {term['acceleration']:+.1f}")


def _fmt_cluster(cluster: dict) -> str:
    lines = [f"  - cluster {cluster['cluster_id']} — {cluster['member_count']} posts "
             f"(first seen {cluster['first_seen_date']})"]
    for ex in cluster['examples']:
        lines.append(f"      • {_source_label(ex)}: {ex['title'][:80]} ({ex['permalink']})")
    return '\n'.join(lines)


def _fmt_velocity_post(post: dict) -> str:
    return (f"  - {_source_label(post)}: {post['title'][:80]} "
            f"(score {post['score']}/{post['score_velocity']:.1f} per hr, "
            f"comments {post['num_comments']}/{post['comment_velocity']:.1f} per hr) "
            f"{post['permalink']}")


def build_digest(date: str, top_terms: list[dict], new_clusters: list[dict],
                  velocity_posts: list[dict], stats: dict,
                  trending: dict[str, list[str]] = None) -> str:
    lines = [f"📡 *Trendscout — {date}*", '']

    if trending:
        lines.append("📈 *Trending now*")
        for region, terms in trending.items():
            lines.append(f"  - {region}: {', '.join(terms)}")
        lines.append('')

    lines.append("🔥 *Emerging terms* (low volume, accelerating)")
    if top_terms:
        for term in top_terms:
            lines.append(_fmt_term(term))
    else:
        lines.append("  - none flagged today")
    lines.append('')

    lines.append("🧩 *New/growing semantic clusters*")
    if new_clusters:
        for cluster in new_clusters:
            lines.append(_fmt_cluster(cluster))
    else:
        lines.append("  - none crossed the size threshold today")
    lines.append('')

    lines.append("⚡ *Fastest-growing posts*")
    if velocity_posts:
        for post in velocity_posts:
            lines.append(_fmt_velocity_post(post))
    else:
        lines.append("  - no posts ingested today")
    lines.append('')

    lines.append(f"_Stats: {stats['posts_ingested']} fetched, {stats['posts_new']} new, "
                  f"{stats['terms_tracked']} terms tracked, {stats['flagged_terms']} flagged_")

    return '\n'.join(lines)
