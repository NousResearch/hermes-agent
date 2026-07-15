#!/usr/bin/env python3
"""
Reddit ingestion via the public read-only JSON API.

No auth required, but Reddit aggressively rate-limits unidentified clients —
always send a descriptive User-Agent and pace requests.
"""

import json
import time
import urllib.error
import urllib.request

USER_AGENT = "trendscout/0.1 (personal trend-monitoring script; contact: david@refineit.com.au)"
BASE_URL = "https://www.reddit.com"
REQUEST_DELAY_SECONDS = 1.5


def _fetch_listing(subreddit: str, sort: str, extra_params: str = "", limit: int = 100, retries: int = 3) -> list[dict]:
    url = f"{BASE_URL}/r/{subreddit}/{sort}.json?limit={limit}{extra_params}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read())
                return payload.get("data", {}).get("children", [])
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            return []
        except Exception:
            return []
    return []


def _post_from_child(child: dict, subreddit: str) -> dict | None:
    d = child.get("data", {})
    post_id = d.get("id")
    if not post_id:
        return None
    return {
        "id": post_id,
        "source": "reddit",
        "subreddit": subreddit,
        "title": d.get("title", ""),
        "selftext": d.get("selftext", "") or "",
        "author": d.get("author"),
        "permalink": f"https://reddit.com{d.get('permalink', '')}",
        "url": d.get("url"),
        "created_utc": d.get("created_utc"),
        "score": d.get("score", 0),
        "num_comments": d.get("num_comments", 0),
    }


def fetch_subreddit_posts(subreddit: str) -> list[dict]:
    """Fetch /new and /top?t=day for a subreddit, deduped by post id."""
    posts: dict[str, dict] = {}

    for child in _fetch_listing(subreddit, "new"):
        post = _post_from_child(child, subreddit)
        if post:
            posts[post["id"]] = post
    time.sleep(REQUEST_DELAY_SECONDS)

    for child in _fetch_listing(subreddit, "top", extra_params="&t=day"):
        post = _post_from_child(child, subreddit)
        if post:
            posts.setdefault(post["id"], post)
    time.sleep(REQUEST_DELAY_SECONDS)

    return list(posts.values())


def fetch_all(subreddits: list[str]) -> list[dict]:
    """Fetch posts for every subreddit in the list, in order."""
    all_posts: list[dict] = []
    for subreddit in subreddits:
        all_posts.extend(fetch_subreddit_posts(subreddit))
    return all_posts
