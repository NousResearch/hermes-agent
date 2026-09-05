---
title: "Reddit Reading — Read Reddit: subreddits, search, threads, users"
sidebar_label: "Reddit Reading"
description: "Read Reddit: subreddits, search, threads, users"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Reddit Reading

Read Reddit: subreddits, search, threads, users. No browser.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/social-media/reddit-reading` |
| Version | `1.0.0` |
| Author | Teknium (teknium1), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Reddit`, `Social Media`, `Research`, `Discussions`, `Community` |
| Related skills | [`rss-feeds`](/docs/user-guide/skills/bundled/research/research-rss-feeds), [`grounded-citations`](/docs/user-guide/skills/bundled/research/research-grounded-citations), [`blocked-page-recovery`](/docs/user-guide/skills/bundled/web/web-blocked-page-recovery), [`xurl`](/docs/user-guide/skills/bundled/social-media/social-media-xurl) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Reddit Reading Skill

Reads Reddit content — subreddit listings, site or subreddit search, full threads with
comments, and user activity — from a server or headless machine where the normal routes
are dead. It does not post, vote, or log in as a user. Idea credit: the per-platform
backend routing in [Agent Reach](https://github.com/Panniantong/Agent-Reach).

## When to Use

- "What is r/LocalLLaMA saying about X", "find Reddit threads on Y", "summarise this
  Reddit thread", "what has u/someone posted lately".
- Any `reddit.com` URL the user shares. `web_extract`, `browser_navigate` and the
  `.json` endpoints all fail from server IPs (403 or a "Prove your humanity" wall);
  this skill is the working path.
- Not for posting, voting, messaging, or anything needing a user login.

## Prerequisites

None for the anonymous path. For anything beyond a handful of calls per task, create
a free Reddit "script" app at https://www.reddit.com/prefs/apps and put the two values
in `~/.hermes/.env`:

```
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
```

The script picks the OAuth backend automatically when both are set (~100 requests per
minute, scores, comment nesting, `num_comments`). Without them it uses Reddit's Atom
feeds, which are the only unauthenticated endpoints still served to non-residential IPs.

## How to Run

Run every command through `terminal` with the skill-relative script path:

```bash
python3 scripts/reddit.py doctor                                  # which backend, current rate-limit window
python3 scripts/reddit.py sub LocalLLaMA --sort hot --limit 15
python3 scripts/reddit.py search "hermes agent" --sub LocalLLaMA --sort new
python3 scripts/reddit.py thread https://www.reddit.com/r/x/comments/abc123/slug/ --limit 40
python3 scripts/reddit.py user spez --limit 10
python3 scripts/reddit.py --json search "topic"                  # machine-readable
```

## Quick Reference

| Need | Command | Anonymous | OAuth |
|---|---|---|---|
| Subreddit front page | `sub NAME --sort hot\|new\|top\|rising [--time week]` | ✔ | ✔ |
| Search all of Reddit | `search "q" --sort relevance\|new\|top\|comments` | ✔ | ✔ |
| Search one subreddit | `search "q" --sub NAME` | ✔ | ✔ |
| Thread + comments | `thread URL --limit N` | ✔ top-level only, no scores | ✔ nested, scores |
| User posts/comments | `user NAME` | ✔ | ✔ |
| Backend + rate limit | `doctor` | ✔ | ✔ |

## Procedure

① `doctor` once per task if you have not called it this session — it tells you which
backend is live and how many seconds remain in the anonymous window.

② Plan your calls before making them. Anonymous Reddit allows roughly **one request per
minute per IP**; the script sleeps until the window resets on a 429 and retries once, so
a five-call plan costs about five minutes. Prefer one `search --sub` over several `sub`
listings, and read one thread rather than the whole listing.

③ For "what is the community saying" questions, read the thread bodies (`thread`) rather
than stopping at titles; the listing only carries the first ~300 characters of each post.

④ Cite the permalink (`url` field), not the listing page, when the result feeds a report.
`grounded-citations` registers these URLs like any other source.

⑤ If the user needs sustained Reddit access (monitoring, more than ~10 calls), stop and
ask them to add the OAuth credentials rather than grinding through the throttle.

## Pitfalls

- `www.reddit.com/…/.json`, `api.reddit.com` and `old.reddit.com` return 403 or an
  empty "Welcome to Reddit" shell for datacentre IPs. Do not fall back to them; do not
  spoof a browser User-Agent (also 403).
- `r.jina.ai` and the `browser_navigate` tool hit the same block ("blocked by network
  security" / humanity check). `blocked-page-recovery`'s Wayback route can still recover
  an **old** thread that was archived; it cannot fetch fresh ones.
- Anonymous thread feeds only contain the post plus top-level comments (Reddit caps the
  feed at a handful of entries); scores and reply nesting are OAuth-only.
- Reddit's `limit` on feeds is advisory — expect 5–25 entries regardless of what you ask.
- Never paste `REDDIT_CLIENT_SECRET` into a chat or log; the script reads it from the
  environment only.

## Verification

`python3 scripts/reddit.py doctor` prints `anonymous_feed: ok` and an
`x-ratelimit-reset` value; `sub announcements --limit 1` returns one entry with a
`reddit.com/r/announcements/comments/` URL. With credentials set, `doctor` prints
`active_backend: oauth` and `thread …` output shows numeric scores.
