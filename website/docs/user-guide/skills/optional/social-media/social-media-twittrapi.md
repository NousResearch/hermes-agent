---
title: "Twittrapi — Read public X data: profiles, tweets, search, followers"
sidebar_label: "Twittrapi"
description: "Read public X data: profiles, tweets, search, followers"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Twittrapi

Read public X data: profiles, tweets, search, followers.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/social-media/twittrapi` |
| Path | `optional-skills/social-media/twittrapi` |
| Version | `1.0.0` |
| Author | TwittrAPI (twittrapi) |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Twitter`, `X`, `Social Media`, `Research`, `API` |
| Related skills | [`xurl`](../../bundled/social-media/social-media-xurl.md), [`reddit-reading`](../../optional/social-media/social-media-reddit-reading.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# TwittrAPI Skill

Reads public X (Twitter) data as JSON through the TwittrAPI REST API: profiles, timelines,
followers and following, search with X operators, tweets with replies, lists, communities and
trends. It is read-only: it does not post, like, follow or send DMs, and cannot see protected
accounts. TwittrAPI is a paid third-party service, not affiliated with X Corp.

## When to Use

- The task needs exact X records rather than a summary: a user's last 200 tweets, a follower
  list, every post matching `from:user since:2026-01-01`, the members of a list, current trends.
- Compared with the built-in `x_search` tool: no xAI account or Grok in the loop, and you get
  the raw items with IDs, counts and cursors instead of a synthesized answer with citations.
- Compared with the `xurl` skill: no X developer account, app or OAuth setup, and no X API
  tier needed for reads. Use `xurl` for anything that writes or acts as the user's account.
- Not for posting, replying, liking, DMs or private data.

## Prerequisites

An API key in `TWITTRAPI_API_KEY` (the CLI prompts for it securely when the skill loads; on
messaging platforms run `hermes setup`). Create a key at https://app.twittrapi.com. Billing is
pay per call from a prepaid balance, from $0.065 per 1,000 calls.
Never print the key, put it in a URL or ask the user to paste it into chat.

## How to Run

Run the helper through `terminal`, with the skill-relative path. It sends
`X-API-KEY: $TWITTRAPI_API_KEY` to `https://api.twittrapi.com`, always adds `format=simple`,
prints the JSON body, and prints `X-Remaining-Balance` and `X-Price-Per-Call` to stderr:

```bash
python3 scripts/twittrapi.py /user/by/username/XDevelopers
python3 scripts/twittrapi.py /search "searchItem=from:XDevelopers -filter:replies" category=Latest
python3 scripts/twittrapi.py /user/-1/followers username=XDevelopers --pages 3
```

`--pages N` follows `nextCursor` for up to N billed calls and merges the `data` arrays. Without
the helper, any HTTP client works: `GET https://api.twittrapi.com/<path>?format=simple` with the
`X-API-KEY` header.

## Quick Reference

All endpoints are `GET`. Paginated ones take `count` (default 20) and `cursor`.

| Need | Path | Paged |
|---|---|---|
| Profile by handle / by ID | `/user/by/username/{username}`, `/user/{userId}` | no |
| Handle to numeric ID | `/username/to/id/{username}` | no |
| Up to 500 profiles | `/user/users/by/ids` with `ids=1,2,3` | no |
| Tweets / tweets and replies / media | `/user/{userId}/tweets`, `/tweetsAndReplies`, `/media` | yes |
| Followers / following / verified followers | `/user/{userId}/followers`, `/followings`, `/verifiedFollowers` | yes |
| Search | `/search` with `searchItem=<query>`, `category=Top\|Latest\|People\|Photos\|Videos` | yes |
| Tweet with reply thread / single tweet | `/tweet/{tweetId}`, `/simpletweet/{tweetId}` | yes / no |
| Retweeters / quote tweets | `/tweet/{tweetId}/retweets`, `/tweet/{tweetId}/quotes` | yes |
| List details / tweets / members | `/lists/details`, `/lists/tweets`, `/lists/members` with `listId=` | no / yes / yes |
| Find communities / details / tweets | `/communities/search/{term}`, `/communities/{id}`, `/communities/{id}/tweets` | yes / no / yes |
| Trends for a place / all places | `/trends/{woeid}` (`1` worldwide, `23424977` US), `/misc/locations` | no |
| Balance and price per call (free) | `/balance` | no |

`/user/{userId}/...` paths accept `-1` as the ID plus `username=<handle>`, which saves the ID
lookup. The tweet ID is the number at the end of `https://x.com/<user>/status/<tweetId>`.

`searchItem` takes X search operators: `from:`, `to:`, `"exact phrase"`, `lang:en`,
`since:2026-01-01`, `until:`, `min_faves:100`, `min_retweets:`, `filter:media`,
`-filter:replies`, `OR`, `-word`. Maximum 512 characters.

## Procedure

1. Resolve the target: handle, numeric ID, tweet ID from the URL, list ID or WOEID.
2. Estimate the calls before paging. Each page is one billed call and holds roughly 20 items
   (followers often 50 to 70), so 1,000 followers is about 20 to 50 calls. Tell the user before
   a job that will cost more than a few dollars; `/balance` is free.
3. Fetch the first page. The simple format returns `{data, nextCursor, previousCursor,
   hasNextPage}`; single objects come as `{data: {...}}`. Tweets carry `id`, `url`, `text`,
   `createdAt`, `author`, `counts` (likes, retweets, replies, quotes, bookmarks, views),
   `media`, `quoted`, `retweetOf`. Users carry `username`, `name`, `description`,
   `counts` (followers, following, tweets), `verified`, `createdAt`.
4. For more pages, pass `nextCursor` as `cursor` (or use `--pages`). Stop when `hasNextPage` is
   false, `nextCursor` is empty, or it repeats the cursor you sent.
5. Cite tweets by their `url` field when the result feeds a report.

## Pitfalls

- `count` is a hint to X, not a limit: pages can hold more or fewer items, and an empty page
  is not an error.
- 402 means the balance is used up (not charged): stop and ask the user to top up. Do not retry.
- 429 is the key's rate limit (not charged) and has no `Retry-After`: back off exponentially.
- 400, 408, 409 and 5xx are refunded (`X-Refunded: true`): retry with backoff, at most ~4 times.
- 401/403 is a missing, wrong or deactivated key; 404 a deleted, suspended or protected
  account or tweet; 422 bad input (non-numeric ID, `[` or `]` or an unclosed quote in the query).
- X shows who liked a tweet only to its author. `/tweet/{tweetId}/favorites` needs a `POST`
  with that author's cookie and a proxy, so this skill does not cover it; use the tweet's
  `counts.likes` instead.
- The simple format omits cards, polls, edit history and community notes. Drop `format=simple`
  only when you need those, and expect X's large nested JSON.

## Verification

`python3 scripts/twittrapi.py /balance` prints the balance and price per call without charge.
`python3 scripts/twittrapi.py /user/by/username/XDevelopers` returns `data.username` equal to
`XDevelopers` and `X-Remaining-Balance` on stderr.
