---
title: "Communication Platforms"
sidebar_label: "Communication Platforms"
description: "Communication platform umbrella: email via Himalaya, X/Twitter via xurl, Yuanbao groups/DMs, and other external-message workflows"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Communication Platforms

Communication platform umbrella: email via Himalaya, X/Twitter via xurl, Yuanbao groups/DMs, and other external-message workflows.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/communication-platforms` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Communication Platforms

Use this skill when reading, drafting, sending, searching, or managing messages through external communication platforms.

## Universal rules

1. Identify the platform, account, recipient/audience, and side-effect scope before sending.
2. Read/search current state before replying, posting, or bulk-updating.
3. Draft first when content is sensitive, public, or irreversible.
4. Verify sends/posts by reading the returned message ID, URL, or thread state.
5. Never expose credentials, auth tokens, private contact data, or unrelated message content.

## Email via Himalaya

- Use for IMAP/SMTP email from the terminal.
- Typical flow: confirm account/config, search or list messages, read the relevant thread, draft reply/new message, send only after scope is clear, then verify sent state.
- Keep message composition concise and preserve quoted-thread context only when needed.

## X/Twitter via xurl

- Use for posting, searching, DMs, media uploads, and X API calls through the official CLI wrapper.
- Treat public posts as high-side-effect actions: draft, confirm exact text/media, then post.
- For API queries, capture rate limits and returned IDs/URLs.

## Yuanbao groups and DMs

- Use for querying Yuanbao group info/members, @mentioning users, and sending private messages.
- Resolve user/group identity before sending; do not guess mention IDs.
- For group messages, make the target audience and mention behavior explicit.

## Verification checklist

- [ ] Correct platform/account selected.
- [ ] Recipient/channel resolved from platform data.
- [ ] Message or query content is scoped to the user's ask.
- [ ] Side effects have a returned handle or readback confirmation.
## Support files

- `references/absorbed-skills.md` — list of original skill packages consolidated into this umbrella and where to recover full archived content.
