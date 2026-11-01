---
title: "Social HAR API Connectivity — Capture social platform login + session for agentic posting"
sidebar_label: "Social HAR API Connectivity"
---

import SkillMeta from '@site/optional-skills/social-media/social-har-api-connectivity/_meta.md';

# Social HAR API Connectivity

Connect your Hermes Agent to any social platform's API by capturing a real browser login session — no official API key needed.

<SkillMeta />

## Usage

1. Run the skill: `Use the social-har-api-connectivity skill to capture a session for Bluesky`
2. The agent opens Chrome in visible mode
3. Log in normally through the browser
4. The agent captures the session and builds a client
5. Use the client to post or read on that platform

## Supported Platforms

See the skill's readme for platform-specific guidance.

## See Also

- [Skill source](pathname:///optional-skills/social-media/social-har-api-connectivity/SKILL.md)
- [`xurl` skill](../social-media/social-media-xurl.md) — X/Twitter via official API
- [`linkedin-posting` skill](../social-media/social-media-linkedin-posting.md) — LinkedIn via OAuth
