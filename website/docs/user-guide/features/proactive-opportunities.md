---
title: Proactive Opportunities
---

Proactive Opportunities are an opt-in inbox for useful next steps Hermes can
offer after it sees repeated work. Instead of silently learning, creating
profiles, scheduling jobs, or spawning agent systems, Hermes stores a pending
proposal that you can accept or dismiss.

Accepted opportunities run through existing Hermes surfaces:

- **Learn a skill** with the same authoring path as `/learn`.
- **Create a vertical bundle** backed by ordinary skill files and
  `~/.hermes/skill-bundles/*.yaml`.
- **Create a specialist profile** with profile-local identity, skills, and
  config.
- **Design a Kanban/swarm ecosystem** for durable multi-agent workflows.
- **Set up a recurring automation** through blueprints or the `cronjob` tool.

There is no new model tool and no system-prompt mutation. The scanner runs
outside the active model loop, creates only pending records, and accepting a
record submits a normal user turn so the usual tool approvals and skill write
approval gates still apply.

## Enable It

Proactive scanning is off by default.

```yaml
proactive:
  enabled: false
  notify: true
  max_pending: 8
  scan_interval_hours: 24
  scan_recent_messages: 120
  min_repeats: 3
```

Turn it on from chat:

```text
/opportunities enable
```

Or edit `~/.hermes/config.yaml` and set `proactive.enabled: true`.

## Use The Inbox

```text
/opportunities
/opportunities status
/opportunities scan
/opportunities seed
/opportunities accept 1
/opportunities dismiss 1
```

- `/opportunities` lists pending proposals.
- `/opportunities scan` manually scans recent user messages, even if automatic
  scanning is off.
- `/opportunities seed` adds starter ideas for learning, verticalization,
  profiles, agent ecosystems, and automation.
- `/opportunities accept N` starts the selected proposal as a normal Hermes
  turn.
- `/opportunities dismiss N` latches the proposal so it is not offered again.

## What The Scanner Looks For

The first scanner is deliberately conservative. It reads recent user messages
from the local session database and looks for repeated task shapes such as
several "review authentication ..." requests, or repeated domain clusters such
as PRDs/backlogs, dashboards/metrics, agents/profiles/Kanban, or schedules and
digests.

It does not create skills or jobs directly. It adds opportunities such as:

- "Learn recurring review authentication workflow"
- "Create a product manager vertical"
- "Create an analyst vertical"
- "Create a specialized-agent ecosystem"
- "Turn recurring work into an automation"

## Why This Is Separate From Cron Suggestions

`/suggestions` is specifically for ready-to-run scheduled jobs. Proactive
Opportunities are broader: they can lead to `/learn`, skill bundles, profiles,
Kanban/swarm templates, or cron jobs. Scheduled automation proposals still use
the existing cron machinery once accepted.
