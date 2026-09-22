---
title: DevTeam Discord visibility pilot
---

# DevTeam Discord visibility pilot

The DevTeam pilot separates human milestones from operational activity without
removing technical evidence.

## Routing policy

- The epic post contains only decisions, human-readable state transitions,
  short summaries, and direct GitHub issue/PR URLs.
- Each sub-issue uses one durable operational context for handoff and follow-up.
  In the approved pilot, `#demandas` is the operational parent and the existing
  dedicated context ID recorded as `Thread: <id>` in the GitHub issue is the
  canonical return destination. Do not require or create an extra thread below
  a normal text or announcement channel.
- The GitHub issue is the durable proof record: commands, focused-test output,
  blockers, the draft PR URL, and final delivery report belong there.
- Discord tool-progress telemetry is quiet by default. This prevents automatic
  tool cards and previews from burying the human milestones in a shared post.
  A dedicated operational deployment can explicitly set
  `display.platforms.discord.tool_progress: new` or `all` when its participants
  need live progress.

## Two-front pilot checklist

For each of the two participating sub-issues:

1. Reuse the canonical context recorded as `Thread: <thread_id>` in the GitHub
   issue. In this pilot it is an existing dedicated context under `#demandas`;
   retrying must return there rather than creating a second destination.
2. Record `Thread: <thread_id>` in the first GitHub issue comment so a resumed
   handoff returns to the same context.
3. Put the issue URL and, once created, the draft PR URL in the human milestone
   message. Do not replace these links with tool-output excerpts.
4. Put the command and test receipt in the GitHub issue, then post only the
   review milestone and PR URL to Discord.

The default is intentionally quiet for all Discord conversations because a
platform-only setting cannot safely infer whether a message is a private
operational thread or a shared project record. Explicit configuration is the
opt-in boundary for contexts that need live telemetry.

## Approved topology and limitations

The pilot uses `#demandas` as the operational parent. Its dedicated context is
already canonical once its ID is stored in the GitHub issue as
`Thread: <thread_id>`; resuming work must target that ID and must not create a
replacement context under a normal text or announcement channel.

The policy does not infer a context from message text or a display title. The
stored thread ID and the GitHub issue/PR URLs are the recoverable audit links.
If a canonical context is unavailable, fail closed and record the blocker in the
issue rather than silently opening a second one.

## Rollback

No existing Discord message, issue comment, or mapping is deleted by this
policy. To undo the behavior, revert the change that makes Discord tool progress
quiet, then configure the prior explicit `display.platforms.discord.tool_progress`
mode if required. The durable GitHub proof and the existing dedicated-context
mapping remain intact.
