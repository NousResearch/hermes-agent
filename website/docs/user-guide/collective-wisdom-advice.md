---
title: Collective Wisdom advice
---

Collective Wisdom lets your active Hermes conversation explain how a team
skill fits your setup before asking you to install, update, or share it.
Agent-written summaries are the default once Collective Wisdom is enabled.
Fixed notifications are available as a local-profile opt-out.

## Choose notification copy

First sign into your team and complete the existing `hermes wisdom setup`
disclosure. Profiles without a delivery-mode setting use agent-written advice.
To opt out, set this in that profile's configuration:

```yaml
wisdom:
  notifications:
    delivery_mode: fixed
```

Keep the profile's other Wisdom settings. Restart its messaging gateway and
open a new local session after updating Hermes. Set `delivery_mode: agent`
to return to agent-written advice. Existing saved `fixed` settings are preserved
on update; change that value explicitly to use the new default. This setting
does not enable sharing, change organization policy, or change your installed
skills' update policies.

## Advice and consent

Hermes uses the selected conversation's model and bounded conversation
context. Its automatic assessment cannot run shell commands, browse, read
arbitrary files, delegate, or install anything. Publisher text is untrusted;
usefulness and overlap are suggestions, not security or compatibility facts.

One recent, authorized private conversation receives the proactive advice.
Local qualifications stay with their originating conversation. If no eligible
conversation is active, activity waits. Other surfaces show the same advice
passively; opening them does not run another assessment.

The active conversation is a user-facing session, not a delegated subagent or
background task. Delegated agents, their subprocesses, scheduled jobs, and
review forks cannot request consent cards, register as active delivery sessions,
or claim the parent's queued advice. They return findings to the main conversation;
only that conversation can request the user's native confirmation control.

When multiple publication events for a skill are waiting, Hermes keeps only
the newest version eligible for automatic advice. Duplicate events do not
produce another recommendation. Historical notices and delivery receipts stay
available, and explicitly requested reviews are not discarded. A newer arrival
cannot retract a message already sent.

- Use **Review first** to inspect canonical checks and requirements.
- Use the native **Install**, **Update**, or sharing control to consent.
- In the native or Dashboard CLI, use `/wisdom inbox`, then its exact
  `/wisdom consent` action. The standalone `hermes wisdom consent` command
  requires an interactive confirmation before applying.
- A conversational "yes" asks Hermes to present the control; it does not
  apply the operation.
- **Not Now** suppresses further recommendations for this unchanged skill
  across your clients in the organization. It does not delete the skill or
  prevent you from reviewing it manually. Offline changes remain pending until
  they sync to the Gateway.

The proposed-package file reviewer also offers **Not Now**. It keeps the package
available for later review without uploading or publishing it. Page navigation,
including **Back to first page**, is read-only; publication still requires the
separate **Approve exact package** control.

New agent-prepared sharing packages can include **Publisher usage
(client-reported)** in the editable author description. This is a snapshot of
the local skill's recorded invocations and days used during the seven profile
calendar dates ending on preparation day. It can span local revisions and does
not verify successful outcomes. Without recorded usage, no summary is added.
Only totals and the date range are included, not paths, conversation excerpts,
per-day records, or private qualification rationale.

The summary stays local with the prepared package until you approve its upload.
Review it in the author description, and edit or remove it before sharing.
Editing changes the approval hash and requires a fresh confirmation. Subsequent
invocations and packaging retries do not update an existing review. If retained,
the approved description accompanies that exact published version to teammates;
it is not live usage telemetry or a Gateway certification. Manual private-draft
creation does not automatically add usage evidence.

Proactive Telegram and Slack advice includes **Notification settings**. Opening
it reads the current state without muting anything; choose a duration separately.
Use **Back** to return to the current inbox. Local notifications offer
`/wisdom mute` for the same controls. Muting does not disable manual review,
installation, updates, or sharing.

Older recommendation buttons open current review or notification settings.
They cannot start packaging, install a skill, or change preferences themselves.
Review and confirm with the new controls; a prior button does not carry consent
forward. The `hermes wisdom act` compatibility command likewise returns current
navigation without applying an operation.

Changed packages, local edits, additional requirements, and expired consent
require fresh review. Reopening an expired card or its checks shows **Expired**
with **Recheck** instead of a confirmation button. Recheck fetches a fresh plan;
it does not approve it or revive the old confirmation. Existing opted-in
automatic updates continue unchanged.

## Recovery and limits

Signing out of Nous retires this profile's unfinished advice and pending
confirmation controls. Sign in and re-verify your team with `hermes wisdom setup`
before continuing Wisdom; this does not revive the old queued advice or its
controls. Signing out of an unrelated model provider does not disconnect Wisdom.
Completed operations and delivery evidence remain available for recovery. A send
already dispatched can still arrive after sign-out; its receipt is retained for
settlement without sending the card again.

Wisdom also retires pending advice when credential refresh records that the Nous
session was revoked. A cached client cannot keep that advice active. Ordinary
token expiry and temporary connection/server failures do not cancel it.
Cached feed notices are retired on sign-out without deleting their history or
delivery receipts. A feed response started before sign-out cannot restore those
notices or advance the saved cursor, even after you re-verify the same team.
Re-verifying after sign-out quietly catches up the feed before making Wisdom
active. Announcements accumulated while signed out do not become new advice;
published skills remain available to browse. A failed catch-up leaves the
profile unverified, and rerunning setup resumes from the saved page. Changing
teams or accounts uses that identity's feed scope, not the previous cursor.
This does not remove installed skills or discard operation and delivery recovery
records; current installation and moderation state still reconcile normally.

Assessment claims are local to one profile and organization, with three-minute
leases and three bounded attempts. Idle sessions poll at most once a minute
per profile/org, and proactive routing uses a ten-minute recent-activity window.
Advice is saved before it is sent. Provider failures eventually produce a
deterministic review notice instead of an invented recommendation.

If a messaging send times out after it might have succeeded, Hermes does not
blindly send it again. The advice remains in the inbox. An interrupted apply
is reconciled against its exact operation journal; ambiguous results remain
visible for review rather than being applied again. Use the ordinary Wisdom
setup/recovery and review commands for operations that still need attention.

Assessment ownership remains local to the profile. Delivery reservations and
notification preferences are coordinated through the Gateway, without syncing
private candidate names, usage, or advice. This is not a general-purpose mailbox.
The weekly agent review uses existing real skill usage and organization policy;
fixed delivery does not run this review. Changing notification copy does not
change the organization's qualification thresholds.
