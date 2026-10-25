---
title: Skill write review
---

# Skill write review

When the existing `skills.write_approval` setting stages a `skill_manage` write,
Telegram immediately receives a comparison for each affected file. Cards show the
absolute pathname and the full original and replacement, with native quotation,
strikethrough and bold entities. Long text is split into contiguous numbered parts;
the final part holds **Approve**, **Disapprove**, **Snooze**, and **Edit**.

`/skills`, `/skills pending`, and `/skills diff <pending-id>` open these same cards
on Telegram. No outstanding proposals means no messages. Existing CLI and other
platform commands remain available. Previously staged, unprepared proposals are
opened explicitly rather than automatically flooding a newly upgraded gateway.

## Transactions and edits

A `skill_manage operations[]` transaction remains atomic. Repeated patches to one
file produce one final comparison. Each file is approved separately; the last
approval applies the existing transaction through `apply_skill_pending`. The UI
explains that files apply together and Disapprove cancels the entire proposal.
Nothing is written while other file approvals are outstanding.

Edit takes the full replacement text in an explicit reply to the edit request,
or a UTF-8 text attachment. Near-limit pastes are rejected because Telegram may
split them; attachments allow complete large replacements. The reply is bound to
the requesting actor, chat, topic, instruction message and revision. An edit clears
all transaction approvals and opens new comparisons. It never applies a write.
Whole-skill deletion can be approved or disapproved, but cannot be edited into a
partial deletion: request a replacement proposal for that case. Binary files in a
whole-skill deletion are identified by pathname, byte count and SHA-256.

Files are frozen at staging. Approval refuses changed content, redirected skill
resolution, changed creation destinations, and added files in a package deletion.
A failed preview remains blocked: it cannot silently rebase when conditions change.
Recreate the proposal instead. Existing frontmatter, size, security, pinning and
write-policy checks remain in the skill engine, as do rollback, provenance,
configured creation/external directories and deferred cache invalidation.

## Backend for another UI

`tools.skill_review` has no Telegram dependency. A desktop integration can use it
under the authenticated owning profile's runtime-home scope:

```python
from tools import skill_review, write_approval

pending = write_approval.list_pending(write_approval.SKILLS)
record = skill_review.inspect(pending_id)
record = skill_review.revise(pending_id, revision, file_index, replacement_text)
result = skill_review.decide(pending_id, revision, file_index, "approve")
# Or decide(..., "reject") to cancel the entire transaction.
```

`inspect` returns a detached record with `revision`, and `skill_review.files` with
absolute `path`, `before`, `after`, `base_hash` and operation `indices`. A null
`before` means creation; a null `after` means removal. `approved` contains file
indices approved at this revision. Inspect does not write skill contents, although
it durably prepares a legacy pending record. `revise` increments the revision and
clears approvals. `decide` returns `record`, human-readable `status`, and `finished`.
Invalid/stale sources or revisions raise `ValueError`; write failure leaves the
proposal pending. Existing CLI explicit approval covers the entire transaction
and shares the same lock and validation.

This is a Python boundary, not an authenticated desktop RPC. A desktop contributor
must add its transport, authorization and UI, bind decisions to the displayed
revision, and handle runtime-home scope and errors. It must not replay the payload
directly or mutate the active conversation's prompt cache.

## Delivery

The existing Telegram delivery primitive accepts caller-provided cards and buttons.
Skill receipts are persisted per file and part, with bot/chat/topic/revision binding.
The existing profile scheduler retries failed delivery and wakes snoozed proposals
using `tools.write_approval.nag_hours`. Approved files are not prompted again.
Completed groups in an interrupted delivery are not duplicated. Telegram has no
send idempotency key: a request accepted by Telegram whose response is lost remains
ambiguous and may be repeated on retry.

Decision cards and edit requests request audible notifications. macOS Dock bouncing
still depends on the Telegram client's notification settings; the server cannot
force an operating-system attention animation.
