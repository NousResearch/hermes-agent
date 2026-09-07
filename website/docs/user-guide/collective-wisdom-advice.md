---
title: Collective Wisdom advice
---

Collective Wisdom can let your active Hermes conversation explain how a team
skill fits your setup before asking you to install, update, or share it.
This is an opt-in, local-profile delivery mode. Fixed notifications remain
the default.

## Enable agent-mediated delivery

First sign into your team and complete the existing `hermes wisdom setup`
disclosure. Then set this in that profile's configuration:

```yaml
wisdom:
  notifications:
    delivery_mode: agent
```

Keep the profile's other Wisdom settings. Restart its messaging gateway and
open a new local session after updating Hermes. Set `delivery_mode: fixed`
to return to fixed notifications. This setting does not enable sharing,
change organization policy, or change your installed skills' update policies.

## Advice and consent

Hermes uses the selected conversation's model and bounded conversation
context. Its automatic assessment cannot run shell commands, browse, read
arbitrary files, delegate, or install anything. Publisher text is untrusted;
usefulness and overlap are suggestions, not security or compatibility facts.

One recent, authorized private conversation receives the proactive advice.
Local qualifications stay with their originating conversation. If no eligible
conversation is active, activity waits. Other surfaces show the same advice
passively; opening them does not run another assessment.

- Use **Review first** to inspect canonical checks and requirements.
- Use the native **Install**, **Update**, or sharing control to consent.
- In the native or Dashboard CLI, use `/wisdom inbox`, then its exact
  `/wisdom consent` action. The standalone `hermes wisdom consent` command
  requires an interactive confirmation before applying.
- A conversational "yes" asks Hermes to present the control; it does not
  apply the operation.
- **Not Now** defers only the current surface. It does not reject the skill
  for your profile or organization.

Changed packages, local edits, additional requirements, and expired consent
require fresh review. Existing opted-in automatic updates continue unchanged.

## Recovery and limits

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

This mode does not coordinate between separate devices. It does not create a
general-purpose mailbox or run a periodic skill qualification scan.
