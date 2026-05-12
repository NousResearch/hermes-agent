---
title: Google Messages for Web
---

# Google Messages for Web checker

Hermes can inspect Jake's existing Pixel/Android text-message inbox through Google Messages for Web using a local Playwright browser profile. This is separate from the Twilio-backed SMS gateway: Twilio lets people text a Hermes-owned number, while this checker previews the inbox already paired to Google Messages on the phone.

Milestone 1 is intentionally read-only and boring in the best possible way:

- open `https://messages.google.com/web` in a dedicated persistent browser profile;
- report whether manual QR pairing/login appears needed;
- extract only the visible conversation list: sender/name, timestamp if visible, snippet, and a defensive unread-ish flag;
- never open individual threads by default;
- never type, draft, or send messages.

## Setup

Install Playwright if it is not already available in the Hermes environment:

```bash
pip install playwright
python -m playwright install chromium
```

Enable or request the `google_messages` toolset for the Hermes session/profile that should use it. The toolset exposes:

- `google_messages_status` — opens Google Messages for Web and reports `ready`, `pairing_required`, or `unknown`.
- `google_messages_conversations` — returns visible conversation-list previews only.

By default both tools use this persistent profile:

```text
~/.hermes/browser-profiles/google-messages
```

That dedicated profile keeps Google Messages cookies/session state away from the normal browser automation profile and makes it easier to revoke, inspect, or delete the pairing later.

## Manual QR pairing

The first status check should usually run non-headless so the QR code can be seen:

```text
google_messages_status(headless=false)
```

On the Pixel:

1. Open Google Messages.
2. Open device pairing / Messages for web.
3. Scan the QR code shown in the Hermes-controlled browser window.

After pairing, future checks can reuse the persistent profile. If the profile is deleted, if Google expires the session, or if the phone unpairs the browser, `google_messages_status` should report that pairing appears needed again.

## Privacy and read-status caveats

This checker is for preview triage, not durable archiving.

Important caveats:

- Conversation-list snippets may still contain private message text and may enter Hermes transcripts/logs if an agent includes them in a response. Do not store more than necessary.
- Opening an individual conversation may mark it read. Milestone 1 does not open threads for exactly this reason.
- Google can change the Messages Web DOM at any time. Selectors and unread detection are defensive best-effort, not a contract.
- RCS/SMS/MMS behavior depends on what Google Messages exposes through the paired web UI.
- Sending is out of scope. Future reply support should stay draft-only/manual until read-status and consent boundaries are proven.

## Removing local state

To force a fresh pairing, close any checker browser window and remove the dedicated profile directory:

```bash
rm -rf ~/.hermes/browser-profiles/google-messages
```

Also unpair the browser from the Pixel's Google Messages device-pairing screen when you no longer want Hermes to have access.
