---
title: Hermes Lens
description: Collect web evidence in a visual board and compare selected sources with Hermes.
---

# Hermes Lens

Hermes Lens opens a research popup from the Desktop browser toolbar. Pin readable blocks
from different websites, add notes, revisit sources, and give Hermes a selected
set of evidence to investigate.

![Hermes Lens workflow](/img/hermes-lens.svg)

## Capture and compare

1. Open a website in the Desktop browser.
2. Select text within one paragraph, listing, or article. Click **Hermes Lens**
   in the browser toolbar, then **Pin selected block**. Lens captures the
   enclosing block, including the surrounding text needed to make sense of it.
   **Pin page** captures the main readable section instead.
3. Repeat on other websites. Cards keep the source URL, title, capture time,
   last-check time, and an optional note.
4. Select up to eight cards, enter a question, and click **Ask Hermes**.
   The selected evidence is appended to the active chat draft. Review and send
   it through the normal composer. Nothing is sent to a model merely by pinning.

The popup is available from any Desktop browser tab. It stays anchored to the
Lens toolbar button, with the webpage visible behind it. Press Escape or click
outside to dismiss it. Closing Lens leaves the
browser mounted, preserving its history and login state.

## Refresh a source

Use **Open source** to revisit a card's page. Then open Lens and click
**Refresh**. Lens reloads that already-open source, reads the captured block,
and retains the previous text when the content changes. **Last checked** shows
when that read completed; it does not imply continuous monitoring.

If the source is closed, redirected, or the block no longer exists, Lens asks
you to open or pin it again instead of replacing the evidence with unrelated
content. Locators are based on DOM IDs or element paths: a site redesign or
reordered list can require repinning. Check the source when interpreting a
change. Embedded frames, canvas drawings, images, and charts without readable
text are not captured in this version.

## Storage and platform support

Lens uses the existing Electron browser and renderer on Windows and macOS,
with no OS-specific commands, browser extension, external service, or new model
tool. The same code path also runs on Linux.

Cards are saved in Desktop's local browser storage, scoped to the connection
and profile of the current workspace using the browser rail's existing scope.
Profile rename and deletion migrate or remove the associated board. Another
window on the same installation sees card updates through storage events.
Cards are not synchronized to another computer or to the agent backend.

Each workspace holds up to 60 cards; a capture stores at most 6,000 characters
and explicitly shows truncation. A comparison includes up to eight cards.
Remove cards with **Remove**. Clearing Desktop browser storage removes boards.
Only HTTP(S) sources are supported. Private-page captures may contain private
information: they remain local until you choose to add them to a chat and send.

## Development verification

From the repository root, install the locked workspace dependencies with
`npm ci`. From `apps/desktop`, run:

```sh
npm run test:ui -- src/features/lens src/store/preview-tabs-scope.test.ts src/app/chat/right-rail/preview-browser-bar.test.tsx
npx playwright test --config e2e/lens/playwright.config.ts
```

The native integration test uses a disposable Electron user-data directory and
a local fixture website. It exercises real selection, pinning, notes, relaunch
persistence, source reload/change detection, and the composer insertion bus.
No provider credentials or model calls are required. The Lens workflow runs
these checks on Windows and macOS.
