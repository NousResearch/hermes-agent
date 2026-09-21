---
title: Third-Party Titlebar Buttons
sidebar_label: Third-Party Titlebar Buttons
---

# Third-Party Titlebar Buttons

This fork carries a Windows Desktop fix for caption-button tools provided by products such as DisplayFusion, Actual Window Manager, and AutoHotkey. Those tools add buttons to the native titlebar, but Chromium's `windowControlsOverlay` only reports Electron's own caption controls. Without an extra reservation, Hermes titlebar controls can sit underneath third-party buttons.

## User setting

The control is in:

**Settings → Appearance → Window Layout → Third-Party Titlebar Buttons**

Choose **0–4** buttons; the default is **0**. Hermes derives the reserved width from the measured native overlay, so it tracks DPI and UI scaling.

## Implementation

The feature is split deliberately:

- `src/store/titlebar-external-buttons.ts` persists the 0–4 choice.
- `src/app/settings/appearance-settings.tsx` renders the control inside the redesigned `window-layout` Appearance subpage.
- `src/app/contrib/wiring.tsx` translates the choice and native overlay width into titlebar CSS values.
- `src/app/shell/titlebar.ts` calculates the external-button reservation and preserves the session-title inset.

## Maintaining the local installation

Hermes desktop updates rebuild from upstream `main` and park local source edits. Use the companion tool at `C:\Scripts\HermesDesktopTitlebarFix` after an update.

Its `Reapply-Hermes-Desktop-Fix.ps1` checks four states before reporting success:

1. titlebar source logic;
2. Appearance → Window Layout setting placement;
3. built renderer asset marker;
4. installed app asset marker.

It then applies the regenerated 12-file patch with `git apply -3`, builds a staged `release-next` package, and swaps it in after Hermes closes.

Do not retire this local tool merely because a patch happens to apply cleanly. Retire it only after an upstream release includes equivalent titlebar reservation **and** the visible Appearance → Window Layout control.
