---
title: "Terminal Input"
description: "Visual cursor movement, selection ownership, clipboard receipts, and paste presentation"
---

# Terminal Input

The classic prompt_toolkit CLI and modern Ink TUI have separate input/rendering paths.
Their user-facing arrow-key and clipboard preferences are described in
[the CLI guide](../user-guide/cli.md#arrow-keys-and-text-selection).

## Ownership

| Concern | Classic CLI | Modern TUI |
| --- | --- | --- |
| Visual-row navigation | `hermes_cli/cli_cursor_navigation.py` uses rendered buffer mappings | `ui-tui/src/lib/inputMetrics.ts` maps wrapped rows to source offsets |
| Prompt edits/selection | `hermes_cli/cli_tui_mixin.py` | `ui-tui/src/components/textInput.tsx` |
| History boundary | Prompt navigation helper | `inputSelectionStore.ts` and `useInputHandlers.ts` |
| Printed reply selection | Terminal emulator | `ui-tui/packages/hermes-ink/src/ink/` |
| Clipboard transport | Terminal/toolkit path | Native writer in `lib/clipboard.ts`; Ink selection copy has native/OSC52 routing |
| Copy/cut receipts | Terminal-owned copies cannot be observed | `app/clipboardFeedbackStore.ts`, rendered by `StatusRule` after sessions |

All TUI paths in this table are relative to `ui-tui/src/` unless a full path is shown.

## Behavioral contracts

Up/Down move between visible rows before entering history. `display.arrow_key_mode: cursor`
keeps arrows inside the draft. Navigation preserves a preferred visual column through short
rows. The TUI maps normalized wrapped output back to original source offsets so decomposed
accents and emoji do not corrupt edit positions.

`display.tui_copy_on_select` controls application-owned mouse-selection copies independently
of terminal-native selection settings. Prompt selection colors use the skin's `selection_bg`.

`writeClipboard` only performs transport. `writeClipboardText` reports a completed copy.
Transactional cuts in `lib/inputClipboard.ts` use the transport directly, then announce
`cut` only after removal succeeds. A failed write preserves the draft. If the selected
range or its contents change while the write is pending, the completed clipboard write is
reported as a copy and the current text is preserved. Successful receipts count grapheme
clusters, last two seconds, and do not replace activity or credit notices.

Paste presentation is independent of copying. Existing collapse thresholds decide whether
the composer displays full text or a compact token. Submission expands retained paste
tokens into their original content. Bracketed paste replaces an active prompt selection
without submitting the draft.

## Verification

Cursor/input regressions live in `tests/hermes_cli/test_cli_visual_cursor.py` and the TUI's
`visualCursorNavigation`, `textInputVisualNavigation`, `textInputCut`, `clipboardFeedback`,
`inputSelectionClipboard`, `clipboard`, and `appChromeStatusRule` tests. Ink transport
receipts have a colocated `copy-selection.test.ts`. Existing attachment/submission tests
cover compact-paste expansion into the model payload.

`terminalInteractionEdges.test.tsx` drives the mounted composer through bracketed paste,
selection replacement, multiline editing, compact-token expansion, Unicode editing,
preferred-column navigation and narrow/wide layout changes. Paste handlers receive a
snapshot with the selected range removed; the editable draft changes only when paste
processing returns a successful result.

The real-process tests in `tests/e2e/core/terminal/` cover transcript integrity and
streaming resize on both interfaces. Their isolated HOME uses the prebuilt TUI and
an explicit Node executable, so terminal checks do not provision a toolchain.
`test_terminal_clipboard_pty.py` drives SGR mouse selection during a streamed reply,
checks the terminal clipboard payload and receipt placement/expiry, then resizes.
The glyph CLI's intentional Thought footer is excluded when checking for duplicated
reasoning body text.

Run Python tests through `scripts/run_tests.sh`, TUI tests through Vitest, and the normal
typecheck/build/repository checks. Validate actual terminal selection, clipboard round-trips,
resizing, and Unicode input by hand: unit tests cannot establish emulator/font behavior.
