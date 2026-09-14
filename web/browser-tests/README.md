# Native PTY input regression tests

The production path remains `ChatPage → xterm → onData → WebSocket → PTY → Ink`.
`installPtyBrowserInput` is the single owner of native textarea edits. It uses
xterm's public `element`, `textarea`, `input`, `paste`, and event APIs; the only
CSS coupling is xterm's existing `.composition-view` preedit display. Stock
`@xterm/xterm` remains pinned at 6.0.0; no patched installed bundle or external
checkout is required.

## Ownership contract

- Capture on xterm's ancestor element, **not** a later textarea listener. This
  prevents its keypress/input/composition and legacy keyCode-229 producers from
  emitting the same edit. Never use a timer or word equality to decide ownership.
- Keep textarea value as the acknowledged editable suffix. `beforeinput`
  captures value, selection, operation, original event, and the oldest pending
  printable key. `input` reconciles the actual browser mutation. Key release is
  the fallback boundary, with FIFO ordering under rollover.
- Native selection/caret movement does not move the PTY cursor. Replacements
  delete the changed terminal tail and reinsert the entire new tail, including
  unchanged suffix text. DEL units match the embedded Ink TextInput's graphemes
  (`ui-tui/src/components/textInput.tsx`, `graphemeStops`/`prevPos`), **not**
  UTF-16 units or prompt_toolkit's base-character erase rule.
- Composition owns its preedit until commit, with no independently armed
  fallback sender. A keyless line break commits composition before a single CR.
- Terminal selection/clipboard helper text is not input. Rebase it before the
  browser default mutation; never send helper newlines or deletes. Native
  textarea selection retains its own context menu and copy behavior.
- Navigation, protocol controls and paste invalidate the editable suffix.
  Unsupported browser operations do not invent a terminal edit. Clipboard
  paste retains xterm's line-ending and bracketed-paste handling. Programmatic
  consumers must call the returned adapter's `paste(text)`, not `term.paste`:
  an `onData` callback is too late to order older native edits before paste.
- Only plausibly keyed edits take a pending FIFO key. Keyless controls flush
  older keys before their own operation. Ctrl/Cmd+C with a native textarea
  selection bypasses xterm without preventing the browser's copy default.
- An overtaken edit claims only its expected input type, data and post-edit
  value (or the boundary-cleared insertion value). Mismatch, a newer edit,
  matching key release or lifecycle reset expires the claim; it is not a
  general next-input suppression flag.
- Input-only insertion never uses a whole-value destructive diff: without a
  pre-edit range, accept only an isolated insertion payload and rebase the
  mirror to that suffix. Ambiguous other mutations invalidate the mirror.
- Composition removal/reinsertion belongs to the composition owner, not an
  ordinary pending delete. The removal-before-finalization regression is a
  **synthetic legal event shape, not a recorded Safari trace**.
- The dashboard's connection gate rejects browser edits before they can become
  acknowledged state. Unmount disposes this owner before disposing xterm.

## Reproduce

From the repository root:

```sh
npm ci --workspace web --include-workspace-root
npm run --workspace web test
npm run --workspace web typecheck
npm run --workspace web build
npm run --workspace web lint
cd web
PLAYWRIGHT_BROWSERS_PATH="$PWD/.playwright-browsers" npx playwright install chromium --only-shell
PLAYWRIGHT_BROWSERS_PATH="$PWD/.playwright-browsers" npm run test:browser -- --project=chromium
```

The optional WebKit project uses `npx playwright install webkit` and
`npm run test:browser -- --project=webkit` with the same browser path. It needs
Playwright's Linux host libraries. Do not install system packages in a
repository-only task; report missing libraries instead. An existing Chrome
executable can alternatively be selected via `CHROMIUM_EXECUTABLE`.

The fixtures mount the **real shipped xterm package** and production browser
owner. Hardware typing uses Playwright's browser keyboard. Dictation/IME traces
replay native DOM event order and post-default textarea snapshots; these are
synthetic traces, not recordings from an iPhone. Chromium normalizes unsupported
`InputEvent.inputType` constructor strings to an empty string, so the
WebKit-only trailing-composition trace explicitly preserves its input type.
The jsdom ChatPage test additionally asserts the production socket-forwarding
integration (its terminal renderer and socket are fakes).

## Device acceptance still required

These tests do not provide a microphone, iPhone dictation service, Safari
selection handles, or the actual remote Ink process. Before deployment, verify
on an iPhone: repeated dictated words, phrase revisions and punctuation,
long-press-space middle edits, selection replacement, native copy/paste,
composition followed immediately by Return, emoji/combining edits, and
reconnection while input is pending. Check actual PTY text, not only textarea
contents. Full-screen programs can define different erase semantics; this
adapter is specifically for Hermes's embedded Ink input.

During local validation, Node 26 stalled while extracting Playwright archives;
using the installed Node 24 for `node node_modules/playwright/cli.js install …`
completed installation. Chromium launched and ran the tests. WebKit installation
completed but launch was blocked by missing GTK4, Graphene, GStreamer, AVIF,
Harfbuzz-ICU, Wayland, Manette, Enchant, Hyphen, Secret and WOFF2 libraries.
