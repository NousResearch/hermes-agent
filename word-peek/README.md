# Word Peek — Explore & build

Desktop entry: `<HERMES_HOME>/desktop-plugins/word-peek/plugin.js`.

The installed file is generated. Editable sources and tests live in `dev/` in the repository. From that directory run `npm install`, then `npm run build`. Copy `dev/dist/plugin.js` to the plugin's root `plugin.js` when preparing a release. The parser is bundled; the installed entry requires no Node packages.

For a local installation, set `HERMES_HOME` to the existing Hermes data root and run `node build.cjs --install`. This overwrites only the plugin entry, leaving dictionary data and stored glossary entries untouched. Reload the plugin or restart Hermes afterward. Without an override, the development scripts look for `~/.hermes`, then `%LOCALAPPDATA%/hermes`. `HERMES_CHECKOUT` can point to a Hermes source checkout and supply its installed development dependencies.

Optional `wn-dict.json` and `wn-thes.json` data files are not redistributed in this PR. Existing installations retain them. On a fresh installation without those files, the contextual glossary, inline questions, personal notes, and HTML editing still work; offline WordNet dictionary lookup reports unavailable.

## Workflow

1. Select text in one paragraph, heading, list item, or table cell of a local HTML preview, then right-click.
2. Select **Explore**, or enter a question and select **Ask**.
3. Read the answer and ask follow-up questions in the panel. **Use as my note** adds your own text without a model call.
4. Select **Expandable note** (default), **Insert below**, or **Replace selection**, then **Preview in HTML**.
5. Select **Apply to HTML**. The local file is backed up, written, checked, and its preview reloaded. **Undo last HTML change** restores the backup if the file has not changed since that save.

The panel keeps question history per captured passage and chat/profile during the current plugin lifetime. HTML additions persist in the file. Glossary entries and last-edit undo metadata persist in plugin storage. Backup files are kept next to the HTML with `.word-peek-<timestamp>.bak` suffixes.

## Scope and integrity

- Inline answers go through the main Hermes agent loop: the question is submitted as a normal chat turn to the passage's session, so the answer appears in the chat with full context and tools, and the panel shows it inline as well.
- Answers can carry reference links already present in the selected document; those links are not automatically fetched or verified, and answers citing any other link are rejected. The answer and saved note retain that attribution.
- Saving supports local `file:` HTML documents up to 900,000 characters, declared as UTF-8 when an encoding is present. Remote sites and opaque sandbox frames are not writable through this workflow.
- Source anchors include block text, a unique ID or structural path, and exact selection offsets. Edits preserve the rest of the source; replacement may normalize markup inside the edited block.
- Saves and undo reread and compare before overwriting. The desktop bridge does not provide an atomic compare-and-swap operation, so another program writing in the final read/write interval cannot be locked out.
- Generated text and links are escaped; model-provided HTML and unprovided source URLs are rejected. Interactive blocks and stale or ambiguous anchors require a new selection.

## Verification

Run `npm test` from `dev/`. Browser tests require installed Microsoft Edge. Set `HERMES_CHECKOUT` to a Hermes checkout containing `apps/desktop/src/contrib/runtime-loader.ts` for the loader regression test.

The browser suite uses Edge with the real bundled UI, real DOM selection, and fixture HTML reads/writes. The Hermes model and desktop bridge are mocked to their inspected API contracts. It covers follow-ups, citations, preview before apply, backup/undo, concurrent-edit refusal, replacement, and narrow layouts. The loader test executes the actual Hermes runtime loader with mocked surrounding application services, including its import validation and blob-module evaluation.

Live gateway/model and native Electron webview behavior still require a manual smoke test. The original browser tests bypassed Hermes' import scanner and missed a load failure: a quoted stop word was mistaken for an import specifier. The stop-word representation now avoids that scanner false positive, and the actual-loader test covers load and registration.

