# Native TUI session tabs

The terminal UI keeps directory-scoped views of opened conversations. Opening, selecting or closing a view is not a prompt submission and does not delete conversation history or interrupt a background turn.

## Commands and controls

- `/new [title]` or `/tabnew`: open another view without resetting the current chat.
- `/tabs` or `/tab`: list views in the current canonical directory.
- `/tab <number|title|ID>`, `/tab next`, `/tab prev`, `/tab new`, `/tab close`.
- `/tabnext` / `/next`; `/tabprev` / `/prev` / `/previous`; `/tabclose`.
- Click a label, **+**, or **×**. Alt+N opens, Alt+W closes, Alt+1–9 selects.
- Ctrl+PageUp/Down and Ctrl+Tab/Ctrl+Shift+Tab cycle when the terminal reports them. Plain Tab and Alt+Left/Right retain completion/word-navigation behavior; Ctrl+T and Ctrl+W retain agents/word-deletion behavior.

Local commands are derived from the dispatch registry for autocomplete as well as execution. Exact/prefix command-name matches rank ahead of skill-description matches. Argument/path completion and inline skill-only references keep their existing behavior. `/new` no longer aliases the destructive `/clear` command in the TUI; classic CLI behavior is unchanged.

## Persistence and boundaries

Layout metadata and unsent drafts are atomically saved under the launch profile's `ui-tui/session-tabs/` directory, keyed by canonical-directory hash. Files are mode 0600, and include plaintext draft input, multiline buffers and paste/image tokens. Never publish these files as plugin assets.

Relaunch in the same profile/directory to restore order, the selected durable session, and drafts. A layout read does not replay historical tools or submit a prompt. Invalid, corrupt, wrong-directory and oversized layouts are ignored. Explicit remote connections and hosted dashboard connections do not load local saved layouts. Closing a view with a draft is refused; closing the last view opens a new one.

The shared owner's existing roster supplies current running/blocked state. Previously saved busy flags are not treated as live after restart. State aggregation uses only the open views in this directory, not the global history list.

## Public launcher capability contract

```sh
hermes --tui-capabilities
```

After the standard Python bootstrap, the standalone flag returns JSON before normal CLI startup, update-debt recovery, plugin registration or chat dispatch. The normal bootstrap's interrupted-install safeguards still apply. It does not require a live gateway or model request. Other flags may be combined through argparse's terminating introspection action. Schema 1 has this shape:

```json
{"schema":1,"ui":"hermes-tui","features":{"session_tabs":1,"tab_commands":1,"tab_persistence":1,"background_activity":1,"slash_completion":1}}
```

Feature integers are contract versions, not a Hermes release-number claim. Launchers should first look for the flag in `hermes --help`, then validate the schema and required features. A missing flag or unsupported contract means native support cannot be assumed. The [CLI Session Tabs plugin](https://github.com/cryptobredda/hermes-cli-tabs) is the concrete consumer and keeps its existing isolated tmux fallback for older core builds. It never installs a source patch or accesses the native stores.

## Herdr

When a complete inherited Herdr pane identity is present, a client-only bridge reports aggregate state through Herdr's public CLI. Direct nanostore subscriptions avoid relying on an unfocused React paint. Reports are single-flight and coalesced to the latest state; a disconnected/unavailable Herdr must not break chat. Cleanup releases only this bridge's own reporting source.

Working in a background tab is visible even if the active tab is idle. Blocked state takes priority over working, and working over idle. No focused-pane guess is used.

A Hermes relaunch restoring its layout is distinct from Herdr relaunching Hermes on a whole-server restart. Herdr 0.9.1 ignores custom-source resume references, so automatic agent relaunch is not promised for that host version. The core contribution neither updates nor restarts Herdr.
