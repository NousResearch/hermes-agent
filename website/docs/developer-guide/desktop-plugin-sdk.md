---
sidebar_label: "Desktop Plugin SDK"
title: "Desktop Plugin SDK (@hermes/plugin-sdk)"
description: "Extend the native Hermes Desktop app — panes, pages, sidebar nav, status bar, palette commands, keybinds, themes, and a scoped backend namespace, with one import and no build step."
---

# Desktop Plugin SDK

The native [Hermes Desktop](/user-guide/desktop) app is contribution-driven: every
surface in the window — panes, routes, sidebar nav, status-bar items, palette
entries, keybinds, themes — registers into one central registry. Core registers
its surfaces exactly the way a plugin does, so the plugin story is the real one,
not a bolted-on afterthought.

A **desktop plugin** is a single ESM file that default-exports a `HermesPlugin`.
It imports one module — `@hermes/plugin-sdk` — and gets everything: the app's
live state, the gateway JSON-RPC door, a scoped REST/socket backend namespace,
React Query, and the app's own UI kit so plugin UI looks native by default. No
repo clone, no `npm run build`, no patching app source. Drop the file in
`$HERMES_HOME/desktop-plugins/<id>/plugin.js` and the app loads it within seconds
and hot-reloads every save.

:::warning This is not the web-dashboard plugin SDK
"Plugin" means several unrelated things across Hermes. This page is the **native
desktop app** (`hermes desktop`) SDK — the `@hermes/plugin-sdk` module and
`$HERMES_HOME/desktop-plugins/`. The **web dashboard** (`hermes dashboard`) has
its own, unrelated plugin system on `window.__HERMES_PLUGIN_SDK__` with a
`manifest.json` — documented at
[Extending the Dashboard](/user-guide/features/extending-the-dashboard). Python
CLI/gateway plugins are documented at [Build a Hermes Plugin](/developer-guide/plugins).
The three do not share code, APIs, or delivery. Only the backend `plugin_api.py`
namespace (`/api/plugins/<id>`) is shared between the desktop and dashboard SDKs.
:::

## Mental model

The SDK follows the VS Code module model. A plugin author imports exactly one
module and never touches app internals (they are lint-fenced out of a bundled
plugin, and fail to resolve in a disk plugin). Capability comes in tiers:

- **`host.state.*`** — readonly views over the app's live state (nanostore
  atoms): active session, per-session turn-busy, cwd, gateway socket status,
  model, profile, viewport. `gateway` is the WebSocket, not turn-busy.
- **`host.*` actions** — curated safe verbs: toast, navigate, tail logs,
  restart the gateway, subscribe to the gateway event stream.
- **`host.request`** — the gateway JSON-RPC door: sessions, config, skills,
  cron — everything the app itself calls.
- **`ctx.rest` / `ctx.socket`** — your plugin's own backend namespace
  (`/api/plugins/<id>`) if you ship a `plugin_api.py`.
- **`ui.*`** — the design language: the app's real components, theme variables,
  icons, and formatters, so your UI matches the app pixel-for-pixel.

## Two delivery modes

| Mode | Where | Who | Build step |
|------|-------|-----|------------|
| **Disk** (recommended) | `$HERMES_HOME/desktop-plugins/<id>/plugin.js` | users, agents | none — plain ESM, loaded uncompiled |
| **Unified package** | `$HERMES_HOME/plugins/<id>/desktop/plugin.js` | plugins that also ship agent-side code | none — same disk pipeline |
| **Bundled** | `apps/desktop/src/plugins/<id>/plugin.tsx` | in-tree, shipped with the app | the app's own Vite build |

All three take the same `HermesPlugin` contract, appear in **Settings → Plugins**,
and enable/disable live. A unified package is just the disk door scanning inside
your agent plugin's folder — see
[One package, both SDKs](#one-package-both-sdks). Everything on this page is
written against the disk door (what you and the agent write);
[Bundled plugins](#bundled-plugins) notes the two
differences. No desktop plugins ship in the core tree today — reference demos
live in the companion
[`hermes-example-plugins`](https://github.com/NousResearch/hermes-example-plugins)
repo.

## Quick start — your first plugin

Create `$HERMES_HOME/desktop-plugins/hello/plugin.js` (that's `~/.hermes/...`
by default, or `~/.hermes/profiles/<name>/...` under a named profile). The folder
name must equal the plugin `id`.

```javascript
// ~/.hermes/desktop-plugins/hello/plugin.js
import { host, haptic, useValue } from '@hermes/plugin-sdk'
import { jsx, jsxs } from 'react/jsx-runtime'

function HelloPane() {
  const gateway = useValue(host.state.gateway)

  return jsxs('div', {
    className: 'flex h-full flex-col gap-2 p-3 text-sm',
    children: [
      jsx('div', { className: 'font-medium', children: 'Hello, Hermes' }),
      jsx('div', {
        className: 'text-(--ui-text-tertiary)',
        children: `gateway: ${gateway}`
      })
    ]
  })
}

export default {
  id: 'hello', // must match the folder name
  name: 'Hello',
  register(ctx) {
    ctx.register({
      id: 'pane',
      area: 'panes',
      title: 'hello',
      data: { placement: 'right', width: '260px' },
      render: () => jsx(HelloPane, {})
    })
    ctx.register({
      id: 'chip',
      area: 'statusBar.right',
      order: 130,
      render: () =>
        jsx('button', {
          type: 'button',
          className: 'px-1.5 text-[0.6875rem] text-(--ui-text-tertiary)',
          onClick: () => {
            haptic('tap')
            host.notify({ kind: 'info', message: 'Hello from my plugin!' })
          },
          children: 'hello'
        })
    })
  }
}
```

Save it. The app watches `desktop-plugins/`, loads the file within a few seconds,
and hot-reloads every later save in place. If it doesn't appear, run ⌘K →
**Reload desktop plugins**. If loading fails, a toast names the error — fix and
save again.

:::note No JSX, no build
The disk file is loaded **uncompiled**, so JSX syntax will not parse. Write UI
with `jsx()` / `jsxs()` calls from `react/jsx-runtime` (or `React.createElement`).
The only importable specifiers are `@hermes/plugin-sdk`, `react`, and
`react/jsx-runtime` — everything else fails to resolve, on purpose.
:::

## The plugin contract

A plugin default-exports a `HermesPlugin`:

```ts
interface HermesPlugin {
  /** Stable slug — becomes the `plugin:<id>` source and the id namespace. */
  id: string
  /** Human name for Settings / about UI. Defaults to `id`. */
  name?: string
  /** Registers on load when the user hasn't chosen (default true). Set false
   *  for opt-in plugins: they inventory in Settings ▸ Plugins, off until the
   *  user flips the switch. */
  defaultEnabled?: boolean
  /** Called once at load; wire contributions through `ctx`. */
  register: (ctx: PluginContext) => void
}
```

`register` receives a **scoped** `PluginContext`. It never touches the registry
directly — the context auto-tags provenance (`source: 'plugin:<id>'`) and
namespaces every contribution id (`<id>:<localId>`), so two plugins can never
collide.

```ts
interface PluginContext {
  /** Resolved source tag, e.g. `'plugin:hello'`. */
  readonly source: string
  /** Register one contribution (id namespaced, source stamped). Returns a disposer. */
  register: (c: PluginContribution) => () => void
  /** Register several at once; the returned disposer removes all of them. */
  registerMany: (cs: PluginContribution[]) => () => void
  /** REST to this plugin's own backend namespace (`/api/plugins/<id>`). */
  rest: <T>(path: string, opts?: PluginRestOptions) => Promise<T>
  /** Live WebSocket to this plugin's own namespace. Returns a disposer. */
  socket: (path: string, onMessage: (data: unknown) => void) => () => void
  /** The curated OS door: native notification, open-external, reveal-in-file-manager, clipboard. */
  os: PluginOs
  /** Plugin-scoped JSON persistence (keys live under `hermes.plugin.<id>.`). */
  storage: PluginStorage
}
```

A **contribution** is the one primitive every surface shares:

```ts
interface Contribution {
  id: string          // you write the local id; the host namespaces it
  area: string        // WHERE it goes (a contribution-area constant)
  title?: string
  order?: number      // sort within the area (lower = earlier)
  when?: () => boolean // dynamic visibility; re-evaluated by the area
  enabled?: boolean
  render?: () => ReactNode  // the component to mount
  data?: unknown      // area-specific payload (see the cookbook)
}
```

You provide `render`, `data`, or both, depending on the area.

## Contribution areas — the cookbook

Import the area constants from the SDK; each area has its own `data` payload.

| Surface | `area` | You provide |
|---------|--------|-------------|
| Layout pane | `PANES_AREA` (`'panes'`) | `title` + `render` + `data: { placement, dock?, width?, height? }` |
| Full page | `ROUTES_AREA` | `data: { path }` + `render` |
| Sidebar nav | `SIDEBAR_NAV_AREA` | `data: { path, label, codicon }` |
| Status bar | `STATUSBAR_AREAS.left` / `.right` | `render` (or `data` as `StatusbarItem`) |
| Title bar | `TITLEBAR_AREAS.left` / `.center` / `.right` | `data` as `TitlebarTool`, or a mount-scoped `<Contribute>` |
| ⌘K palette | `PALETTE_AREA` | `data: PaletteContribution` |
| Keybind | `KEYBINDS_AREA` | `data: KeybindContribution` |
| Theme | `THEMES_AREA` | `data` as a `DesktopTheme` |
| Composer | `COMPOSER_AREAS.*` | render slots, or middleware / attachment providers |

### Panes

A pane is a tile in the layout tree. `placement` is the semantic role — the pane
stacks (as tabs) with existing panes of that role; the user can drag it anywhere
afterward.

```javascript
ctx.register({
  id: 'pane',
  area: 'panes',
  title: 'my pane',
  data: { placement: 'right', width: '260px' },
  render: () => jsx(MyPane, {})
})
```

`placement` is `'main' | 'left' | 'right' | 'top' | 'bottom'`. To land on a
specific **edge** instead of stacking, add a `dock` gesture — the same thing as
dragging onto a pane's drop chip:

```javascript
// Below the conversation, 200px tall.
data: {
  placement: 'bottom',
  dock: { pane: 'workspace', pos: 'bottom' },
  height: '200px'
}
```

`dock.pane` is any pane id (`workspace` is the main thread; also `sessions`,
`terminal`, `files`, `review`, `logs`); `dock.pos` is
`'top' | 'bottom' | 'left' | 'right' | 'center'`. Declare a `width`/`height` so
the pane doesn't claim half the zone.

Closing the only pane contributed by a plugin disables that plugin, which can
be re-enabled from **Settings → Plugins**. When a plugin contributes multiple
panes, closing one dismisses only that pane and leaves the plugin's other panes,
commands, and middleware active. **Reset layout** restores dismissed contributed
panes.

### Pages and sidebar nav

A route mounts a full page in the workspace pane, like any built-in view. Pair it
with a sidebar nav row (and/or a palette command) to make it reachable.

```javascript
import { ROUTES_AREA, SIDEBAR_NAV_AREA } from '@hermes/plugin-sdk'

ctx.registerMany([
  {
    id: 'page',
    area: ROUTES_AREA,
    data: { path: '/my-page' },
    render: () => jsx(MyPage, {})
  },
  {
    id: 'nav',
    area: SIDEBAR_NAV_AREA,
    data: { path: '/my-page', label: 'My Page', codicon: 'project' }
  }
])
```

`codicon` is a [VS Code codicon](https://microsoft.github.io/vscode-codicons/dist/codicon.html)
id. Navigate to a route from anywhere with `host.navigate('/my-page')`.

### Status bar and title bar

Status-bar items render into the left or right cluster of the bottom bar.
Simplest is a `render` function; for a plain button use `data` as a
`StatusbarItem` (`{ id, label?, icon?, detail?, variant?, menuItems?, … }`).

```javascript
import { STATUSBAR_AREAS, TITLEBAR_AREAS } from '@hermes/plugin-sdk'

ctx.register({
  id: 'count',
  area: STATUSBAR_AREAS.right,
  order: 120,
  render: () => jsx(MyStatus, {})
})
```

Title-bar tools live in `TITLEBAR_AREAS.left | .center | .right` as `TitlebarTool`
data (`{ id, label, icon, active?, onSelect? }`).

### Palette commands and keybinds

```javascript
import { PALETTE_AREA, KEYBINDS_AREA } from '@hermes/plugin-sdk'

ctx.registerMany([
  {
    id: 'open',
    area: PALETTE_AREA,
    data: {
      id: 'my-page.open',
      label: 'Open My Page',
      keywords: ['my', 'page'],
      run: () => host.navigate('/my-page')
    }
  },
  {
    id: 'refresh',
    area: KEYBINDS_AREA,
    data: {
      id: 'my-page.refresh',
      label: 'Refresh My Page',
      category: 'My Plugin',
      defaults: ['mod+shift+r'],
      run: () => void doRefresh()
    }
  }
])
```

Keybinds are user-rebindable in settings; `defaults` is just the initial binding.

### Themes

A theme contribution ships a full `DesktopTheme` as its `data` (name, label,
colors, …). It appears in the theme picker like a built-in.

```javascript
import { THEMES_AREA } from '@hermes/plugin-sdk'

ctx.register({ id: 'noir', area: THEMES_AREA, data: myDesktopTheme })
```

Registering a theme lists it; it does not select it. `useTheme()` reads the
painted appearance (`theme`, `themeName`, `availableThemes`, `resolvedMode`) and
changes it (`setTheme`, `setMode`, `previewTheme`) from a component:

```javascript
import { Button, useTheme } from '@hermes/plugin-sdk'

function ThemePicker() {
  const { availableThemes, setTheme, themeName } = useTheme()

  return availableThemes.map(t => (
    <Button key={t.name} disabled={t.name === themeName} onClick={() => setTheme(t.name)}>
      {t.label}
    </Button>
  ))
}
```

A switch driven by something other than a render — a gateway connecting, a
socket event, any `host.onEvent` callback — has no component to hang the hook
on. Use `requestTheme(name)` there. An unresolvable name is refused rather than
coerced to the default skin, so the return value doubles as the availability
check and a wrong name can never silently reset someone's appearance:

```javascript
import { host, requestTheme } from '@hermes/plugin-sdk'

host.onEvent('gateway.ready', () => {
  if (!requestTheme('noir')) {
    host.notifyError('Connected, but the noir theme is not installed.')
  }
})
```

Both doors persist per profile, so a plugin-driven switch sticks exactly like a
manual pick. To tint the *active* theme rather than replace it, use
`setAccentOverride(hex)` and clear it in `ctx.onDispose` — the bundled `accent`
plugin is the worked example.

### Composer extensions

`COMPOSER_AREAS` (`top`, `bottom`, `leading`, `actions`, `attachments`,
`middleware`) let a plugin add controls around the message composer, provide an
attachment source, or transform a draft before it is sent (`ComposerMiddleware`
with a `handler(draft) => draft | null`).

### Transcript directives — inline components the model addresses

`TRANSCRIPT_DIRECTIVE_AREA` makes the transcript itself a contribution area.
Register a named directive and the agent can render your component inline in
an assistant message by emitting a paragraph of the form `::name{key="value"}`:

```javascript
import { TRANSCRIPT_DIRECTIVE_AREA } from '@hermes/plugin-sdk'

ctx.register({
  id: 'task-card',
  area: TRANSCRIPT_DIRECTIVE_AREA,
  data: {
    name: 'task', // the model writes ::task{id="BB-12"}
    render: ({ attrs, streaming }) => jsx(TaskCard, { taskId: attrs.id, streaming })
  }
})
```

Rules the host enforces so the surface stays safe:

- The directive must be the **entire paragraph** — `::name` mid-prose stays
  prose, so plugin components can never hijack running text.
- Attributes are **untrusted model output** (`key="value"` pairs, string-only).
  Validate your own fields; render nothing on garbage rather than guessing.
- An **unclaimed** directive (no plugin registered for the name) renders as
  the plain paragraph it always was — nothing breaks when a plugin is off.
- Renders are wrapped in the contribution error boundary: a throw degrades to
  an inline error chip, never a dead message.
- First registration wins on a name collision; namespace adventurous names
  with your slug (`myplugin-board`, not `board`).

Core ships one directive as the reference consumer: `::preview{file="…"}`
renders the workspace HTML file **live inside the message** — a sandboxed
`srcdoc` iframe with an opaque origin (scripts run and the widget is fully
interactive; no reach into the app, its storage, or the bridge). The frame
sizes itself to the content (height live, width adopted from the content's
intrinsic span, flush left in the message flow), and a theme prelude hands
the document the app's resolved tokens (`--foreground`, `--muted-foreground`,
`--accent`, `--border`, `--card`), the app font, and a transparent
background — so widget-shaped HTML reads as native while a full page keeps
its own design. Non-HTML targets and remote gateways fall back to the
classic preview card. Tell the agent about your directive in a skill (that's
how it learns to emit it).

Previewed widgets can also **talk back**. Inside the frame,
`window.hermes.send('get-price eth')` (or a declarative
`<button data-hermes-send="get-price eth">` — no script needed) hands that
prompt to the agent as a user turn, off-screen: no bubble takes up the
transcript, the widget updating is the visible response. The turn is still
real — it wakes the agent, rides the composer's steer/queue rules, and
persists (typed `hidden`) so resume and the session DB keep the full record.
Prompts are trimmed, capped at 500 chars, and throttled to one per second
per frame.

### Mount-scoped chrome (`Contribute`)

`ctx.register` is for **permanent** contributions. When chrome should live and
die with a component that's already on screen (a page's own title-bar control
leaves when the page unmounts), render `<Contribute>` inside it instead:

```javascript
import { Contribute, TITLEBAR_AREAS } from '@hermes/plugin-sdk'

jsx(Contribute, {
  area: TITLEBAR_AREAS.center,
  id: 'my-page:switcher', // namespace with your slug
  children: jsx(MySwitcher, {})
})
```

It registers on mount and disposes on unmount automatically.

## Host API

Everything on `host` is reachable from anywhere in a plugin. State atoms are
readonly — read with `.get()` in handlers, subscribe with `useValue(atom)` in
components.

```ts
host.state.activeSessionId  // ReadableAtom<string | null>
host.state.awaitingResponse // ReadableAtom<boolean>  true until the first assistant payload
host.state.busy             // ReadableAtom<boolean>  focused chat is working after a send
host.state.busyBySession    // ReadableAtom<Record<string, boolean>>  runtime id → mid-turn
host.state.focusedSessionId // ReadableAtom<string | null>  (runtime id of the FOCUSED session — tile-aware; prefer for session.* RPC)
host.state.focusedSessionProfile // ReadableAtom<string>  (owner profile of the focused chat — prefer over `profile` for per-bot/profile readouts)
host.state.focusedStoredSessionId // ReadableAtom<string | null>  (durable id — navigation / session-list matching)
host.state.focusedUsage     // ReadableAtom<UsageStats | null>  (live streamed usage of the focused session, no RPC needed)
host.state.cwd              // ReadableAtom<string>
host.state.gateway          // ReadableAtom<string>  socket state ('idle' | 'connecting' | 'open' | …)
host.state.model            // ReadableAtom<string>
host.state.profile          // ReadableAtom<string>
host.state.viewport         // ReadableAtom<{ width, height, narrow }>
```

`host.state.gateway` is the WebSocket connection, not whether a chat turn is
running. A session can be mid-turn while the socket is `open`; another session
can be idle at the same time. Disable composer or plugin actions from the
**focused session's** turn-busy (`host.state.busyBySession[sessionId]`, or that
session's `view.$busy`) — never from `gateway`, and never from a process-global
busy flag.

```ts
host.notify({ kind, message, title?, detail?, action? })  // toast; returns id
host.notifyError(error, fallbackMessage)                   // toast an error
ctx.os.notify({ title, body?, silent?, icon?, activate?, onActivate?, actions? })
                                           // native OS notification (attributed to your plugin)
ctx.os.openExternal(url)                   // OS default handler (browser, mail, spotify:) → Promise<boolean>
ctx.os.revealPath(path)                    // reveal in Finder / Explorer → Promise<boolean>
ctx.os.writeClipboard(text)                // system clipboard → Promise<boolean>
host.navigate('/route')                    // hash-route navigation
host.openSession(id, { profile?, intent? }) // open a stored session core-style;
                                           //   profile: soft-swap to that profile's backend first
                                           //   intent: 'in-place' (default) | 'stack' | 'tab' | 'window'
host.newChat(profile?)                     // fresh chat draft, optionally in another profile
host.openWorkspace(id, { render, title?, minWidth?, onClose? })
                                           // dock a plugin-rendered tab into the MAIN
                                           //   workspace zone and reveal it; returns a disposer
host.paneVisibility(paneId)                // ReadableAtom<boolean> — is a contributed pane
                                           //   actually on screen (its zone's active tab)?
host.onEvent(type, fn)                     // gateway event stream ('*' = all); returns disposer
host.logs(...)                             // tail an app log file
host.status()                              // one-shot system status snapshot
host.restartGateway()                      // restart the backend gateway
host.profileRoutes()                       // [{ profile, targetProfile, connectionId, mode }]
host.requestProfile<T>(route, method, params?)   // registry-routed RPC; no foreground swap
host.requestProfile<T>(profile, method, params?) // legacy v1/local overload
host.request<T>(method, params?)           // active-gateway JSON-RPC — the real power
```

`host.request` is the same JSON-RPC the app itself uses (sessions, config, skills,
cron, kanban, …). `host.requestProfile` accepts a descriptor from
`host.profileRoutes()` and routes that RPC through its exact registry source and
profile without changing the active chat or gateway. The profile-only overload is
retained only for the sole-local/legacy topology; registry-aware plugins should pass
the descriptor so two sources exposing the same profile name cannot collide.

`host.openWorkspace(id, { render, title?, minWidth?, onClose? })` docks a
plugin-rendered view into the **main workspace zone** — the same center area
session tiles and previews use — as a tab, and reveals it. Re-calling it with
the same `id` refreshes the content in place and re-fronts the tab instead of
opening a duplicate. Closing the tab (the tab's Close control or ⌘W) tears the
registration down and fires your `onClose`; the returned disposer closes it
programmatically. Feature-detect it (`typeof host.openWorkspace ===
'function'`) and fall back to a regular contributed pane on older desktop
builds — Bot Mode's group-chat rooms are the reference consumer (main-window
takeover when available, in-panel view otherwise).

`host.paneVisibility(paneId)` returns a readonly reactive atom that is `true`
while a contributed pane is actually on screen: present in the layout tree,
not dismissed or hidden, its zone un-minimized, and holding its zone's active
tab slot (a lone pane in its own zone counts). The id is the
contribution-scoped pane id, `<pluginId>:<paneId>`. Atoms are memoized per id,
so calling it in render is safe. Use it to register companion UI only while
your pane is visible — Bot Mode's Cronjobs pane is the reference consumer: it
registers while the Bots pane holds the sidebar tab and unregisters when the
user tabs back to Sessions. Feature-detect on older desktops
(`typeof host.paneVisibility === 'function'`) and fall back to
always-registered behavior.

`host.profileRoutes()` inventories every registered source in the current connection
registry. Connect-on-demand SSH sources expose a credential-free `default` seed
route without opening a tunnel, so a plugin can be the first caller that dials them;
an SSH `remoteProfile` remains the route's backend `targetProfile`. `connectionId`
is the registry routing identity;
pair it with `profile` for keys and persistence. Endpoint, token, SSH host/key, and
other raw connection fields never cross the plugin IPC boundary. `profile` is the
source-local route used
for requests; `targetProfile` is the backend Hermes profile served by that route.
They differ when a route explicitly maps to another backend profile (for example an
SSH `remoteProfile` override or a legacy per-profile URL alias). This distinction
preserves backend identity without exposing connection secrets.

Profile-shaped plugins get first-class methods too:
`profiles.list` (each profile + its most recent conversation as
`last_session`; pass `include_sessions: false` to skip the per-profile DB
probe; pass `preferred_session_ids: { profileName: sessionId }` for an
exact, existence-checked lookup of one pinned session per profile — each
named row gains a `preferred_session` summary that resolves hidden rows
and compression lineages to their live tip, or `null` when the id is
definitively gone; older gateways ignore the param and omit the field)
and `profiles.create` (`name`, `description`, `clone_from`,
`clone_all`, `no_skills`, `soul`, optional `model` + `provider` pin) — the
ws twins of the dashboard's `/api/profiles` REST routes.
`host.state.busy` is the focused chat's live turn (thinking and streaming).
`host.state.awaitingResponse` stays true from send until the first assistant
payload. Both follow the chat the user is actually looking at — the focused
session tile when one holds focus, else the primary workspace chat (the same
signal the statusbar's busy pulse reads). Subscribe in a component:

```javascript
const busy = useValue(host.state.busy)
```

For token-level detail, listen with `host.onEvent` (`message.start`,
`message.delta`, `message.complete`).

`host.onEvent` streams live gateway events (message deltas,
session lifecycle, tool activity). Listeners are isolated — a throw in your
listener can't affect app dispatch. Every `host` door is async-safe: a sync throw
from an internal helper (e.g. no desktop bridge in a plain browser) becomes a
rejection your `.catch()` sees, never an error-boundary crash.

`ctx.os` is the curated OS door — every way a plugin reaches outside the app
window, in one namespace attributed to your plugin. `ctx.os.notify` posts a
**native OS notification** — the same Electron pipeline the app's own
approval/turn alerts use. It fires only while the user is away from Hermes
(backgrounded / unfocused); use `host.notify` for the in-app toast when
they're looking at the app. Users can silence it per device under Settings ▸
Notifications ▸ "Plugin notifications", and repeats from the same plugin are
throttled, so treat it as a signal for genuinely notable events — not a log.

Rich presentation + activation (extends the original `ctx.os` door):

```ts
ctx.os.notify({
  title: 'New match found',
  body: 'Someone matched your signal',
  icon: '/abs/path/to/icon.png', // Electron Notification icon
  // Body click → focus Hermes + navigate. Same vocabulary as OS deep links:
  activate: 'hermes://index-network/intent/1',
  // or: activate: '/index-network/intent/1'
  // or: activate: { path: '/index-network/intent/1' }
  onActivate: () => focusLocalState('1'), // optional renderer callback
  actions: [
    { id: 'open', label: 'Open', activate: 'hermes://index-network/intent/1' },
    { id: 'dismiss', label: 'Dismiss', onAction: () => dismiss('1') },
  ],
})
```

`activate` is deeplink-compatible: `hermes://index-network/intent/1` and the
hash path `/index-network/intent/1` resolve to the same in-app route (and the
same `hermes://…` URL works as an OS deep link). Action buttons only render on
signed macOS builds; elsewhere the body click still activates. Navigation only
happens on user click — never from a background event alone.

The other doors (`openExternal`, `revealPath`, `writeClipboard`) resolve
`false` instead of throwing when the capability isn't available (older desktop
shell, plain browser) — branch on the result rather than sniffing the bridge.

## Data layer — React Query + nanostores

Plugins share the app's single `QueryClient`, so plugin queries cache, dedupe,
poll, and invalidate exactly like core screens — never hand-roll a fetch loop.

```javascript
import { useQuery, useMutation, useQueryClient, atom, computed, useValue } from '@hermes/plugin-sdk'

function MyPanel() {
  const { data, isLoading } = useQuery({
    queryKey: ['my-plugin', 'items'],
    queryFn: () => host.request('my.list', {})
  })
  // …
}
```

For state shared between a trigger and its panel (or a poll loop), use `atom` /
`computed` — the same primitive `host.state` uses. Subscribe in the leaf that
renders the value with `useValue`. To invalidate a query from **outside** React
(e.g. a `ctx.socket` frame arriving), import the shared `queryClient`:

```javascript
import { queryClient } from '@hermes/plugin-sdk'

ctx.socket('/events', () => {
  queryClient.invalidateQueries({ queryKey: ['my-plugin', 'items'] })
})
```

## The UI kit and theming

Import the app's real components directly so your UI is native by default:

> `Button`, `Input`, `Textarea`, `Select*`, `Switch`, `Checkbox`,
> `SegmentedControl`, `Tabs*`, `Dialog*`, `ConfirmDialog`, `DropdownMenu*`,
> `ContextMenu*`, `Popover*`, `Tip`/`Tooltip*`, `Badge`, `Kbd`/`KbdGroup`,
> `SearchField`, `ScrollArea`, `Separator`, `Skeleton`, `GlyphSpinner`, `Loader`,
> `EmptyState`, `ErrorState`, `CopyButton`, `StatusDot`, `LogView`, `Codicon`,
> `DecodeText`.

Plus helpers: `cn` (class merge), `icons.*` (the app's lucide set), `haptic`,
`profileColor` / `profileColorSoft` (deterministic identity colors), the time
formatters `relativeTime` / `fmtDateTime` / `fmtDayTime` / `coarseElapsed`,
`useI18n` (localized copy — your plugin stays translatable), and
`evaluateRuntimeReadiness`.

**Style with theme variables, never hardcoded colors.** Panes already sit on the
app's editor background — leave the background alone and use vars for everything
else: `var(--ui-text-secondary)`, `var(--ui-text-tertiary)`,
`var(--ui-text-quaternary)`, `var(--ui-stroke-secondary)`, `var(--ui-accent)`.
For canvas drawing, resolve them once with
`getComputedStyle(canvas).getPropertyValue('--ui-accent')`. This is what makes a
plugin reskin automatically with every theme.

## A backend for your plugin

If your plugin needs server-side work, ship a Python `plugin_api.py` and reach it
through `ctx.rest` / `ctx.socket` — a namespace scoped to your plugin **by
construction**.

### One package, both SDKs {#one-package-both-sdks}

A feature that needs a desktop UI **and** agent-side code (a Python plugin, its
backend routes, skills) doesn't have to ship as two co-dependent installs. The
desktop app also scans `$HERMES_HOME/plugins/<id>/` — the regular agent-plugin
root — for a `desktop/plugin.js`, and loads it through the exact same pipeline
as the standalone disk door (hot reload included):

```
~/.hermes/plugins/<id>/           # ONE installable folder
├── plugin.yaml                   # the agent half: tools, hooks, commands
├── skills/…
├── dashboard/
│   ├── manifest.json             # { "name": "<id>", "api": "plugin_api.py" }
│   └── plugin_api.py             # backend routes → /api/plugins/<id>/
└── desktop/
    └── plugin.js                 # the desktop half: panes, commands, ctx.rest
```

The `desktop/plugin.js` half is an ordinary disk plugin — same contract, same
imports, same `ctx.rest('/…')` reaching the `plugin_api.py` sitting beside it.
Installing, sharing, or removing the feature is one folder.

Two enable switches still apply, on purpose, and both default to **off**: the
desktop half ships opt-in — it inventories in **Settings → Plugins** but stays
disabled until the user toggles it, or ticks **Desktop UI** when installing the
package from **Settings → Plugins**, which counts as that toggle the same way the
install modal's **Enable agent plugin** switch does for the Python half — matching
the Python half's `plugins.enabled` gate in `config.yaml` (the security boundary
below). Dropping
a package into `~/.hermes/plugins` is inert on every surface until the user
says otherwise. The desktop half degrades gracefully when the backend half is
off — `ctx.rest` returns errors, not crashes.

:::note
The scan is local to the machine the desktop app runs on. Against a remote
backend, the remote box's `~/.hermes/plugins` is not reachable as a filesystem —
only locally installed packages contribute a desktop half (same rule as the
standalone door).
:::

### Distributing with an install link {#install-link}

Ship your plugin repo (agent half, desktop half, or both) and link to it with
the `hermes://` scheme — a plain anchor on your website or README:

```html
<a href="hermes://plugin/install?repo=owner/repo&enable=1">Install in Hermes</a>
```

The user gets a confirmation dialog (repo id, source links, a probe of what
the repo ships) and picks components before anything is installed — deep links
never auto-install. `force=1` replaces an existing install; dev builds use
`hermes-dev://`. Full link reference:
[One-click install links](/user-guide/features/plugins#one-click-install-links-desktop).

### The Python side

Desktop plugins reuse the dashboard plugin backend mount. Put the backend in a
`dashboard/` subfolder of a regular Hermes plugin and declare it in a
`manifest.json`:

```
~/.hermes/plugins/<id>/
└── dashboard/
    ├── manifest.json      # { "name": "<id>", "api": "plugin_api.py" }
    └── plugin_api.py      # exports `router = APIRouter()`
```

```python
# plugin_api.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/board")
async def board():
    return {"items": ["one", "two", "three"]}

@router.post("/action")
async def action(body: dict):
    return {"ok": True, "received": body}
```

Routes mount under `/api/plugins/<id>/` (`GET /api/plugins/<id>/board`, …).
Backend code runs inside the gateway process, so it can import from the
hermes-agent codebase directly (`hermes_state`, `hermes_cli.config`, …). See
[Extending the Dashboard → Backend API routes](/user-guide/features/extending-the-dashboard#backend-api-routes)
for the full backend reference — the mount is identical.

:::caution The Python backend is gated separately
Enabling a plugin in the desktop **Settings → Plugins** panel is a renderer-side
choice; it does **not** import Python. A user plugin's `plugin_api.py` is
imported only when the plugin is in the `plugins.enabled` allow-list in
`config.yaml` (and not in `plugins.disabled`). Project plugins (`./.hermes/`)
never auto-import Python. This is a security boundary, not an oversight
(GHSA-mcfc-hp25-cjv7).
:::

### Calling it from the plugin

```javascript
register(ctx) {
  // REST — namespace-relative path.
  const load = () => ctx.rest('/board')                 // GET /api/plugins/<id>/board
  const act  = () => ctx.rest('/action', { method: 'POST', body: { go: true } })

  // Live twin — a WebSocket to your own namespace.
  const stop = ctx.socket('/events', frame => {
    queryClient.invalidateQueries({ queryKey: [ctx.source, 'board'] })
  })
}
```

`ctx.rest` is profile-aware and rejects path traversal (`..`) so you can never
address another plugin's API or a core route through it. `PluginRestOptions` is
`{ method?, body?, upload?: { filename, contentType?, bytes }, timeoutMs? }`.

`ctx.socket` auto-reconnects with backoff until disposed. **It resolves to a no-op
on OAuth remotes** (single-use WS tickets are core-managed) — treat the socket as
an accelerator over polling, never a replacement. Every consumer needs a polling
fallback anyway, since any socket can drop.

For gateway-wide data (not your own namespace), use `host.request` (JSON-RPC) and
`host.onEvent` (the gateway event stream) instead.

## Settings, enable state, and storage

Every plugin — enabled or not — inventories in **Settings → Plugins**, where the
user toggles it live (no app restart), reveals its folder, or rescans. The user's
choice is remembered:

- No choice yet → the plugin's own `defaultEnabled` (default `true`). Set
  `defaultEnabled: false` to ship an opt-in plugin that stays dark until the user
  flips it on.
- Explicit choice → persisted and honored across restarts. A disabled plugin
  stays disabled — don't fight it; the user turned you off.

Persist your own state with `ctx.storage`, namespaced to your plugin
(`hermes.plugin.<id>.*`) so plugins can't read or clobber each other:

```javascript
ctx.storage.set('lastTab', 'board')
const tab = ctx.storage.get('lastTab', 'summary')
ctx.storage.remove('lastTab')
```

## Bundled plugins

A plugin can ship in-tree at `apps/desktop/src/plugins/<id>/plugin.tsx` (default
export a `HermesPlugin`). It's discovered by `discoverBundledPlugins()` at boot —
no import, no registry edit — and shares the exact inventory + live
enable/disable contract as a disk plugin. The two differences:

1. It goes through the app's Vite build, so you can write **real JSX** and import
   the SDK by its `@hermes/plugin-sdk` alias.
2. It's still lint-fenced to `@hermes/plugin-sdk` + `react` only — no `@/…` app
   internals.

No desktop plugins ship in the core tree today; the shipped app stays uncluttered
and demos live in the
[`hermes-example-plugins`](https://github.com/NousResearch/hermes-example-plugins)
companion repo.

## Security model

A loaded plugin is evaluated as ESM in the renderer realm with **full app
authority** — the React singleton, the whole SDK (`host.request` gateway RPC,
`ctx.rest`, storage, `navigate`). The isolation the loader provides is **error
isolation only**: a plugin can't crash the app (contributions are error-bounded,
listeners isolated), but it can do anything the app can.

This is acceptable for **local** sources — a disk file can already run code on
your machine — which is why the disk door only loads local files you (or your
agent) wrote. The optional `integrity` (`sha256-…`) check only proves the bytes
match a hash; it does **not** sandbox. A future remote-source door will need a
real boundary (iframe/worker + CSP + capability gating) before it can land; do
not treat this pipeline as a trust boundary.

## Pitfalls

- **JSX won't parse in a disk plugin.** The file loads uncompiled — use `jsx()` /
  `jsxs()` (or `React.createElement`), not JSX syntax. (Bundled plugins are built,
  so JSX is fine there.)
- **Only three specifiers resolve:** `@hermes/plugin-sdk`, `react`,
  `react/jsx-runtime`. Any other import surfaces an up-front load error.
- **Never hardcode colors** (`#000`, `black`, `rgb(...)`). Leave the background
  alone; use theme variables (`var(--ui-*)`) for everything.
- **Reference only what you imported.** A component you forgot to import (e.g.
  `StatusDot`) is a `ReferenceError` at render — double-check every identifier in
  your `jsx()` calls appears in the import line.
- **Read state imperatively in handlers** (`$atom.get()`), never from a render
  closure — rapid events will otherwise see stale values. Subscribe (`useValue`)
  only in the leaf that renders the value.
- **Canvas panes must track their container** with a `ResizeObserver` and resize
  the canvas (width/height attributes, not just CSS) — panes resize constantly.
- **Don't poll faster than a few seconds** with `host.request`; prefer
  `host.onEvent` / `ctx.socket` and let React Query dedupe.
- **`ctx.socket` is a no-op on OAuth remotes.** Always have a polling fallback.

## Reference

### SDK exports at a glance

| Category | Exports |
|----------|---------|
| Host | `host` (`.state.*`, `.notify`, `.notifyError`, `.navigate`, `.onEvent`, `.logs`, `.status`, `.restartGateway`, `.request`) |
| Plugin contract | `HermesPlugin`, `PluginContext`, `PluginContribution`, `PluginStorage`, `PluginOs`, `PluginRestOptions`, `PluginNativeNotificationInput`, `PluginNotificationAction`, `HermesOpenTarget`, `Contribution` |
| Area constants | `PANES_AREA`, `ROUTES_AREA`, `SIDEBAR_NAV_AREA`, `STATUSBAR_AREAS`, `TITLEBAR_AREAS`, `PALETTE_AREA`, `KEYBINDS_AREA`, `THEMES_AREA`, `COMPOSER_AREAS` |
| Area payloads | `RouteContribution`, `SidebarNavContribution`, `StatusbarItem`, `TitlebarTool`, `PaletteContribution`, `KeybindContribution`, `ComposerMiddleware`, `ComposerAttachmentProvider` |
| React / state | `useValue`, `atom`, `computed`, `useQuery`, `useMutation`, `useQueryClient`, `queryClient`, `Contribute` |
| Theming | `useTheme`, `requestTheme`, `setAccentOverride`, `$accentOverride`, `retintTheme`, `themeHue`, `DesktopTheme`, `DesktopThemeColors`, plus OKLCH math (`hexToOklch`, `oklchToHex`, `oklchToSrgb255`, `mixOklab`, `maxChroma`, `hueDelta`, `contrastRatio`, `readableOn`, `normalizeHex`) |
| UI kit | `Button`, `Input`, `Textarea`, `Select*`, `Switch`, `Checkbox`, `SegmentedControl`, `Tabs*`, `Dialog*`, `ConfirmDialog`, `DropdownMenu*`, `ContextMenu*`, `Popover*`, `Tip`/`Tooltip*`, `Badge`, `Kbd`/`KbdGroup`, `SearchField`, `ScrollArea`, `Separator`, `Skeleton`, `GlyphSpinner`, `Loader`, `EmptyState`, `ErrorState`, `CopyButton`, `StatusDot`, `LogView`, `Codicon`, `DecodeText` |
| Helpers | `cn`, `icons`, `haptic`, `useI18n`, `profileColor`, `profileColorSoft`, `relativeTime`, `fmtDateTime`, `fmtDayTime`, `coarseElapsed`, `evaluateRuntimeReadiness` |

The canonical, always-current export list is `apps/desktop/src/sdk/index.ts`.

### Agents: the `hermes-desktop-plugins` skill

When an agent writes a desktop plugin, it should load the bundled
**`hermes-desktop-plugins`** skill — it carries the same contract as this page in
agent-facing form, with a ready-to-copy `templates/plugin.js`. This page is the
human/developer reference; the skill is the working checklist.

## Troubleshooting

**My plugin doesn't appear.** Confirm the file is at
`$HERMES_HOME/desktop-plugins/<id>/plugin.js` and the folder name matches the
export `id`. Run ⌘K → **Reload desktop plugins**. Check the app for an error
toast naming the failure, and tail `hermes logs gui -f`.

**"unsupported import" on load.** A disk plugin may only import
`@hermes/plugin-sdk`, `react`, and `react/jsx-runtime`. Remove any other import.

**A `jsx` element renders nothing / throws `ReferenceError`.** An identifier used
in a `jsx()` call isn't imported. Add it to the import line.

**`ctx.rest` returns 404.** The backend isn't mounted: confirm
`~/.hermes/plugins/<id>/dashboard/manifest.json` has `"api": "plugin_api.py"`,
that the plugin is in `plugins.enabled` in `config.yaml`, and restart the gateway
(backend routes mount at startup). Tail `~/.hermes/logs/errors.log` for
`Failed to load plugin <id> API routes`.

**`ctx.socket` never fires.** On an OAuth remote it's a no-op by design — use your
polling fallback. Otherwise verify the backend exposes the matching
`@router.websocket(...)` route under its namespace.

**Colors look wrong after a theme switch.** You hardcoded a color. Replace it with
a `var(--ui-*)` theme variable.
