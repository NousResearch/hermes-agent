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
  atoms): active session, cwd, gateway status, model, profile, viewport.
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
| **Bundled** | `apps/desktop/src/plugins/<id>/plugin.tsx` | in-tree, shipped with the app | the app's own Vite build |

Both take the same `HermesPlugin` contract, appear in **Settings → Plugins**, and
enable/disable live. Everything on this page is written against the disk door
(what you and the agent write); [Bundled plugins](#bundled-plugins) notes the two
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

### Composer extensions

`COMPOSER_AREAS` (`top`, `bottom`, `leading`, `actions`, `attachments`,
`middleware`) let a plugin add controls around the message composer, provide an
attachment source, or transform a draft before it is sent (`ComposerMiddleware`
with a `handler(draft) => draft | null`).

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
host.state.cwd              // ReadableAtom<string>
host.state.gateway          // ReadableAtom<string>  ('idle' | 'connecting' | 'open' | …)
host.state.model            // ReadableAtom<string>
host.state.profile          // ReadableAtom<string>
host.state.viewport         // ReadableAtom<{ width, height, narrow }>

host.notify({ kind, message, title?, detail?, action? })  // toast; returns id
host.notifyError(error, fallbackMessage)                   // toast an error
host.navigate('/route')                    // hash-route navigation
host.onEvent(type, fn)                     // gateway event stream ('*' = all); returns disposer
host.logs(...)                             // tail an app log file
host.status()                              // one-shot system status snapshot
host.restartGateway()                      // restart the backend gateway
host.request<T>(method, params?)           // gateway JSON-RPC — the real power
```

`host.request` is the same JSON-RPC the app itself uses (sessions, config, skills,
cron, kanban, …). `host.onEvent` streams live gateway events (message deltas,
session lifecycle, tool activity). Listeners are isolated — a throw in your
listener can't affect app dispatch. Every `host` door is async-safe: a sync throw
from an internal helper (e.g. no desktop bridge in a plain browser) becomes a
rejection your `.catch()` sees, never an error-boundary crash.

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
| Plugin contract | `HermesPlugin`, `PluginContext`, `PluginContribution`, `PluginStorage`, `PluginRestOptions`, `Contribution` |
| Area constants | `PANES_AREA`, `ROUTES_AREA`, `SIDEBAR_NAV_AREA`, `STATUSBAR_AREAS`, `TITLEBAR_AREAS`, `PALETTE_AREA`, `KEYBINDS_AREA`, `THEMES_AREA`, `COMPOSER_AREAS` |
| Area payloads | `RouteContribution`, `SidebarNavContribution`, `StatusbarItem`, `TitlebarTool`, `PaletteContribution`, `KeybindContribution`, `ComposerMiddleware`, `ComposerAttachmentProvider` |
| React / state | `useValue`, `atom`, `computed`, `useQuery`, `useMutation`, `useQueryClient`, `queryClient`, `Contribute` |
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
