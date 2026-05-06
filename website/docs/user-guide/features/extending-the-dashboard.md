---
sidebar_position: 17
title: "Extending the Dashboard"
description: "Build themes and plugins for the Hermes web dashboard — palettes, typography, layouts, custom tabs, shell slots, page-scoped slots, and backend API routes"
---

# Extending the Dashboard

The Hermes web dashboard (`hermes dashboard`) is built to be reskinned and extended without forking the codebase. Three layers are exposed:

1. **Themes** — YAML files that repaint the dashboard's palette, typography, layout, and per-component chrome. Drop a file in `~/.hermes/dashboard-themes/`; it appears in the theme switcher.
2. **UI plugins** — a directory with `manifest.json` + a JavaScript bundle that registers a tab, replaces a built-in page, augments one via page-scoped slots, or injects components into named shell slots.
3. **Backend plugins** — a Python file inside that plugin directory that exposes a FastAPI `router`; routes are mounted under `/api/plugins/<name>/` and called from the plugin's UI.

All three are **drop-in at runtime**: no repo clone, no `npm run build`, no patching the dashboard source. This page is the canonical reference for all three.

If you just want to use the dashboard, see [Web Dashboard](./web-dashboard). If you want to reskin the terminal CLI (not the web dashboard), see [Skins & Themes](./skins) — the CLI skin system is unrelated to dashboard themes.

:::note How the pieces compose
Themes and plugins are independent but synergistic. A theme can stand alone (just a YAML file). A plugin can stand alone (just a tab). Together they let you build a complete visual reskin with custom HUDs — the bundled `strike-freedom-cockpit` demo does exactly that. See [Combined theme + plugin demo](#combined-theme--plugin-demo).
:::

---

## Table of contents

- [Themes](#themes)
  - [Quick start — your first theme](#quick-start--your-first-theme)
  - [Palette, typography, layout](#palette-typography-layout)
  - [Layout variants](#layout-variants)
  - [Theme assets (images as CSS vars)](#theme-assets-images-as-css-vars)
  - [Component chrome overrides](#component-chrome-overrides)
  - [Color overrides](#color-overrides)
  - [Raw `customCSS`](#raw-customcss)
  - [Built-in themes](#built-in-themes)
  - [Full theme YAML reference](#full-theme-yaml-reference)
- [Plugins](#plugins)
  - [Quick start — your first plugin](#quick-start--your-first-plugin)
  - [Directory layout](#directory-layout)
  - [Manifest reference](#manifest-reference)
  - [The Plugin SDK](#the-plugin-sdk)
  - [Shell slots](#shell-slots)
  - [Replacing built-in pages (`tab.override`)](#replacing-built-in-pages-taboverride)
  - [Augmenting built-in pages (page-scoped slots)](#augmenting-built-in-pages-page-scoped-slots)
  - [Slot-only plugins (`tab.hidden`)](#slot-only-plugins-tabhidden)
  - [Backend API routes](#backend-api-routes)
  - [Custom CSS per plugin](#custom-css-per-plugin)
  - [Plugin discovery & reload](#plugin-discovery--reload)
- [Combined theme + plugin demo](#combined-theme--plugin-demo)
- [API reference](#api-reference)
- [Troubleshooting](#troubleshooting)

---

## Themes

Themes are YAML files stored in `~/.hermes/dashboard-themes/`. The file name doesn't matter (the theme's `name:` field is what the system uses), but convention is `<name>.yaml`. Every field is optional — missing keys fall back to the built-in `default` theme, so a theme can be as small as one color.

### Quick start — your first theme

```bash
mkdir -p ~/.hermes/dashboard-themes
```

```yaml
# ~/.hermes/dashboard-themes/neon.yaml
name: neon
label: Neon
description: Pure magenta on black

palette:
  background: "#000000"
  midground: "#ff00ff"
```

Refresh the dashboard. Click the palette icon in the header and pick **Neon**. The background goes black, text and accents go magenta, and every derived color (card, border, muted, ring, etc.) is recomputed from that 2-color triplet via `color-mix()` in CSS.

That's the whole onboarding: one file, two colors. Everything below is optional refinement.

### Palette, typography, layout

These three blocks are the heart of a theme. Each is independent — override one, leave the others.

#### Palette (3-layer)

The palette is a triplet of color layers plus a warm-glow vignette color and a noise-grain multiplier. The dashboard's design-system cascade derives every shadcn-compatible token (card, popover, muted, border, primary, destructive, ring, etc.) from this triplet via CSS `color-mix()`. Overriding three colors cascades into the whole UI.

| Key | Description |
|-----|-------------|
| `palette.background` | Deepest canvas color — typically near-black. Drives the page background and card fill. |
| `palette.midground` | Primary text and accent. Most UI chrome reads this (foreground text, button outlines, focus rings). |
| `palette.foreground` | Top-layer highlight. The default theme sets this to white at alpha 0 (invisible); themes that want a bright accent on top can raise its alpha. |
| `palette.warmGlow` | `rgba(...)` string used as the vignette color by `<Backdrop />`. |
| `palette.noiseOpacity` | 0–1.2 multiplier on the grain overlay. Lower = softer, higher = grittier. |

Each layer accepts either `{hex: "#RRGGBB", alpha: 0.0–1.0}` or a bare hex string (alpha defaults to 1.0).

```yaml
palette:
  background:
    hex: "#05091a"
    alpha: 1.0
  midground: "#d8f0ff"          # bare hex, alpha = 1.0
  foreground:
    hex: "#ffffff"
    alpha: 0                    # invisible top layer
  warmGlow: "rgba(255, 199, 55, 0.24)"
  noiseOpacity: 0.7
```

#### Typography

| Key | Type | Description |
|-----|------|-------------|
| `fontSans` | string | CSS font-family stack for body copy (applied to `html`, `body`). |
| `fontMono` | string | CSS font-family stack for code blocks, `<code>`, `.font-mono` utilities. |
| `fontDisplay` | string | Optional heading/display stack. Falls back to `fontSans`. |
| `fontUrl` | string | Optional external stylesheet URL. Injected as `<link rel="stylesheet">` in `<head>` on theme switch. Same URL is never injected twice. Works with Google Fonts, Bunny Fonts, self-hosted `@font-face` sheets — anything linkable. |
| `baseSize` | string | Root font size — controls the rem scale. E.g. `"14px"`, `"16px"`. |
| `lineHeight` | string | Default line-height. E.g. `"1.5"`, `"1.65"`. |
| `letterSpacing` | string | Default letter-spacing. E.g. `"0"`, `"0.01em"`, `"-0.01em"`. |

```yaml
typography:
  fontSans: '"Orbitron", "Eurostile", "Impact", sans-serif'
  fontMono: '"Share Tech Mono", ui-monospace, monospace'
  fontDisplay: '"Orbitron", "Eurostile", sans-serif'
  fontUrl: "https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700&family=Share+Tech+Mono&display=swap"
  baseSize: "14px"
  lineHeight: "1.5"
  letterSpacing: "0.04em"
```

#### Layout

| Key | Values | Description |
|-----|--------|-------------|
| `radius` | any CSS length (`"0"`, `"0.25rem"`, `"0.5rem"`, `"1rem"`, ...) | Corner-radius token. Maps to `--radius` and cascades into `--radius-sm/md/lg/xl` — every rounded element shifts together. |
| `density` | `compact` \| `comfortable` \| `spacious` | Spacing multiplier applied as the `--spacing-mul` CSS var. `compact = 0.85×`, `comfortable = 1.0×` (default), `spacious = 1.2×`. Scales Tailwind's base spacing, so padding, gap, and space-between utilities all shift proportionally. |

```yaml
layout:
  radius: "0"
  density: compact
```

### Layout variants

`layoutVariant` picks the overall shell layout. Defaults to `"standard"` when absent.

| Variant | Behaviour |
|---------|-----------|
| `standard` | Single column, 1600px max-width (default). |
| `cockpit` | Left sidebar rail (260px) + main content. Populated by plugins via the `sidebar` slot — see [Shell slots](#shell-slots). Without a plugin the rail shows a placeholder. |
| `tiled` | Drops the max-width clamp so pages can use the full viewport width. |

```yaml
layoutVariant: cockpit
```

The current variant is exposed as `document.documentElement.dataset.layoutVariant`, so raw CSS in `customCSS` can target it via `:root[data-layout-variant="cockpit"] ...`.

### Theme assets (images as CSS vars)

Ship artwork URLs with a theme. Each named slot becomes a CSS var (`--theme-asset-<name>`) that the built-in shell and any plugin can read. The `bg` slot is automatically wired into the backdrop; other slots are plugin-facing.

```yaml
assets:
  bg: "https://example.com/hero-bg.jpg"           # auto-wired into <Backdrop />
  hero: "/my-images/strike-freedom.png"           # for plugin sidebars
  crest: "/my-images/crest.svg"                   # for header-left plugins
  logo: "/my-images/logo.png"
  sidebar: "/my-images/rail.png"
  header: "/my-images/header-art.png"
  custom:
    scanLines: "/my-images/scanlines.png"         # → --theme-asset-custom-scanLines
```

Values accept:

- Bare URLs — wrapped in `url(...)` automatically.
- Pre-wrapped `url(...)`, `linear-gradient(...)`, `radial-gradient(...)` expressions — used as-is.
- `"none"` — explicit opt-out.

Every asset is also emitted as `--theme-asset-<name>-raw` (the unwrapped URL), in case a plugin needs to pass it to `<img src>` instead of `background-image`.

Plugins read these with plain CSS or JS:

```javascript
// In a plugin slot
const hero = getComputedStyle(document.documentElement)
  .getPropertyValue("--theme-asset-hero").trim();
```

### Component chrome overrides

`componentStyles` restyles individual shell components without writing CSS selectors. Each bucket's entries become CSS vars (`--component-<bucket>-<kebab-property>`) that the shell's shared components read. So `card:` overrides apply to every `<Card>`, `header:` to the app bar, etc.

```yaml
componentStyles:
  card:
    clipPath: "polygon(12px 0, 100% 0, 100% calc(100% - 12px), calc(100% - 12px) 100%, 0 100%, 0 12px)"
    background: "linear-gradient(180deg, rgba(10, 22, 52, 0.85), rgba(5, 9, 26, 0.92))"
    boxShadow: "inset 0 0 0 1px rgba(64, 200, 255, 0.28)"
  header:
    background: "linear-gradient(180deg, rgba(16, 32, 72, 0.95), rgba(5, 9, 26, 0.9))"
  tab:
    clipPath: "polygon(6px 0, 100% 0, calc(100% - 6px) 100%, 0 100%)"
  sidebar: {}
  backdrop: {}
  footer: {}
  progress: {}
  badge: {}
  page: {}
```

Supported buckets: `card`, `header`, `footer`, `sidebar`, `tab`, `progress`, `badge`, `backdrop`, `page`.

Property names use camelCase (`clipPath`) and are emitted as kebab (`clip-path`). Values are plain CSS strings — anything CSS accepts (`clip-path`, `border-image`, `background`, `box-shadow`, `animation`, ...).

### Color overrides

Most themes won't need this — the 3-layer palette derives every shadcn token. Use `colorOverrides` when you want a specific accent the derivation won't produce (a softer destructive red for a pastel theme, a specific success green for a brand).

```yaml
colorOverrides:
  primary: "#ffce3a"
  primaryForeground: "#05091a"
  accent: "#3fd3ff"
  ring: "#3fd3ff"
  destructive: "#ff3a5e"
  border: "rgba(64, 200, 255, 0.28)"
```

Supported keys: `card`, `cardForeground`, `popover`, `popoverForeground`, `primary`, `primaryForeground`, `secondary`, `secondaryForeground`, `muted`, `mutedForeground`, `accent`, `accentForeground`, `destructive`, `destructiveForeground`, `success`, `warning`, `border`, `input`, `ring`.

Each key maps 1:1 to the `--color-<kebab>` CSS var (e.g. `primaryForeground` → `--color-primary-foreground`). Any key set here wins over the palette cascade for the active theme only — switching to another theme clears the overrides.

### Raw `customCSS`

For selector-level chrome that `componentStyles` can't express — pseudo-elements, animations, media queries, theme-scoped overrides — drop raw CSS into `customCSS`:

```yaml
customCSS: |
  /* Scanline overlay — only visible when cockpit variant is active. */
  :root[data-layout-variant="cockpit"] body::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 100;
    background: repeating-linear-gradient(to bottom,
      transparent 0px, transparent 2px,
      rgba(64, 200, 255, 0.035) 3px, rgba(64, 200, 255, 0.035) 4px);
    mix-blend-mode: screen;
  }
```

The CSS is injected as a single scoped `<style data-hermes-theme-css>` tag on theme apply and cleaned up on theme switch. **Capped at 32 KiB per theme.**

### Built-in themes

Each built-in ships its own palette, typography, and layout — switching produces visible changes beyond color alone.

| Theme | Palette | Typography | Layout |
|-------|---------|------------|--------|
| **Hermes Teal** (`default`) | Dark teal + cream | System stack, 15px | 0.5rem radius, comfortable |
| **Hermes Teal (Large)** (`default-large`) | Same as default | System stack, 18px, line-height 1.65 | 0.5rem radius, spacious |
| **Midnight** (`midnight`) | Deep blue-violet | Inter + JetBrains Mono, 14px | 0.75rem radius, comfortable |
| **Ember** (`ember`) | Warm crimson + bronze | Spectral (serif) + IBM Plex Mono, 15px | 0.25rem radius, comfortable |
| **Mono** (`mono`) | Grayscale | IBM Plex Sans + IBM Plex Mono, 13px | 0 radius, compact |
| **Cyberpunk** (`cyberpunk`) | Neon green on black | Share Tech Mono everywhere, 14px | 0 radius, compact |
| **Rosé** (`rose`) | Pink + ivory | Fraunces (serif) + DM Mono, 16px | 1rem radius, spacious |

Themes that reference Google Fonts (all except Hermes Teal) load the stylesheet on demand — the first time you switch to them a `<link>` tag is injected into `<head>`.

### Full theme YAML reference

Every knob in one file — copy and trim what you don't need:

```yaml
# ~/.hermes/dashboard-themes/ocean.yaml
name: ocean
label: Ocean Deep
description: Deep sea blues with coral accents

# 3-layer palette (accepts {hex, alpha} or bare hex)
palette:
  background:
    hex: "#0a1628"
    alpha: 1.0
  midground:
    hex: "#a8d0ff"
    alpha: 1.0
  foreground:
    hex: "#ffffff"
    alpha: 0.0
  warmGlow: "rgba(255, 107, 107, 0.35)"
  noiseOpacity: 0.7

typography:
  fontSans: "Poppins, system-ui, sans-serif"
  fontMono: "Fira Code, ui-monospace, monospace"
  fontDisplay: "Poppins, system-ui, sans-serif"   # optional
  fontUrl: "https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&family=Fira+Code:wght@400;500&display=swap"
  baseSize: "15px"
  lineHeight: "1.6"
  letterSpacing: "-0.003em"

layout:
  radius: "0.75rem"
  density: comfortable

layoutVariant: standard        # standard | cockpit | tiled

assets:
  bg: "https://example.com/ocean-bg.jpg"
  hero: "/my-images/kraken.png"
  crest: "/my-images/anchor.svg"
  logo: "/my-images/logo.png"
  custom:
    pattern: "/my-images/waves.svg"

componentStyles:
  card:
    boxShadow: "inset 0 0 0 1px rgba(168, 208, 255, 0.18)"
  header:
    background: "linear-gradient(180deg, rgba(10, 22, 40, 0.95), rgba(5, 9, 26, 0.9))"

colorOverrides:
  destructive: "#ff6b6b"
  ring: "#ff6b6b"

customCSS: |
  /* Any additional selector-level tweaks */
```

Refresh the dashboard after creating the file. Switch themes live from the header bar — click the palette icon. Selection persists to `config.yaml` under `dashboard.theme` and is restored on reload.

---

## Plugins

A dashboard plugin is a directory with a `manifest.json`, a pre-built JS bundle, and optionally a CSS file and a Python file with FastAPI routes. Plugins live next to other Hermes plugins in `~/.hermes/plugins/<name>/` — the dashboard extension is a `dashboard/` subfolder inside that plugin directory, so one plugin can extend both the CLI/gateway and the dashboard from a single install.

Plugins don't bundle React or UI components. They use the **Plugin SDK** exposed on `window.__HERMES_PLUGIN_SDK__`. This keeps plugin bundles tiny (typically a few KB) and avoids version conflicts.

### Quick start — your first plugin

Create the directory structure:

```bash
mkdir -p ~/.hermes/plugins/my-plugin/dashboard/dist
```

Write the manifest:

```json
// ~/.hermes/plugins/my-plugin/dashboard/manifest.json
{
  "name": "my-plugin",
  "label": "My Plugin",
  "icon": "Sparkles",
  "version": "1.0.0",
  "tab": {
    "path": "/my-plugin",
    "position": "after:skills"
  },
  "entry": "dist/index.js"
}
```

Write the JS bundle (a plain IIFE — no build step needed):

```javascript
// ~/.hermes/plugins/my-plugin/dashboard/dist/index.js
(function () {
  "use strict";

  const SDK = window.__HERMES_PLUGIN_SDK__;
  const { React } = SDK;
  const { Card, CardHeader, CardTitle, CardContent } = SDK.components;

  function MyPage() {
    return React.createElement(Card, null,
      React.createElement(CardHeader, null,
        React.createElement(CardTitle, null, "My Plugin"),
      ),
      React.createElement(CardContent, null,
        React.createElement("p", { className: "text-sm text-muted-foreground" },
          "Hello from my custom dashboard tab.",
        ),
      ),
    );
  }

  window.__HERMES_PLUGINS__.register("my-plugin", MyPage);
})();
```

Refresh the dashboard — your tab appears in the nav bar, after **Skills**.

:::tip Skip React.createElement
If you prefer JSX, use any bundler (esbuild, Vite, rollup) with React as an external and IIFE output. The only hard requirement is that the final file is a single JS file loadable via `<script>`. React is never bundled; it comes from `SDK.React`.
:::

### Directory layout

```
~/.hermes/plugins/my-plugin/
├── plugin.yaml              # optional — existing CLI/gateway plugin manifest
├── __init__.py              # optional — existing CLI/gateway hooks
└── dashboard/               # dashboard extension
    ├── manifest.json        # required — tab config, icon, entry point
    ├── dist/
    │   ├── index.js         # required — pre-built JS bundle (IIFE)
    │   └── style.css        # optional — custom CSS
    └── plugin_api.py        # optional — backend API routes (FastAPI)
```

A single plugin directory can carry three orthogonal extensions:

- `plugin.yaml` + `__init__.py` — CLI/gateway plugin ([see plugins page](./plugins)).
- `dashboard/manifest.json` + `dashboard/dist/index.js` — dashboard UI plugin.
- `dashboard/plugin_api.py` — dashboard backend routes.

None of them are required; include only the layers you need.

### Manifest reference

```json
{
  "name": "my-plugin",
  "label": "My Plugin",
  "description": "What this plugin does",
  "icon": "Sparkles",
  "version": "1.0.0",
  "tab": {
    "path": "/my-plugin",
    "position": "after:skills",
    "override": "/",
    "hidden": false
  },
  "slots": ["sidebar", "header-left"],
  "entry": "dist/index.js",
  "css": "dist/style.css",
  "api": "plugin_api.py"
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique plugin identifier. Lowercase, hyphens ok. Used in URLs and registration. |
| `label` | Yes | Display name shown in the nav tab. |
| `description` | No | Short description (shown in dashboard admin surfaces). |
| `icon` | No | Lucide icon name. Defaults to `Puzzle`. Unknown names fall back to `Puzzle`. |
| `version` | No | Semver string. Defaults to `0.0.0`. |
| `tab.path` | Yes | URL path for the tab (e.g. `/my-plugin`). |
| `tab.position` | No | Where to insert the tab. `"end"` (default), `"after:<path>"`, or `"before:<path>"` — value after the colon is the **path segment** of the target tab (no leading slash). Examples: `"after:skills"`, `"before:config"`. |
| `tab.override` | No | Set to a built-in route path (`"/"`, `"/sessions"`, `"/config"`, ...) to **replace** that page instead of adding a new tab. See [Replacing built-in pages](#replacing-built-in-pages-taboverride). |
| `tab.hidden` | No | When true, register the component and any slots without adding a tab to the nav. Used by slot-only plugins. See [Slot-only plugins](#slot-only-plugins-tabhidden). |
| `slots` | No | Named shell slots this plugin populates. **Documentation aid only** — actual registration happens from the JS bundle via `registerSlot()`. Listing slots here makes discovery surfaces more informative. |
| `entry` | Yes | Path to the JS bundle relative to `dashboard/`. Defaults to `dist/index.js`. |
| `css` | No | Path to a CSS file to inject as a `<link>` tag. |
| `api` | No | Path to a Python file with FastAPI routes. Mounted at `/api/plugins/<name>/`. |

#### Available icons

Plugins use Lucide icon names. The dashboard maps these by name — unknown names silently fall back to `Puzzle`.

Currently mapped: `Activity`, `BarChart3`, `Clock`, `Code`, `Database`, `Eye`, `FileText`, `Globe`, `Heart`, `KeyRound`, `MessageSquare`, `Package`, `Puzzle`, `Settings`, `Shield`, `Sparkles`, `Star`, `Terminal`, `Wrench`, `Zap`.

Need a different icon? Open a PR to `web/src/App.tsx`'s `ICON_MAP` — pure additive change.

### The Plugin SDK

Everything a plugin needs is on `window.__HERMES_PLUGIN_SDK__`. Plugins should never import React directly.

```javascript
const SDK = window.__HERMES_PLUGIN_SDK__;

// React + hooks
SDK.React                    // the React instance
SDK.hooks.useState
SDK.hooks.useEffect
SDK.hooks.useCallback
SDK.hooks.useMemo
SDK.hooks.useRef
SDK.hooks.useContext
SDK.hooks.createContext

// UI components (shadcn/ui primitives)
SDK.components.Card
SDK.components.CardHeader
SDK.components.CardTitle
SDK.components.CardContent
SDK.components.Badge
SDK.components.Button
SDK.components.Input
SDK.components.Label
SDK.components.Select
SDK.components.SelectOption
SDK.components.Separator
SDK.components.Tabs
SDK.components.TabsList
SDK.components.TabsTrigger
SDK.components.PluginSlot    // render a named slot (useful for nested plugin UIs)

// Hermes API client + raw fetcher
SDK.api                      // typed client — getStatus, getSessions, getConfig, ...
SDK.fetchJSON                // raw fetch for custom endpoints (plugin-registered routes)

// Utilities
SDK.utils.cn                 // Tailwind class merger (clsx + twMerge)
SDK.utils.timeAgo            // "5m ago" from unix timestamp
SDK.utils.isoTimeAgo         // "5m ago" from ISO string

// Hooks
SDK.useI18n                  // i18n hook for multi-language plugins
```

#### Calling your plugin's backend

```javascript
SDK.fetchJSON("/api/plugins/my-plugin/data")
  .then((data) => console.log(data))
  .catch((err) => console.error("API call failed:", err));
```

`fetchJSON` injects the session auth token, surfaces errors as thrown exceptions, and parses JSON automatically.

#### Calling built-in Hermes endpoints

```javascript
// Agent status
SDK.api.getStatus().then((s) => console.log("Version:", s.version));

// Recent sessions
SDK.api.getSessions(10).then((resp) => console.log(resp.sessions.length));
```

See [Web Dashboard → REST API](./web-dashboard#rest-api) for the full list.

### Shell slots

Slots let a plugin inject components into named locations of the app shell — the cockpit sidebar, the header, the footer, an overlay layer — without claiming a whole tab. Multiple plugins can populate the same slot; they render stacked in registration order.

Register from inside the plugin bundle:

```javascript
window.__HERMES_PLUGINS__.registerSlot("my-plugin", "sidebar", MySidebar);
window.__HERMES_PLUGINS__.registerSlot("my-plugin", "header-left", MyCrest);
```

#### Slot catalogue

**Shell-wide slots** (render anywhere in the app chrome):

| Slot | Location |
|------|----------|
| `backdrop` | Inside the `<Backdrop />` layer stack, above the noise layer. |
| `header-left` | Before the Hermes brand in the top bar. |
| `header-right` | Before the theme/language switchers in the top bar. |
| `header-banner` | Full-width strip below the nav. |
| `sidebar` | Cockpit sidebar rail — **only rendered when `layoutVariant === "cockpit"`**. |
| `pre-main` | Above the route outlet (inside `<main>`). |
| `post-main` | Below the route outlet (inside `<main>`). |
| `footer-left` | Footer cell content (replaces default). |
| `footer-right` | Footer cell content (replaces default). |
| `overlay` | Fixed-position layer above everything else. Useful for chrome (scanlines, vignettes) `customCSS` can't achieve alone. |

**Page-scoped slots** (render only on the named built-in page — use these to inject widgets, cards, or toolbars into an existing page without overriding the whole route):

| Slot | Where it renders |
|------|------------------|
| `sessions:top` / `sessions:bottom` | Top / bottom of the `/sessions` page. |
| `analytics:top` / `analytics:bottom` | Top / bottom of the `/analytics` page. |
| `logs:top` / `logs:bottom` | Top (above filter toolbar) / bottom (below log viewer) of `/logs`. |
| `cron:top` / `cron:bottom` | Top / bottom of the `/cron` page. |
| `skills:top` / `skills:bottom` | Top / bottom of the `/skills` page. |
| `config:top` / `config:bottom` | Top / bottom of the `/config` page. |
| `env:top` / `env:bottom` | Top / bottom of the `/env` (Keys) page. |
| `docs:top` / `docs:bottom` | Top (above the iframe) / bottom of `/docs`. |
| `chat:top` / `chat:bottom` | Top / bottom of `/chat` (only active when embedded chat is enabled). |

Example — add a banner card to the top of the Sessions page:

```javascript
function PinnedSessionsBanner() {
  return React.createElement(Card, null,
    React.createElement(CardContent, { className: "py-2 text-xs" },
      "Pinned note injected by my-plugin"),
  );
}

window.__HERMES_PLUGINS__.registerSlot("my-plugin", "sessions:top", PinnedSessionsBanner);
```

Combine page-scoped slots with `tab.hidden: true` if your plugin only augments existing pages and doesn't need a sidebar tab of its own.

The shell only renders `<PluginSlot name="..." />` for the slots above. Additional names are accepted by the registry for nested plugin UIs — a plugin can expose its own slots via `SDK.components.PluginSlot`.

#### Re-registration and HMR

If the same `(plugin, slot)` pair is registered twice, the later call replaces the earlier one — this matches how React HMR expects plugin re-mounts to behave.

### Replacing built-in pages (`tab.override`)

Setting `tab.override` to a built-in route path makes the plugin's component replace that page instead of adding a new tab. Useful when a theme wants a custom home page (`/`) but wants to keep the rest of the dashboard intact.

```json
{
  "name": "my-home",
  "label": "Home",
  "tab": {
    "path": "/my-home",
    "override": "/",
    "position": "end"
  },
  "entry": "dist/index.js"
}
```

With `override` set:

- The original page component at `/` is removed from the router.
- Your plugin renders at `/` instead.
- No nav tab is added for `tab.path` (the override is the point).

Only one plugin can override a given path. If two plugins claim the same override, the first wins and the second is ignored with a dev-mode warning.

If you only need to add a card or toolbar to an existing page without taking it over, use [page-scoped slots](#augmenting-built-in-pages-page-scoped-slots) instead.

### Augmenting built-in pages (page-scoped slots)

Full replacement via `tab.override` is heavy — your plugin now owns the entire page, including any future updates we ship to it. Most of the time you just want to add a banner, card, or toolbar to an existing page. That's what **page-scoped slots** are for.

Every built-in page exposes `<page>:top` and `<page>:bottom` slots rendered at the top and bottom of its content area. Your plugin populates one by calling `registerSlot()` — the built-in page keeps working normally, and your component renders alongside it.

Available slots: `sessions:*`, `analytics:*`, `logs:*`, `cron:*`, `skills:*`, `config:*`, `env:*`, `docs:*`, `chat:*` (each with `:top` and `:bottom`). See the full catalogue in [Shell slots → Slot catalogue](#slot-catalogue).

Minimal example — pin a banner to the top of the Sessions page:

```json
// ~/.hermes/plugins/session-notes/dashboard/manifest.json
{
  "name": "session-notes",
  "label": "Session Notes",
  "tab": { "path": "/session-notes", "hidden": true },
  "slots": ["sessions:top"],
  "entry": "dist/index.js"
}
```

```javascript
// ~/.hermes/plugins/session-notes/dashboard/dist/index.js
(function () {
  const SDK = window.__HERMES_PLUGIN_SDK__;
  const { React } = SDK;
  const { Card, CardContent } = SDK.components;

  function Banner() {
    return React.createElement(Card, null,
      React.createElement(CardContent, { className: "py-2 text-xs" },
        "Remember to label important sessions before archiving."),
    );
  }

  // Placeholder for the hidden tab.
  window.__HERMES_PLUGINS__.register("session-notes", function () { return null; });

  // The real work.
  window.__HERMES_PLUGINS__.registerSlot("session-notes", "sessions:top", Banner);
})();
```

Key points:

- `tab.hidden: true` keeps the plugin out of the sidebar — it has no standalone page.
- The `slots` manifest field is documentation only. The actual binding happens in the JS bundle via `registerSlot()`.
- Multiple plugins can claim the same page-scoped slot. They render stacked in registration order.
- Zero footprint when no plugin registers: the built-in page renders exactly as before.

The bundled `example-dashboard` plugin ships a live demo that injects a banner into `sessions:top` — install it to see the pattern end-to-end.

### Slot-only plugins (`tab.hidden`)

When `tab.hidden: true`, the plugin registers its component (for direct URL visits) and any slots, but never adds a tab to the navigation. Used by plugins that only exist to inject into slots — a header crest, a sidebar HUD, an overlay.

```json
{
  "name": "header-crest",
  "label": "Header Crest",
  "tab": {
    "path": "/header-crest",
    "position": "end",
    "hidden": true
  },
  "slots": ["header-left"],
  "entry": "dist/index.js"
}
```

The bundle still calls `register()` with a placeholder component (good practice in case someone hits the URL directly) and then `registerSlot()` to do the real work.

### Backend API routes

Plugins can register FastAPI routes by setting `api` in the manifest. Create the file and export a `router`:

```python
# ~/.hermes/plugins/my-plugin/dashboard/plugin_api.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/data")
async def get_data():
    return {"items": ["one", "two", "three"]}

@router.post("/action")
async def do_action(body: dict):
    return {"ok": True, "received": body}
```

Routes are mounted under `/api/plugins/<name>/`, so the above becomes:

- `GET  /api/plugins/my-plugin/data`
- `POST /api/plugins/my-plugin/action`

Plugin API routes bypass session-token authentication since the dashboard server binds to localhost by default. **Don't expose the dashboard on a public interface with `--host 0.0.0.0` if you run untrusted plugins** — their routes become reachable too.

#### Accessing Hermes internals

Backend routes run inside the dashboard process, so they can import from the hermes-agent codebase directly:

```python
from fastapi import APIRouter
from hermes_state import SessionDB
from hermes_cli.config import load_config

router = APIRouter()

@router.get("/session-count")
async def session_count():
    db = SessionDB()
    try:
        count = len(db.list_sessions(limit=9999))
        return {"count": count}
    finally:
        db.close()

@router.get("/config-snapshot")
async def config_snapshot():
    cfg = load_config()
    return {"model": cfg.get("model", {})}
```

### Custom CSS per plugin

If your plugin needs styles beyond Tailwind classes and inline `style=`, add a CSS file and reference it in the manifest:

```json
{
  "css": "dist/style.css"
}
```

The file is injected as a `<link>` tag on plugin load. Use specific class names to avoid conflicts with the dashboard's styles, and reference the dashboard's CSS vars to stay theme-aware:

```css
/* dist/style.css */
.my-plugin-chart {
  border: 1px solid var(--color-border);
  background: var(--color-card);
  color: var(--color-card-foreground);
  padding: 1rem;
}
.my-plugin-chart:hover {
  border-color: var(--color-ring);
}
```

The dashboard exposes every shadcn token as `--color-*` plus theme extras (`--theme-asset-*`, `--component-<bucket>-*`, `--radius`, `--spacing-mul`). Reference those and your plugin automatically reskins with the active theme.

### Plugin discovery & reload

The dashboard scans three directories for `dashboard/manifest.json`:

| Priority | Directory | Source label |
|----------|-----------|--------------|
| 1 (wins on conflict) | `~/.hermes/plugins/<name>/dashboard/` | `user` |
| 2 | `<repo>/plugins/memory/<name>/dashboard/` | `bundled` |
| 2 | `<repo>/plugins/<name>/dashboard/` | `bundled` |
| 3 | `./.hermes/plugins/<name>/dashboard/` | `project` — only when `HERMES_ENABLE_PROJECT_PLUGINS` is set |

Discovery results are cached per dashboard process. After adding a new plugin, either:

```bash
# Force a rescan without restart
curl http://127.0.0.1:9119/api/dashboard/plugins/rescan
```

…or restart `hermes dashboard`.

#### Plugin load lifecycle

1. Dashboard loads. `main.tsx` exposes the SDK on `window.__HERMES_PLUGIN_SDK__` and the registry on `window.__HERMES_PLUGINS__`.
2. `App.tsx` calls `usePlugins()` → fetches `GET /api/dashboard/plugins`.
3. For each manifest: CSS `<link>` is injected (if declared), then a `<script>` tag loads the JS bundle.
4. The plugin's IIFE runs and calls `window.__HERMES_PLUGINS__.register(name, Component)` — and optionally `.registerSlot(name, slot, Component)` for each slot.
5. The dashboard resolves the registered component against the manifest, adds the tab to navigation (unless `hidden`), and mounts the component as a route.

Plugins have up to **2 seconds** after their script loads to call `register()`. After that the dashboard stops waiting and finishes initial render. If a plugin later registers, it still appears — the nav is reactive.

If a plugin's script fails to load (404, syntax error, exception during IIFE), the dashboard logs a warning to the browser console and continues without it.

---

## Combined theme + plugin demo

The repo ships `plugins/strike-freedom-cockpit/` as a complete reskin demo. It pairs a theme YAML with a slot-only plugin to produce a cockpit-style HUD without forking the dashboard.

**What it demonstrates:**

- A full theme using palette, typography, `fontUrl`, `layoutVariant: cockpit`, `assets`, `componentStyles` (notched card corners, gradient backgrounds), `colorOverrides`, and `customCSS` (scanline overlay).
- A slot-only plugin (`tab.hidden: true`) that registers into three slots:
  - `sidebar` — an MS-STATUS panel with live telemetry bars driven by `SDK.api.getStatus()`.
  - `header-left` — a faction crest that reads `--theme-asset-crest` from the active theme.
  - `footer-right` — a custom tagline replacing the default org line.
- The plugin reads theme-supplied artwork via CSS vars, so swapping themes changes the hero/crest without plugin code changes.

**Install:**

```bash
# Theme
cp plugins/strike-freedom-cockpit/theme/strike-freedom.yaml \
   ~/.hermes/dashboard-themes/

# Plugin
cp -r plugins/strike-freedom-cockpit ~/.hermes/plugins/
```

Open the dashboard, pick **Strike Freedom** from the theme switcher. The cockpit sidebar appears, the crest shows in the header, the tagline replaces the footer. Switch back to **Hermes Teal** and the plugin remains installed but invisible (the `sidebar` slot only renders under the `cockpit` layout variant).

Read the plugin source (`plugins/strike-freedom-cockpit/dashboard/dist/index.js`) to see how it reads CSS vars, guards against older dashboards without slot support, and registers three slots from one bundle.

---

## API reference

### Theme endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dashboard/themes` | GET | List available themes + active name. Built-ins return `{name, label, description}`; user themes also include a `definition` field with the full normalised theme object. |
| `/api/dashboard/theme` | PUT | Set active theme. Body: `{"name": "midnight"}`. Persists to `config.yaml` under `dashboard.theme`. |

### Plugin endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dashboard/plugins` | GET | List discovered plugins (with manifests, minus internal fields). |
| `/api/dashboard/plugins/rescan` | GET | Force re-scan the plugin directories without restarting. |
| `/dashboard-plugins/<name>/<path>` | GET | Serve static assets from a plugin's `dashboard/` directory. Path traversal is blocked. |
| `/api/plugins/<name>/*` | * | Plugin-registered backend routes. |

### SDK on `window`

| Global | Type | Provider |
|--------|------|----------|
| `window.__HERMES_PLUGIN_SDK__` | object | `registry.ts` — React, hooks, UI components, API client, utils. |
| `window.__HERMES_PLUGINS__.register(name, Component)` | function | Register a plugin's main component. |
| `window.__HERMES_PLUGINS__.registerSlot(name, slot, Component)` | function | Register into a named shell slot. |

---

## Troubleshooting

**My theme doesn't appear in the picker.**
Check that the file is in `~/.hermes/dashboard-themes/` and ends in `.yaml` or `.yml`. Refresh the page. Run `curl http://127.0.0.1:9119/api/dashboard/themes` — your theme should be in the response. If the YAML has a parse error, the dashboard logs to `errors.log` under `~/.hermes/logs/`.

**My plugin's tab doesn't show up.**
1. Check the manifest is at `~/.hermes/plugins/<name>/dashboard/manifest.json` (note the `dashboard/` subdirectory).
2. `curl http://127.0.0.1:9119/api/dashboard/plugins/rescan` to force re-discovery.
3. Open browser dev tools → Network — confirm `manifest.json`, `index.js`, and any CSS loaded without 404s.
4. Open browser dev tools → Console — look for errors during the IIFE or `window.__HERMES_PLUGINS__ is undefined` (indicates the SDK didn't initialize, usually a React render crash earlier).
5. Verify your bundle calls `window.__HERMES_PLUGINS__.register(...)` with the **same name** as `manifest.json:name`.

**Slot-registered components don't render.**
The `sidebar` slot only renders when the active theme has `layoutVariant: cockpit`. Other slots always render. If you're registering into a slot with no hits, add `console.log` inside `registerSlot` to confirm the plugin bundle ran at all.

**Plugin backend routes return 404.**
1. Confirm the manifest has `"api": "plugin_api.py"` pointing to an existing file inside `dashboard/`.
2. Restart `hermes dashboard` — plugin API routes are mounted once at startup, **not** on rescan.
3. Check that `plugin_api.py` exports a module-level `router = APIRouter()`. Other export names are not picked up.
4. Tail `~/.hermes/logs/errors.log` for `Failed to load plugin <name> API routes` — import errors are logged there.

**Theme change drops my color overrides.**
`colorOverrides` are scoped to the active theme and cleared on theme switch — that's by design. If you want overrides that persist, put them in your theme's YAML, not in the live switcher.

**Theme customCSS gets truncated.**
The `customCSS` block is capped at 32 KiB per theme. Split large stylesheets across multiple themes, or switch to a plugin that injects a full stylesheet via its `css` field (no size cap).

**I want to ship a plugin on PyPI.**
Dashboard plugins are installed by directory layout, not by pip entry point. The cleanest distribution path today is a git repo the user clones into `~/.hermes/plugins/`. A pip-based installer for dashboard plugins is not currently wired up.
