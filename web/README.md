# Hermes Agent — Web UI

Browser-based dashboard for managing Hermes Agent configuration, API keys, and monitoring active sessions.

## Stack

- **Vite** + **React 19** + **TypeScript**
- **Tailwind CSS v4** with custom dark theme
- **shadcn/ui**-style components (hand-rolled, no CLI dependency)

## Development

```bash
# Start the backend API server
cd ../
python -m hermes_cli.main web --no-open

# In another terminal, start the Vite dev server (with HMR + API proxy)
cd web/
npm install
npm run dev
```

Open the **Vite URL** printed in the terminal (usually `http://localhost:5173`). That is the live-reload UI.

`hermes dashboard` on port 9119 serves the **built** bundle from `hermes_cli/web_dist/`, not the Vite dev server — changes in `web/src/` will not appear there until you run `npm run build` and restart the dashboard (or use `web --no-open` + Vite as above).

The Vite dev server proxies `/api` requests to `http://127.0.0.1:9119` (the FastAPI backend).

## Build

```bash
npm run build
```

This outputs to `../hermes_cli/web_dist/`, which the FastAPI server serves as a static SPA. The built assets are included in the Python package via `pyproject.toml` package-data.

## Structure

```
src/
├── components/              # Shared UI components used across pages
│   ├── AuthWidget           # OAuth "logged in as" sidebar affordance
│   ├── AutoField            # Dynamic form field renderer (select/switch/input)
│   ├── AutomationBlueprints # Automation blueprint browser and launcher
│   ├── ChatSessionList      # Conversation switcher for the Chat tab
│   ├── ChatSidebar          # Structured-events panel next to the terminal
│   ├── LanguageSwitcher     # Locale selector dropdown
│   ├── Markdown             # Lightweight LLM output renderer
│   ├── ModelInfoCard        # Model capability and metadata card
│   ├── ModelPickerDialog    # Fuzzy model search and selection dialog
│   ├── OAuthLoginModal      # OAuth device-flow login modal
│   ├── OAuthProvidersCard   # OAuth provider status and management
│   ├── PlatformsCard        # Connected platform status card with connectivity badges
│   ├── ProfileScopeBanner   # Amber banner shown when a non-default management profile is active
│   ├── ProfileSwitcher      # Management profile selector
│   ├── ReasoningPicker      # Model reasoning effort selector
│   ├── ScheduleBuilder      # Cron schedule builder UI
│   ├── SidebarFooter        # Sidebar bottom status and version info
│   ├── SidebarStatusStrip   # Agent status strip in sidebar
│   ├── SkillEditorDialog    # Skill create and edit dialog
│   ├── SlashPopover         # Slash command autocomplete popover
│   ├── ThemeSwitcher        # Theme and font override picker
│   ├── ToolCall             # Tool call result display with expand/collapse
│   └── ToolsetConfigDrawer  # Toolset enable/disable configuration drawer
├── contexts/                # React contexts and their hooks
│   ├── PageHeaderProvider   # Page title injection into the app shell header
│   ├── ProfileProvider      # Active management profile scope
│   └── SystemActions        # Global system action dispatch and toast feedback
├── hooks/                   # Standalone React hooks
│   ├── useModalBehavior     # Escape-key and focus trap for modals
│   └── useSidebarStatus     # Polling hook for agent status strip
├── i18n/                    # Internationalisation — 18 locales
│   ├── types.ts             # Translations interface and Locale union type
│   ├── context.tsx          # I18nProvider and useI18n hook
│   └── *.ts                 # Per-locale translation maps (en, zh, ja, ru …)
├── lib/                     # Pure utilities and API layer
│   ├── api.ts               # Typed fetch wrappers for all backend endpoints
│   ├── cron-job.ts          # Cron job form state types and mutation payload helpers
│   ├── gatewayClient.ts     # WebSocket client for tui_gateway JSON-RPC
│   ├── fuzzy.ts             # Subsequence scorer for picker filtering
│   ├── reasoning-effort.ts  # Reasoning-effort normalisation helpers
│   ├── schedule.ts          # Cron schedule builder helpers
│   └── utils.ts             # cn() and themedBody helpers for Tailwind
├── pages/                   # Route-level page components
│   ├── AnalyticsPage        # Usage analytics and token stats
│   ├── ChannelsPage         # Notification channel management
│   ├── ChatPage             # Embedded hermes --tui terminal
│   ├── ConfigPage           # Dynamic config editor (reads schema from backend)
│   ├── CronPage             # Scheduled job management
│   ├── DocsPage             # Documentation viewer
│   ├── EnvPage              # API key management with save/clear
│   ├── FilesPage            # Working directory file browser
│   ├── LogsPage             # Agent log viewer
│   ├── McpPage              # MCP server management
│   ├── ModelsPage           # Model catalogue and selection
│   ├── PairingPage          # Device pairing and access management
│   ├── PluginsPage          # Dashboard plugin management
│   ├── ProfileBuilderPage   # Profile creation wizard
│   ├── ProfilesPage         # Profile list and management
│   ├── SessionsPage         # Active and recent session list
│   ├── SkillsPage           # Skills browser and editor
│   ├── SystemPage           # System health and diagnostics
│   └── WebhooksPage         # Webhook configuration and management
├── plugins/                 # Dashboard plugin system
│   ├── registry.ts          # Plugin SDK exposure and component registry
│   ├── slots.ts             # Named injection slots (header, sidebar, backdrop)
│   ├── usePlugins.ts        # Hook — fetches manifests and injects plugin bundles
│   └── sdk.d.ts             # Public type surface for window.__HERMES_PLUGIN_SDK__
├── themes/                  # Dashboard theming layer
│   ├── presets.ts           # Built-in theme definitions
│   ├── fonts.ts             # Curated UI-font catalog for font override
│   └── context.tsx          # ThemeProvider and useTheme hook
├── App.tsx                  # Main layout, routing, and navigation
├── main.tsx                 # React entry point
└── index.css                # Tailwind imports and theme variables
```

## Typography & contrast rules

Read before adding or editing UI styles. These rules keep the dashboard legible across all built-in themes and stop drift back into the patterns the design system was just refactored out of.

### Text size floor

- **Minimum body size: `text-xs` (12px / 0.75rem).** Do not use arbitrary `text-[0.6rem]`, `text-[0.65rem]`, `text-[9px]`, `text-[10px]`, or `text-[11px]` on copy, hints, labels, counts, or badges. Use the standard scale: `text-xs`, `text-sm`, `text-base`.
- Smaller sizes are only acceptable on **decorative overlays** (chart stripes, empty-state icons) — never on text the user is meant to read.

### Opacity floor on text

- **Never apply opacity below 0.7 to text.** No `opacity-30`, `opacity-50`, `opacity-60` on `<span>`s, `<p>`s, labels, etc.
- **Do not stack opacity tokens.** Patterns like `text-muted-foreground/60`, `text-midground/70`, `text-foreground/50` create unpredictable WCAG failures because the parent token already has alpha.
- Use the **semantic text tokens** from `@nous-research/ui`'s `globals.css`:
  - `text-text-primary` — default body text.
  - `text-text-secondary` — subtitles, meta, inactive nav.
  - `text-text-tertiary` — small chrome labels, counts, footnotes.
  - `text-text-disabled` — disabled states.
  - `text-text-on-accent` — text on filled accent surfaces.

### Brand uppercase via `text-display`, not raw `uppercase`

- The dashboard preserves the Nous brand uppercase aesthetic, but it is **opt-in per element, not global**.
- Apply uppercase via the DS utility `text-display` on **brand chrome only** — page titles, nav section headings, badges, brand wordmark. DS components (`Button`, `Badge`, `Tabs`, `Segmented`, etc.) already self-apply `text-display`.
- **Do not introduce new `uppercase`** (the literal Tailwind class) in `hermes-agent/web/src`. Prefer `text-display` for new brand chrome. Legacy `uppercase` call sites (e.g. `components/ui/label.tsx`, `card.tsx`) remain until migrated.
- The app shell no longer forces uppercase globally, so blanket `normal-case` opt-outs are unnecessary. Use `normal-case` only where a DS component applies `text-display` but the label should stay sentence case — e.g. dynamic user content (model slugs, theme names) **or** fixed UI copy that is not brand chrome (EnvPage “not configured” toggle, sidebar “New chat”).

### Fonts

Typography is **opt-in per surface**, not global on layout shells — the app shell and page header keep their original theme/expanded fonts; Mondwest applies only where explicitly set.

| Tier | Classes | Use for |
|------|---------|---------|
| Brand chrome | `font-mondwest text-display` (or `themedChrome`) | Sidebar nav, card section headers (`CardTitle`), Segmented filter buttons, filter panel headings |
| Themed body | `font-mondwest normal-case` (or `themedBody`) | Card content (`Card`, `CardDescription`), session/platform rows, analytics tables — **scoped to the component** |
| Page chrome | `font-expanded` | Page header h1 (`PageHeaderProvider`) — sentence case, not `text-display` |
| Wordmark | `Typography` + size/tracking only | Sidebar/mobile “Hermes Agent” — mixed case, no Mondwest, no `text-display` |
| Technical | `font-mono-ui` / `font-mono` / `font-courier` | Model slugs, env keys, schedules, YAML, repo URLs |

- Do **not** put `themedBody` or `themedFont` on `<main>`, `App`, or other layout wrappers — it overrides component-scoped styles.
- **`Card`** applies `themedBody`; **`CardTitle`** uses `text-display` (uppercase chrome); **`CardDescription`** uses `themedBody`.
- **`NouiTypography`** defaults to `font-sans` unless a font prop is passed.
- Do **not** use raw `font-sans` or `font-display` (theme sans variable) on new dashboard UI — prefer Mondwest tiers above where brand-appropriate.

### Color tokens

- Prefer **semantic tokens** (`text-text-*`, `bg-card`, `border-border`, `text-foreground`, `text-destructive`, `text-success`, `text-warning`) over raw layer references (`text-midground`, `text-foreground`).
- `text-muted-foreground` is now wired to `--color-text-secondary`, so existing call sites stay correct, but new code should prefer the semantic name.
- When you genuinely need a non-token color (icon de-emphasis on a chart, terminal foreground via inline style), keep alpha at `≥ 0.7` for any text.

