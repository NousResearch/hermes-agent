import type { DetailsMode, SectionName, SectionVisibility } from '../types.js'

const MODES = ['hidden', 'collapsed', 'expanded'] as const

export const SECTION_NAMES = ['thinking', 'tools', 'subagents', 'activity'] as const

// Out-of-the-box per-section defaults — applied when the user hasn't pinned
// an explicit override and layered ABOVE the global details_mode:
//
//   - thinking / tools: expanded — stream open so the turn reads like a
//     live transcript (reasoning + tool calls side by side) instead of a
//     wall of chevrons the user has to click every turn.
//   - activity: hidden — ambient meta (gateway hints, terminal-parity
//     nudges, background notifications) is noise for typical use.  Tool
//     failures still render inline on the failing tool row, and ambient
//     errors/warnings surface via the floating-alert backstop when every
//     panel resolves to hidden.
//   - subagents: not set — falls through to the global details_mode so
//     Spawn trees stay under a chevron until a delegation actually happens.
//
// Opt out of any of these with `display.sections.<name>` in config.yaml
// or at runtime via `/details <name> collapsed|hidden`.
const SECTION_DEFAULTS: SectionVisibility = {
  thinking: 'expanded',
  tools: 'expanded',
  activity: 'hidden'
}

const THINKING_FALLBACK: Record<string, DetailsMode> = {
  collapsed: 'collapsed',
  full: 'expanded',
  truncated: 'collapsed'
}

const norm = (v: unknown) =>
  String(v ?? '')
    .trim()
    .toLowerCase()

export const parseDetailsMode = (v: unknown): DetailsMode | null => MODES.find(m => m === norm(v)) ?? null

export const isSectionName = (v: unknown): v is SectionName =>
  typeof v === 'string' && (SECTION_NAMES as readonly string[]).includes(v)

export const resolveDetailsMode = (d?: { details_mode?: unknown; thinking_mode?: unknown } | null): DetailsMode =>
  parseDetailsMode(d?.details_mode) ?? THINKING_FALLBACK[norm(d?.thinking_mode)] ?? 'collapsed'

// Build SectionVisibility from a free-form blob.  Unknown section names and
// invalid modes are dropped silently — partial overrides are intentional, so
// missing keys fall through to SECTION_DEFAULTS / global at lookup time.
export const resolveSections = (raw: unknown): SectionVisibility =>
  raw && typeof raw === 'object' && !Array.isArray(raw)
    ? (Object.fromEntries(
        Object.entries(raw as Record<string, unknown>)
          .map(([k, v]) => [k, parseDetailsMode(v)] as const)
          .filter(([k, m]) => !!m && isSectionName(k))
      ) as SectionVisibility)
    : {}

// Effective mode for one section: explicit override → global command mode →
// built-in live-stream defaults → global config mode.
//
// The `commandOverride` flag is set for in-session `/details <mode>` changes.
// That command should immediately apply to every section, including sections
// with built-in defaults like thinking/tools=expanded and activity=hidden. On
// startup/config sync we keep those defaults layered above the persisted global
// config so the TUI still opens live reasoning/tools by default unless the user
// pins explicit per-section overrides.
export const sectionMode = (
  name: SectionName,
  global: DetailsMode,
  sections?: SectionVisibility,
  commandOverride = false
): DetailsMode => sections?.[name] ?? (commandOverride ? global : (SECTION_DEFAULTS[name] ?? global))

export const nextDetailsMode = (m: DetailsMode): DetailsMode => MODES[(MODES.indexOf(m) + 1) % MODES.length]!
