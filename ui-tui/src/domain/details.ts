import type { DetailsMode, SectionName, SectionVisibility } from '../types.js'

const MODES = ['hidden', 'collapsed', 'expanded'] as const

export const SECTION_NAMES = ['thinking', 'tools', 'subagents', 'activity'] as const

// Activity panel = ambient meta (gateway hints, terminal-parity nudges,
// background-process notifications).  Hidden out of the box because tool
// failures already render inline on the failing tool row — the panel itself
// is noise for typical use.  Opt back in via `display.sections.activity` or
// `/details activity collapsed`.
const SECTION_DEFAULTS: SectionVisibility = { activity: 'hidden' }

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

// Effective mode for one section: explicit override → SECTION_DEFAULTS → global.
// Single source of truth for "is this section open by default / rendered at all".
export const sectionMode = (name: SectionName, global: DetailsMode, sections?: SectionVisibility): DetailsMode =>
  sections?.[name] ?? SECTION_DEFAULTS[name] ?? global

export const nextDetailsMode = (m: DetailsMode): DetailsMode => MODES[(MODES.indexOf(m) + 1) % MODES.length]!
