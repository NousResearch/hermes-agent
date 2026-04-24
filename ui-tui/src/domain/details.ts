import type { DetailsMode, SectionName, SectionVisibility } from '../types.js'

const MODES = ['hidden', 'collapsed', 'expanded'] as const
export const SECTION_NAMES: readonly SectionName[] = ['thinking', 'tools', 'subagents', 'activity']

const THINKING_FALLBACK: Record<string, DetailsMode> = {
  collapsed: 'collapsed',
  full: 'expanded',
  truncated: 'collapsed'
}

export const parseDetailsMode = (v: unknown): DetailsMode | null => {
  const s = typeof v === 'string' ? v.trim().toLowerCase() : ''

  return MODES.find(m => m === s) ?? null
}

export const isSectionName = (v: unknown): v is SectionName =>
  typeof v === 'string' && (SECTION_NAMES as readonly string[]).includes(v)

export const resolveDetailsMode = (d?: { details_mode?: unknown; thinking_mode?: unknown } | null): DetailsMode =>
  parseDetailsMode(d?.details_mode) ??
  THINKING_FALLBACK[
    String(d?.thinking_mode ?? '')
      .trim()
      .toLowerCase()
  ] ??
  'collapsed'

// Build a SectionVisibility from a free-form `display.sections` config blob.
// Skips keys that aren't recognized section names or don't parse to a valid
// mode — partial overrides are intentional, missing keys fall through to the
// global details_mode at render time.
export const resolveSections = (raw: unknown): SectionVisibility => {
  const out: SectionVisibility = {}

  if (!raw || typeof raw !== 'object') {
    return out
  }

  for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
    const mode = parseDetailsMode(v)

    if (mode && isSectionName(k)) {
      out[k] = mode
    }
  }

  return out
}

// Built-in per-section defaults applied when the user has no explicit
// override.  The activity panel (gateway hints, terminal-parity nudges,
// background-process notifications) is hidden out of the box — it's noise
// for the typical day-to-day user, who only cares about thinking + tools +
// streamed content.  Tool failures still surface inline on the failing tool
// row; this default only suppresses the ambient meta feed.
//
// Opt back in with `display.sections.activity: collapsed` (under chevron)
// or `expanded` (always open) in `~/.hermes/config.yaml`, or live with
// `/details activity collapsed`.
const SECTION_DEFAULTS: SectionVisibility = { activity: 'hidden' }

// Resolve the effective mode for one section: explicit override wins,
// then the SECTION_DEFAULTS fallback, then the global details_mode.
// Single source of truth — every render site that needs to know "is this
// section open by default" calls this.
export const sectionMode = (
  name: SectionName,
  global: DetailsMode,
  sections?: SectionVisibility
): DetailsMode => sections?.[name] ?? SECTION_DEFAULTS[name] ?? global

export const nextDetailsMode = (m: DetailsMode): DetailsMode => MODES[(MODES.indexOf(m) + 1) % MODES.length]!
