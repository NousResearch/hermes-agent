import { readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/**
 * Deep-merge `overrides` on top of `base`, preserving the base's shape:
 *
 * - a string leaf is replaced only by a string;
 * - nested objects merge recursively;
 * - keys absent from `base` are ignored (no unknown keys injected);
 * - type mismatches are ignored.
 *
 * Returns the same `base` reference when nothing changes, so callers can cheaply
 * skip re-renders on a no-op merge.
 */
export function mergeStrings<T>(base: T, overrides: unknown): T {
  if (!isRecord(base) || !isRecord(overrides)) {
    return base
  }

  let changed = false
  const result: Record<string, unknown> = { ...base }

  for (const [key, overrideValue] of Object.entries(overrides)) {
    if (!(key in base)) {
      continue
    }

    const baseValue = (base as Record<string, unknown>)[key]

    if (typeof baseValue === 'string') {
      if (typeof overrideValue === 'string' && overrideValue !== baseValue) {
        result[key] = overrideValue
        changed = true
      }
      continue
    }

    if (isRecord(baseValue) && isRecord(overrideValue)) {
      const mergedChild = mergeStrings(baseValue, overrideValue)
      if (mergedChild !== baseValue) {
        result[key] = mergedChild
        changed = true
      }
    }
  }

  return changed ? (result as T) : base
}

function hermesHome(): string {
  return process.env.HERMES_HOME?.trim() || join(homedir(), '.hermes')
}

const LANG_RE = /^[A-Za-z]{2,8}(-[A-Za-z]{2,8})*$/

/**
 * Read user-authored TUI locale overrides for `lang` from
 * `<hermes-home>/locale-overrides/tui/<lang>.json`. This lives outside the
 * installed bundle, so it survives updates. Returns the parsed object, or null
 * when absent, unreadable, not valid JSON, or not a JSON object. Never throws.
 *
 * Honors `HERMES_LOCALE_OVERRIDES` (a directory) for parity with the Python
 * layer / tests; the `tui/<lang>.json` suffix is appended to it.
 */
export function readUserOverrides(lang: string): Record<string, unknown> | null {
  if (!LANG_RE.test(lang)) {
    return null
  }

  const baseDir = process.env.HERMES_LOCALE_OVERRIDES?.trim() || join(hermesHome(), 'locale-overrides')
  const file = join(baseDir, 'tui', `${lang}.json`)

  let raw: string
  try {
    raw = readFileSync(file, 'utf8')
  } catch {
    return null
  }

  try {
    const parsed: unknown = JSON.parse(raw)
    return isRecord(parsed) ? parsed : null
  } catch {
    return null
  }
}
