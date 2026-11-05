/**
 * Ctrl+S prompt stash — port of `hermes_cli/prompt_stash.py` for the Ink composer.
 *
 * The classic CLI owns the reference implementation: a session-scoped stack of parked
 * drafts, newest first, with a browse panel. This module keeps the TUI on exactly the
 * same decision table (see `resolveCtrlS`) and the same limits, wording and preview
 * shaping, so the two interfaces cannot drift apart. `tests/fixtures/prompt_stash_vectors.json`
 * is read by both this module's vitest suite and the Python suite next to the original.
 *
 * Everything here is pure — no React, no Ink, no gateway — so the decision table is
 * unit-testable the same way the Python one is.
 */
import type { ComposerToken } from '../app/interfaces.js'

/** Single-line preview length for the browse panel. */
export const STASH_PREVIEW_WIDTH = 60

/** Cap the stack so a user leaning on Ctrl+S can't grow it without bound. */
export const MAX_STASH_ITEMS = 20

const ELLIPSIS = '…'
const NEWLINE_GLYPH = ' ⏎ '

/**
 * Collapse a possibly multi-line draft into one preview line.
 *
 * Mirrors `build_preview`: normalise CRLF/CR, render newlines as `⏎`, tabs as a space,
 * squeeze every whitespace run to one space, then truncate to `width` cells with `…`.
 * Truncation counts code points, not UTF-16 units, because the Python original counts
 * characters — an emoji must not cut one code unit earlier in one surface than the other.
 */
export function buildPreview(text: string, width: number = STASH_PREVIEW_WIDTH): string {
  if (!text) {
    return ''
  }

  const normalised = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n')
  const flattened = normalised.replace(/\n/g, NEWLINE_GLYPH).replace(/\t/g, ' ')

  // Squeeze every whitespace run to one space, as `" ".join(flat.split())` does.
  const flat = flattened
    .split(/\s+/)
    .filter(part => part.length > 0)
    .join(' ')

  if (width > 1) {
    const chars = [...flat]

    if (chars.length > width) {
      return chars.slice(0, width - 1).join('') + ELLIPSIS
    }
  }

  return flat
}

/** One parked draft: exact text plus the chips that were attached to it. */
export interface StashEntry {
  /** Preview line the browse panel renders. */
  preview: string
  /** Wall-clock ms at which the draft was parked (panel age display only). */
  stashedAt: number
  text: string
  /** Paste chips and image tokens, so a collapsed paste survives the round trip. */
  tokens: ComposerToken[]
}

/**
 * Wording for the two placeholder hints. Injected so the TUI can translate them while
 * these defaults stay byte-identical to `PromptStash.placeholder_hint` in Python.
 */
export interface StashHintStrings {
  browse: (count: number) => string
  restore: (preview: string) => string
}

export const DEFAULT_STASH_HINTS: StashHintStrings = {
  browse: count => `Ctrl+S to browse ${count} stashed drafts`,
  restore: preview => `Ctrl+S to restore: ${preview}`
}

/** Wording for the browse panel's age column. Defaults mirror `_fmt_stash_age`. */
export interface StashAgeStrings {
  hours: (hours: number) => string
  justNow: string
  minutes: (minutes: number) => string
  seconds: (seconds: number) => string
}

export const DEFAULT_STASH_AGE: StashAgeStrings = {
  hours: hours => `${hours}h ago`,
  justNow: 'just now',
  minutes: minutes => `${minutes} min ago`,
  seconds: seconds => `${seconds}s ago`
}

/**
 * Age column of one panel row, mirroring `_fmt_stash_age`: sub-10s reads "just now",
 * sub-90s counts seconds, then whole minutes, then whole hours.
 */
export function formatStashAge(
  stashedAt: number,
  now: number,
  strings: StashAgeStrings = DEFAULT_STASH_AGE
): string {
  const secs = Math.trunc((now - stashedAt) / 1000)

  if (secs < 10) {
    return strings.justNow
  }

  if (secs < 90) {
    return strings.seconds(secs)
  }

  const mins = Math.trunc(secs / 60)

  return mins < 60 ? strings.minutes(mins) : strings.hours(Math.trunc(mins / 60))
}

/** Snapshots handed back to the composer on restore. */
export interface StashPayload {
  text: string
  tokens: ComposerToken[]
}

export class PromptStash {
  private readonly clock: () => number
  private readonly maxItems: number
  private items: StashEntry[] = []

  panelCursor = 0
  panelOpen = false

  constructor(options: { clock?: () => number; maxItems?: number } = {}) {
    this.clock = options.clock ?? Date.now
    this.maxItems = Math.max(1, Math.trunc(options.maxItems ?? MAX_STASH_ITEMS))
  }

  get length(): number {
    return this.items.length
  }

  /** Newest-first list of entries (a copy — mutate through the methods). */
  get entries(): StashEntry[] {
    return [...this.items]
  }

  panelRows(): StashEntry[] {
    return [...this.items]
  }

  /** Status-bar indicator: `📌 2` idle, `📌 2 ▲` while browsing, `''` when empty. */
  indicator(): string {
    const n = this.items.length

    return n === 0 ? '' : this.panelOpen ? `📌 ${n} ▲` : `📌 ${n}`
  }

  /** Composer placeholder text advertising the parked draft. Mirrors `placeholder_hint`. */
  placeholderHint(hints: StashHintStrings = DEFAULT_STASH_HINTS): string {
    const n = this.items.length

    if (n === 1) {
      return hints.restore(this.items[0]!.preview)
    }

    return n === 0 ? '' : hints.browse(n)
  }

  /**
   * Push a draft. A blank buffer with no chips is a no-op so Ctrl+S on an empty composer
   * triggers the restore half of the gesture instead of pushing junk.
   */
  stash(text: string, tokens: readonly ComposerToken[] = []): boolean {
    if (!(text ?? '').trim() && tokens.length === 0) {
      return false
    }

    const entry: StashEntry = {
      preview: buildPreview(text ?? '') || '(images only)',
      stashedAt: this.clock(),
      text: text ?? '',
      tokens: [...tokens]
    }

    this.items.unshift(entry)
    this.items = this.items.slice(0, this.maxItems) // drop the oldest past the cap
    this.closePanel()

    return true
  }

  /** Remove and return the entry at `index`, or null. */
  pop(index = 0): StashPayload | null {
    if (index < 0 || index >= this.items.length) {
      return null
    }

    const [entry] = this.items.splice(index, 1)

    if (!this.items.length) {
      this.panelOpen = false
    }

    this.panelCursor = this.clampCursor(this.panelCursor)

    return entry ? { text: entry.text, tokens: [...entry.tokens] } : null
  }

  peek(index = 0): StashEntry | null {
    return index >= 0 && index < this.items.length ? this.items[index]! : null
  }

  clear(): void {
    this.items = []
    this.closePanel()
  }

  // ------------------------------------------------------------ panel state

  private clampCursor(value: number): number {
    return this.items.length ? Math.max(0, Math.min(Math.trunc(value), this.items.length - 1)) : 0
  }

  /** Open the browse panel. False when there is nothing to browse. */
  openPanel(): boolean {
    if (!this.items.length) {
      return false
    }

    this.panelOpen = true
    this.panelCursor = 0

    return true
  }

  closePanel(): void {
    this.panelOpen = false
    this.panelCursor = 0
  }

  /** Move the panel cursor, clamped to the list bounds. Returns the new index. */
  moveCursor(delta: number): number {
    this.panelCursor = this.clampCursor(this.panelCursor + Math.trunc(delta))

    return this.panelCursor
  }

  /** Delete the highlighted entry. False when there was nothing to drop. */
  deleteAtCursor(): boolean {
    if (!this.items.length) {
      return false
    }

    const idx = this.clampCursor(this.panelCursor)

    this.items.splice(idx, 1)

    if (!this.items.length) {
      this.closePanel()
    } else {
      this.panelCursor = this.clampCursor(idx)
    }

    return true
  }

  /** Pop the highlighted entry and close the panel. */
  restoreAtCursor(): StashPayload | null {
    if (!this.items.length) {
      return null
    }

    const result = this.pop(this.clampCursor(this.panelCursor))

    this.closePanel()

    return result
  }
}

/** Outcomes of a single Ctrl+S press. Same names as the Python action constants. */
export type StashAction = 'close_panel' | 'noop' | 'open_panel' | 'restored' | 'stashed'

export const ACTION_NOOP: StashAction = 'noop'
export const ACTION_STASHED: StashAction = 'stashed'
export const ACTION_RESTORED: StashAction = 'restored'
export const ACTION_OPEN_PANEL: StashAction = 'open_panel'
export const ACTION_CLOSE_PANEL: StashAction = 'close_panel'

/**
 * Decide what one Ctrl+S press does, exactly as `resolve_ctrl_s` does:
 *
 *   panel open            → close it (the escape hatch)
 *   text or chips present → park them (onto the stack, so an earlier draft stays reachable)
 *   empty, nothing parked → no-op
 *   empty, one parked     → restore it
 *   empty, several parked → open the browse panel
 */
export function resolveCtrlS(
  stash: PromptStash,
  bufferText: string,
  tokens: readonly ComposerToken[] = []
): { action: StashAction; payload: null | StashPayload } {
  if (stash.panelOpen) {
    stash.closePanel()

    return { action: ACTION_CLOSE_PANEL, payload: null }
  }

  if ((bufferText ?? '').trim() || tokens.length) {
    return { action: stash.stash(bufferText, tokens) ? ACTION_STASHED : ACTION_NOOP, payload: null }
  }

  const count = stash.length

  if (count === 0) {
    return { action: ACTION_NOOP, payload: null }
  }

  if (count === 1) {
    return { action: ACTION_RESTORED, payload: stash.pop(0) }
  }

  stash.openPanel()

  return { action: ACTION_OPEN_PANEL, payload: null }
}
