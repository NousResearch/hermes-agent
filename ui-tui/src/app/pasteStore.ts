import { atom } from 'nanostores'

import { PASTE_SNIPPET_RE } from '../protocol/paste.js'
import type { PasteSection } from '../types.js'

import type { ComposerToken } from './interfaces.js'

const MAX_STAGED = 16
const MAX_STAGED_CHARS = 8 * 1024 * 1024

interface StagedPastes {
  display: string
  pastes: PasteSection[]
}

// The chip display and paste bodies of recent submissions, keyed by the exact
// text the model gets. Whoever paints a submission's user row (at submit, from
// the local queue, or when the authority starts a busy-time admission) holds
// that same text, so the row takes its own bodies and never another
// message's: two submissions under one key sent the same text.
const staged = new Map<string, StagedPastes>()

/** Ctrl+P with no chip in the composer: open (or close) every sent paste section. */
export const $pasteSectionsOpen = atom(false)

export const togglePasteSections = () => $pasteSectionsOpen.set(!$pasteSectionsOpen.get())

const stagedChars = () =>
  [...staged].reduce((sum, [key, { pastes }]) => pastes.reduce((n, p) => n + p.text.length, sum + key.length), 0)

/**
 * Remember the body behind each chip of `display`, paired exactly as
 * `expandTokens` pairs them (left to right, same-label tokens in list order),
 * under `text`: what the model receives.
 */
export function stagePastes(display: string, text: string, tokens: readonly ComposerToken[]) {
  const byLabel = new Map<string, string[]>()

  for (const token of tokens) {
    if (token.kind === 'paste') {
      byLabel.set(token.label, [...(byLabel.get(token.label) ?? []), token.text])
    }
  }

  const pastes = (display.match(new RegExp(PASTE_SNIPPET_RE.source, 'g')) ?? []).flatMap(label => {
    const body = byLabel.get(label)?.shift()

    return body === undefined ? [] : [{ label, text: body }]
  })

  if (!pastes.length) {
    return
  }

  staged.delete(text)
  staged.set(text, { display, pastes })

  // Bounded by entries and by size; the newest always stays.
  for (const key of staged.keys()) {
    if (staged.size <= 1 || (staged.size <= MAX_STAGED && stagedChars() <= MAX_STAGED_CHARS)) {
      break
    }

    staged.delete(key)
  }
}

/** Take the chip display and paste bodies of the submission whose model text is `text`. */
export function takePastes(text: string): StagedPastes | undefined {
  const entry = staged.get(text)
  staged.delete(text)

  return entry
}
