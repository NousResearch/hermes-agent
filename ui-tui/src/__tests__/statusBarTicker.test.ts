import { stringWidth } from '@hermes/ink'
import { afterEach, describe, expect, it } from 'vitest'

import { patchUiState } from '../app/uiStore.js'
import { $tickerVerbs, padVerb, VERB_PAD_LEN, verbPadLen } from '../components/appChrome.js'
import { VERBS } from '../content/verbs.js'

describe('FaceTicker verb padding', () => {
  afterEach(() => {
    patchUiState({ skin: null })
  })

  it('pads every verb to the same width', () => {
    for (const verb of VERBS) {
      expect(padVerb(verb)).toHaveLength(VERB_PAD_LEN)
    }
  })

  it('keeps trailing ellipsis attached', () => {
    for (const verb of VERBS) {
      expect(padVerb(verb).startsWith(`${verb}…`)).toBe(true)
    }
  })

  it('pads skin verbs by terminal display width, not JS length', () => {
    // '汉' is 1 UTF-16 unit but 2 display cells — JS `.length` padding would
    // jitter the status bar for CJK/emoji verbs (the stale-PR blocker).
    const verbs = ['🚀', '汉', 'crafting']
    const pad = verbPadLen(verbs)

    for (const verb of verbs) {
      // Every padded row occupies exactly `pad` terminal cells.
      expect(stringWidth(padVerb(verb, verbs))).toBe(pad)
    }

    // A longer verb list reserves more than the built-in default list.
    expect(verbPadLen(['crafting the plan', 'tempering steel'])).toBeGreaterThan(verbPadLen(VERBS))
  })

  it('uses skin spinner.thinking_verbs when present', () => {
    patchUiState({
      skin: { name: 'ares', spinner: { thinking_verbs: ['forging', 'marching', 'tempering steel'] } }
    })

    expect($tickerVerbs.get()).toEqual(['forging', 'marching', 'tempering steel'])
  })

  it('falls back to built-in verbs when the skin has none', () => {
    patchUiState({ skin: { name: 'mono' } })
    expect($tickerVerbs.get()).toBe(VERBS)

    patchUiState({ skin: null })
    expect($tickerVerbs.get()).toBe(VERBS)
  })
})
