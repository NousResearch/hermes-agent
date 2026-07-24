import { describe, expect, it } from 'vitest'

import type { PasteSnippet } from './interfaces.js'
import { expandSnips } from './useSubmission.js'

const snip = (label: string, text: string): PasteSnippet => ({ label, text })

describe('expandSnips (paste history recall)', () => {
  it('replaces a collapsed paste label with its full content', () => {
    const label = '[[ hello.. [3 lines] .. world ]]'
    const full = `here: ${label} done`
    const expand = expandSnips([snip(label, 'hello\nfoo\nworld')])

    expect(expand(full)).toBe('here: hello\nfoo\nworld done')
  })

  it('is a no-op for already-expanded / label-free text (recall round-trip)', () => {
    const expanded = 'hello\nfoo\nworld'
    // Re-submitting a recalled history entry has no snips and no labels.
    expect(expandSnips([])(expanded)).toBe(expanded)
  })

  it('expands repeated identical labels in submission order', () => {
    const label = '[[ x [1 lines] ]]'
    const expand = expandSnips([snip(label, 'first'), snip(label, 'second')])

    expect(expand(`${label} then ${label}`)).toBe('first then second')
  })

  it('leaves an unmatched label intact', () => {
    const label = '[[ orphan [2 lines] ]]'
    expect(expandSnips([])(label)).toBe(label)
  })
})
