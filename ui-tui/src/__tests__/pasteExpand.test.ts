import { describe, expect, it } from 'vitest'

import type { ComposerToken } from '../app/interfaces.js'
import { stagePastes, takePastes } from '../app/pasteStore.js'
import { prepareSubmission } from '../app/useSubmission.js'
import { collapsePasteSpan, expandPasteChip, insideSpan, shiftSpan } from '../domain/pasteExpand.js'
import { pasteTokenLabel } from '../lib/text.js'

const lines = (middle: string) => Array.from({ length: 60 }, (_, i) => (i === 30 ? middle : `line ${i + 1}`)).join('\n')
const body = lines('line 31')
const label = pasteTokenLabel(body, 60)
const token: ComposerToken = { kind: 'paste', label, text: body }

describe('expanding a collapsed paste in the composer', () => {
  it('sends the text as edited while expanded, under a chip again', () => {
    const draft = `review: ${label}`
    const expanded = expandPasteChip(draft, draft.length, [token])!

    expect(expanded.value).toBe(`review: ${body}`)

    // Edit inside the expanded text, extend its last line, and edit the prose before it.
    const at = expanded.value.indexOf('line 30')
    let value = expanded.value.slice(0, at) + 'EDITED ' + expanded.value.slice(at)
    let span = shiftSpan(expanded.value, value, expanded.span)!

    for (const next of [`${value}!`, `please ${value}!`]) {
      span = shiftSpan(value, next, span)!
      value = next
    }

    const collapsed = collapsePasteSpan(value, span, expanded.tokens)!
    const sent = prepareSubmission(collapsed.value, collapsed.tokens)

    expect(sent.display).toBe(`please review: ${collapsed.tokens[0]!.label}`)
    expect(sent.text).toBe(`please review: ${body.replace('line 30', 'EDITED line 30')}!`)
  })

  it('keeps two same-label pastes on their own bodies through open, edit and fold', () => {
    // Same first/last lines and line count, so the same chip label; different middles.
    const a: ComposerToken = { kind: 'paste', label, text: lines('ALPHA') }
    const b: ComposerToken = { kind: 'paste', label, text: lines('BRAVO') }
    let value = `one ${label} two ${label}`
    let tokens = [a, b]

    // Open the FIRST chip, edit it, fold it back; then the same for the second.
    for (const [first, from, to] of [
      [true, 'ALPHA', 'ALPHA-2'],
      [false, 'BRAVO', 'BRAVO-2']
    ] as const) {
      const expanded = expandPasteChip(value, first ? value.indexOf(label) : value.lastIndexOf(label), tokens)!
      const edited = expanded.value.replace(from, to)
      const span = shiftSpan(expanded.value, edited, expanded.span)!
      const collapsed = collapsePasteSpan(edited, span, expanded.tokens)!
      value = collapsed.value
      tokens = collapsed.tokens
    }

    const sent = prepareSubmission(value, tokens).text

    expect(sent).toBe(`one ${lines('ALPHA-2')} two ${lines('BRAVO-2')}`)
  })

  it('takes a paste made inside an expanded span as plain text, and never folds a chip into one', () => {
    const expanded = expandPasteChip(label, 0, [token])!

    expect(insideSpan(expanded.span, expanded.span.end)).toBe(true)
    expect(insideSpan(expanded.span, expanded.span.start)).toBe(false)

    // A chip that lands inside the span: folding would hide its label inside the outer payload.
    const image: ComposerToken = { index: 1, kind: 'image', label: '[[ Image 1 ]]', path: '/x.png' }
    const value = `${expanded.value}${image.label}`
    const span = shiftSpan(expanded.value, value, expanded.span)!

    expect(collapsePasteSpan(value, span, [image])).toBeNull()
  })
})

describe('sent message paste sections', () => {
  it('gives each user row the bodies behind its own chips, by the text the model got', () => {
    const other = lines('OTHER')
    const display = `review: ${label} vs ${label}`
    const tokens: ComposerToken[] = [token, { kind: 'paste', label, text: other }]
    const sent = prepareSubmission(display, tokens).text

    stagePastes(display, sent, tokens)
    // A later submission with the same chip but a different body never sees these.
    stagePastes(`review: ${label}`, `review: ${other}`, [{ kind: 'paste', label, text: other }])

    expect(takePastes(sent)).toEqual({
      display,
      pastes: [
        { label, text: body },
        { label, text: other }
      ]
    })
    expect(takePastes(sent)).toBeUndefined()
    expect(takePastes(`review: ${other}`)?.pastes).toEqual([{ label, text: other }])
  })
})
