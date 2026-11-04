import { describe, expect, it } from 'vitest'

import type { ComposerToken } from '../app/interfaces.js'
import { attachPastes, stagePastes } from '../app/pasteStore.js'
import { prepareSubmission } from '../app/useSubmission.js'
import { collapsePasteSpan, expandPasteChip, shiftSpan } from '../domain/pasteExpand.js'
import { pasteTokenLabel } from '../lib/text.js'

const body = Array.from({ length: 60 }, (_, i) => `line ${i + 1}`).join('\n')
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

    const collapsed = collapsePasteSpan(value, span)
    const sent = prepareSubmission(collapsed.value, [...expanded.tokens, collapsed.token])

    expect(sent.display).toBe(`please review: ${collapsed.token.label}`)
    expect(sent.text).toBe(`please review: ${body.replace('line 30', 'EDITED line 30')}!`)
  })
})

describe('sent message paste sections', () => {
  it('gives the user row the full payload behind each chip it shows', () => {
    stagePastes([token])

    expect(attachPastes({ role: 'user', text: `review: ${label}` }).pastes).toEqual([{ label, text: body }])
    // Consumed: a later row with the same chip text does not inherit it.
    expect(attachPastes({ role: 'user', text: label }).pastes).toBeUndefined()
  })
})
