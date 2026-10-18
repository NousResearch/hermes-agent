import { describe, expect, it } from 'vitest'

import { completionReplyMessages } from '../app/userMessages.js'

describe('completionReplyMessages', () => {
  it('adds no reply for a deliberately silent wake turn', () => {
    expect(completionReplyMessages({ silent: true, text: '' }, '')).toEqual([])
  })

  it('keeps an empty-looking reply visible when the turn was not marked silent', () => {
    expect(completionReplyMessages({ text: '' }, '')).toEqual([{ role: 'assistant', text: '' }])
  })

  it('never drops real text even if the frame says silent', () => {
    expect(completionReplyMessages({ silent: true }, 'Needs you')).toEqual([{ role: 'assistant', text: 'Needs you' }])
  })
})
