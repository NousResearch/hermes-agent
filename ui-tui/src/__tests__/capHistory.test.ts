import { describe, expect, it } from 'vitest'

import { capHistory } from '../lib/capHistory.js'
import type { Msg } from '../types.js'

const msg = (role: Msg['role'], text = 'x'): Msg => ({ role, text }) as Msg

describe('capHistory', () => {
  it('returns items unchanged when under the cap', () => {
    const items = Array.from({ length: 5 }, (_, i) => msg('user', String(i)))

    expect(capHistory(items, 10)).toHaveLength(5)
  })

  it('preserves the intro message when over the cap', () => {
    const intro = { kind: 'intro', role: 'system', text: '' } as Msg
    const tail = Array.from({ length: 1000 }, (_, i) => msg('assistant', String(i)))
    const out = capHistory([intro, ...tail], 100)

    expect(out).toHaveLength(100)
    expect(out[0]).toBe(intro)
  })

  it('preserves only the intro message at the intro-reservation boundary', () => {
    const intro = { kind: 'intro', role: 'system', text: '' } as Msg
    const tail = Array.from({ length: 3 }, (_, i) => msg('assistant', String(i)))
    const out = capHistory([intro, ...tail], 1)

    expect(out).toEqual([intro])
  })

  it('drops oldest non-intro messages beyond the cap', () => {
    const items = Array.from({ length: 1005 }, (_, i) => msg('user', `m${i}`))
    const out = capHistory(items, 1000)

    expect(out).toHaveLength(1000)
    // First dropped message was index 0 ("m0"); first kept is "m5".
    expect((out[0] as Extract<Msg, { role: 'user' }>).text).toBe('m5')
  })

  it('normalizes invalid and low caps before slicing', () => {
    const items = Array.from({ length: 5 }, (_, i) => msg('user', `m${i}`))

    expect(capHistory(items, 0)).toEqual([items[4]])
    expect(capHistory(items, Number.NaN)).toBe(items)
  })
})
