import { describe, expect, it } from 'vitest'

import {
  advancePrintableBurst,
  applyPrintableInsert,
  type PrintableBurstState,
  shouldRouteMultiCharInputAsPaste
} from '../components/textInput.js'

describe('advancePrintableBurst', () => {
  it('switches to frame batching for a sustained 2ms key stream', () => {
    let state: PrintableBurstState | null = null
    const decisions: boolean[] = []

    for (let index = 0; index < 20; index++) {
      const sample = advancePrintableBurst(state, index * 2)
      state = sample.state
      decisions.push(sample.rapid)
    }

    expect(decisions.slice(0, 7)).toEqual(Array.from({ length: 7 }, () => false))
    expect(decisions.slice(7)).toEqual(Array.from({ length: 13 }, () => true))
  })

  it('resets to normal echo after an idle gap', () => {
    let state: PrintableBurstState | null = null

    for (let index = 0; index < 8; index++) {
      state = advancePrintableBurst(state, index * 2).state
    }

    expect(state.rapid).toBe(true)
    expect(advancePrintableBurst(state, 100).rapid).toBe(false)
  })

  it('does not classify ordinary typing as machine-speed input', () => {
    let state: PrintableBurstState | null = null

    for (let index = 0; index < 20; index++) {
      const sample = advancePrintableBurst(state, index * 20)
      state = sample.state
      expect(sample.rapid).toBe(false)
    }
  })
})

describe('applyPrintableInsert', () => {
  it('applies non-bracketed multi-character bursts immediately', () => {
    const burst = applyPrintableInsert('abc', 3, 'xxxxx')

    const repeated = [...'xxxxx'].reduce((state, ch) => applyPrintableInsert(state.value, state.cursor, ch)!, {
      cursor: 3,
      value: 'abc'
    })

    expect(burst).toEqual({ cursor: 8, value: 'abcxxxxx' })
    expect(burst).toEqual(repeated)
  })

  it('replaces the selected range for burst input', () => {
    expect(applyPrintableInsert('abZZef', 4, 'cd', { end: 4, start: 2 })).toEqual({
      cursor: 4,
      value: 'abcdef'
    })
  })

  it('rejects control or escape-bearing input', () => {
    expect(applyPrintableInsert('abc', 3, '\x1b[200~pasted')).toBeNull()
    expect(applyPrintableInsert('abc', 3, '\t')).toBeNull()
  })
})

describe('shouldRouteMultiCharInputAsPaste', () => {
  it('keeps newline-bearing chunks on the paste path', () => {
    expect(shouldRouteMultiCharInputAsPaste('hello\nworld')).toBe(true)
    expect(shouldRouteMultiCharInputAsPaste('hello\r\nworld'.replace(/\r\n/g, '\n'))).toBe(true)
  })

  it('treats repeated printable key bursts as immediate input', () => {
    expect(shouldRouteMultiCharInputAsPaste('xxxxx')).toBe(false)
  })
})
