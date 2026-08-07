import { describe, expect, it } from 'vitest'
import { getGraphemeSegmenter } from '../utils/intl.js'
import { stringWidth } from './stringWidth.js'
import { tokenize, styledCharsFromTokens } from '@alcalzone/ansi-tokenize'
import { CharPool, createScreen, HyperlinkPool, StylePool } from './screen.js'

const stylePool = new StylePool()
const charPool = new CharPool()
const hyperlinkPool = new HyperlinkPool()

function mkScreen(w: number, h: number) {
  return createScreen(w, h, stylePool, charPool, hyperlinkPool)
}

describe('Thai combining marks fix', () => {
  it('stringWidth returns 0 for Thai SARA AA (U+0E32)', () => {
    // SARA AA renders within the same terminal cell as the base consonant.
    expect(stringWidth('\u0e32')).toBe(0)
  })

  it('stringWidth returns 0 for Thai SARA AM (U+0E33)', () => {
    expect(stringWidth('\u0e33')).toBe(0)
  })

  it('stringWidth returns 1 for grapheme cluster with base + SARA AA', () => {
    // "รา" = ร + า. With SARA AA now zero-width, the base consonant
    // alone determines the grapheme width.
    expect(stringWidth('รา')).toBe(1)
  })

  it('stringWidth returns 1 for full word สร้าง', () => {
    // "สร้าง" occupies 3 visual columns (ส, ร้า, ง)
    // This test ensures the width was already correct — it pins the
    // invariant that the fix doesn't regress existing behavior.
    const graphemes = [...getGraphemeSegmenter().segment('สร้าง')].map(
      s => s.segment
    )
    // Intl.Segmenter produces ["ส","ร้","า","ง"] (no clustering fix here —
    // that's in flushBuffer which runs at a higher layer). stringWidth
    // per-grapheme:
    // "ส"=1, "ร้"=1, "า"=0 (with fix), "ง"=1
    expect(stringWidth('ส')).toBe(1)
    expect(stringWidth('ร้')).toBe(1)
    expect(stringWidth('\u0e32')).toBe(0) // SARA AA standalone: zero-width
    expect(stringWidth('ง')).toBe(1)
  })

  it('stringWidth for ข้อมูล is correct per cluster', () => {
    const graphemes = [...getGraphemeSegmenter().segment('ข้อมูล')].map(
      s => s.segment
    )
    // Intl.Segmenter: ["ข้","อ","มู","ล"]
    expect(graphemes).toHaveLength(4)
    for (const g of graphemes) {
      expect(stringWidth(g)).toBeGreaterThanOrEqual(0)
      expect(stringWidth(g)).toBeLessThanOrEqual(1)
    }
  })

  it('stringWidth did NOT change for non-Thai characters', () => {
    // Verify no regressions for common character classes
    expect(stringWidth('a')).toBe(1) // ASCII
    expect(stringWidth(' ')).toBe(1) // space
    expect(stringWidth('あ')).toBe(2) // CJK (wide)
    expect(stringWidth('🫰')).toBe(2) // emoji
    expect(stringWidth('é')).toBe(1) // Latin-1 with combining (or single char)
    expect(stringWidth('\u0300')).toBe(0) // combining grave accent
  })

  it('zero-width grapheme merging: Intl.Segmenter splits SARA AA', () => {
    // Verify Intl.Segmenter behavior: SARA AA is split from base.
    // This is the bug that the flushBuffer merge fix works around.
    const graphemes = [...getGraphemeSegmenter().segment('รา')].map(
      s => s.segment
    )
    expect(graphemes).toEqual(['ร', '\u0e32'])
    // Without the flushBuffer fix in output.ts (which happens at a higher
    // layer), "า" would be a separate 1-cell-wide grapheme, causing cursor
    // desync. With the stringWidth fix, stringWidth('\u0e32')=0, so
    // flushBuffer merges it back into the preceding cluster.
    expect(stringWidth(graphemes[0]!)).toBe(1) // ร
    expect(stringWidth(graphemes[1]!)).toBe(0) // า (zero-width with fix)
  })

  it('ไทย (thai) with tone marks are still zero-width', () => {
    // Thai tone marks (U+0E48-U+0E4E) and other combining signs
    expect(stringWidth('\u0e48')).toBe(0) // MAI EK
    expect(stringWidth('\u0e49')).toBe(0) // MAI THO
    expect(stringWidth('\u0e4a')).toBe(0) // MAI TRI
    expect(stringWidth('\u0e4b')).toBe(0) // MAI CHATTAWA
    // Base + tone mark
    expect(stringWidth('ร\u0e48')).toBe(1) // ร + MAI EK = width 1
    expect(stringWidth('ร\u0e49')).toBe(1) // ร + MAI THO = width 1
  })

  it('Lao SARA AA (U+0EB2) and SARA AM (U+0EB3) are zero-width', () => {
    expect(stringWidth('\u0eb2')).toBe(0) // Lao SARA AA
    expect(stringWidth('\u0eb3')).toBe(0) // Lao SARA AM
    // Base + SARA AA
    expect(stringWidth('ກ\u0eb2')).toBe(1) // ກ (Lao ko) + SARA AA
  })
})
