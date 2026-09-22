import { describe, expect, it } from 'vitest'

import { isArabicChar, isArabicTashkeel, shapeArabic, shapeArabicCharacters, unshapeArabic } from './arabic.js'

describe('Arabic text shaper', () => {
  it('detects Arabic code points and diacritics', () => {
    expect(isArabicChar(0x0627)).toBe(true) // Alef
    expect(isArabicChar(0x0645)).toBe(true) // Meem
    expect(isArabicChar(0x0041)).toBe(false) // Latin 'A'
    expect(isArabicTashkeel(0x064e)).toBe(true) // Fatha
    expect(isArabicTashkeel(0x0627)).toBe(false)
  })

  it('shapes basic Arabic word "مرحبا" into contextual forms', () => {
    // م (Meem initial: \uFEE3)
    // ر (Reh final: \uFEAE)
    // ح (Hah initial: \uFEA3)
    // ب (Beh medial: \uFE92)
    // ا (Alef final: \uFE8E)
    const shaped = shapeArabic('مرحبا')
    expect(shaped).toBe('\uFEE3\uFEAE\uFEA3\uFE92\uFE8E')
  })

  it('shapes "صديقي" with dual-joining and right-joining letters', () => {
    // ص (Sad initial: \uFEBB)
    // د (Dal final: \uFEAA)
    // ي (Yeh initial: \uFEF3)
    // ق (Qaf medial: \uFED8)
    // ي (Yeh final: \uFEF2)
    const shaped = shapeArabic('صديقي')
    expect(shaped).toBe('\uFEBB\uFEAA\uFEF3\uFED8\uFEF2')
  })

  it('handles non-connecting letters correctly (e.g. د, ر, و, ا)', () => {
    // "ورد" (Waw, Reh, Dal - all right-joining)
    // و (isolated: \uFEED)
    // ر (isolated: \uFEAD)
    // د (isolated: \uFEA9)
    const shaped = shapeArabic('ورد')
    expect(shaped).toBe('\uFEED\uFEAD\uFEA9')
  })

  it('preserves Tashkeel diacritics without breaking letter connections', () => {
    // بَ (Beh initial with Fatha: \uFE91 + \u064E)
    // ت (Teh final: \uFE96)
    const shaped = shapeArabic('بَت')
    expect(shaped).toBe('\uFE91\u064E\uFE96')
  })

  it('preserves ClusteredChar metadata (width, styleId, hyperlink)', () => {
    const chars = [
      { value: 'م', width: 1, styleId: 42, hyperlink: 'http://hermes.ai' },
      { value: 'ر', width: 1, styleId: 42, hyperlink: 'http://hermes.ai' }
    ]

    const result = shapeArabicCharacters(chars)
    expect(result[0]!.value).toBe('\uFEE3')
    expect(result[0]!.styleId).toBe(42)
    expect(result[0]!.hyperlink).toBe('http://hermes.ai')
    expect(result[1]!.value).toBe('\uFEAE')
    expect(result[1]!.styleId).toBe(42)
  })

  it('un-shapes presentation forms back to canonical Unicode characters', () => {
    const shaped = shapeArabic('أهلاً بيك يا صديقي!')
    const unshaped = unshapeArabic(shaped)
    expect(unshaped).toBe('أهلاً بيك يا صديقي!')
  })
})
