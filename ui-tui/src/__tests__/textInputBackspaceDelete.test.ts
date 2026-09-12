import { describe, expect, it } from 'vitest'

import { backspaceDelete, canFastBackspaceShape } from '../components/textInput.js'

// Upstream #94512: the TUI composer snapped Backspace to a grapheme-cluster
// boundary (prevPos → Intl.Segmenter), so deleting a Thai syllable like กิ
// (ก U+0E01 + ◌ิ U+0E34) removed the whole cluster — base consonant included.
// Native text fields delete one code point at a time, so a vowel/tone typo
// only removes the trailing mark. These tests pin the corrected semantics:
// per-code-point Backspace at end-of-input over a combining mark, with the
// deliberate grapheme-cluster model preserved everywhere else.

const KO_KAI = '\u0e01' // ก  Thai base consonant
const SARA_I = '\u0e34' // ◌ิ  Thai combining vowel
const MAI_THO = '\u0e49' // ◌้  Thai combining tone

// Mirror of the source's graphemeStops (Intl.Segmenter, granularity grapheme,
// always including 0 and s.length). Used to verify the snapPos no-op invariant.
const graphemeStops = (s: string): number[] => {
  const stops = [0]

  for (const { index } of new Intl.Segmenter(undefined, { granularity: 'grapheme' }).segment(s)) {
    if (index > 0) {
      stops.push(index)
    }
  }

  if (stops.at(-1) !== s.length) {
    stops.push(s.length)
  }

  return stops
}

const atGraphemeBoundary = (s: string, p: number): boolean => graphemeStops(s).includes(p)


describe('backspaceDelete', () => {
  it('deletes one combining mark at a time at end-of-input (Thai กิ → ก)', () => {
    const thai = KO_KAI + SARA_I // กิ

    expect(backspaceDelete(thai, thai.length)).toEqual({ cursor: 1, value: KO_KAI })
  })

  it('peels a base + 2 combining marks one Backspace per mark', () => {
    const cluster = KO_KAI + SARA_I + MAI_THO // ก�้

    const afterOne = backspaceDelete(cluster, cluster.length) // ก�
    const afterTwo = backspaceDelete(afterOne.value, afterOne.value.length) // ก

    expect(afterOne).toEqual({ cursor: 2, value: KO_KAI + SARA_I })
    expect(afterTwo).toEqual({ cursor: 1, value: KO_KAI })
  })

  it('extends to Devanagari vowel signs (क + ि → क)', () => {
    const deva = '\u0915\u093f' // क + ि (U+093F DEVANAGARI VOWEL SIGN I)

    expect(backspaceDelete(deva, deva.length)).toEqual({ cursor: 1, value: '\u0915' })
  })

  it('leaves plain ASCII backspace unchanged (ab → a)', () => {
    expect(backspaceDelete('ab', 2)).toEqual({ cursor: 1, value: 'a' })
    expect(backspaceDelete('a', 1)).toEqual({ cursor: 0, value: '' })
  })

  it('never splits a surrogate pair: one Backspace removes a whole emoji', () => {
    const value = 'a😀' // U+1F600 = two UTF-16 units

    expect(value.length).toBe(3)
    expect(backspaceDelete(value, value.length)).toEqual({ cursor: 1, value: 'a' })
    expect(backspaceDelete('😀', 2)).toEqual({ cursor: 0, value: '' })
  })

  it('keeps grapheme-cluster delete for ZWJ emoji sequences (non-combining tail)', () => {
    // man + ZWJ + woman + ZWJ + girl — one grapheme; the tail code point is an
    // emoji, not a combining mark, so the whole sequence still deletes at once.
    const family = '👨\u200d👩\u200d👧'

    expect(backspaceDelete(family, family.length)).toEqual({ cursor: 0, value: '' })
  })

  it('keeps grapheme-cluster delete for mid-text Backspace (only end-of-input changes)', () => {
    // Cursor before the trailing 'x' is NOT at end-of-input, so the ก� cluster
    // is still removed whole — cursor movement/selection granularity is intact.
    const value = KO_KAI + SARA_I + 'x'

    expect(backspaceDelete(value, 2)).toEqual({ cursor: 0, value: 'x' })
  })

  it('mid-text Backspace between two Thai combining clusters deletes the whole preceding cluster', () => {
    // กิกิ, cursor at 2 = between the two syllables (after ก◌ิ, before ก). Not
    // end-of-input, so the grapheme-cluster model holds even though the
    // preceding code point IS a combining mark — the sharpest mid-text case the
    // AI review asked to pin: same Backspace semantics everywhere unless the
    // cursor is at the very end of the input.
    const two = KO_KAI + SARA_I + KO_KAI + SARA_I // กิกิ

    expect(backspaceDelete(two, 2)).toEqual({ cursor: 0, value: KO_KAI + SARA_I })
  })

  it('is a no-op at cursor 0', () => {
    expect(backspaceDelete(KO_KAI + SARA_I, 0)).toEqual({ cursor: 0, value: KO_KAI + SARA_I })
  })
})

describe('backspaceDelete ↔ fast-echo path consistency (#94512)', () => {
  it('excludes a Thai combining cluster from the raw \\b \\b fast path', () => {
    // The fast-echo bypass writes "\b \b" and only clears one cell, so it must
    // never run for a combining cluster. canFastBackspaceShape already rejects
    // non-ASCII removed graphemes; pin that so the per-codepoint normal path
    // (backspaceDelete) is what handles Thai.
    const thai = KO_KAI + SARA_I

    expect(canFastBackspaceShape(thai, thai.length)).toBe(false)
  })
})

describe('backspaceDelete → snapPos no-op invariant (#94512 AI review)', () => {
  // The keydown handler runs the result of backspaceDelete straight into
  // commit (textInput.tsx:1606 → commit at 1746), and commit snaps every
  // cursor with `const c = snapPos(next, nextCur)` (textInput.tsx:1156).
  // snapPos clamps to the largest grapheme stop ≤ p, so for the returned
  // cursor to survive unchanged — and for the per-codepoint delete to actually
  // land where backspaceDelete says — every branch must return a position on a
  // grapheme boundary of the NEW value. These tests pin that contract.
  it('end-of-input combining delete lands the cursor on a grapheme boundary', () => {
    const r = backspaceDelete(KO_KAI + SARA_I, 2) // กิ → ก

    expect(atGraphemeBoundary(r.value, r.cursor)).toBe(true)
    expect(r.cursor).toBe(r.value.length) // end of string is always a boundary
  })

  it('mid-text grapheme delete also lands on a grapheme boundary', () => {
    const two = KO_KAI + SARA_I + KO_KAI + SARA_I // กิกิ
    const r = backspaceDelete(two, 2)

    expect(atGraphemeBoundary(r.value, r.cursor)).toBe(true)
  })

  it('ASCII, emoji, ZWJ and no-op cases all land on grapheme boundaries', () => {
    const cases = [
      backspaceDelete('ab', 2), // → a
      backspaceDelete('a😀', 3), // → a
      backspaceDelete('😀', 2), // → ''
      backspaceDelete('👨\u200d👩\u200d👧', '👨\u200d👩\u200d👧'.length), // → ''
      backspaceDelete(KO_KAI + SARA_I, 0), // no-op
    ]

    for (const r of cases) {
      expect(atGraphemeBoundary(r.value, r.cursor)).toBe(true)
    }
  })
})
