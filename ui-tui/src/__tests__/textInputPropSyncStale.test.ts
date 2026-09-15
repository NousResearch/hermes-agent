import { describe, expect, it } from 'vitest'

import { shouldAdoptPropValue } from '../components/textInput.js'

// The composer pushes edits to its parent on a FRAME_BATCH_MS (16 ms)
// deferral and keeps accepting edits while that flush is in flight, so a
// flushed `value` prop can arrive carrying a string the buffer has already
// moved past. Adopting it rewinds the buffer one edit, and the next erase in
// the same IME recompose burst then re-applies an already-applied deletion.
//
// Regression intent: Vietnamese Telex sends a recompose as a burst of
// backspaces followed by the finished syllable, all within a few
// milliseconds (EVKey/Unikey/OpenKey). A rewound buffer turns
// "chinh" + backspace x3 + "ính" into "chiính" instead of "chính".
//
// The guard must stay narrow: only values this component itself pushed are
// stale by construction. A slash-command rewrite, history recall, or any
// programmatic setValue must still reset the composer.

describe('shouldAdoptPropValue', () => {
  it('ignores a prop the component pushed and has since moved past', () => {
    // A deferred commit pushed "...chin" while the buffer was still there;
    // the buffer has since advanced to "...chi" when the prop lands.
    expect(shouldAdoptPropValue('okey giờ chin', 'okey giờ chi', 'okey giờ chin', false)).toBe(false)
  })

  it('ignores a prop equal to the live buffer', () => {
    expect(shouldAdoptPropValue('abc', 'abc', 'abc', false)).toBe(false)
  })

  it('ignores the echo of a synchronous commit', () => {
    expect(shouldAdoptPropValue('abc', 'abcd', 'abcd', true)).toBe(false)
  })

  it('adopts a genuinely external value', () => {
    // Slash-command rewrite and history recall are not strings we pushed.
    expect(shouldAdoptPropValue('/help', 'draft text', 'draft text', false)).toBe(true)
    expect(shouldAdoptPropValue('recalled', 'draft text', null, false)).toBe(true)
  })

  it('adopts an external value even when an older push had the same length', () => {
    expect(shouldAdoptPropValue('/model', 'okey giờ chi', 'okey giờ chin', false)).toBe(true)
  })
})

describe('deferred-flush race (the "chiính" regression)', () => {
  it('rejects the stale flush that the pre-fix guards accepted', () => {
    const stale = 'okey giờ chin'
    const local = 'okey giờ chi'

    // Pre-fix the only guards were the one-shot echo flag and equality, so a
    // stale prop was adopted and the buffer rewound one edit.
    const legacyAdopts = !(false || stale === local)
    expect(legacyAdopts).toBe(true)

    // With the guard, the string we pushed is never a reason to rewind.
    expect(shouldAdoptPropValue(stale, local, stale, false)).toBe(false)
  })

  it('does not let the echo flag alone carry the contract', () => {
    // self.current is consumed by the first render, so the second delivery of
    // the same stale value arrives with ownEcho already false.
    expect(shouldAdoptPropValue('okey giờ chin', 'okey giờ chi', 'okey giờ chin', true)).toBe(false)
    expect(shouldAdoptPropValue('okey giờ chin', 'okey giờ chi', 'okey giờ chin', false)).toBe(false)
  })
})
