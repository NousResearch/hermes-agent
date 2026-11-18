import { describe, expect, it } from 'vitest'

import { stripOsc8 } from './squash-text-nodes.js'

const ESC = '\u001b'
const BEL = '\u0007'
const OPEN = `${ESC}]8;`

// mulberry32: 32-bit integer math via Math.imul, so every output bit is mixed.
const seeded = (seed: number) => () => {
  seed = (seed + 0x6d2b79f5) | 0
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t

  return ((t ^ (t >>> 14)) >>> 0) / 4294967296
}

describe('stripOsc8', () => {
  it('never leaves an OSC 8 opener, for any mix of escape fragments, in linear time', () => {
    // Whole tokens as well as single chars, so openers, terminators and their fragments all
    // show up often enough to exercise the strip (single chars alone almost never spell one).
    const alphabet = [OPEN, `${ESC}]8`, ']8;', ESC, ']', '8', ';', BEL, `${ESC}\\`, 'a', 'https://x']
    const rand = seeded(1)
    let withOpener = 0
    let spliced = 0

    for (let run = 0; run < 20000; run++) {
      const input = Array.from(
        { length: 1 + Math.floor(rand() * 12) },
        () => alphabet[Math.floor(rand() * alphabet.length)]
      ).join('')

      if (input.includes(OPEN)) {
        withOpener++

        // A single regex pass leaves an opener here: one only a removal can create.
        // eslint-disable-next-line no-control-regex -- the escape sequence is what is matched
        spliced += Number(input.replace(/\u001b\]8;[^\u0007\u001b]*(?:\u0007|\u001b\\)?/g, '').includes(OPEN))
      }

      expect(stripOsc8(input)).not.toContain(OPEN)
    }

    // The corpus must reach the stripping path, including openers spliced by a removal.
    expect(withOpener).toBeGreaterThan(100)
    expect(spliced).toBeGreaterThan(0)

    const nested = ESC.repeat(5000) + `]8;;x${BEL}`.repeat(5000) + 'label'
    const started = performance.now()

    expect(stripOsc8(nested)).toBe('label')
    expect(performance.now() - started).toBeLessThan(50)
  })
})
