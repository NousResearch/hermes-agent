import { describe, expect, it } from 'vitest'

import { stripOsc8 } from './squash-text-nodes.js'

const ESC = '\u001b'
const BEL = '\u0007'

describe('stripOsc8', () => {
  it('never leaves an OSC 8 opener, for any mix of escape fragments, in linear time', () => {
    // Every char the opener and its terminators are built from, plus label text.
    const alphabet = [ESC, ']', '8', ';', BEL, '\\', 'a', ':']
    let seed = 1

    const rand = (n: number) => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff

      return seed % n
    }

    for (let run = 0; run < 20000; run++) {
      const input = Array.from({ length: 1 + rand(24) }, () => alphabet[rand(alphabet.length)]).join('')

      expect(stripOsc8(input)).not.toContain(`${ESC}]8;`)
    }

    const nested = ESC.repeat(5000) + `]8;;x${BEL}`.repeat(5000) + 'label'
    const started = performance.now()

    expect(stripOsc8(nested)).toBe('label')
    expect(performance.now() - started).toBeLessThan(50)
  })
})
