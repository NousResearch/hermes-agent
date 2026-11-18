import { describe, expect, it } from 'vitest'

import { createNode, createTextNode, type DOMElement, type ElementNames } from './dom.js'
import { squashTextNodesToSegments, stripOsc8 } from './squash-text-nodes.js'
import type { TextStyles } from './styles.js'

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

    // Unterminated openers whose payload scan restarts on every splice stay linear too.
    const unterminated = ESC.repeat(20000) + ']8;'.repeat(20000) + 'x'.repeat(20000)
    const begun = performance.now()

    expect(stripOsc8(unterminated)).not.toContain(OPEN)
    expect(performance.now() - begun).toBeLessThan(50)
  })

  it('drops only the opener when the payload has no terminator before a newline or the end', () => {
    expect(stripOsc8(`before ${OPEN};\nline2 ${OPEN};x${BEL}y`)).toBe('before ;\nline2 y')
    expect(stripOsc8(`a ${OPEN};https://x`)).toBe('a ;https://x')
  })
})

const text = (value: string) => createTextNode(value) as unknown as DOMElement

const element = (name: ElementNames, children: DOMElement[], textStyles?: TextStyles, href?: string) => {
  const node = createNode(name)

  node.childNodes = children
  node.textStyles = textStyles

  if (href) {
    node.attributes['href'] = href
  }

  return node
}

describe('squashTextNodesToSegments', () => {
  const segmentsOf = (node: DOMElement) =>
    squashTextNodesToSegments(node).map(({ hyperlink, styles, text }) => ({ hyperlink, styles, text }))

  it('maps the stripped run back to each segment with its own styles', () => {
    // Joined: "ab" ESC | "]8;;u" BEL "C😀D" | "E" | "f" — the opener spans the first boundary.
    const run = element('ink-text', [
      text(`ab${ESC}`),
      element('ink-virtual-text', [text(`]8;;u${BEL}C😀D`)], { bold: true }),
      element('ink-virtual-text', [text('E')], { italic: true }),
      text('f')
    ])

    expect(segmentsOf(run)).toEqual([
      { hyperlink: undefined, styles: {}, text: 'ab' },
      { hyperlink: undefined, styles: { bold: true }, text: 'C😀D' },
      { hyperlink: undefined, styles: { italic: true }, text: 'E' },
      { hyperlink: undefined, styles: {}, text: 'f' }
    ])
  })

  it('never lets a payload skip run past the text node it started in', () => {
    const run = element('ink-text', [
      text(`before ${OPEN};`),
      element('ink-virtual-text', [text(`${BEL} WARNING`)], { color: 'ansi:red' }),
      element('ink-link', [text('TRUSTED')], undefined, 'https://docs.example.com/real')
    ])

    expect(segmentsOf(run)).toEqual([
      { hyperlink: undefined, styles: {}, text: 'before ;' },
      { hyperlink: undefined, styles: { color: 'ansi:red' }, text: `${BEL} WARNING` },
      { hyperlink: 'https://docs.example.com/real', styles: {}, text: 'TRUSTED' }
    ])
  })
})
