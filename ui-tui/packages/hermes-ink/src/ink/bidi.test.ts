import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { getParagraphDirection, reorderBidi, setNeedsBidiForTesting } from './bidi.js'

describe('BiDi text reordering and direction detection', () => {
  beforeEach(() => {
    setNeedsBidiForTesting(true)
  })

  afterEach(() => {
    setNeedsBidiForTesting(undefined)
  })

  describe('getParagraphDirection', () => {
    it('detects LTR for tool call lines', () => {
      expect(getParagraphDirection('• Terminal("echo ""!مرحبا يا صديقي""!) (0.2s)')).toBe('ltr')
      expect(getParagraphDirection('Terminal("tail -5 /opt/data/logs/gateway.log")')).toBe('ltr')
    })

    it('detects LTR for command prompts, paths, and code syntax', () => {
      expect(getParagraphDirection('$ echo مرحبا')).toBe('ltr')
      expect(getParagraphDirection('> npm test')).toBe('ltr')
      expect(getParagraphDirection('/opt/data/logs')).toBe('ltr')
    })

    it('detects RTL for conversational Arabic text', () => {
      expect(getParagraphDirection('أهلاً بيك يا صديقي!')).toBe('rtl')
      expect(getParagraphDirection('تحية وبدء محادثة جديدة')).toBe('rtl')
      expect(getParagraphDirection('• مرحبا بكم')).toBe('rtl')
      expect(getParagraphDirection('- خدمة ثوانية أنا معاك')).toBe('rtl')
    })
  })

  describe('reorderBidi', () => {
    function makeChars(str: string) {
      return Array.from(str).map(c => ({
        value: c,
        width: 1,
        styleId: 0,
        hyperlink: undefined
      }))
    }

    function charsToString(chars: { value: string }[]) {
      return chars.map(c => c.value).join('')
    }

    it('leaves pure LTR strings untouched', () => {
      const input = makeChars('Hello world 123!')
      const result = reorderBidi(input)
      expect(charsToString(result)).toBe('Hello world 123!')
    })

    it('shapes and visually reorders Arabic word "مرحبا"', () => {
      // Logical "مرحبا" -> shaped into ﻣ (\uFEE3) ﺮ (\uFEAE) ﺣ (\uFEA3) ﺒ (\uFE92) ﺎ (\uFE8E)
      // Visual reversal: ﺎ (\uFE8E) ﺒ (\uFE92) ﺣ (\uFEA3) ﺮ (\uFEAE) ﻣ (\uFEE3)
      const input = makeChars('مرحبا')
      const result = reorderBidi(input)
      expect(charsToString(result)).toBe('\uFE8E\uFE92\uFEA3\uFEAE\uFEE3')
    })

    it('preserves code wrappers in tool calls without flipping parentheses or command names', () => {
      const input = makeChars('Terminal("مرحبا")')
      const result = reorderBidi(input)
      const str = charsToString(result)

      // Prefix "Terminal(\"" must stay LTR at the start
      expect(str.startsWith('Terminal("')).toBe(true)
      // Suffix "\")" must stay LTR at the end
      expect(str.endsWith('")')).toBe(true)
      // Arabic inside must be shaped and visually reversed
      expect(str).toBe('Terminal("\uFE8E\uFE92\uFEA3\uFEAE\uFEE3")')
    })

    it('mirrors paired brackets within RTL runs so they enclose the text correctly on screen', () => {
      const input = makeChars('(مرحبا)')
      const result = reorderBidi(input)
      const str = charsToString(result)

      // On screen from left to right: opening bracket '(' on the left, closing bracket ')' on the right
      expect(str.startsWith('(')).toBe(true)
      expect(str.endsWith(')')).toBe(true)
    })
  })
})
