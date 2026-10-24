import { describe, expect, it } from 'vitest'
import wrapAnsiNpm from 'wrap-ansi'

import { stringWidth } from './stringWidth.js'
import { unbreakableUnits, wrapAnsi } from './wrapAnsi.js'

const widths = (text: string) => text.split('\n').map(line => stringWidth(line))

describe('wrapAnsi CJK wrapping', () => {
  // wrap-ansi splits on ' ' only, so a space-less CJK clause is one
  // unbreakable word: the row ended at the last space (~col 60 of 120) and
  // the rest moved down whole, leaving a half-empty transcript line.
  const paragraph =
    '这条来自我最早那个 PID 写错的监听器（P=295447），但它读的是同一份 topdirs19.txt，数据与真监听器（23:37:28）完全一致——验证结论不变。'

  it('fills a line instead of breaking only at the previous space', () => {
    const lines = wrapAnsi(paragraph, 120, { trim: false, hard: true }).split('\n')

    expect(lines.length).toBeGreaterThan(1)
    expect(widths(lines[0]!)[0]).toBeGreaterThanOrEqual(118)
    expect(Math.max(...widths(lines.join('\n')))).toBeLessThanOrEqual(120)
  })

  it('keeps every row inside the column budget', () => {
    for (const columns of [20, 33, 60, 80, 120]) {
      const out = wrapAnsi(paragraph, columns, { trim: false, hard: true })

      expect(Math.max(...widths(out))).toBeLessThanOrEqual(columns)
    }
  })

  it('does not start a row with closing punctuation', () => {
    const lines = wrapAnsi('这是一句用于测试标点不能出现在行首的中文，后面继续写一些字。', 20, {
      trim: false,
      hard: true
    }).split('\n')

    for (const line of lines.slice(1)) {
      expect(line.startsWith('，')).toBe(false)
      expect(line.startsWith('。')).toBe(false)
    }
  })

  it('hard-breaks a CJK run longer than the whole row', () => {
    const out = wrapAnsi('测试'.repeat(30), 9, { trim: false, hard: true })

    expect(Math.max(...widths(out))).toBeLessThanOrEqual(9)
  })

  it('reports ideographs as break opportunities and glues closing punctuation', () => {
    expect(unbreakableUnits('中文，abc def')).toEqual(['中', '文，', 'abc', 'def'])
    expect(unbreakableUnits('对半配置包的检查')).toEqual(['对', '半', '配', '置', '包', '的', '检', '查'])
  })

  it('leaves Latin-only text on the stock wrap-ansi path', () => {
    const text = 'the quick brown fox jumps over the lazy dog and keeps running past the fence'
    const options = { trim: false, hard: true }

    expect(wrapAnsi(text, 24, options)).toBe(wrapAnsiNpm(text, 24, options))
  })

  it('leaves ANSI text on the stock path (escapes carry no layout width)', () => {
    const text = '\u001B[31m这是一行带颜色的中文\u001B[39m'

    expect(wrapAnsi(text, 40, { trim: false, hard: true })).toBe(wrapAnsiNpm(text, 40, { trim: false, hard: true }))
  })
})
