import { PassThrough } from 'stream'

import { Box, renderSync, stringWidth } from '@hermes/ink'
import chalk from 'chalk'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { AUDIO_DIRECTIVE_RE, INLINE_RE, Md, MEDIA_LINE_RE, stripInlineMarkup } from '../components/markdown.js'
import { __resetLinkTitleCache, fetchLinkTitle } from '../lib/externalLink.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME, LIGHT_THEME } from '../theme.js'

afterEach(() => {
  __resetLinkTitleCache()
  vi.unstubAllGlobals()
})

// Stub the network and warm the shared title cache, so a subsequent render
// has the resolved title available synchronously.
const stubFetchedTitle = (url: string, title: string) => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(
      new Response(`<html><head><title>${title}</title></head></html>`, {
        headers: { 'content-type': 'text/html' },
        status: 200
      })
    )
  )

  return fetchLinkTitle(url)
}

const matches = (text: string) => [...text.matchAll(INLINE_RE)].map(m => m[0])
const BEL = String.fromCharCode(7)
const ESC = String.fromCharCode(27)
const CSI_RE = new RegExp(`${ESC}\\[[0-?]*[ -/]*[@-~]`, 'g')
const OSC_RE = new RegExp(`${ESC}\\][\\s\\S]*?(?:${BEL}|${ESC}\\\\)`, 'g')

const renderPlain = (node: React.ReactNode, columns = 80) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns, isTTY: false, rows: 24 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(node, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  const frame = output

  instance.unmount()
  instance.cleanup()

  return frame
    .replace(OSC_RE, '')
    .split('\n')
    .map(line => stripAnsi(line).replace(CSI_RE, '').trimEnd())
}

describe('INLINE_RE emphasis', () => {
  it('matches word-boundary italic/bold', () => {
    expect(matches('say _hi_ there')).toEqual(['_hi_'])
    expect(matches('very __bold move__ today')).toEqual(['__bold move__'])
    expect(matches('(_paren_) and [_bracket_]')).toEqual(['_paren_', '_bracket_'])
  })

  it('keeps intraword underscores literal', () => {
    const path = '/home/me/.hermes/cache/screenshots/browser_screenshot_ecc1c3feab.png'

    expect(matches(path)).toEqual([])
    expect(matches('snake_case_var and MY_CONST')).toEqual([])
    expect(matches('foo__bar__baz')).toEqual([])
  })

  it('keeps Python dunder identifiers literal', () => {
    expect(matches('if __name__ == "__main__":')).toEqual([])
    expect(matches('def __init__(self):')).toEqual([])
    expect(matches('print(__file__)')).toEqual([])
  })

  it('still matches asterisk emphasis intraword', () => {
    expect(matches('a*b*c')).toEqual(['*b*'])
    expect(matches('a**bold**c')).toEqual(['**bold**'])
  })

  it('matches short alphanumeric subscript (H~2~O, CO~2~, X~n~)', () => {
    expect(matches('H~2~O')).toEqual(['~2~'])
    expect(matches('CO~2~ levels')).toEqual(['~2~'])
    expect(matches('the X~n~ term')).toEqual(['~n~'])
  })

  it('ignores kaomoji-style ~! and ~? punctuation', () => {
    // Kimi / Qwen / GLM emit these as decorators and the whole span between
    // two tildes used to get collapsed into one dim blob.
    expect(matches('Aww ~! Building step by step, I love it ~!')).toEqual([])
    expect(matches('cool ~? yeah ~?')).toEqual([])
    expect(matches('mixed ~! and ~? flow')).toEqual([])
  })

  it('ignores tilde spans that contain spaces or punctuation', () => {
    // Real subscript doesn't contain spaces; a tilde followed by words-then-
    // tilde is almost always conversational. Matching it swallows text.
    expect(matches('hello ~good idea~ there')).toEqual([])
    expect(matches('x ~oh no!~ y')).toEqual([])
  })

  it('does not let strikethrough eat subscript', () => {
    expect(matches('~~strike~~ and H~2~O')).toEqual(['~~strike~~', '~2~'])
  })
})

describe('stripInlineMarkup', () => {
  it('strips word-boundary emphasis only', () => {
    expect(stripInlineMarkup('say _hi_ there')).toBe('say hi there')
    expect(stripInlineMarkup('browser_screenshot_ecc.png')).toBe('browser_screenshot_ecc.png')
    expect(stripInlineMarkup('__bold move__ and foo__bar__')).toBe('bold move and foo__bar__')
  })

  it('preserves Python dunder identifiers', () => {
    expect(stripInlineMarkup('if __name__ == "__main__":')).toBe('if __name__ == "__main__":')
    expect(stripInlineMarkup('class X: def __init__(self): pass')).toBe('class X: def __init__(self): pass')
  })

  it('leaves ~!/~? kaomoji alone and still handles real subscript', () => {
    expect(stripInlineMarkup('Yay ~! nice work ~!')).toBe('Yay ~! nice work ~!')
    expect(stripInlineMarkup('H~2~O and CO~2~')).toBe('H_2O and CO_2')
  })

  it('strips inline math delimiters but keeps the formula text', () => {
    expect(stripInlineMarkup('$\\mathbb{Z}$ is a ring')).toBe('\\mathbb{Z} is a ring')
    expect(stripInlineMarkup('see \\(a + b\\) ok')).toBe('see a + b ok')
  })
})

describe('INLINE_RE inline math', () => {
  it('matches single-dollar math and beats emphasis at the same start', () => {
    // Without math handling, `*b*` would have matched as italics and
    // corrupted the formula. With math added to INLINE_RE, the leftmost
    // match at column 0 (`$P=a*b*c$`) wins.
    expect(matches('$P=a*b*c$')).toEqual(['$P=a*b*c$'])
    expect(matches('see $\\mathbb{Z}$ here')).toEqual(['$\\mathbb{Z}$'])
  })

  it('does not match currency-style prose', () => {
    expect(matches('it costs $5 and $10')).toEqual([])
    expect(matches('paid $5')).toEqual([])
  })

  it('does not let inline math swallow a $$ display fence', () => {
    // `$$x$$` is a display block, not two abutting inline-math spans.
    expect(matches('$$x$$')).toEqual([])
  })

  it('matches \\(...\\) inline math', () => {
    expect(matches('foo \\(x + y\\) bar')).toEqual(['\\(x + y\\)'])
  })

  it('does not corrupt subscripts/superscripts inside math', () => {
    // `_n` and `^r` are markdown emphasis/superscript markers in prose, but
    // inside a `$...$` span the entire formula is captured as a single
    // inline-math token so the inner regexes never see those characters.
    expect(matches('$P=a_n x^n + a_0$')).toEqual(['$P=a_n x^n + a_0$'])
    expect(matches('$\\beta_1,\\dots,\\beta_r$')).toEqual(['$\\beta_1,\\dots,\\beta_r$'])
  })

  it('places math content in the correct capture group (regression: m[16] is bare URL)', () => {
    // When `m[16]` was the bare URL group AND the inline-math `$...$`
    // group simultaneously (because the bare URL pattern lacked its own
    // capturing parens), MdInline rendered `$\\mathbb{R}$` as an
    // underlined autolink instead of italic amber math. Lock down the
    // numbering: math goes in m[17] / m[18], URLs go in m[16].
    const url = [...'see https://example.com here'.matchAll(INLINE_RE)][0]!
    const dollarMath = [...'$\\mathbb{R}$'.matchAll(INLINE_RE)][0]!
    const parenMath = [...'\\(\\pi\\)'.matchAll(INLINE_RE)][0]!

    expect(url[16]).toBe('https://example.com')
    expect(url[17]).toBeUndefined()
    expect(url[18]).toBeUndefined()

    expect(dollarMath[16]).toBeUndefined()
    expect(dollarMath[17]).toBe('\\mathbb{R}')
    expect(dollarMath[18]).toBeUndefined()

    expect(parenMath[16]).toBeUndefined()
    expect(parenMath[17]).toBeUndefined()
    expect(parenMath[18]).toBe('\\pi')
  })
})

describe('protocol sentinels', () => {
  it('captures MEDIA: paths with surrounding quotes or backticks', () => {
    expect('MEDIA:/tmp/a.png'.match(MEDIA_LINE_RE)?.[1]).toBe('/tmp/a.png')
    expect('  MEDIA: /home/me/.hermes/cache/screenshots/browser_screenshot_ecc.png  '.match(MEDIA_LINE_RE)?.[1]).toBe(
      '/home/me/.hermes/cache/screenshots/browser_screenshot_ecc.png'
    )
    expect('`MEDIA:/tmp/a.png`'.match(MEDIA_LINE_RE)?.[1]).toBe('/tmp/a.png')
    expect('"MEDIA:C:\\files\\a.png"'.match(MEDIA_LINE_RE)?.[1]).toBe('C:\\files\\a.png')
  })

  it('ignores MEDIA: tokens embedded in prose', () => {
    expect('here is MEDIA:/tmp/a.png for you'.match(MEDIA_LINE_RE)).toBeNull()
    expect('the media: section is empty'.match(MEDIA_LINE_RE)).toBeNull()
  })

  it('matches the [[audio_as_voice]] directive', () => {
    expect(AUDIO_DIRECTIVE_RE.test('[[audio_as_voice]]')).toBe(true)
    expect(AUDIO_DIRECTIVE_RE.test('  [[audio_as_voice]]  ')).toBe(true)
    expect(AUDIO_DIRECTIVE_RE.test('audio_as_voice')).toBe(false)
  })
})

describe('fenced code panels', () => {
  const renderCode = (text: string, width = 40, compact = false) =>
    renderPlain(
      React.createElement(
        Box,
        { width },
        React.createElement(Md, { cols: width, compact, t: DEFAULT_THEME, text })
      ),
      width
    )

  it('separates highlighted fenced code with a theme-aware language header', () => {
    const lines = renderCode('```python\ndef hello():\n    print("hello")\n```')

    expect(lines[0]).toMatch(/^╭─ python ─+╮$/)
    expect(lines.some(line => /^│ def hello\(\):\s+│$/.test(line))).toBe(true)
    expect(lines.some(line => /^│ {5}print\("hello"\)\s+│$/.test(line))).toBe(true)
    expect(lines.some(line => /^╰─+╯$/.test(line))).toBe(true)
  })

  it('renders language-less, diff, and empty fences as panels', () => {
    expect(renderCode('```\nplain\n```')[0]).toMatch(/^╭─+╮$/)

    const diff = renderCode('```diff\n@@ -1 +1 @@\n-old\n+new\n```')
    expect(diff[0]).toMatch(/^╭─ diff ─+╮$/)
    expect(diff.some(line => /^│ @@ -1 \+1 @@\s+│$/.test(line))).toBe(true)
    expect(diff.some(line => /^│ -old\s+│$/.test(line))).toBe(true)
    expect(diff.some(line => /^│ \+new\s+│$/.test(line))).toBe(true)

    const empty = renderCode('```\n```')
    expect(empty[0]).toMatch(/^╭─+╮$/)
    expect(empty.some(line => /^│\s+│$/.test(line))).toBe(true)
    expect(empty.some(line => /^╰─+╯$/.test(line))).toBe(true)
  })

  it('wraps long and wide-Unicode code within the inner panel width', () => {
    const lines = renderCode('```text\n한국어🙂 abcdefghijklmnopqrstuvwxyz\n```', 24)

    expect(lines[0]).toBe('╭─ text ───────────────╮')
    expect(lines).toContain('│ 한국어🙂 abcdefghijk │')
    expect(lines).toContain('│ lmnopqrstuvwxyz      │')
    expect(lines.some(line => line === '╰──────────────────────╯')).toBe(true)
    expect(lines.every(line => stringWidth(line) <= 24)).toBe(true)
  })

  it('falls back to a compact left accent in narrow terminals and compact mode', () => {
    const narrow = renderCode('```python\nprint("hello world")\n```', 12)
    expect(narrow).toEqual(expect.arrayContaining(['│ python', '│ print("hel', '│ lo world")']))
    expect(narrow.some(line => line.includes('╭'))).toBe(false)
    expect(narrow.every(line => stringWidth(line) <= 12)).toBe(true)

    const compact = renderCode('```python\nprint(1)\n```', 40, true)
    expect(compact).toEqual(expect.arrayContaining(['│ python', '│ print(1)']))
    expect(compact.some(line => line.includes('╭'))).toBe(false)
  })

  it('keeps unsupported language labels, tabs, and consecutive blocks isolated', () => {
    const lines = renderCode('```brainfuck\n\t++>\n```\n\n```\nsecond\n```')

    expect(lines.filter(line => line.startsWith('╭')).length).toBe(2)
    expect(lines[0]).toMatch(/^╭─ brainfuck ─+╮$/)
    expect(lines.join('\n')).toContain('++>')
    expect(lines.join('\n')).toContain('second')
  })

  // Width-safe label regressions. The previous implementation passed the raw
  // `lang` string to Ink's `borderText`; Ink's border-embedding path
  // (`packages/hermes-ink/src/ink/render-border.ts::embedTextInBorder`)
  // truncates with JS `substring` when `stringWidth(text) >= borderLength - 2`,
  // which corrupts a CJK / emoji label mid-glyph. Each of these cases used
  // to either overflow the top border or produce broken cells. The fix
  // pre-truncates by display width with `…` appended inside the budget.
  it('truncates a long Korean language label to the safe border budget at the 20-col threshold', () => {
    const lines = renderCode('```한국어 라벨 매우 깁니다 길어\nx\n```', 20)

    // Every visible line must fit the panel width.
    expect(lines.every(line => stringWidth(line) <= 20)).toBe(true)

    // The header line is the first row. The exact format depends on
    // whether Ink leaves a trailing dash before the `╮` corner; the
    // invariants we care about are: corners present, ellipsis present,
    // and the label fits inside the safe border budget.
    const header = lines[0]!
    expect(header.startsWith('╭─')).toBe(true)
    expect(header.endsWith('╮')).toBe(true)
    expect(header).toContain('…')

    // The untruncated label is ~28 cells wide; 20-col panel can hold at
    // most 15 cells in the label slot, so the visible label must fit.
    // Strip the corners (`╭`/`╮`) and surrounding dashes/spaces, then
    // measure what remains.
    const label = header.slice(2, -1).replace(/^─\s+|\s+─$/g, '').trim()
    expect(stringWidth(label)).toBeLessThanOrEqual(20 - 5)
  })

  it('truncates an emoji-containing language label without producing broken surrogate pairs', () => {
    const width = 22
    // The synthetic label mixes emoji, CJK, and ASCII and would overflow
    // the 22-col budget. Pre-fix Ink's JS-substring cut could leave a
    // `�` (U+FFFD) replacement char or a lone surrogate in the panel.
    const text = '```python_emoji_😀_korean_한국어_superlong\nx\n```'
    const rendered = renderCode(text, width)

    expect(rendered.every(line => stringWidth(line) <= width)).toBe(true)
    expect(rendered[0]).not.toMatch(/\uFFFD/)
  })

  it('truncates a mixed Korean/emoji/ASCII label that exceeds the panel width', () => {
    const text = '```한국어_이모지_😀_ASCII_label_that_is_unreasonably_long\nx\n```'
    const width = 24
    const lines = renderCode(text, width)

    expect(lines.every(line => stringWidth(line) <= width)).toBe(true)
    expect(lines[0]).not.toMatch(/\uFFFD/)
  })

  it('truncates a label wider than the entire panel and still produces a closed top border', () => {
    const text = '```' + 'A'.repeat(120) + '\nx\n```'
    const width = 20
    const lines = renderCode(text, width)

    expect(lines[0]).toMatch(/^╭─.+╮$/)
    expect(lines.every(line => stringWidth(line) <= width)).toBe(true)
    // The label inside the top border must end with the ellipsis (truncated)
    // and never include a U+FFFD replacement char.
    expect(lines[0]).toContain('…')
    expect(lines[0]).not.toMatch(/\uFFFD/)
  })

  it('keeps the full label in normal mode when it fits the budget', () => {
    const lines = renderCode('```python\nx = 1\n```', 40)

    // Short label, well within budget — should appear verbatim.
    expect(lines[0]).toMatch(/python/)
    expect(lines[0]).not.toMatch(/…/)
  })

  it('stays width-safe at the boundary widths 19 (narrow), 20 (normal threshold), 21 (normal)', () => {
    // 19 cols: narrow left-accent view, single-line truncated label.
    const narrow19 = renderCode('```python\nx\n```', 19)
    expect(narrow19.every(line => stringWidth(line) <= 19)).toBe(true)
    expect(narrow19.some(line => line.startsWith('╭'))).toBe(false)

    // 20 cols: normal panel, full border, label truncated to safe width.
    const normal20 = renderCode('```python_long_label\nx\n```', 20)
    expect(normal20.every(line => stringWidth(line) <= 20)).toBe(true)
    expect(normal20.some(line => line.startsWith('╭'))).toBe(true)

    // 21 cols: normal panel, slightly more room.
    const normal21 = renderCode('```python_long_label\nx\n```', 21)
    expect(normal21.every(line => stringWidth(line) <= 21)).toBe(true)
    expect(normal21.some(line => line.startsWith('╭'))).toBe(true)
  })
})

describe('Md wrapping', () => {
  it('trims spaces from word-wrap continuation lines', () => {
    const lines = renderPlain(
      React.createElement(Box, { width: 5 }, React.createElement(Md, { t: DEFAULT_THEME, text: 'Let me' }))
    )

    expect(lines).toContain('Let')
    expect(lines).toContain('me')
    expect(lines).not.toContain(' me')
  })

  it('keeps nested list and quote indentation out of trim-sensitive text', () => {
    const lines = renderPlain(
      React.createElement(
        Box,
        { flexDirection: 'column', width: 24 },
        React.createElement(Md, { t: DEFAULT_THEME, text: '  - nested bullet' }),
        React.createElement(Md, { t: DEFAULT_THEME, text: '>> nested quote' })
      )
    )

    expect(lines).toContain('  • nested bullet')
    expect(lines).toContain('  │ nested quote')
  })

  it('preserves original inline-code edge spaces', () => {
    const lines = renderPlain(
      React.createElement(Box, { width: 24 }, React.createElement(Md, { t: DEFAULT_THEME, text: '` hi ` ok' }))
    )

    expect(lines.some(line => line.startsWith(' hi  ok'))).toBe(true)
  })

  it('renders Python dunder identifiers literally outside code fences', () => {
    const lines = renderPlain(
      React.createElement(
        Box,
        { width: 80 },
        React.createElement(Md, {
          t: DEFAULT_THEME,
          text: 'if __name__ == "__main__":\n    obj.__init__()'
        })
      )
    )

    const rendered = lines.join('\n')

    expect(rendered).toContain('if __name__ == "__main__":')
    expect(rendered).toContain('obj.__init__()')
  })
})

describe('Md link labels', () => {
  it('renders bare URLs with readable slug labels', () => {
    const lines = renderPlain(
      React.createElement(
        Box,
        { width: 120 },
        React.createElement(Md, {
          t: DEFAULT_THEME,
          text: 'see https://www.expedia.com/things-to-do/puerto-rico-el-yunque-rainforest-adventure for details'
        })
      )
    )

    const rendered = lines.join('\n')

    expect(rendered).toContain('Puerto Rico El Yunque Rainforest Adventure')
    expect(rendered).not.toContain('https://www.expedia.com/things-to-do/puerto-rico-el-yunque-rainforest-adventure')
  })

  it('keeps the authored markdown label even when a page title resolves', async () => {
    const url = 'https://www.expedia.com/things-to-do/puerto-rico-el-yunque-rainforest-adventure'

    // Warm the shared cache so `useLinkTitle` would have a title to render
    // synchronously — the label must still win.
    await stubFetchedTitle(url, 'El Yunque Rainforest Adventure | Expedia')

    const lines = renderPlain(
      React.createElement(
        Box,
        { width: 80 },
        React.createElement(Md, { t: DEFAULT_THEME, text: `[Trip details](${url})` })
      )
    )

    const rendered = lines.join('\n')

    expect(rendered).toContain('Trip details')
    expect(rendered).not.toContain('El Yunque Rainforest Adventure | Expedia')
  })

  it('still resolves titles for links whose label is just the URL', async () => {
    const url = 'https://www.expedia.com/things-to-do/puerto-rico-el-yunque-rainforest-adventure'

    await stubFetchedTitle(url, 'Rainforest Adventure Tour')

    const lines = renderPlain(
      React.createElement(Box, { width: 120 }, React.createElement(Md, { t: DEFAULT_THEME, text: `[${url}](${url})` }))
    )

    expect(lines.join('\n')).toContain('Rainforest Adventure Tour')
  })
})

describe('renderTable CJK width alignment', () => {
  it('column starts share the same display offset across CJK rows', async () => {
    const { stringWidth } = await import('@hermes/ink')

    const md = [
      '| 配置 | Config | 状态 |',
      '|------|--------|------|',
      '| Vicuna (report) | dense | × |',
      '| ChatGLM | chat | ✓ |',
      '| 通义千问 | qwen | × |'
    ].join('\n')

    // Pre-fix bug: ` `.repeat(w - stripInlineMarkup(...).length) used
    // UTF-16 code units, so a CJK header cell padded to 2 cells while
    // the body cell padded to 4, drifting subsequent columns by 2
    // cells per CJK char.
    //
    // Post-fix contract: the prefix preceding the start of column N
    // has the same display width across the header and every body row
    // (deduped to skip the divider, which renders independently).
    const lines = renderPlain(
      React.createElement(Box, null, React.createElement(Md, { compact: true, t: DEFAULT_THEME, text: md }))
    ).filter(line => line.trim().length > 0)

    // Heuristic: a "data row" line either contains 'Config' (header)
    // or one of the body labels; a divider is all box-drawing.  Use
    // the substring 'Config' / 'dense' / 'chat' / 'qwen' as the
    // unique anchor for column 2's start position on each row.
    const colStarts = (line: string, anchor: string): number => {
      const idx = line.indexOf(anchor)

      return idx < 0 ? -1 : stringWidth(line.slice(0, idx))
    }

    const headerCol2 = lines.map(l => colStarts(l, 'Config')).find(v => v >= 0)
    const denseCol2 = lines.map(l => colStarts(l, 'dense')).find(v => v >= 0)
    const chatCol2 = lines.map(l => colStarts(l, 'chat')).find(v => v >= 0)
    const qwenCol2 = lines.map(l => colStarts(l, 'qwen')).find(v => v >= 0)

    expect(headerCol2).toBeDefined()
    expect(denseCol2).toBe(headerCol2)
    expect(chatCol2).toBe(headerCol2)
    // The CJK row is the one that drifted before the fix.  It must
    // align with the rest now.
    expect(qwenCol2).toBe(headerCol2)
  })
})

describe('body prose stays in the theme palette', () => {
  // Prose used to render in the terminal's DEFAULT foreground while inline
  // tokens beside it carried a theme color, so one line mixed two inks.
  // Because an inline token can match mid-word, so could a single word.
  // LIGHT_THEME is the vehicle here because every tone in it is hex, so
  // emitted SGR maps back to palette entries without format juggling.
  const foregroundRuns = (text: string): string[] => {
    // chalk is a singleton and defaults to level 0 under vitest (no TTY),
    // which would emit no SGR at all and make every assertion here vacuous.
    const savedLevel = chalk.level
    chalk.level = 3

    const stdout = new PassThrough()
    const stdin = new PassThrough()
    const stderr = new PassThrough()
    let output = ''

    Object.assign(stdout, { columns: 80, isTTY: true, rows: 24 })
    Object.assign(stdin, { isTTY: false })
    Object.assign(stderr, { isTTY: false })
    stdout.on('data', chunk => {
      output += chunk.toString()
    })

    const instance = renderSync(
      React.createElement(Box, { width: 70 }, React.createElement(Md, { cols: 68, t: LIGHT_THEME, text })),
      {
        patchConsole: false,
        stderr: stderr as NodeJS.WriteStream,
        stdin: stdin as NodeJS.ReadStream,
        stdout: stdout as NodeJS.WriteStream
      }
    )

    instance.unmount()
    instance.cleanup()
    chalk.level = savedLevel

    return [...output.matchAll(new RegExp(`${ESC}\\[38;2;(\\d+);(\\d+);(\\d+)m`, 'g'))].map(
      m =>
        '#' +
        m
          .slice(1, 4)
          .map(v => Number(v).toString(16).padStart(2, '0'))
          .join('')
    )
  }

  const PALETTE = new Set(
    Object.values(LIGHT_THEME.color)
      .filter((v): v is string => typeof v === 'string' && v.startsWith('#'))
      .map(v => v.toLowerCase())
  )

  const INK = LIGHT_THEME.color.text.toLowerCase()

  it('opens a paragraph with the theme ink, not the terminal default', () => {
    expect(foregroundRuns('plain prose line')[0]).toBe(INK)
  })

  it('keeps every foreground on a mixed-token line inside the palette', () => {
    // `render_terminal_output` trips the underscore-italic token mid-word —
    // the exact shape that split one word across two inks.
    const fg = foregroundRuns('set the `flag` and re-render_terminal_output for the run')

    expect(fg.length).toBeGreaterThan(0)

    for (const c of fg) {
      expect(PALETTE.has(c)).toBe(true)
    }
  })

  it('returns to the theme ink after an inline token, not to the terminal default', () => {
    const fg = foregroundRuns('before `code` after')

    expect(fg[0]).toBe(INK)
    expect(fg.at(-1)).toBe(INK)
  })

  it('themes list-item prose too', () => {
    for (const text of ['- a bullet item', '1. a numbered item']) {
      expect(foregroundRuns(text)).toContain(INK)
    }
  })
})
