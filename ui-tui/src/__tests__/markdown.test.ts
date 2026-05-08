import { describe, expect, it } from 'vitest'

import { AUDIO_DIRECTIVE_RE, INLINE_RE, MEDIA_LINE_RE, stripInlineMarkup } from '../components/markdown.js'

const matches = (text: string) => [...text.matchAll(INLINE_RE)].map(m => m[0])

describe('INLINE_RE emphasis', () => {
  it('matches word-boundary italic/bold', () => {
    expect(matches('say _hi_ there')).toEqual(['_hi_'])
    expect(matches('very __bold__ move')).toEqual(['__bold__'])
    expect(matches('(_paren_) and [_bracket_]')).toEqual(['_paren_', '_bracket_'])
  })

  it('keeps intraword underscores literal', () => {
    const path = '/home/me/.hermes/cache/screenshots/browser_screenshot_ecc1c3feab.png'

    expect(matches(path)).toEqual([])
    expect(matches('snake_case_var and MY_CONST')).toEqual([])
    expect(matches('foo__bar__baz')).toEqual([])
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
    expect(stripInlineMarkup('__bold__ and foo__bar__')).toBe('bold and foo__bar__')
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
