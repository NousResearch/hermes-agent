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
