import { describe, expect, it } from 'vitest'

import { Parser } from './parser.js'

const renderedText = (actions: ReturnType<Parser['feed']>): string =>
  actions
    .filter(action => action.type === 'text')
    .flatMap(action => action.graphemes)
    .map(grapheme => grapheme.value)
    .join('')

describe('output parser line endings after ESC', () => {
  it.each(['\r', '\n'])('preserves %j received in the same chunk as ESC', lineEnding => {
    const actions = new Parser().feed(`before\x1b${lineEnding}after`)

    expect(renderedText(actions).replaceAll('\x1b', '')).toBe(`before${lineEnding}after`)
  })

  it.each(['\r', '\n'])('preserves %j received in the chunk after ESC', lineEnding => {
    const parser = new Parser()
    const actions = [...parser.feed('before\x1b'), ...parser.feed(`${lineEnding}after`)]

    expect(renderedText(actions).replaceAll('\x1b', '')).toBe(`before${lineEnding}after`)
  })
})
