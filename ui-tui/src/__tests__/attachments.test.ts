import { describe, expect, it } from 'vitest'

import type { ComposerToken } from '../app/interfaces.js'
import { droppedTokens, expandTokens, imageToken, nextImageIndex } from '../domain/attachments.js'

const paste = (label: string, text: string): ComposerToken => ({ kind: 'paste', label, text })

const image = (index: number, path = `/tmp/img${index}.png`): ComposerToken => ({
  index,
  kind: 'image',
  label: imageToken(index),
  path
})

describe('expandTokens (what the agent actually receives)', () => {
  it('replaces a collapsed paste label with its full content', () => {
    const label = '[[ hello.. [3 lines] .. world ]]'
    const expand = expandTokens([paste(label, 'hello\nfoo\nworld')])

    expect(expand(`here: ${label} done`)).toBe('here: hello\nfoo\nworld done')
  })

  it('is a no-op for already-expanded / token-free text (recall round-trip)', () => {
    const expanded = 'hello\nfoo\nworld'
    expect(expandTokens([])(expanded)).toBe(expanded)
  })

  it('expands repeated identical labels in submission order', () => {
    const label = '[[ x [1 lines] ]]'
    const expand = expandTokens([paste(label, 'first'), paste(label, 'second')])

    expect(expand(`${label} then ${label}`)).toBe('first then second')
  })

  it('leaves an unmatched label intact', () => {
    const label = '[[ orphan [2 lines] ]]'
    expect(expandTokens([])(label)).toBe(label)
  })

  it('drops an image token from the text — the gateway already holds the file', () => {
    const expand = expandTokens([image(1)])

    expect(expand(`what is in ${imageToken(1)}`)).toBe('what is in')
  })

  it('leaves no double space where an image token sat mid-sentence', () => {
    const expand = expandTokens([image(1)])

    expect(expand(`before ${imageToken(1)} after`)).toBe('before after')
  })

  it('resolves an image-only message to empty text', () => {
    expect(expandTokens([image(1)])(imageToken(1))).toBe('')
  })

  it('resolves pastes and images in one pass', () => {
    const label = '[[ log.. [9 lines] ]]'
    const expand = expandTokens([paste(label, 'stack\ntrace'), image(2)])

    expect(expand(`${label} and ${imageToken(2)}`)).toBe('stack\ntrace and')
  })
})

describe('nextImageIndex (user-facing numbering)', () => {
  it('starts at 1', () => {
    expect(nextImageIndex([])).toBe(1)
  })

  it('counts past existing images', () => {
    expect(nextImageIndex([image(1), image(2)])).toBe(3)
  })

  it('ignores paste tokens', () => {
    expect(nextImageIndex([paste('[[ x ]]', 'y')])).toBe(1)
  })

  it('does not reuse an index after an earlier image is deleted', () => {
    // [[ Image 1 ]] was erased; the next attach must not become Image 1 again
    // or expandTokens would resolve two different files to one label.
    expect(nextImageIndex([image(2)])).toBe(3)
  })
})

describe('droppedTokens (deleting the token unattaches the thing)', () => {
  it('reports an image whose token was erased from the text', () => {
    expect(droppedTokens([image(1)], 'just text now')).toEqual([image(1)])
  })

  it('reports nothing while the token is still present', () => {
    expect(droppedTokens([image(1)], `look at ${imageToken(1)}`)).toEqual([])
  })

  it('keeps a surviving token when a sibling is erased', () => {
    expect(droppedTokens([image(1), image(2)], imageToken(2))).toEqual([image(1)])
  })
})
