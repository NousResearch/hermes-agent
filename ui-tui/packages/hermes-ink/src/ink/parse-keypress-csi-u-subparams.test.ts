import { describe, expect, it } from 'vitest'

import { INITIAL_STATE, parseMultipleKeypresses } from './parse-keypress.js'

const parseOne = (sequence: string) => parseMultipleKeypresses(INITIAL_STATE, sequence)[0]

describe('CSI-u sub-parameters', () => {
  it.each([
    ['plain', '\x1b[97;2u'],
    ['alternate keys', '\x1b[97:65;2u'],
    ['event type', '\x1b[97;2:1u'],
    ['both', '\x1b[97:65;2:1u'],
  ])('parses Shift+A from the %s spelling', (_label, sequence) => {
    expect(parseOne(sequence)).toEqual([
      expect.objectContaining({ name: 'a', shift: true, ctrl: false, meta: false }),
    ])
  })

  it('resolves a key release to no name, so input-event swallows it', () => {
    expect(parseOne('\x1b[97;2:3u')).toEqual([expect.objectContaining({ name: undefined })])
  })
})
