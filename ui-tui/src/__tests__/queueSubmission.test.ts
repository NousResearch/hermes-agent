import { describe, expect, it } from 'vitest'

import type { ComposerToken } from '../app/interfaces.js'
import { expandPasteTokens, queueItemFromSlash } from '../app/useSubmission.js'
import { imageToken } from '../domain/attachments.js'

describe('/queue collapsed paste submission', () => {
  it('keeps the collapsed argument for display and the full multiline payload for execution', () => {
    const display = '[[ first.. [3 lines] .. last ]]'

    expect(queueItemFromSlash(`/queue ${display}`, '/queue first\nmiddle\nlast')).toEqual({
      display,
      text: 'first\nmiddle\nlast'
    })
  })

  it('supports the /q alias and rejects an empty queue command', () => {
    expect(queueItemFromSlash('/q [[ payload ]]', '/q complete payload')).toEqual({
      display: '[[ payload ]]',
      text: 'complete payload'
    })
    expect(queueItemFromSlash('/queue', '/queue')).toBeUndefined()
  })

  it('expands paste tokens without consuming image tokens', () => {
    const paste: ComposerToken = { kind: 'paste', label: '[[ paste [2 lines] ]]', text: 'one\ntwo' }
    const image: ComposerToken = { kind: 'image', index: 1, label: imageToken(1), path: '/tmp/image.png' }

    expect(expandPasteTokens([paste, image])(`${paste.label} and ${image.label}`)).toBe(`one\ntwo and ${image.label}`)
  })
})
