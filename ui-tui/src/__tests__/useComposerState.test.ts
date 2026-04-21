import { describe, expect, it } from 'vitest'

import { looksLikeDroppedPath } from '../app/useComposerState.js'

describe('looksLikeDroppedPath', () => {
  it('recognizes macOS screenshot temp paths and file URIs', () => {
    expect(looksLikeDroppedPath('/var/folders/x/T/TemporaryItems/Screenshot\\ 2026-04-21\\ at\\ 1.04.43 PM.png')).toBe(true)
    expect(looksLikeDroppedPath('file:///var/folders/x/T/TemporaryItems/Screenshot%202026-04-21%20at%201.04.43%20PM.png')).toBe(true)
  })

  it('rejects normal multiline or plain text paste', () => {
    expect(looksLikeDroppedPath('hello world')).toBe(false)
    expect(looksLikeDroppedPath('line one\nline two')).toBe(false)
  })
})
