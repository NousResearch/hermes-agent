import { describe, expect, it, vi } from 'vitest'

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

  it('recognizes common image file extensions', () => {
    expect(looksLikeDroppedPath('/Users/me/Desktop/photo.jpg')).toBe(true)
    expect(looksLikeDroppedPath('/Users/me/Desktop/diagram.png')).toBe(true)
    expect(looksLikeDroppedPath('/tmp/capture.webp')).toBe(true)
    expect(looksLikeDroppedPath('/tmp/image.gif')).toBe(true)
  })

  it('recognizes file:// URIs with various extensions', () => {
    expect(looksLikeDroppedPath('file:///home/user/doc.pdf')).toBe(true)
    expect(looksLikeDroppedPath('file:///tmp/screenshot.png')).toBe(true)
  })

  it('recognizes paths with spaces (not backslash-escaped)', () => {
    expect(looksLikeDroppedPath('/var/folders/x/T/TemporaryItems/Screenshot 2026-04-21 at 1.04.43 PM.png')).toBe(true)
  })

  it('rejects empty/whitespace-only input', () => {
    expect(looksLikeDroppedPath('')).toBe(false)
    expect(looksLikeDroppedPath('   ')).toBe(false)
    expect(looksLikeDroppedPath('\n')).toBe(false)
  })

  it('rejects URLs that are not file:// URIs', () => {
    expect(looksLikeDroppedPath('https://example.com/image.png')).toBe(false)
    expect(looksLikeDroppedPath('http://localhost/file.pdf')).toBe(false)
  })

  it('treats leading-slash strings as potential paths (server-side validates)', () => {
    // The heuristic is intentionally broad — starts with / could be a path.
    // Server-side image.attach / input.detect_drop does real validation.
    expect(looksLikeDroppedPath('/help')).toBe(true)
    expect(looksLikeDroppedPath('/model sonnet')).toBe(true)
  })
})
