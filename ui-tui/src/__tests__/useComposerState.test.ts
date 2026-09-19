import { describe, expect, it, vi } from 'vitest'

import { looksLikeDroppedPath, requestClipboardImage } from '../app/useComposerState.js'

describe('requestClipboardImage', () => {
  it('uploads local image bytes when the TUI is attached to a remote gateway', async () => {
    const response = { attached: true, name: 'upload.png', path: '/gateway/upload.png' }
    const gw = { request: vi.fn().mockResolvedValue(response) }
    const readImage = vi.fn().mockResolvedValue({ contentBase64: 'cG5n', filename: 'clipboard.png' })

    await expect(
      requestClipboardImage(gw, 'session-1', readImage, { HERMES_TUI_GATEWAY_URL: 'ws://gateway.test/api/ws' })
    ).resolves.toEqual(response)
    expect(gw.request).toHaveBeenCalledWith('image.attach_bytes', {
      content_base64: 'cG5n',
      filename: 'clipboard.png',
      session_id: 'session-1'
    })
  })

  it('keeps gateway-side clipboard extraction for a spawned local gateway', async () => {
    const response = { attached: true, path: '/local/clip.png' }
    const gw = { request: vi.fn().mockResolvedValue(response) }
    const readImage = vi.fn()

    await expect(requestClipboardImage(gw, 'session-1', readImage, {})).resolves.toEqual(response)
    expect(readImage).not.toHaveBeenCalled()
    expect(gw.request).toHaveBeenCalledWith('clipboard.paste', { session_id: 'session-1' })
  })

  it('does not inspect the remote gateway clipboard when the client has no image', async () => {
    const gw = { request: vi.fn() }
    const readImage = vi.fn().mockResolvedValue(null)

    await expect(
      requestClipboardImage(gw, 'session-1', readImage, { HERMES_TUI_GATEWAY_URL: 'ws://gateway.test/api/ws' })
    ).resolves.toBeNull()
    expect(gw.request).not.toHaveBeenCalled()
  })
})

describe('looksLikeDroppedPath', () => {
  it('recognizes macOS screenshot temp paths and file URIs', () => {
    expect(looksLikeDroppedPath('/var/folders/x/T/TemporaryItems/Screenshot\\ 2026-04-21\\ at\\ 1.04.43 PM.png')).toBe(
      true
    )
    expect(
      looksLikeDroppedPath('file:///var/folders/x/T/TemporaryItems/Screenshot%202026-04-21%20at%201.04.43%20PM.png')
    ).toBe(true)
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

  it('rejects short slash-like strings without path structure', () => {
    // No second '/' or '.' → not a plausible file path
    expect(looksLikeDroppedPath('/help')).toBe(false)
    expect(looksLikeDroppedPath('/model sonnet')).toBe(false)
    expect(looksLikeDroppedPath('/api')).toBe(false)
  })

  it('accepts absolute paths with directory separators or extensions', () => {
    expect(looksLikeDroppedPath('/usr/bin/test')).toBe(true)
    expect(looksLikeDroppedPath('/tmp/file.txt')).toBe(true)
    expect(looksLikeDroppedPath('/etc/hosts')).toBe(true) // has second /
  })
})
