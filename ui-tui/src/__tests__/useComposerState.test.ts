import { describe, expect, it } from 'vitest'

import { looksLikeDroppedPath, mergeAttachmentResult } from '../app/useComposerState.js'

describe('mergeAttachmentResult (stale-capture guard)', () => {
  it('keeps the captured value when the line did not change', () => {
    const next = { value: '/image /tmp/a.png[[ Image 1 ]]', cursor: 22 }

    expect(mergeAttachmentResult('/image /tmp/a.png', '/image /tmp/a.png', next)).toBe(
      '/image /tmp/a.png[[ Image 1 ]]'
    )
  })

  it('drops the stale slash text when the submit cleared the line', () => {
    // A `/image` slash submits the slash text and clears the line before the
    // attach RPC resolves; the resolved token must not resurrect the
    // submitted `/image <path>` text (the 2^n-1 marker-accumulation bug).
    const next = { value: '/image /tmp/a.png[[ Image 1 ]]', cursor: 22 }

    expect(mergeAttachmentResult('/image /tmp/a.png', '', next)).toBe('[[ Image 1 ]]')
  })

  it('preserves text typed while the attach was in flight', () => {
    const next = { value: 'a[[ Image 1 ]]', cursor: 1 }

    expect(mergeAttachmentResult('a', 'abc', next)).toBe('abc[[ Image 1 ]]')
  })

  it('keeps the caption that followed the path on a drag-drop attach', () => {
    // Drag-drop of "~/shot.png look at this" attaches with a caption; the
    // line is not cleared in that flow, so the full value is kept.
    const next = { value: '~/shot.png[[ Image 1 ]] look at this', cursor: 31 }

    expect(mergeAttachmentResult('~/shot.png', '~/shot.png', next)).toBe(
      '~/shot.png[[ Image 1 ]] look at this'
    )
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
