import { describe, expect, it } from 'vitest'

import { looksLikeDroppedPath, mergeAttachmentResult } from '../app/useComposerState.js'

describe('mergeAttachmentResult (stale-capture guard)', () => {
  // Fixtures mirror what `appendAttachment`'s attach functions actually build:
  // the token is inserted into the CAPTURED value via `insertAtCursor`, which
  // adds a space before it when the preceding char is not whitespace (the
  // captured path ends in a non-space, hence `…a.png [[ Image 1 ]]`).

  it('keeps the captured value when the line did not change', () => {
    const next = { value: '/image /tmp/a.png [[ Image 1 ]]', cursor: 31 }

    expect(mergeAttachmentResult('/image /tmp/a.png', '/image /tmp/a.png', next)).toBe(
      '/image /tmp/a.png [[ Image 1 ]]'
    )
  })

  it('drops the stale slash text when the submit cleared the line', () => {
    // A `/image` slash submits the slash text and clears the line before the
    // attach RPC resolves; the resolved token must not resurrect the
    // submitted `/image <path>` text (the 2^n-1 marker-accumulation bug).
    // `next` was still built from the captured path, so only the trailing
    // token plus `insertAtCursor`'s leading space survives.
    const next = { value: '/image /tmp/a.png [[ Image 1 ]]', cursor: 31 }

    expect(mergeAttachmentResult('/image /tmp/a.png', '', next)).toBe(' [[ Image 1 ]]')
  })

  it('preserves text typed while the attach was in flight', () => {
    const next = { value: 'a [[ Image 1 ]]', cursor: 13 }

    expect(mergeAttachmentResult('a', 'abc', next)).toBe('abc [[ Image 1 ]]')
  })

  it('keeps the caption that followed the path on a drag-drop attach', () => {
    // Drag-drop of "~/shot.png look at this" attaches with a caption; the
    // line is not cleared in that flow, so the full value is kept.
    const next = { value: '~/shot.png [[ Image 1 ]] look at this', cursor: 33 }

    expect(mergeAttachmentResult('~/shot.png', '~/shot.png', next)).toBe(
      '~/shot.png [[ Image 1 ]] look at this'
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
