import { describe, expect, it } from 'vitest'

import { findStableBoundary } from '../components/streamingMarkdown.js'
// We test the pure boundary logic by rendering the component's ref
// behaviour through repeated calls. Since React isn't being rendered here,
// we reach into the module to test findStableBoundary via its exported
// behaviour — but the pure helper isn't exported. So test the component's
// observable output: pass sequential text values and verify the stable
// prefix never retreats.
//
// Strategy: mount StreamingMd in isolation and observe which <Md>
// instances it renders (by text prop). Without a DOM renderer that's
// heavy, so we validate the helper behaviour by directly invoking the
// fence/boundary logic via a re-exported surface.
import { DEFAULT_THEME } from '../theme.js'

describe('findStableBoundary', () => {
  it('returns -1 when no blank line exists yet', () => {
    expect(findStableBoundary('partial line with no newline yet')).toBe(-1)
  })

  it('returns -1 when only single newlines exist', () => {
    expect(findStableBoundary('line one\nline two\nline three')).toBe(-1)
  })

  it('splits after the last blank line separator', () => {
    // 'first\n\nsecond\n\nthird' → last blank = before 'third'
    const text = 'first paragraph\n\nsecond paragraph\n\nthird'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('first paragraph\n\nsecond paragraph\n\n')
    expect(text.slice(idx)).toBe('third')
  })

  it('refuses to split inside an open fenced block', () => {
    // Fence opens, contains a blank line inside the code, no close yet.
    const text = '```ts\nfn();\n\nmore code here'

    expect(findStableBoundary(text)).toBe(-1)
  })

  it('splits before an open fenced block but not inside', () => {
    const text = 'intro paragraph\n\n```ts\nfn();\n\nmore code'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('intro paragraph\n\n')
    expect(text.slice(idx).startsWith('```ts')).toBe(true)
  })

  it('allows splitting after a fenced block closes', () => {
    const text = '```ts\nfn();\n```\n\nnarration continues'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('```ts\nfn();\n```\n\n')
    expect(text.slice(idx)).toBe('narration continues')
  })

  it('walks backwards through nested fence boundaries safely', () => {
    // Two closed fences + narration + one new open fence. The only legal
    // split is before the open fence, not between the closed ones.
    const text = '```js\na\n```\n\nmid text\n\n```python\nstill open'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('```js\na\n```\n\nmid text\n\n')
  })

  it('handles empty input', () => {
    expect(findStableBoundary('')).toBe(-1)
  })

  it('refuses to split inside an open $$ math block', () => {
    // Display math has been opened but not closed; the only blank line
    // sits inside the open block, so there's no safe boundary yet.
    const text = '$$\nx + y\n\nmore math'

    expect(findStableBoundary(text)).toBe(-1)
  })

  it('allows splitting after a $$ math block closes', () => {
    const text = '$$\nx + y = z\n$$\n\nnarration continues'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('$$\nx + y = z\n$$\n\n')
    expect(text.slice(idx)).toBe('narration continues')
  })

  it('splits before an open $$ block but not inside', () => {
    // Mirror of the existing fenced-code test: prose, then an unclosed
    // math block. The only safe boundary is the blank line BEFORE `$$`.
    const text = 'intro paragraph\n\n$$\nx + y\n\nmore'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('intro paragraph\n\n')
    expect(text.slice(idx).startsWith('$$')).toBe(true)
  })

  it('treats single-line $$x$$ as zero net toggle', () => {
    // `$$x = y$$` opens AND closes on one line, so the stable boundary
    // after it is allowed.
    const text = 'intro\n\n$$x = y$$\n\nnarration'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('intro\n\n$$x = y$$\n\n')
    expect(text.slice(idx)).toBe('narration')
  })

  it('refuses to split inside an open \\[ math block', () => {
    const text = '\\[\nx + y\n\nmore'

    expect(findStableBoundary(text)).toBe(-1)
  })
})

describe('streaming theme assumption', () => {
  it('theme is exportable (component import sanity check)', () => {
    // Sanity that the theme we pass doesn't change shape. Component import
    // already happens above — this is a smoke test that the module graph
    // for streamingMarkdown wires up without cycles.
    expect(DEFAULT_THEME.color.accent).toBeTruthy()
  })
})
