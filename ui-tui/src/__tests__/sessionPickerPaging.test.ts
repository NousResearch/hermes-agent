import { describe, expect, it } from 'vitest'

import { buildResumeViewport, moveResumeSelection, pageForSelection } from '../components/sessionPickerPaging.js'

const makeItems = (count: number) =>
  Array.from({ length: count }, (_, i) => ({
    id: `sid-${i + 1}`,
    message_count: i + 1,
    preview: `preview ${i + 1}`,
    source: 'cli',
    started_at: 1_700_000_000 - i * 60,
    title: `Session ${i + 1}`
  }))

describe('resume picker paging helpers', () => {
  it('keeps a fixed 10-row page while the selection moves within it', () => {
    const items = makeItems(25)

    expect(buildResumeViewport(items, 0, 10)).toMatchObject({
      end: 10,
      offset: 0,
      page: 1,
      totalPages: 3
    })
    expect(buildResumeViewport(items, 9, 10)).toMatchObject({
      end: 10,
      offset: 0,
      page: 1,
      totalPages: 3
    })
    expect(buildResumeViewport(items, 10, 10)).toMatchObject({
      end: 20,
      offset: 10,
      page: 2,
      totalPages: 3
    })
    expect(buildResumeViewport(items, 24, 10)).toMatchObject({
      end: 25,
      offset: 20,
      page: 3,
      totalPages: 3
    })
  })

  it('supports moving by row and by whole page without leaving bounds', () => {
    expect(moveResumeSelection(0, -1, 25)).toBe(0)
    expect(moveResumeSelection(0, 1, 25)).toBe(1)
    expect(moveResumeSelection(9, 1, 25)).toBe(10)
    expect(moveResumeSelection(10, -1, 25)).toBe(9)
    expect(moveResumeSelection(7, 10, 25)).toBe(17)
    expect(moveResumeSelection(17, -10, 25)).toBe(7)
    expect(moveResumeSelection(24, 10, 25)).toBe(24)
  })

  it('computes page numbers from selection indices', () => {
    expect(pageForSelection(0, 10)).toBe(1)
    expect(pageForSelection(9, 10)).toBe(1)
    expect(pageForSelection(10, 10)).toBe(2)
    expect(pageForSelection(19, 10)).toBe(2)
    expect(pageForSelection(20, 10)).toBe(3)
  })
})
