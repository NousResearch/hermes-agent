import { PassThrough } from 'node:stream'

import { Box, renderSync, Text } from '@hermes/ink'
import React from 'react'
import stripAnsi from 'strip-ansi'
import { describe, expect, it } from 'vitest'

import { COMPOSER_RAIL_WIDTH, composerInputColumns, ComposerRail } from '../components/composerRail.js'
import { stableComposerColumns } from '../lib/inputMetrics.js'

const PAINTED_COLUMNS = 80
const RAIL_GLYPH = '\u2502'

/**
 * Rows the transcript window has to budget for a painted draft region. Ink
 * repaints and then clears on unmount, so only the first frame describes the
 * layout; the trailing frames would double-count every row.
 */
const paintRows = (element: React.ReactElement): string[] => {
  const stdout = Object.assign(new PassThrough(), { columns: PAINTED_COLUMNS, rows: 40 })
  const frames: string[] = []

  stdout.on('data', chunk => frames.push(chunk.toString()))

  const view = renderSync(element, {
    stdin: new PassThrough() as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream
  })

  view.unmount()
  view.cleanup()

  return stripAnsi(frames[0] ?? '')
    .split('\n')
    .filter(line => line.trim().length > 0)
}

const draftRows = (count: number): React.ReactElement => (
  <Box flexDirection="column">
    {Array.from({ length: count }, (_, i) => (
      <Text key={i}>draft line {i}</Text>
    ))}
  </Box>
)

describe('composer boundary', () => {
  // The composer pane is flexShrink=0, so a row it gains is a row taken out of
  // the transcript window. The boundary has to be paid for in columns only: the
  // "full box" the issue rules out would add a top and bottom border and fail
  // this by rendering two rows more than the draft actually has.
  it.each([1, 2, 5])('draws a rail on every draft row without changing the row count (%i rows)', count => {
    const bare = paintRows(draftRows(count))

    const railed = paintRows(
      <ComposerRail color="#888888" width={PAINTED_COLUMNS - 2}>
        {draftRows(count)}
      </ComposerRail>
    )

    expect(bare).toHaveLength(count)
    expect(railed).toHaveLength(bare.length)
    expect(railed.every(line => line.startsWith(RAIL_GLYPH))).toBe(true)
    // The draft is still there, just one column in.
    expect(railed.join('\n')).toContain('draft line 0')
  })

  // Must hold across the scrollbar-gutter threshold in stableComposerColumns
  // (total - prompt >= 24), where shrinking the total would hand a column back.
  it.each([40, 80, 120])('spends exactly one input column at %i columns', cols => {
    for (const promptWidth of [2, 8, 16, 22, 30]) {
      const base = stableComposerColumns(cols, promptWidth)
      const railed = composerInputColumns(cols, promptWidth)

      expect(railed).toBeGreaterThanOrEqual(1)
      expect(base - railed).toBe(base === 1 ? 0 : COMPOSER_RAIL_WIDTH)
    }
  })
})
