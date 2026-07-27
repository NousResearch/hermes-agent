import { PassThrough } from 'stream'

import { Box, renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { usePet } from '../app/usePet.js'
import { PetPane } from '../components/appLayout.js'
import { stripAnsi } from '../lib/text.js'

vi.mock('../app/usePet.js', () => ({
  usePet: vi.fn()
}))

const opaqueCell = [255, 0, 0, 255, 0, 0, 255, 255]

const PET_GLYPHS = new Set(['▀', '▄', '█'])

const firstGlyphCol = (line: string) => [...line].findIndex(ch => PET_GLYPHS.has(ch))

const renderFrame = (element: React.ReactElement, columns = 40) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns, isTTY: false, rows: 12 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(element, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  instance.unmount()
  instance.cleanup()

  return stripAnsi(output)
    .split('\n')
    .map(line => line.replace(/\s+$/, ''))
}

describe('PetPane', () => {
  afterEach(() => {
    vi.mocked(usePet).mockReset()
  })

  it('overlays the bottom-right corner with a flat, right-aligned sprite', () => {
    const columns = 40
    vi.mocked(usePet).mockReturnValue({
      enabled: true,
      grid: [
        [opaqueCell, opaqueCell],
        [opaqueCell, opaqueCell]
      ],
      kitty: null
    })

    const lines = renderFrame(
      <Box flexDirection="column" height={8} position="relative" width={columns}>
        <PetPane />
      </Box>,
      columns
    )

    const cols = lines.map(firstGlyphCol).filter(col => col >= 0)
    expect(cols.length).toBeGreaterThanOrEqual(2)
    // Flat (no per-row drift) and right-aligned (a corner block, not full width).
    expect(new Set(cols).size).toBe(1)
    expect(cols[0]).toBeGreaterThan(columns / 2)
  })

  it('renders nothing when disabled', () => {
    vi.mocked(usePet).mockReturnValue({ enabled: false, grid: null, kitty: null })

    const lines = renderFrame(
      <Box flexDirection="column" height={8} position="relative" width={40}>
        <PetPane />
      </Box>
    )

    expect(lines.every(line => firstGlyphCol(line) < 0)).toBe(true)
  })
})
