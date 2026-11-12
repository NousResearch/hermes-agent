import { EventEmitter } from 'events'

import React from 'react'
import { describe, expect, it } from 'vitest'

import Box from './components/Box.js'
import Text from './components/Text.js'
import Ink from './ink.js'

class FakeTty extends EventEmitter {
  chunks: string[] = []
  columns = 40
  rows = 8
  isTTY = true

  write(chunk: string | Uint8Array, cb?: (err?: Error | null) => void): boolean {
    this.chunks.push(typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8'))
    cb?.()

    return true
  }
}

const paint = (node: React.ReactElement) => {
  const stdout = new FakeTty()
  const stdin = new FakeTty()
  const stderr = new FakeTty()

  const ink = new Ink({
    exitOnCtrlC: false,
    patchConsole: false,
    stderr: stderr as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream
  })

  ink.render(node)
  ink.onRender()
  const frame = stdout.chunks.join('')
  ink.unmount()

  return frame
}

// The composer's floating panels (session switcher, model picker, …) are
// absolute `bottom: 100%` children of a relative Box whose only OTHER children
// — the input rows — unmount while an overlay is open. That leaves the host box
// at height 0 with a sibling on the same row, which is exactly the shape the
// same-row ghost guard skips. The guard must not take the escaping absolute
// child with it: it paints outside the host's bounds and can never ghost.
describe('absolute children of a zero-height box', () => {
  it('paints an absolute bottom:100% panel when its host box collapses to h=0', () => {
    const frame = paint(
      <Box flexDirection="column">
        <Text>transcript</Text>

        <Box flexDirection="column" position="relative">
          <Box bottom="100%" flexDirection="column" left={0} position="absolute" right={0}>
            <Text>PANEL-CONTENT</Text>
          </Box>
        </Box>

        <Text>footer</Text>
      </Box>
    )

    expect(frame).toContain('PANEL-CONTENT')
  })

  // NOT covered here: that the guard still SUPPRESSES the same-row ghost it
  // exists for (a squeezed box and its sibling both writing one row, leaving
  // the longer content's tail behind). That path has no test upstream either,
  // and the obvious candidates are vacuous — they pass with the guard deleted
  // outright, which would silently reintroduce the ghost. Asserting it needs a
  // tree where Yoga actually squeezes a node to h=0 onto a sibling's row (the
  // HelpV2 shortcuts column is the known real case); worth adding separately.
})
