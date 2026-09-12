import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { MessageLine } from '../components/messageLine.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'
import type { Msg, ToolTrailPosition } from '../types.js'

const ANSWER = 'Both files are already covered by the suite.'

const MSG: Msg = {
  role: 'assistant',
  text: ANSWER,
  tools: ['Read Files ("a.ts") ✓', 'Read Files ("b.ts") ✓', 'Search Files ("describe") ✓']
}

/** Render one MessageLine offscreen and return its settled plain-text frame. */
const frameFor = async (toolTrailPosition?: ToolTrailPosition) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 72, isTTY: false, rows: 30 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(
    <MessageLine
      cols={72}
      msg={MSG}
      sections={{ tools: 'expanded' }}
      t={DEFAULT_THEME}
      {...(toolTrailPosition ? { toolTrailPosition } : {})}
    />,
    {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    }
  )

  // ToolTrail syncs its open/closed state in a mount effect; read the frame
  // only once those have flushed, like the MoA visibility test does.
  await new Promise(resolve => setImmediate(resolve))
  await new Promise(resolve => setImmediate(resolve))

  instance.unmount()

  return stripAnsi(output)
}

/** Row index of the first line containing `needle`, or -1. */
const rowOf = (frame: string, needle: string) => frame.split('\n').findIndex(line => line.includes(needle))

describe('MessageLine — display.tool_trail_position', () => {
  it('renders the trail above the answer when the option is unset', async () => {
    const frame = await frameFor()
    const trail = rowOf(frame, 'Tool calls')
    const answer = rowOf(frame, ANSWER)

    expect(trail).toBeGreaterThanOrEqual(0)
    expect(answer).toBeGreaterThanOrEqual(0)
    expect(trail).toBeLessThan(answer)
    expect(frame).toContain('Response')
  })

  it('is byte-identical to the unset default when set explicitly to above', async () => {
    expect(await frameFor('above')).toBe(await frameFor())
  })

  it('renders the trail below the answer under below', async () => {
    const frame = await frameFor('below')
    const trail = rowOf(frame, 'Tool calls')
    const answer = rowOf(frame, ANSWER)

    expect(trail).toBeGreaterThanOrEqual(0)
    expect(answer).toBeGreaterThanOrEqual(0)
    expect(trail).toBeGreaterThan(answer)
  })

  it('drops the Response rule under below but keeps every tool row', async () => {
    const frame = await frameFor('below')

    expect(frame).not.toContain('Response')

    for (const tool of ['a.ts', 'b.ts', 'describe']) {
      expect(frame).toContain(tool)
    }
  })

  it('keeps hidden trails hidden in both positions', async () => {
    // Visibility is orthogonal to position: `tools: hidden` wins either way,
    // and with no trail there is no separator to draw in either mode.
    const hidden = async (toolTrailPosition: ToolTrailPosition) => {
      const stdout = new PassThrough()
      const stdin = new PassThrough()
      const stderr = new PassThrough()
      let output = ''

      Object.assign(stdout, { columns: 72, isTTY: false, rows: 30 })
      Object.assign(stdin, { isTTY: false })
      Object.assign(stderr, { isTTY: false })
      stdout.on('data', chunk => {
        output += chunk.toString()
      })

      const instance = renderSync(
        <MessageLine
          cols={72}
          msg={MSG}
          sections={{ tools: 'hidden' }}
          t={DEFAULT_THEME}
          toolTrailPosition={toolTrailPosition}
        />,
        {
          patchConsole: false,
          stderr: stderr as NodeJS.WriteStream,
          stdin: stdin as NodeJS.ReadStream,
          stdout: stdout as NodeJS.WriteStream
        }
      )

      await new Promise(resolve => setImmediate(resolve))
      await new Promise(resolve => setImmediate(resolve))
      instance.unmount()

      return stripAnsi(output)
    }

    const above = await hidden('above')
    const below = await hidden('below')

    expect(above).not.toContain('Tool calls')
    expect(below).not.toContain('Tool calls')
    expect(above).toContain(ANSWER)
    expect(below).toBe(above)
  })
})
