import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { ToolTrail } from '../components/thinking.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

describe('ToolTrail — MoA reference panel visibility (#64701)', () => {
  it('stays expanded after mount effects settle when reasoningAlwaysVisible is set, even under sections.thinking: hidden', async () => {
    const stdout = new PassThrough()
    const stdin = new PassThrough()
    const stderr = new PassThrough()
    let output = ''

    Object.assign(stdout, { columns: 60, isTTY: false, rows: 20 })
    Object.assign(stdin, { isTTY: false })
    Object.assign(stderr, { isTTY: false })
    stdout.on('data', chunk => {
      output += chunk.toString()
    })

    const instance = renderSync(
      <ToolTrail
        reasoning="Reference model output that must stay visible on first paint."
        reasoningAlwaysVisible
        sections={{ thinking: 'hidden' }}
        t={DEFAULT_THEME}
      />,
      {
        patchConsole: false,
        stderr: stderr as NodeJS.WriteStream,
        stdin: stdin as NodeJS.ReadStream,
        stdout: stdout as NodeJS.WriteStream
      }
    )

    // Let queued passive effects (and any re-render they trigger) flush
    // before reading the frame — the #64701 bug is specifically that the
    // re-sync effect fires AFTER the first paint and clobbers the
    // reasoningAlwaysVisible mount value, so asserting on the pre-effect
    // frame alone would miss the regression entirely.
    await new Promise(resolve => setImmediate(resolve))
    await new Promise(resolve => setImmediate(resolve))

    const frame = stripAnsi(output)

    instance.unmount()
    instance.cleanup()

    // Open chevron (▾) means the panel is still expanded once effects have
    // settled, as the reasoningAlwaysVisible-seeded useState value intends.
    // A collapsed (▸) render here means the re-sync effect fired on mount
    // and clobbered it — the exact #64701 regression.
    expect(frame).toContain('▾ ')
    expect(frame).toContain('Thinking')
    expect(frame).not.toContain('▸ ')
  })
})
