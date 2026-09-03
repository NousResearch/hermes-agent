import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { ToolTrail } from '../components/thinking.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

const flushEffects = async () => {
  // Passive effects + the re-render they trigger need a few macrotask
  // turns (React's scheduler uses MessageChannel) before the next frame
  // paints — setTimeout(0)-class waits, not setImmediate (which can land
  // in the wrong phase and observe the pre-effect frame).
  for (let i = 0; i < 10; i++) {
    await new Promise(resolve => setTimeout(resolve, 5))
  }
}

const mountTrail = (reasoningActive: boolean, sections?: Record<string, string>) => {
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
      reasoning="Live reasoning text."
      reasoningActive={reasoningActive}
      sections={sections ?? { thinking: 'collapsed' }}
      t={DEFAULT_THEME}
    />,
    {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    }
  )

  // The PassThrough accumulates every repaint, and a collapsed panel stops
  // repainting entirely once settled — so assert on the FINAL chevron state
  // in the accumulated output rather than the tail after a clear().
  const finalChevronOpen = () => stripAnsi(output).lastIndexOf('▾ ') > stripAnsi(output).lastIndexOf('▸ ')

  return { finalChevronOpen, instance }
}

describe('ToolTrail — collapsed mode auto-expands while reasoning is live', () => {
  it('opens (▾) when reasoningActive is true under sections.thinking: collapsed', async () => {
    const { finalChevronOpen, instance } = mountTrail(true)

    await flushEffects()

    expect(finalChevronOpen()).toBe(true)

    instance.unmount()
    instance.cleanup()
  })

  it('collapses (▸) when reasoningActive is false under sections.thinking: collapsed', async () => {
    const { finalChevronOpen, instance } = mountTrail(false)

    await flushEffects()

    expect(finalChevronOpen()).toBe(false)

    instance.unmount()
    instance.cleanup()
  })

  it('closes the panel when the reasoning phase ends mid-turn (rerender)', async () => {
    const { finalChevronOpen, instance } = mountTrail(true)

    await flushEffects()

    expect(finalChevronOpen()).toBe(true)

    // Reasoning phase finished (final answer / tool call started) — the
    // turn's reasoningActive drops and the panel must collapse.
    instance.rerender(
      <ToolTrail
        reasoning="Live reasoning text."
        reasoningActive={false}
        sections={{ thinking: 'collapsed' }}
        t={DEFAULT_THEME}
      />
    )

    await flushEffects()

    expect(finalChevronOpen()).toBe(false)

    instance.unmount()
    instance.cleanup()
  })

  it('leaves expanded-mode panels fully manual (no forced collapse)', async () => {
    const { finalChevronOpen, instance } = mountTrail(false, { thinking: 'expanded' })

    await flushEffects()

    // `expanded` is a manual preference: reasoningActive=false must NOT
    // force it closed (the auto behavior only applies to `collapsed`).
    expect(finalChevronOpen()).toBe(true)

    instance.unmount()
    instance.cleanup()
  })
})
