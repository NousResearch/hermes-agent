import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import { stripAnsi } from '@hermes/shared/ansi'
import React from 'react'
import { describe, expect, it } from 'vitest'

import type { CompletionItem } from '../app/interfaces.js'
import { CompletionMenuPanel } from '../components/appOverlays.js'
import { DEFAULT_THEME } from '../theme.js'

const completion = (i: number): CompletionItem => ({
  display: `/cmd${i.toString().padStart(2, '0')}`,
  meta: `description ${i}`,
  text: `/cmd${i}`
})

const renderPanel = async (completions: CompletionItem[], compIdx = 0) => {
  const stdout = new PassThrough()
  Object.assign(stdout, { columns: 80, isTTY: false, rows: 30 })
  let output = ''
  stdout.on('data', chunk => {
    output += String(chunk)
  })

  const instance = renderSync(<CompletionMenuPanel cols={80} compIdx={compIdx} completions={completions} t={DEFAULT_THEME} />, {
    patchConsole: false,
    stdin: new PassThrough() as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream,
    stderr: new PassThrough() as unknown as NodeJS.WriteStream
  })

  // Ink throttles the first paint; give the frame a turn to land before
  // reading the captured output.
  await new Promise(resolve => setTimeout(resolve, 100))
  instance.unmount()

  return stripAnsi(output)
}

describe('completion menu hidden-count footer', () => {
  it('shows how many completions the window truncated', async () => {
    const output = await renderPanel(Array.from({ length: 20 }, (_, i) => completion(i)))

    expect(output).toContain('/cmd00')
    expect(output).toContain('…and 4 more')
  })

  it('counts hidden items above the viewport too, not just the tail', async () => {
    // compIdx deep in the list scrolls the window; items hidden above the
    // window still count toward the footer total.
    const output = await renderPanel(Array.from({ length: 20 }, (_, i) => completion(i)), 15)

    expect(output).toContain('…and 4 more')
  })

  it('shows no footer when the list fits inside the window', async () => {
    const output = await renderPanel(Array.from({ length: 16 }, (_, i) => completion(i)))

    expect(output).toContain('/cmd15')
    expect(output).not.toContain('…and')
  })

  it('shows no footer for a short list', async () => {
    const output = await renderPanel(Array.from({ length: 5 }, (_, i) => completion(i)))

    expect(output).toContain('/cmd04')
    expect(output).not.toContain('…and')
  })
})
