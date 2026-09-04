import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { ToolTrail } from '../components/thinking.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

const flushEffects = async () => {
  for (let i = 0; i < 10; i++) {
    await new Promise(resolve => setTimeout(resolve, 5))
  }
}

const mountTrail = (props: Record<string, unknown>) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 80, isTTY: false, rows: 24 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(
    <ToolTrail t={DEFAULT_THEME} {...props} />,
    {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    }
  )

  const text = () => stripAnsi(output)

  return { instance, text }
}

describe('ToolTrail compact cards', () => {
  it('renders a collapsed terminal card with name, duration, and 1-line preview', async () => {
    const { instance, text } = mountTrail({
      sections: { tools: 'expanded' },
      trail: ['Terminal("ls -la") (0.3s) :: total 12 ✓']
    })

    await flushEffects()

    const out = text()
    expect(out).toMatch(/▸/)
    expect(out).toMatch(/Terminal/)
    expect(out).toMatch(/0\.3s/)
    expect(out).toMatch(/total 12/)
    expect(out).not.toMatch(/●/)
    expect(out).not.toMatch(/Args:/)

    instance.unmount()
    instance.cleanup()
  })

  it('keeps all tools collapsed when tools section is expanded', async () => {
    const { instance, text } = mountTrail({
      sections: { tools: 'expanded' },
      trail: ['Read File("foo.ts") (0.1s) :: export const x ✓']
    })

    await flushEffects()

    const out = text()
    expect(out).toMatch(/▸/)
    expect(out).toMatch(/Read File/)
    expect(out).not.toMatch(/●/)

    instance.unmount()
    instance.cleanup()
  })

  it('hides tool cards when display.sections.tools is hidden', async () => {
    const { instance, text } = mountTrail({
      sections: { tools: 'hidden' },
      trail: ['Terminal("ls") (0.1s) :: ok ✓']
    })

    await flushEffects()

    const out = text()
    expect(out).not.toMatch(/Terminal/)

    instance.unmount()
    instance.cleanup()
  })

  it('shows a live terminal tool as a collapsed card with elapsed time', async () => {
    const { instance, text } = mountTrail({
      sections: { tools: 'expanded' },
      tools: [{ id: 't1', name: 'terminal', context: 'sleep 2', startedAt: Date.now() - 1500 }]
    })

    await flushEffects()

    const out = text()
    expect(out).toMatch(/▸/)
    expect(out).toMatch(/Terminal/)
    expect(out).toMatch(/sleep 2/)

    instance.unmount()
    instance.cleanup()
  })
})
