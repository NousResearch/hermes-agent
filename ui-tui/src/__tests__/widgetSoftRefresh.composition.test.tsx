import { PassThrough } from 'node:stream'

import { Box, render, Text } from '@hermes/ink'
import { createElement } from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { resetOverlayState } from '../app/overlayStore.js'
import { AmbientDock, openWidget, softUpdateWidget } from '../sdk/host.js'
import { defineWidgetApp } from '../sdk/registry.js'

const makeStdout = () => {
  const s = new PassThrough()
  Object.defineProperty(s, 'columns', { configurable: true, value: 100 })
  Object.defineProperty(s, 'rows', { configurable: true, value: 20 })
  Object.defineProperty(s, 'isTTY', { configurable: true, value: false })
  let captured = ''
  s.on('data', (c: Buffer) => {
    captured += c.toString()
  })

  return {
    read: () => captured,
    stdout: s as unknown as NodeJS.WriteStream
  }
}

const wait = (ms: number) => new Promise(r => setTimeout(r, ms))

describe('AmbientDock soft value redraw (composition)', () => {
  beforeEach(() => {
    resetOverlayState()
  })

  afterEach(() => {
    resetOverlayState()
  })

  it('updates only the changed ambient card text while siblings stay mounted', async () => {
    const renders = { a: 0, b: 0 }

    const appA = defineWidgetApp<{ v: string }>({
      help: 'a',
      id: 'soft-a',
      mode: 'ambient',
      zone: 'dock-bottom',
      init: () => ({ v: 'A0' }),
      reduce: s => s,
      render: ({ state }) => {
        renders.a += 1

        return createElement(Text, null, `CARD_A:${state.v}`)
      }
    })

    const appB = defineWidgetApp<{ v: string }>({
      help: 'b',
      id: 'soft-b',
      mode: 'ambient',
      zone: 'dock-bottom',
      init: () => ({ v: 'B0' }),
      reduce: s => s,
      render: ({ state }) => {
        renders.b += 1

        return createElement(Text, null, `CARD_B:${state.v}`)
      }
    })

    openWidget(appA, { v: 'A0' })
    openWidget(appB, { v: 'B0' })

    const { stdout, read } = makeStdout()

    const inst = await render(
      createElement(
        Box,
        { flexDirection: 'column' },
        createElement(Text, null, 'STATUS_MARKER'),
        createElement(AmbientDock, { placement: 'dock-bottom' })
      ),
      { stdin: process.stdin, stderr: process.stderr, stdout }
    )

    await wait(80)
    expect(read()).toContain('STATUS_MARKER')
    expect(read()).toContain('CARD_A:A0')
    expect(read()).toContain('CARD_B:B0')

    const aBefore = renders.a
    const bBefore = renders.b

    softUpdateWidget(appA, { v: 'A1' })
    await wait(80)

    const frame = read()
    expect(frame).toContain('CARD_A:A1')
    expect(frame).toContain('CARD_B:B0')
    expect(frame).toContain('STATUS_MARKER')

    // Soft path: A re-renders with new value; B should not re-render from the value tick.
    expect(renders.a).toBeGreaterThan(aBefore)
    expect(renders.b).toBe(bBefore)

    inst.unmount()
    await wait(20)
  })
})
