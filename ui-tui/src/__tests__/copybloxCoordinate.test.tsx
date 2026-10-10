import { PassThrough } from 'stream'

import { Box, renderSync, Text } from '@hermes/ink'
import { stripAnsi } from '@hermes/shared/ansi'
import React from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { CopyBlox } from '../components/copyblox.js'
import { DEFAULT_THEME } from '../theme.js'

const { copyText } = vi.hoisted(() => ({ copyText: vi.fn() }))
vi.mock('../lib/copyText.js', () => ({ copyText }))

const clickHandlers: Array<(e: never) => void | undefined> = []
vi.mock('@hermes/ink', async importOriginal => {
  const actual = await importOriginal<typeof HermesInk>()

  return {
    ...actual,
    NoSelect: (props: React.ComponentProps<typeof actual.NoSelect>) => {
      clickHandlers.push(props.onClick)

      return React.createElement(actual.NoSelect, props)
    }
  }
})
import type * as HermesInk from '@hermes/ink'
// Coordinate dispatch through the real Ink hit-test path: render, then
// resolve the mounted DOM and dispatchClick at the header row.
import { dispatchClick } from '@hermes/ink'
import { peekInkInstance } from '@hermes/ink'

describe('E2E: coordinate-dispatched click → exact clipboard', () => {
  beforeEach(() => {
    copyText.mockReset()
    clickHandlers.length = 0
  })

  let passedStdout: PassThrough

  const render = (node: React.ReactNode, cols: number) => {
    passedStdout = new PassThrough()
    const stdout = passedStdout
    let out = ''
    Object.assign(stdout, { columns: cols, isTTY: false, rows: 40 })
    stdout.on('data', c => {
      out += c.toString()
    })

    const inst = renderSync(node, {
      patchConsole: false,
      stdout,
      stdin: new PassThrough() as never,
      stderr: new PassThrough() as never
    })

    return { inst, lines: out.split('\n').map(l => stripAnsi(l).trimEnd()) }
  }

  it('clicking header row (coordinate dispatch) copies exact raw source; clicking body does not', async () => {
    const raw = 'def f():\n\treturn "世界 🌍"\n'
    copyText.mockResolvedValue({ method: 'native', success: true })

    const node = React.createElement(
      Box,
      { flexDirection: 'column' },
      React.createElement(
        CopyBlox,
        { closed: true, cols: 40, language: 'python', rawContent: raw, theme: DEFAULT_THEME },
        React.createElement(Text, null, 'def f():')
      ),
      React.createElement(Text, null, 'adjacent noninteractive')
    )

    const { inst } = render(node, 40)
    // Grab the Ink root node from the instance registry keyed on our stdout.
    const inkInstance = peekInkInstance(passedStdout as never)
    expect(inkInstance).toBeTruthy()
    const root = inkInstance!.rootNode
    expect(root).toBeTruthy()
    const fired = dispatchClick(root as never, 5, 0)
    expect(fired).toBe(true)
    await vi.waitFor(() => expect(copyText).toHaveBeenCalledWith(raw))

    // Body row (row 2) click must NOT copy
    copyText.mockClear()
    dispatchClick(root as never, 5, 2)
    await new Promise(r => setTimeout(r, 30))
    expect(copyText).not.toHaveBeenCalled()
  })

  it('unmount during pending copy: no state update after unmount (no crash)', async () => {
    let release!: () => void
    copyText.mockImplementation(
      () =>
        new Promise<void>(r => {
          release = r as unknown as () => void
        })
    )

    const { inst } = render(
      React.createElement(
        CopyBlox,
        { closed: true, cols: 40, language: 'py', rawContent: 'x', theme: DEFAULT_THEME },
        React.createElement(Text, null, 'x')
      ),
      40
    )

    const handler = clickHandlers.at(-1)!
    handler({ stopImmediatePropagation: vi.fn() } as never)
    inst.unmount()
    inst.cleanup()
    release()
    await new Promise(r => setTimeout(r, 30))
  })
})
