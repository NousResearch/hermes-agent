import { PassThrough } from 'node:stream'

import type * as InkModule from '@hermes/ink'
import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { patchUiState, resetUiState } from '../app/uiStore.js'
import { TextInput } from '../components/textInput.js'

// Under display.tui_quiet_idle the composer must not react to a terminal
// focus-out: that report arrives on every tab switch, and any repaint it
// triggers lands in the tab the user just LEFT, which Konsole / tmux
// monitor-activity flag as fresh output.

const focus = vi.hoisted(() => ({ value: true }))
const ttyWrites = vi.hoisted(() => [] as string[])

vi.mock('@hermes/ink', async importOriginal => {
  const actual = await importOriginal<typeof InkModule>()

  const ttyStdout = {
    columns: 80,
    isTTY: true,
    rows: 24,
    write: (s: string) => {
      ttyWrites.push(String(s))

      return true
    }
  }

  return {
    ...actual,
    useStdout: () => ({ stdout: ttyStdout, write: ttyStdout.write }),
    useTerminalFocus: () => focus.value
  }
})

const INVERSE = '\x1b[7m'

async function renderComposer(terminalFocused: boolean) {
  focus.value = terminalFocused
  ttyWrites.length = 0

  const stdin = Object.assign(new PassThrough(), { isTTY: false, setRawMode: () => {} })
  const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
  const stderr = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
  let frame = ''

  stdout.on('data', (d: Buffer) => {
    frame += d.toString()
  })

  const instance = renderSync(<TextInput columns={60} onChange={() => {}} onSubmit={() => {}} value="hi" />, {
    patchConsole: false,
    stderr: stderr as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream
  })

  await new Promise(resolve => setTimeout(resolve, 20))
  instance.unmount()
  instance.cleanup()

  return { frame, hidCursor: ttyWrites.includes('\x1b[?25l') }
}

describe('TextInput terminal blur', () => {
  afterEach(() => {
    resetUiState()
  })

  it('by default restyles on blur: hides the hardware cursor and draws an inverse block', async () => {
    const focused = await renderComposer(true)
    const blurred = await renderComposer(false)

    expect(focused.hidCursor).toBe(false)
    expect(focused.frame).not.toContain(INVERSE)
    expect(blurred.hidCursor).toBe(true)
    expect(blurred.frame).toContain(INVERSE)
  })

  it('renders identically focused or blurred under display.tui_quiet_idle', async () => {
    patchUiState({ quietIdle: true })

    const focused = await renderComposer(true)
    const blurred = await renderComposer(false)

    expect(blurred.hidCursor).toBe(false)
    expect(blurred.frame).not.toContain(INVERSE)
    expect(blurred.frame).toBe(focused.frame)
  })
})
