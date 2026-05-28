import { EventEmitter } from 'events'
import React from 'react'
import { describe, expect, it } from 'vitest'

import Text from './components/Text.js'
import Ink from './ink.js'
import { CURSOR_HOME, ERASE_SCREEN, ERASE_SCROLLBACK } from './termio/csi.js'

class FakeTty extends EventEmitter {
  chunks: string[] = []
  columns = 20
  rows = 5
  isTTY = true

  write(chunk: string | Uint8Array, cb?: (err?: Error | null) => void): boolean {
    this.chunks.push(typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8'))
    cb?.()
    return true
  }
}

const tick = () => new Promise<void>(resolve => queueMicrotask(resolve))

function createInk(stdout = new FakeTty()) {
  const stdin = new FakeTty()
  const stderr = new FakeTty()
  const ink = new Ink({
    exitOnCtrlC: false,
    patchConsole: false,
    stderr: stderr as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream
  })

  return { ink, stdout }
}

async function withEnv(updates: Record<string, string | undefined>, fn: () => Promise<void>): Promise<void> {
  const original = new Map<string, string | undefined>()

  for (const key of Object.keys(updates)) {
    original.set(key, process.env[key])
    const value = updates[key]

    if (value === undefined) {
      delete process.env[key]
    } else {
      process.env[key] = value
    }
  }

  try {
    await fn()
  } finally {
    for (const [key, value] of original) {
      if (value === undefined) {
        delete process.env[key]
      } else {
        process.env[key] = value
      }
    }
  }
}

describe('Ink resize healing', () => {
  it('heals same-dimension alt-screen resize events with an erase before repaint', async () => {
    const { ink, stdout } = createInk()

    try {
      ink.setAltScreenActive(true)
      ink.render(React.createElement(Text, null, 'hello'))
      ink.onRender()
      stdout.chunks = []

      stdout.emit('resize')
      ink.onRender()
      await tick()

      expect(stdout.chunks.join('')).toContain(ERASE_SCREEN + CURSOR_HOME)
    } finally {
      ink.unmount()
    }
  })

  it('heals same-dimension tmux inline resize events with a visible-screen erase and full repaint', async () => {
    await withEnv({ HERMES_TUI_MAIN_SCREEN_RESIZE_REPAINT: undefined, TMUX: '/tmp/tmux-501/default,1,0' }, async () => {
      const { ink, stdout } = createInk()

      try {
        ink.render(React.createElement(Text, null, 'hello'))
        ink.onRender()
        stdout.chunks = []

        stdout.emit('resize')
        ink.onRender()
        await tick()

        const output = stdout.chunks.join('')
        expect(output).toContain(ERASE_SCREEN + CURSOR_HOME)
        expect(output).toContain('hello')
        expect(output).not.toContain(ERASE_SCROLLBACK)
      } finally {
        ink.unmount()
      }
    })
  })

  it('heals dimension-changing tmux inline resize events without clearing scrollback', async () => {
    await withEnv({ HERMES_TUI_MAIN_SCREEN_RESIZE_REPAINT: undefined, TMUX: '/tmp/tmux-501/default,1,0' }, async () => {
      const { ink, stdout } = createInk()

      try {
        ink.render(React.createElement(Text, null, 'hello'))
        ink.onRender()
        stdout.chunks = []
        stdout.columns = 30
        stdout.rows = 8

        stdout.emit('resize')
        ink.onRender()
        await tick()

        const output = stdout.chunks.join('')
        expect(output).toContain(ERASE_SCREEN + CURSOR_HOME)
        expect(output).toContain('hello')
        expect(output).not.toContain(ERASE_SCROLLBACK)
      } finally {
        ink.unmount()
      }
    })
  })

  it('ignores same-dimension main-screen resize events outside tmux', async () => {
    await withEnv({ HERMES_TUI_MAIN_SCREEN_RESIZE_REPAINT: undefined, TMUX: undefined }, async () => {
      const { ink, stdout } = createInk()

      try {
        ink.render(React.createElement(Text, null, 'hello'))
        ink.onRender()
        stdout.chunks = []

        stdout.emit('resize')
        ink.onRender()
        await tick()

        expect(stdout.chunks.join('')).toBe('')
      } finally {
        ink.unmount()
      }
    })
  })
})
