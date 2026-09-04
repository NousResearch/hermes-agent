import { EventEmitter } from 'events'

import React from 'react'
import { describe, expect, it } from 'vitest'

import Box from './components/Box.js'
import Text from './components/Text.js'
import Ink from './ink.js'
import { ERASE_SCREEN, ERASE_SCROLLBACK } from './termio/csi.js'
import { DISABLE_MOUSE_TRACKING } from './termio/dec.js'

/**
 * Focus-regain recovery (DECSET 1004 focus-in).
 *
 * Two properties are asserted against the RESULTING SCREEN, not against which
 * bytes were emitted:
 *
 *  1. Healing — a row that is stale on the physical screen but BLANK in the
 *     new frame must be gone. The cell diff skips blank-over-blank, so a
 *     buffer-only reset leaves it behind; the clear is what removes it.
 *  2. Atomicity — the clear must ride in the SAME write() as the repaint, so
 *     no frame can be presented between "screen cleared" and "content drawn".
 *     A separate erase write is the visible flash on an ordinary tab switch.
 *
 * Both hold on the alt screen and on the main screen (INLINE_MODE / Termux).
 */

/** Minimal terminal emulator: replays ANSI into a cell grid. */
class TermModel {
  private readonly rows: string[][]
  private cx = 0
  private cy = 0

  constructor(
    private readonly width: number,
    private readonly height: number
  ) {
    this.rows = Array.from({ length: height }, () => Array.from({ length: width }, () => ' '))
  }

  private put(ch: string): void {
    if (this.cy >= 0 && this.cy < this.height && this.cx >= 0 && this.cx < this.width) {
      this.rows[this.cy]![this.cx] = ch
    }

    this.cx++

    if (this.cx >= this.width) {
      this.cx = 0
      this.cy++
    }
  }

  write(data: string): void {
    let i = 0

    while (i < data.length) {
      const ch = data[i]!

      if (ch === '\x1b') {
        if (data[i + 1] !== '[') {
          i += 2

          continue
        }

        let j = i + 2

        while (j < data.length && !/[A-Za-z]/.test(data[j]!)) {
          j++
        }

        const final = data[j]
        const params = data.slice(i + 2, j)
        i = j + 1

        // DEC private modes (mouse, sync, cursor visibility) don't move cells.
        if (params.startsWith('?')) {
          continue
        }

        const nums = params.split(';').map(p => (p === '' ? undefined : Number(p)))
        const n = nums[0] ?? 1

        switch (final) {
          case 'H':
            this.cy = (nums[0] ?? 1) - 1
            this.cx = (nums[1] ?? 1) - 1

            break

          case 'J':
            if ((nums[0] ?? 0) === 2 || (nums[0] ?? 0) === 3) {
              for (const row of this.rows) {
                row.fill(' ')
              }
            }

            break

          case 'K':
            if (this.cy >= 0 && this.cy < this.height) {
              for (let x = this.cx; x < this.width; x++) {
                this.rows[this.cy]![x] = ' '
              }
            }

            break

          case 'A':
            this.cy -= n

            break

          case 'B':
            this.cy += n

            break

          case 'C':
            this.cx += n

            break

          case 'D':
            this.cx -= n

            break

          case 'G':
            this.cx = n - 1

            break

          default:
            break
        }

        continue
      }

      if (ch === '\r') {
        this.cx = 0
        i++

        continue
      }

      if (ch === '\n') {
        this.cy++
        this.cx = 0
        i++

        continue
      }

      this.put(ch)
      i++
    }
  }

  text(): string {
    return this.rows.map(r => r.join('').trimEnd()).join('\n')
  }
}

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

  drain(): string {
    const out = this.chunks.join('')
    this.chunks = []

    return out
  }
}

type InkPrivate = {
  handleTerminalFocusChange: (isFocused: boolean) => void
}

const peek = (ink: Ink): InkPrivate => ink as unknown as InkPrivate
const tick = () => new Promise<void>(resolve => queueMicrotask(resolve))

const STALE = 'STATUSROW downloading 42%'

// Tall frame -> short frame: the vacated row is BLANK in the new frame, which
// is exactly the case the cell diff skips.
const tall = () =>
  React.createElement(
    Box,
    { flexDirection: 'column' },
    React.createElement(Text, null, 'hello'),
    React.createElement(Text, null, STALE)
  )

const short = () => React.createElement(Box, { flexDirection: 'column' }, React.createElement(Text, null, 'hello'))

async function focusRegain(altScreen: boolean, env?: Record<string, string>) {
  const restore: Array<[string, string | undefined]> = []

  for (const [k, v] of Object.entries(env ?? {})) {
    restore.push([k, process.env[k]])
    process.env[k] = v
  }

  try {
    return await runFocusRegain(altScreen)
  } finally {
    for (const [k, v] of restore) {
      if (v === undefined) {
        delete process.env[k]
      } else {
        process.env[k] = v
      }
    }
  }
}

async function runFocusRegain(altScreen: boolean) {
  const stdout = new FakeTty()
  const stdin = new FakeTty()
  const stderr = new FakeTty()
  const term = new TermModel(40, 8)

  const ink = new Ink({
    exitOnCtrlC: false,
    patchConsole: false,
    stderr: stderr as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream
  })

  if (altScreen) {
    ink.setAltScreenActive(true, 'all')
  }

  ink.render(tall())
  ink.onRender()
  await tick()
  term.write(stdout.drain())

  // Hidden/throttled tab: Ink emits the shrunk frame, the emulator drops it.
  // Ink's virtual frame now says "short"; the physical screen still shows the
  // status row.
  ink.render(short())
  ink.onRender()
  await tick()
  stdout.drain()

  const beforeFocus = term.text()

  peek(ink).handleTerminalFocusChange(true)
  await tick()

  const chunks = [...stdout.chunks]
  term.write(stdout.drain())
  ink.unmount()

  return { beforeFocus, afterFocus: term.text(), chunks }
}

describe.each([
  { altScreen: true, name: 'alt screen' },
  { altScreen: false, name: 'main screen (INLINE_MODE)' }
])('Ink focus recovery — $name', ({ altScreen }) => {
  it('clears the stale row and repaints the current frame', async () => {
    const { beforeFocus, afterFocus } = await focusRegain(altScreen)

    // Precondition: the physical screen really is stale before focus-in.
    expect(beforeFocus).toContain(STALE)

    expect(afterFocus).not.toContain(STALE)
    expect(afterFocus).toContain('hello')
    // The repaint must not duplicate content it just redrew.
    expect(afterFocus.match(/hello/g)).toHaveLength(1)
  })

  it('emits the clear in the same write as the repaint', async () => {
    const { chunks } = await focusRegain(altScreen)

    const eraseChunks = chunks.filter(c => c.includes(ERASE_SCREEN))

    expect(eraseChunks).toHaveLength(1)
    // Atomic: clear + content in one write, so no blank frame can be shown.
    expect(eraseChunks[0]).toContain('hello')
  })

  it('never erases scrollback (CSI 3J) on an ordinary focus regain', async () => {
    // Apple Terminal opts into a scrollback-deep erase, but only to clear
    // reflow artifacts after a RESIZE. A tab switch must not take the user's
    // history with it.
    const { chunks } = await focusRegain(altScreen, { TERM_PROGRAM: 'Apple_Terminal' })

    expect(chunks.join('')).not.toContain(ERASE_SCROLLBACK)
  })

  it('re-asserts terminal modes so mouse tracking survives a hidden pane', async () => {
    // An emulator that dropped the DEC mouse modes while hidden would
    // otherwise stay dead until the DECRQM watchdog's next probe. Mouse
    // tracking is alt-screen-scoped (reassertTerminalModes returns early on
    // main screen, where altScreenMouseTracking is always 'off'), so only
    // assert the re-arm where tracking exists.
    const { chunks } = await focusRegain(altScreen)

    if (altScreen) {
      expect(chunks.join('')).toContain(DISABLE_MOUSE_TRACKING)
    }
  })
})
