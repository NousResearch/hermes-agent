import { EventEmitter } from 'events'

import { renderSync } from '@hermes/ink'
import React, { useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { TextInput } from '../components/textInput.js'
import type { InputCursorSnapshot } from '../components/textInput.js'

// Regression coverage for the stale own-echo rewind (#111934): while a
// deferred key-burst flush is in flight, the parent's re-render can hand the
// TextInput the value it emitted BEFORE the user typed further characters.
// The `[value]` effect used to treat any non-equal incoming value as an
// external assignment, rewinding local keystrokes (cursor jumps backward,
// freshly typed letters vanish).
//
// Only setTimeout/setInterval/Date are faked — setImmediate stays real so
// React's scheduler still commits renders between ticks.

class FakeTty extends EventEmitter {
  chunks: string[] = []
  columns = 80
  rows = 24
  isTTY = true
  isRaw = false
  private pendingReads: string[] = []
  ref(): void {}
  unref(): void {}
  read(): string | null {
    return this.pendingReads.shift() ?? null
  }
  send(chunk: string): void {
    this.pendingReads.push(chunk)
    this.emit('readable')
  }
  setEncoding(): this {
    return this
  }
  setRawMode(mode: boolean): this {
    this.isRaw = mode

    return this
  }
  write(chunk: string | Uint8Array, cb?: (err?: Error | null) => void): boolean {
    this.chunks.push(typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8'))
    cb?.()

    return true
  }
}

const tick = () => new Promise<void>(resolve => setImmediate(resolve))

function Harness({
  onValue,
  setValueRef,
  snapshotRef
}: {
  onValue: (value: string) => void
  setValueRef?: React.MutableRefObject<((next: string) => void) | null>
  snapshotRef: React.RefObject<InputCursorSnapshot | null>
}) {
  const [value, setValue] = useState('')

  if (setValueRef) {
    // Stands in for history navigation / a slash completion replacing the draft.
    setValueRef.current = setValue
  }

  return React.createElement(TextInput, {
    cursorSnapshotRef: snapshotRef,
    onChange: (next: string) => {
      setValue(next)
      onValue(next)
    },
    value
  })
}

describe('stale parent own-echo during deferred key-burst flush', () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'setInterval', 'Date'] })
    // useStdout() resolves to process.stdout (not the FakeTty passed to
    // renderSync), so the fast-echo bypass has to be armed on the real stream.
    ;(process.stdout as { isTTY?: boolean }).isTTY = true
    vi.spyOn(process.stdout, 'write').mockImplementation(() => true)
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllEnvs()
    vi.restoreAllMocks()
    ;(process.stdout as { isTTY?: boolean }).isTTY = undefined
  })

  it('does not rewind local keystrokes when the parent echoes a value older than vRef', async () => {
    vi.stubEnv('TERM_PROGRAM', 'iTerm.app')
    vi.stubEnv('TMUX', '')

    const stdout = new FakeTty()
    const stdin = new FakeTty()
    const stderr = new FakeTty()
    const values: string[] = []
    const snapshotRef = useRefBridge()

    const instance = renderSync(React.createElement(Harness, { onValue: v => values.push(v), snapshotRef }), {
      patchConsole: false,
      stderr: stderr as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream
    })

    try {
      await tick()

      // 1. Type "a", "b", "c". "a" commits synchronously (fast-append needs a
      //    non-empty line); "b" and "c" ride the deferred 16ms key-burst path,
      //    so the parent still holds "a".
      for (const ch of ['a', 'b', 'c']) {
        stdin.send(ch)
        await tick()
      }

      // 2. Flush the burst: the parent receives "abc" and schedules a re-render.
      vi.advanceTimersByTime(16)

      // 3. "d" lands BEFORE that re-render commits — vRef is now "abcd" while
      //    the incoming echo will still say "abc".
      stdin.send('d')

      // 4. The echoed "abc" commits and the [value] effect runs.
      await tick()
      await tick()

      // 5. One more keystroke must survive too, not land on a rewound value.
      stdin.send('e')
      await tick()
      vi.advanceTimersByTime(16)
      await tick()
      await tick()
    } finally {
      instance.unmount()
      instance.cleanup()
    }

    // unmount published the final {cursor, value} into the snapshot ref.
    expect(values.at(-1)).toBe('abcde')
    expect(snapshotRef.current).toEqual({ cursor: 5, value: 'abcde' })
  })

  it('drops a pending key-burst flush when an external value replaces the draft', async () => {
    vi.stubEnv('TERM_PROGRAM', 'iTerm.app')
    vi.stubEnv('TMUX', '')

    const stdout = new FakeTty()
    const stdin = new FakeTty()
    const stderr = new FakeTty()
    const values: string[] = []
    const snapshotRef = useRefBridge()
    const setValueRef: React.MutableRefObject<((next: string) => void) | null> = { current: null }

    const instance = renderSync(
      React.createElement(Harness, { onValue: v => values.push(v), setValueRef, snapshotRef }),
      {
        patchConsole: false,
        stderr: stderr as unknown as NodeJS.WriteStream,
        stdin: stdin as unknown as NodeJS.ReadStream,
        stdout: stdout as unknown as NodeJS.WriteStream
      }
    )

    try {
      await tick()

      for (const ch of ['a', 'b', 'c']) {
        stdin.send(ch)
        await tick()
      }

      vi.advanceTimersByTime(16)
      await tick()
      await tick()

      // "d" arms a new deferred flush; history navigation then replaces the
      // draft before that flush fires.
      stdin.send('d')
      await tick()
      setValueRef.current?.('history draft')
      await tick()
      await tick()

      // The stale burst must not be handed to the parent 16ms later.
      vi.advanceTimersByTime(16)
      await tick()
      await tick()
    } finally {
      instance.unmount()
      instance.cleanup()
    }

    expect(values).toEqual(['a', 'abc'])
    expect(snapshotRef.current).toEqual({ cursor: 'history draft'.length, value: 'history draft' })
  })
})

// The snapshot ref is read after unmount in the assertion above; the helper
// keeps the harness a plain function component without hooks-order concerns.
function useRefBridge(): React.RefObject<InputCursorSnapshot | null> {
  return { current: null }
}
