import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import stripAnsi from 'strip-ansi'
import { beforeEach, describe, expect, it } from 'vitest'

import type { FindOverlay } from '../app/interfaces.js'
import { $findFocus, $findScrollRestore, findSnippet, findTranscriptHits } from '../app/transcriptFindStore.js'
import { TranscriptFind } from '../components/transcriptFind.js'
import { DEFAULT_THEME } from '../theme.js'
import type { Msg } from '../types.js'

const rows: Msg[] = [
  { kind: 'intro', role: 'system', text: 'Hermes banner retry' },
  { role: 'user', text: 'why does the retry loop spin?' },
  { role: 'assistant', text: 'The backoff resets on every\nretry   loop pass.' },
  { role: 'user', text: 'fix it' },
  { role: 'assistant', text: 'Patched the RETRY loop.' },
  { kind: 'slash', role: 'system', text: '/find retry loop' }
]

const flush = (ms = 30) => new Promise(resolve => setTimeout(resolve, ms))

class FakeInput extends EventEmitter {
  chunks: string[] = []
  isRaw = false
  isTTY = true
  readableLength = 0
  ref() {}
  setEncoding() {}
  setRawMode() {}
  unref() {}

  read() {
    const next = this.chunks.shift() ?? null
    this.readableLength = this.chunks.length

    return next
  }

  send(chunk: string) {
    this.chunks.push(chunk)
    this.readableLength = this.chunks.length
    this.emit('readable')
  }
}

function mount(req: FindOverlay) {
  const stdin = new FakeInput()
  const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
  let closed = 0
  let frame = ''

  stdout.on('data', chunk => (frame += chunk.toString()))

  const instance = renderSync(
    React.createElement(TranscriptFind, { cols: 80, onClose: () => (closed += 1), req, t: DEFAULT_THEME }),
    { patchConsole: false, stdin: stdin as any, stdout: stdout as any }
  )

  const press = async (data: string) => {
    stdin.send(data)
    await flush(data === '\u001b' ? 80 : 30)
  }

  return { closed: () => closed, frame: () => stripAnsi(frame), instance, press }
}

beforeEach(() => {
  $findFocus.set(null)
  $findScrollRestore.set(false)
})

describe('/find transcript search', () => {
  it('matches case- and whitespace-insensitively across line breaks, never the banner or the /find echo', () => {
    expect(findTranscriptHits(rows, 'RETRY loop')).toEqual([1, 2, 4])
    expect(findTranscriptHits(rows, 'nothing here')).toEqual([])
    expect(findTranscriptHits(rows, '   ')).toEqual([])
    expect(findSnippet(rows[2]!.text, 'retry loop', 80)).toBe('The backoff resets on every retry loop pass.')
    expect(findSnippet(rows[2]!.text, 'retry loop', 20)).toBe('…every retry loop pas…')
  })

  it('selects the newest match, steps with ↑↓, and Esc asks the transcript to scroll back', async () => {
    const { closed, frame, instance, press } = mount({ getRows: () => rows, query: 'retry loop' })

    await flush()
    expect($findFocus.get()).toBe(4)
    expect(frame()).toContain('3/3')
    await press('\u001b[A')
    expect($findFocus.get()).toBe(2)
    await press('\u001b[A')
    await press('\u001b[A')
    expect($findFocus.get()).toBe(1)
    await press('x')
    expect($findFocus.get()).toBe(-1)
    expect(frame()).toContain('no matches')
    await press('\u007f')
    expect($findFocus.get()).toBe(4)
    await press('\u001b')
    await flush()
    expect(closed()).toBe(1)
    expect($findScrollRestore.get()).toBe(true)
    instance.unmount()
    expect($findFocus.get()).toBeNull()
  })
})
