import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { liveSessionInflightMessages } from '../app/useSessionLifecycle.js'
import { MessageLine } from '../components/messageLine.js'
import { toTranscriptMessages } from '../domain/messages.js'
import { upsert } from '../lib/messages.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

describe('toTranscriptMessages', () => {
  it('preserves assistant tool-call rows so resume does not drop prior turns', () => {
    const rows = [
      { role: 'user', text: 'first prompt' },
      { role: 'tool', context: 'repo', name: 'search_files', text: 'ignored raw result' },
      { role: 'assistant', text: 'first answer' },
      { role: 'user', text: 'second prompt' }
    ]

    expect(toTranscriptMessages(rows).map(msg => [msg.role, msg.text])).toEqual([
      ['user', 'first prompt'],
      ['assistant', 'first answer'],
      ['user', 'second prompt']
    ])
    expect(toTranscriptMessages(rows)[1]?.tools?.[0]).toContain('Search Files')
  })

  it('assigns stable ids to resumed transcript rows', () => {
    const rows = [
      { role: 'user', text: 'first prompt' },
      { role: 'assistant', text: 'first answer' },
      { role: 'user', text: 'second prompt' }
    ]

    const messages = toTranscriptMessages(rows)

    expect(messages.every(msg => typeof msg.id === 'string' && msg.id.length > 0)).toBe(true)
    expect(new Set(messages.map(msg => msg.id)).size).toBe(messages.length)
  })
})

describe('liveSessionInflightMessages', () => {
  it('assigns ids to inflight user rows', () => {
    expect(liveSessionInflightMessages({ assistant: '', streaming: false, user: 'resume me' })[0]?.id).toMatch(/^msg-/)
  })
})

describe('MessageLine', () => {
  it('preserves a separator after compound user prompt glyphs in transcript rows', () => {
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

    const t = {
      ...DEFAULT_THEME,
      brand: { ...DEFAULT_THEME.brand, prompt: 'Ψ >' }
    }

    const instance = renderSync(
      React.createElement(MessageLine, {
        cols: 80,
        msg: { role: 'user', text: 'Okay' },
        t
      }),
      {
        patchConsole: false,
        stderr: stderr as NodeJS.WriteStream,
        stdin: stdin as NodeJS.ReadStream,
        stdout: stdout as NodeJS.WriteStream
      }
    )

    instance.unmount()
    instance.cleanup()

    const renderedLine = stripAnsi(output)
      .split('\n')
      .find(line => line.includes('Okay'))

    expect(renderedLine).toContain('Ψ > Okay')
  })
})

describe('upsert', () => {
  it('appends when last role differs', () => {
    expect(upsert([{ role: 'user', text: 'hi' }], 'assistant', 'hello')).toHaveLength(2)
  })

  it('replaces when last role matches', () => {
    const out = upsert([{ id: 'msg-keep', role: 'assistant', text: 'partial' }], 'assistant', 'full')

    expect(out[0]!.text).toBe('full')
    expect(out[0]!.id).toBe('msg-keep')
  })

  it('appends to empty', () => {
    const out = upsert([], 'user', 'first')

    expect(out[0]).toMatchObject({ role: 'user', text: 'first' })
    expect(out[0]!.id).toMatch(/^msg-/)
  })

  it('does not mutate', () => {
    const prev = [{ role: 'user' as const, text: 'hi' }]
    upsert(prev, 'assistant', 'yo')
    expect(prev).toHaveLength(1)
  })
})
