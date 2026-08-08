import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { MessageLine } from '../components/messageLine.js'
import { MAX_HISTORY } from '../config/limits.js'
import { toTranscriptMessages } from '../domain/messages.js'
import { capTranscriptHistory, stampHumanMessage, streamingAssistantMessage, upsert } from '../lib/messages.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

describe('toTranscriptMessages', () => {
  it('carries persisted epoch timestamps into display messages as milliseconds', () => {
    const result = toTranscriptMessages([{ role: 'user', text: 'hello', timestamp: 1_786_210_645 }])

    expect(result[0]?.timestamp).toBe(1_786_210_645_000)
  })

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

  it('skips hidden display_kind rows entirely', () => {
    const rows = [
      { role: 'user', text: 'visible prompt' },
      { role: 'user', text: '[CONTEXT COMPACTION — REFERENCE ONLY]', display_kind: 'hidden' },
      { role: 'assistant', text: 'visible reply' }
    ]

    const result = toTranscriptMessages(rows)
    expect(result.map(msg => msg.text)).toEqual(['visible prompt', 'visible reply'])
    expect(result.every(m => !m.text?.includes('COMPACTION'))).toBe(true)
  })

  it('projects model_switch as an event with replaced text', () => {
    const rows = [
      { role: 'user', text: 'hello' },
      { role: 'user', text: '[System: model changed to gpt-5]', display_kind: 'model_switch' },
      { role: 'assistant', text: 'hi' }
    ]

    const result = toTranscriptMessages(rows)
    expect(result.map(msg => [msg.kind, msg.role, msg.text])).toEqual([
      [undefined, 'user', 'hello'],
      ['event', 'system', 'model changed'],
      [undefined, 'assistant', 'hi']
    ])
  })

  it('projects async_delegation_complete with task_count metadata', () => {
    const rows = [
      { role: 'user', text: 'do work' },
      { role: 'assistant', text: 'done' },
      {
        role: 'user',
        text: '[IMPORTANT: delegation done]',
        display_kind: 'async_delegation_complete',
        display_metadata: { task_count: 3 }
      },
      { role: 'assistant', text: 'merged' }
    ]

    const result = toTranscriptMessages(rows)
    expect(result.map(msg => [msg.kind, msg.text])).toEqual([
      [undefined, 'do work'],
      [undefined, 'done'],
      ['event', '3 background agents finished'],
      [undefined, 'merged']
    ])
  })

  it('projects async_delegation_complete without metadata as generic text', () => {
    const rows = [{ role: 'user', text: 'event', display_kind: 'async_delegation_complete' }]

    const result = toTranscriptMessages(rows)
    expect(result[0]?.kind).toBe('event')
    expect(result[0]?.text).toBe('background agent work finished')
  })
})

describe('MessageLine', () => {
  it('visibly renders the configured timestamp for a human message', () => {
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

    const timestamp = new Date(2026, 7, 8, 15, 4, 5).getTime()

    const instance = renderSync(
      React.createElement(MessageLine, {
        cols: 80,
        msg: { role: 'user', text: 'Hello', timestamp },
        showTimestamps: true,
        t: DEFAULT_THEME,
        timestampFormat: '%Y-%m-%d %H:%M:%S'
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

    expect(stripAnsi(output)).toContain('2026-08-08 15:04:05')
  })

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
    expect(upsert([{ role: 'assistant', text: 'partial' }], 'assistant', 'full')[0]!.text).toBe('full')
  })

  it('appends to empty', () => {
    expect(upsert([], 'user', 'first')).toEqual([{ role: 'user', text: 'first' }])
  })

  it('does not mutate', () => {
    const prev = [{ role: 'user' as const, text: 'hi' }]
    upsert(prev, 'assistant', 'yo')
    expect(prev).toHaveLength(1)
  })
})

describe('stampHumanMessage', () => {
  it('stamps only user and assistant display rows without replacing stored timestamps', () => {
    const now = 1_786_165_445_000

    expect(stampHumanMessage({ role: 'user', text: 'prompt' }, now).timestamp).toBe(now)
    expect(stampHumanMessage({ role: 'assistant', text: 'reply', timestamp: 123 }, now).timestamp).toBe(123)
    expect(stampHumanMessage({ role: 'system', text: 'notice' }, now).timestamp).toBeUndefined()
    expect(stampHumanMessage({ role: 'tool', text: 'progress' }, now).timestamp).toBeUndefined()
    expect(stampHumanMessage({ role: 'assistant', text: '   ' }, now).timestamp).toBeUndefined()
  })

  it('gives a streamed assistant bubble one deterministic turn timestamp', () => {
    const timestamp = 1_786_165_445_000

    expect(streamingAssistantMessage('partial', timestamp, ['web_search'])).toEqual({
      role: 'assistant',
      text: 'partial',
      timestamp,
      tools: ['web_search']
    })
  })
})

describe('capTranscriptHistory', () => {
  it('keeps the intro and the newest bounded display rows', () => {
    const intro = { kind: 'intro' as const, role: 'system' as const, text: '' }
    const rows = Array.from({ length: 1_005 }, (_, index) => ({ role: 'user' as const, text: `m${index}` }))
    const capped = capTranscriptHistory([intro, ...rows])

    expect(capped).toHaveLength(MAX_HISTORY)
    expect(capped[0]).toBe(intro)
    expect(capped[1]?.text).toBe(`m${rows.length - (MAX_HISTORY - 1)}`)
    expect(capped.at(-1)?.text).toBe('m1004')
  })
})
