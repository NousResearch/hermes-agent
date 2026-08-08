import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { MessageLine } from '../components/messageLine.js'
import { MAX_HISTORY } from '../config/limits.js'
import { toTranscriptMessages } from '../domain/messages.js'
import { capTranscriptHistory, upsert } from '../lib/messages.js'
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

  it('keeps reasoning on a resumed assistant turn that also has visible text', () => {
    const rows = [
      { role: 'user', text: 'explain qubits' },
      { role: 'assistant', text: 'A qubit holds superposition.', reasoning: 'Start from the classical bit.' }
    ]

    const result = toTranscriptMessages(rows)
    expect(result).toHaveLength(2)
    expect(result[1]?.role).toBe('assistant')
    expect(result[1]?.text).toBe('A qubit holds superposition.')
    expect(result[1]?.thinking).toBe('Start from the classical bit.')
    expect(result[1]?.thinkingTokens).toBeGreaterThan(0)
  })

  it('retains a reasoning-only assistant turn as a trail block instead of dropping it', () => {
    const rows = [
      { role: 'user', text: 'think first' },
      { role: 'assistant', text: '', reasoning_content: 'Weighing two approaches before answering.' },
      { role: 'assistant', text: 'Approach B.' }
    ]

    const result = toTranscriptMessages(rows)
    expect(result.map(msg => [msg.kind, msg.role, msg.text])).toEqual([
      [undefined, 'user', 'think first'],
      ['trail', 'system', ''],
      [undefined, 'assistant', 'Approach B.']
    ])
    expect(result[1]?.thinking).toBe('Weighing two approaches before answering.')
  })

  it('still drops a genuinely empty assistant row', () => {
    const rows = [
      { role: 'user', text: 'hi' },
      { role: 'assistant', text: '' },
      { role: 'assistant', text: '   ', reasoning: '', reasoning_details: [], codex_reasoning_items: [] },
      { role: 'assistant', text: 'hello' }
    ]

    expect(toTranscriptMessages(rows).map(msg => [msg.role, msg.text])).toEqual([
      ['user', 'hi'],
      ['assistant', 'hello']
    ])
  })

  it('flattens structured reasoning_details entries instead of stringifying them', () => {
    const rows = [
      {
        role: 'assistant',
        text: '',
        reasoning_details: [
          { type: 'reasoning.summary', summary: 'Checked the failing test first.' },
          { type: 'reasoning.encrypted_content', encrypted_content: 'AAAAB3NzaC1yc2E' },
          { type: 'reasoning.text', text: 'Then read the stack trace.' }
        ]
      }
    ]

    const result = toTranscriptMessages(rows)
    expect(result[0]?.thinking).toBe('Checked the failing test first.\n\nThen read the stack trace.')
    expect(result[0]?.thinking).not.toContain('object Object')
    expect(result[0]?.thinking).not.toContain('AAAAB3NzaC1yc2E')
  })

  it('reads Codex reasoning items through their summary parts and skips encrypted-only items', () => {
    const rows = [
      {
        role: 'assistant',
        text: 'done',
        codex_reasoning_items: [
          {
            type: 'reasoning',
            id: 'rs_a',
            encrypted_content: 'enc_blob_a',
            summary: [{ type: 'summary_text', text: 'Planned the edit.' }]
          },
          { type: 'reasoning', id: 'rs_b', encrypted_content: 'enc_blob_b' }
        ]
      }
    ]

    const result = toTranscriptMessages(rows)
    expect(result[0]?.thinking).toBe('Planned the edit.')
    expect(result[0]?.thinking).not.toContain('enc_blob')
  })

  it('does not double a Claude turn that reports the same thinking twice', () => {
    const rows = [
      {
        role: 'assistant',
        text: 'answer',
        reasoning: 'Let me re-read the contract.',
        reasoning_details: [
          { type: 'thinking', thinking: 'Let me re-read the contract.', signature: 'sig-abc' },
          { type: 'redacted_thinking', data: 'opaque-payload' }
        ]
      }
    ]

    expect(toTranscriptMessages(rows)[0]?.thinking).toBe('Let me re-read the contract.')
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
