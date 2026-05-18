import { renderSync } from '@hermes/ink'
import React from 'react'
import { PassThrough } from 'stream'
import { describe, expect, it } from 'vitest'

import { effortLabel, modelLabel, StatusRule } from '../components/appChrome.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

const renderStatusRule = (statusBarSegments: readonly string[]) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 120, isTTY: false, rows: 24 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(
    React.createElement(StatusRule, {
      bgCount: 0,
      busy: false,
      cols: 120,
      cwdLabel: '~/workspace',
      model: 'gpt-5.5',
      modelFast: false,
      modelReasoningEffort: 'medium',
      sessionStartedAt: null,
      showCost: false,
      status: 'ready',
      statusBarSegments,
      statusColor: DEFAULT_THEME.color.statusFg,
      t: DEFAULT_THEME,
      turnStartedAt: null,
      usage: {
        context_max: 272_000,
        context_percent: 0,
        context_used: 0,
        total: 0
      },
      voiceLabel: 'voice off'
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

  return stripAnsi(output)
}

describe('status bar model label', () => {
  it('keeps medium reasoning visible as @med', () => {
    expect(modelLabel('gpt-5.5', 'medium')).toBe('gpt 5.5@med')
  })

  it('shortens minimal reasoning and keeps explicit efforts adjacent to the model', () => {
    expect(modelLabel('openai/gpt-5.5', 'minimal')).toBe('gpt 5.5@min')
    expect(modelLabel('openai/gpt-5.5', 'xhigh')).toBe('gpt 5.5@xhigh')
  })

  it('omits only neutral/default effort labels', () => {
    expect(effortLabel('default')).toBe('')
    expect(effortLabel('normal')).toBe('')
    expect(modelLabel('gpt-5.5')).toBe('gpt 5.5')
  })

  it('keeps fast as a separate suffix after the effort', () => {
    expect(modelLabel('gpt-5.5', 'medium', true)).toBe('gpt 5.5@med fast')
  })
})

describe('StatusRule segment rendering', () => {
  it('can hide only the context meter while keeping tokens and percent', () => {
    const output = renderStatusRule(['indicator', 'model', 'context_tokens', 'context_percent', 'voice'])

    expect(output).toContain('ready')
    expect(output).toContain('gpt 5.5@med')
    expect(output).toContain('0/272k')
    expect(output).toContain('0%')
    expect(output).toContain('voice off')
    expect(output).not.toContain('[░░░░░░░░░░]')
  })

  it('renders the context meter only when context_bar is enabled', () => {
    const output = renderStatusRule(['context_tokens', 'context_bar', 'context_percent'])

    expect(output).toContain('0/272k')
    expect(output).toContain('[░░░░░░░░░░] 0%')
  })
})
