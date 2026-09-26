import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import { stripAnsi } from '@hermes/shared/ansi'
import chalk from 'chalk'
import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { Thinking } from '../components/thinking.js'
import { DEFAULT_THEME } from '../theme.js'

const colorLevel = chalk.level

beforeEach(() => {
  chalk.level = 3
  // Force real ANSI dim; several terminals (VTE, Apple Terminal) remap dim
  // to a fallback color instead of emitting SGR 2.
  vi.stubEnv('HERMES_TUI_DIM', '1')
})

afterEach(() => {
  chalk.level = colorLevel
  vi.unstubAllEnvs()
})

const flushEffects = async () => {
  for (let i = 0; i < 10; i++) {
    await new Promise(resolve => setTimeout(resolve, 5))
  }
}

const renderThinking = async (props: { mode?: 'full' | 'truncated'; reasoning: string }) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 60, isTTY: false, rows: 20 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(
    <Thinking branch="last" mode={props.mode ?? 'truncated'} rails={[]} reasoning={props.reasoning} t={DEFAULT_THEME} />,
    {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    }
  )

  await flushEffects()

  instance.unmount()
  instance.cleanup()

  return output
}

// The dim/italic codes must wrap the thinking body text itself, not just
// appear anywhere (the tree rails already carry their own dim codes).
const DIM_ITALIC_BODY = /ESC\[2m[\s\S]*?ESC\[3m(?:ESC\[[0-9;]*m)*thinking-body-dim-marker[\s\S]*?ESC\[22m/

describe('Thinking body renders dim + italic', () => {
  it('truncated mode: thinking body carries dim and italic styling', async () => {
    const output = await renderThinking({ mode: 'truncated', reasoning: 'thinking-body-dim-marker' })

    expect(stripAnsi(output)).toContain('thinking-body-dim-marker')
    expect(output.replaceAll('', 'ESC')).toMatch(DIM_ITALIC_BODY)
  })

  it('full mode: thinking body carries dim and italic styling', async () => {
    const output = await renderThinking({ mode: 'full', reasoning: 'thinking-body-dim-marker' })

    expect(stripAnsi(output)).toContain('thinking-body-dim-marker')
    expect(output.replaceAll('', 'ESC')).toMatch(DIM_ITALIC_BODY)
  })
})
