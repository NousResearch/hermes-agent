import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import chalk from 'chalk'
import React from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { MessageLine } from '../components/messageLine.js'
import { isUserPrompt, userPromptShade } from '../lib/userPromptShade.js'
import { DEFAULT_THEME, type Theme } from '../theme.js'
import type { Msg } from '../types.js'

const SHADE = '#2a2a33'
const SHADE_SGR = '48;2;42;42;51'

const colorLevel = chalk.level

beforeEach(() => {
  chalk.level = 3
})
afterEach(() => {
  chalk.level = colorLevel
})

const theme: Theme = { ...DEFAULT_THEME, color: { ...DEFAULT_THEME.color, userMessageBg: SHADE } }

async function renderFrame(msg: Msg): Promise<string> {
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

  const instance = renderSync(<MessageLine cols={60} msg={msg} t={theme} />, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  await new Promise(resolve => setImmediate(resolve))

  instance.unmount()
  instance.cleanup()

  return output
}

describe('user prompt shade bar', () => {
  it('applies only to prompts the user typed', () => {
    expect(isUserPrompt({ role: 'user', text: 'hi' })).toBe(true)
    expect(isUserPrompt({ kind: 'slash', role: 'user', text: '/help' })).toBe(false)
    expect(isUserPrompt({ role: 'assistant', text: 'hi' })).toBe(false)
    expect(userPromptShade({ role: 'assistant', text: 'hi' }, theme)).toEqual({})
    expect(userPromptShade({ role: 'user', text: 'hi' }, theme)).toEqual({ backgroundColor: SHADE, paddingY: 1 })
  })

  it('paints the theme shade behind a user prompt', async () => {
    expect(await renderFrame({ role: 'user', text: 'find the bug' })).toContain(SHADE_SGR)
  })

  it('leaves assistant replies on the terminal background', async () => {
    expect(await renderFrame({ role: 'assistant', text: 'found it' })).not.toContain(SHADE_SGR)
  })
})
