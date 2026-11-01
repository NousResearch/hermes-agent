import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import stripAnsi from 'strip-ansi'
import { expect, it } from 'vitest'

import { ClarifyPrompt } from '../components/prompts.js'
import { DEFAULT_THEME } from '../theme.js'

// Minimal raw-mode TTY stdin: the prompt's useInput needs setRawMode to mount.
class FakeInput extends EventEmitter {
  isRaw = false
  isTTY = true
  readableLength = 0
  read = () => null
  ref = () => {}
  setEncoding = () => {}
  setRawMode = (enabled: boolean) => {
    this.isRaw = enabled
  }
  unref = () => {}
}

const paint = (node: React.ReactElement) => {
  const stdout = Object.assign(new PassThrough(), { columns: 100, isTTY: false, rows: 30 })
  const frames: string[] = []
  stdout.on('data', chunk => frames.push(chunk.toString()))

  const view = renderSync(node, {
    patchConsole: false,
    stdin: new FakeInput() as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream
  })

  view.unmount()
  view.cleanup()

  return stripAnsi(frames.join(''))
}

it('renders clarify questions and choices as Markdown, not literal markers', () => {
  const req = {
    questions: [
      {
        choices: ['**Rebase** onto `main`', 'Merge'],
        qid: 'q1',
        question: 'Which **branch strategy**\nfor `feature/x`?'
      }
    ],
    requestId: 'r1'
  }

  const out = paint(
    <ClarifyPrompt onCancel={() => {}} onQuestionAnswer={() => {}} req={req as never} t={DEFAULT_THEME} />
  )

  expect(out).toContain('Which branch strategy')
  expect(out).toContain('for feature/x?')
  expect(out).toContain('1. Rebase onto main')
  expect(out).not.toContain('**')
  expect(out).not.toContain('`')
})
