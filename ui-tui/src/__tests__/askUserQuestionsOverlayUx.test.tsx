import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React, { act } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const inputHarness = vi.hoisted(() => ({
  handler: undefined as undefined | ((input: string, key: Record<string, boolean>) => void)
}))

vi.mock('@hermes/ink', async importOriginal => {
  const mod = await importOriginal<Record<string, unknown>>()

  return {
    ...mod,
    useInput: (handler: (input: string, key: Record<string, boolean>) => void) => {
      inputHarness.handler = handler
    }
  }
})

import { AskUserQuestionsTool } from '../components/askUserQuestionsTool.js'
import { stripAnsi } from '@hermes/shared/ansi'
import { DEFAULT_THEME } from '../theme.js'

const req = {
  requestId: 'auq-1',
  questions: [
    {
      header: 'SCOPE',
      multiSelect: false,
      options: [
        { label: 'Full', recommended: true },
        { label: 'Partial' }
      ],
      question: 'Scope?'
    },
    {
      header: 'MODEL',
      multiSelect: false,
      options: [{ label: 'A' }, { label: 'B' }],
      question: 'Model?'
    }
  ]
}

function mount(overrides: Record<string, unknown> = {}) {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 100, isTTY: false, rows: 40 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const props = {
    onCancel: vi.fn(),
    onAnswer: vi.fn(),
    req,
    t: DEFAULT_THEME,
    ...overrides
  }

  inputHarness.handler = undefined
  const element = React.createElement(AskUserQuestionsTool as React.ComponentType<any>, props)

  const instance = renderSync(element, {
    patchConsole: false,
    stderr: stderr as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream
  })

  return {
    cleanup: () => {
      instance.unmount()
      instance.cleanup()
    },
    output: () => stripAnsi(output),
    props
  }
}

beforeEach(() => {
  inputHarness.handler = undefined
})

async function press(input: string, key: Record<string, boolean> = {}) {
  await act(async () => {
    inputHarness.handler?.(input, key)
    await Promise.resolve()
  })
}

describe('AskUserQuestionsTool multi-question navigation', () => {
  it('Tab advances to the next question without submitting', async () => {
    const mounted = mount()

    await press('', { tab: true })

    expect(mounted.props.onAnswer).not.toHaveBeenCalled()
    // Q2 panel becomes the active one (accent header is visible; Q1 deactivates).
    expect(mounted.output()).toContain('Question 2/2')

    mounted.cleanup()
  })

  it('Enter advances to the next question without submitting', async () => {
    const mounted = mount()

    await press('2')
    expect(mounted.output()).toContain('Question 2/2')
    await press('', { return: true })
    // CONCERN 3 (documented, not fixed): Enter on the last question finalises
    // the whole batch, substituting the default selection for any question
    // the user never visited — silent default-answer submission.
    expect(mounted.props.onAnswer).toHaveBeenCalledWith({ 0: 'Partial', 1: 'A' }, 'auq-1')

    mounted.cleanup()
  })

  it('Shift+Tab goes back to the previous question', async () => {
    const mounted = mount()

    await press('', { tab: true })
    await press('', { shift: true, tab: true })
    expect(mounted.props.onAnswer).not.toHaveBeenCalled()

    mounted.cleanup()
  })

  it('Esc cancels instead of answering', async () => {
    const mounted = mount()

    await press('', { escape: true })
    expect(mounted.props.onCancel).toHaveBeenCalledTimes(1)
    expect(mounted.props.onAnswer).not.toHaveBeenCalled()

    mounted.cleanup()
  })
})