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

vi.mock('../components/textInput.js', () => ({ TextInput: () => null }))

import { ClarifyPrompt } from '../components/prompts.js'
import { stripAnsi } from '@hermes/shared/ansi'
import { DEFAULT_THEME } from '../theme.js'

const req = {
  answers: {},
  choices: null,
  question: '',
  requestId: 'req-1',
  questions: [
    { choices: ['A', 'B'], multiSelect: false, qid: 'q0', question: 'First?' },
    { choices: ['X', 'Y'], multiSelect: false, qid: 'q1', question: 'Second?' }
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
    cols: 100,
    onAnswer: vi.fn(),
    onBatchCancel: vi.fn(),
    onBatchSubmit: vi.fn(),
    onCancel: vi.fn(),
    onQuestionAnswer: vi.fn(),
    req,
    t: DEFAULT_THEME,
    ...overrides
  }

  inputHarness.handler = undefined
  const element = React.createElement(ClarifyPrompt as React.ComponentType<any>, props)

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

describe('ClarifyPrompt shared batch UX', () => {
  it('number keys select only and one final Enter submits every answer', async () => {
    const mounted = mount()

    await press('2')
    expect(mounted.props.onQuestionAnswer).not.toHaveBeenCalled()
    expect(mounted.props.onBatchSubmit).not.toHaveBeenCalled()

    await press('', { return: true })
    expect(mounted.props.onQuestionAnswer).not.toHaveBeenCalled()
    expect(mounted.props.onBatchSubmit).not.toHaveBeenCalled()

    await press('2')
    await press('', { return: true })
    expect(mounted.props.onBatchSubmit).not.toHaveBeenCalled()
    expect(mounted.output()).toContain('Review answers')

    await press('', { return: true })
    expect(mounted.props.onBatchSubmit).toHaveBeenCalledWith({ q0: 'B', q1: 'Y' })
    mounted.cleanup()
  })

  it('supports checkbox multi-select and submits a JSON array answer', async () => {
    const mounted = mount({
      req: {
        ...req,
        questions: [{ choices: ['API', 'UI'], multiSelect: true, qid: 'q0', question: 'Areas?' }]
      }
    })

    await press('1')
    await press(' ')
    await press('2')
    await press(' ')
    await press('', { return: true })
    await press('', { return: true })

    expect(mounted.props.onBatchSubmit).toHaveBeenCalledWith({
      q0: JSON.stringify(['API', 'UI'])
    })
    mounted.cleanup()
  })

  it('re-enables final submit after a transient RPC failure', async () => {
    const onBatchSubmit = vi.fn().mockRejectedValueOnce(new Error('offline')).mockResolvedValueOnce(undefined)

    const mounted = mount({
      onBatchSubmit,
      req: {
        ...req,
        questions: [{ choices: ['A', 'B'], multiSelect: false, qid: 'q0', question: 'Only?' }]
      }
    })

    await press('', { return: true })
    await press('', { return: true })
    expect(onBatchSubmit).toHaveBeenCalledTimes(1)

    await press('', { return: true })
    expect(onBatchSubmit).toHaveBeenCalledTimes(2)
    mounted.cleanup()
  })

  it('cancel returns staged answers instead of discarding the batch', async () => {
    const mounted = mount()

    await press('', { return: true })
    await press('', { escape: true })

    expect(mounted.props.onBatchCancel).toHaveBeenCalledWith({ q0: 'A' })
    mounted.cleanup()
  })

  it('keeps inactive choices collapsed and exposes deadline context', () => {
    const mounted = mount({
      req: { ...req, expiresAt: Date.now() / 1000 + 120 }
    })

    const output = mounted.output()

    expect(output).toContain('First?')
    expect(output).toContain('Second?')
    expect(output).toContain('A')
    expect(output).toContain('B')
    expect(output).not.toContain('1. X')
    expect(output).toContain('expires in')
    mounted.cleanup()
  })
})
