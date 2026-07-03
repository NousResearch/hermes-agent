import { describe, expect, it } from 'vitest'

import { startPromptLiveSession } from '../app/useMainApp.js'

describe('startPromptLiveSession', () => {
  it('starts a kept-live session with generated id/title, applies selected model, then dispatches the prompt', async () => {
    const calls: Array<[string, unknown]> = []

    const sid = await startPromptLiveSession({
      dispatchSubmission: prompt => calls.push(['dispatch', prompt]),
      maybeWarn: value => calls.push(['warn', value]),
      modelArg: 'kimi-k2.6 --provider ollama-cloud',
      newLiveSession: async (message, title) => {
        calls.push(['new', { message, title }])

        return 'abc123'
      },
      onModelSwitched: (value, result) => calls.push(['model-switched', { result, value }]),
      prompt: '  Build the thing  ',
      rpc: async (method, params) => {
        calls.push(['rpc', { method, params }])

        return { value: 'kimi-k2.6', warning: '' }
      },
      sys: text => calls.push(['sys', text])
    })

    expect(sid).toBe('abc123')
    expect(calls).toEqual([
      ['new', { message: 'new live session started', title: undefined }],
      [
        'rpc',
        {
          method: 'config.set',
          params: { key: 'model', session_id: 'abc123', value: 'kimi-k2.6 --provider ollama-cloud' }
        }
      ],
      ['sys', 'model → kimi-k2.6'],
      ['warn', { value: 'kimi-k2.6', warning: '' }],
      ['model-switched', { result: { value: 'kimi-k2.6', warning: '' }, value: 'kimi-k2.6' }],
      ['dispatch', 'Build the thing']
    ])
  })

  it('does not start a session for an empty prompt', async () => {
    const calls: string[] = []

    const sid = await startPromptLiveSession({
      dispatchSubmission: () => calls.push('dispatch'),
      maybeWarn: () => calls.push('warn'),
      newLiveSession: async () => {
        calls.push('new')

        return 'abc123'
      },
      prompt: '   ',
      rpc: async () => ({ value: 'unused' }),
      sys: () => calls.push('sys')
    })

    expect(sid).toBeNull()
    expect(calls).toEqual([])
  })
})
