import { describe, expect, it } from 'vitest'

import { patchUiState } from '../app/uiStore.js'
import { startPromptLiveSession } from '../app/useMainApp.js'

describe('startPromptLiveSession', () => {
  it('keeps the created target through a delayed model switch without publishing into the new focus', async () => {
    patchUiState({ sid: 'created' })
    let finish!: (value: { model: string }) => void
    const dispatched: unknown[] = []
    const notices: string[] = []

    const pending = startPromptLiveSession({
      dispatchSubmission: (text, destination) => dispatched.push({ text, destination }),
      maybeWarn: () => notices.push('warn'),
      modelArg: 'chosen',
      newLiveSession: async () => 'created',
      onModelSwitched: () => notices.push('model'),
      prompt: 'private prompt',
      rpc: async method =>
        method === 'session.resume'
          ? { revision: 8, execution_generation: 4 }
          : new Promise(resolve => {
              finish = resolve
            }),
      sys: text => notices.push(text)
    })

    await new Promise(resolve => setImmediate(resolve))
    patchUiState({ sid: 'other' })
    finish({ model: 'chosen' })
    await pending
    expect(dispatched).toEqual([{ text: 'private prompt', destination: expect.objectContaining({ sid: 'created' }) }])
    expect(notices).toEqual([])
  })

  it('starts a kept-live session with generated id/title, applies selected model, then dispatches the prompt', async () => {
    const calls: Array<[string, unknown]> = []

    const sid = await startPromptLiveSession({
      dispatchSubmission: prompt => calls.push(['dispatch', prompt]),
      maybeWarn: value => calls.push(['warn', value]),
      modelArg: 'kimi-k2.6 --provider ollama-cloud',
      newLiveSession: async (message, title) => {
        calls.push(['new', { message, title }])

        patchUiState({ sid: 'abc123' })

        return 'abc123'
      },
      onModelSwitched: (value, result) => calls.push(['model-switched', { result, value }]),
      prompt: '  Build the thing  ',
      rpc: async (method, params) => {
        calls.push(['rpc', { method, params }])

        return method === 'session.resume'
          ? { revision: 8, execution_generation: 4 }
          : { model: 'kimi-k2.6', warning: '' }
      },
      sys: text => calls.push(['sys', text])
    })

    expect(sid).toBe('abc123')
    expect(calls).toEqual([
      ['new', { message: 'new live session started', title: undefined }],
      ['rpc', { method: 'session.resume', params: { session_id: 'abc123' } }],
      [
        'rpc',
        {
          method: 'session.mutate',
          params: {
            session_id: 'abc123',
            request_id: expect.any(String),
            expected_revision: 8,
            expected_generation: 4,
            operation: 'model',
            payload: { model: 'kimi-k2.6', provider: 'ollama-cloud' }
          }
        }
      ],
      ['sys', 'model → kimi-k2.6'],
      ['warn', { model: 'kimi-k2.6', value: 'kimi-k2.6', warning: '' }],
      ['model-switched', { result: { model: 'kimi-k2.6', value: 'kimi-k2.6', warning: '' }, value: 'kimi-k2.6' }],
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

        patchUiState({ sid: 'abc123' })

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
