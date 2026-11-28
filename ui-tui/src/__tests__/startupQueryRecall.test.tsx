import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { patchUiState, resetUiState } from '../app/uiStore.js'
import { useComposerState } from '../app/useComposerState.js'
import { useSubmission } from '../app/useSubmission.js'

const history = vi.hoisted(() => ({ items: ['an older prompt from a previous session'] }))

vi.mock('../lib/history.js', () => ({
  load: () => history.items,
  append: (line: string) => void history.items.push(line.trim())
}))

it('the startup -q prompt is the first entry Up recalls', async () => {
  resetUiState()
  patchUiState({ sid: 'owner' })
  const request = vi.fn(async (_method: string, _params?: unknown) => ({ status: 'started' }))
  const gw = { request, isCanonical: true } as any
  const submitRef = { current: (_value: string) => {} }
  const slashRef = { current: (_value: string) => false }
  let composer!: ReturnType<typeof useComposerState>
  let submission!: ReturnType<typeof useSubmission>

  function Harness() {
    composer = useComposerState({ gw, submitRef, sys: vi.fn() })
    submission = useSubmission({ gw, submitRef, slashRef, sys: vi.fn(), composerActions: composer.actions,
      composerRefs: composer.refs, composerState: composer.state, appendMessage: vi.fn(), setLastUserMsg: vi.fn() })

    return <Text>{composer.state.input}</Text>
  }

  const instance = renderSync(<Harness />, {
    stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false }) as any,
    stderr: new PassThrough() as any,
    patchConsole: false
  })

  try {
    submission.submitLiteral('Reply with only the word: hi')
    await vi.waitFor(() => expect(request.mock.calls.some(([method]) => method === 'prompt.submit')).toBe(true))
    expect(composer.refs.historyRef.current.at(-1)).toBe('Reply with only the word: hi')
  } finally {
    instance.unmount()
    instance.cleanup()
    resetUiState()
  }
})
