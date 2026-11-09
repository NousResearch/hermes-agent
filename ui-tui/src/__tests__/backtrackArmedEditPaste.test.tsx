import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { $pendingRewind } from '../app/backtrackStore.js'
import { createSlashHandler } from '../app/createSlashHandler.js'
import type { ComposerToken } from '../app/interfaces.js'
import { $overlayState, resetOverlayState } from '../app/overlayStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { useSubmission } from '../app/useSubmission.js'
import { pasteTokenLabel } from '../lib/text.js'

const flush = () => new Promise(resolve => setImmediate(resolve))

const rows = [
  { role: 'user', content: 'ALPHA please', text: 'ALPHA please', row_id: 11 },
  { role: 'assistant', content: 'ALPHA', text: 'ALPHA', row_id: 12 },
  { role: 'user', content: 'BETA please', text: 'BETA please', row_id: 13 },
  { role: 'assistant', content: 'BETA', text: 'BETA', row_id: 14 }
]

afterEach(() => {
  $pendingRewind.set(null)
  resetOverlayState()
  resetUiState()
})

// The real submit path: `/rewind` arms the edit, the composer holds a paste chip and an image chip,
// and Enter goes through useSubmission.dispatchSubmission, whose clearIn() empties the tokens before
// the async rewind finishes.
it('an armed edit holding a paste and an image sends the pasted text and the attachment', async () => {
  patchUiState({ sid: 'owner', busy: false })

  const request = vi.fn(async (method: string, params: any) => {
    if (method === 'session.resume') {
      return { session_id: 'owner', revision: 7, execution_generation: 3, messages: rows, info: { model: 'm' } }
    }

    if (method === 'session.mutate' && params.operation === 'rewind') {
      return { session_id: 'owner', operation: 'rewind', revision: 8, execution_generation: 4, rewound_count: 2 }
    }

    throw new Error(`unexpected RPC: ${method}`)
  })

  const gw = { isCanonical: true, request } as any
  const send = vi.fn()

  const slash = createSlashHandler({
    slashFlightRef: { current: 0 },
    gateway: { gw, rpc: request },
    transcript: { page: vi.fn(), panel: vi.fn(), send, sys: vi.fn(), setHistoryItems: vi.fn(), trimLastExchange: (i: unknown[]) => i },
    local: { getHistoryItems: () => [], getLastUserMsg: () => '', maybeWarn: vi.fn() },
    session: { newSession: vi.fn(), resumeById: vi.fn() },
    composer: { enqueue: vi.fn(), setInput: vi.fn() },
    voice: {}
  } as any)

  slash('/rewind')
  await flush()
  const browse = $overlayState.get().rewind!

  browse.onEdit(browse.targets[1]!)
  expect($pendingRewind.get()?.target.rowId).toBe(13)

  const body = Array.from({ length: 7 }, (_, i) => `pasted line ${i + 1}`).join('\n')
  const label = pasteTokenLabel(body, 7)

  const tokensRef: { current: ComposerToken[] } = {
    current: [
      { kind: 'paste', label, text: body },
      { index: 1, kind: 'image', label: '[[ image #1 ]]', mime: 'image/png', path: '/shots/a.png' } as ComposerToken
    ]
  }

  let submission!: ReturnType<typeof useSubmission>

  const noop = () => {}

  function Harness() {
    submission = useSubmission({
      appendMessage: noop,
      composerActions: { clearIn: () => { tokensRef.current = [] }, enqueue: noop, pushHistory: noop } as any,
      composerRefs: { queueEditRef: { current: null }, queueRef: { current: [] }, tokensRef } as any,
      composerState: { completions: [], input: '', inputBuf: [] } as any,
      gw,
      setLastUserMsg: noop,
      slashRef: { current: () => true },
      submitRef: { current: noop },
      sys: noop
    })

    return <Text>x</Text>
  }

  const instance = renderSync(<Harness />, {
    stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 20 }) as any,
    stderr: new PassThrough() as any,
    patchConsole: false
  })

  try {
    const display = `BETA again with ${label} and [[ image #1 ]]`

    submission.dispatchSubmission(display)
    expect(tokensRef.current).toEqual([]) // cleared before the rewind answers
    await flush()
    await flush()
    await flush()

    const rewinds = request.mock.calls.filter(([method]) => method === 'session.mutate')

    expect(rewinds).toHaveLength(1)
    expect(rewinds[0]![1]).toMatchObject({ payload: { target_message_id: 13 } })
    expect(send).toHaveBeenCalledTimes(1)
    const [text, , shown, , opts] = send.mock.calls[0]!

    expect(text).toBe(`BETA again with ${body} and`) // the image travels as an attachment
    expect(text).not.toContain('[[ pasted')
    expect(shown).toBe(display)
    expect(opts).toMatchObject({ attachments: [{ mime: 'image/png', path: '/shots/a.png' }] })
  } finally {
    instance.unmount()
  }
})
