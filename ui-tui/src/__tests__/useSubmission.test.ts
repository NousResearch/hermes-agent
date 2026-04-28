import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('react', () => ({
  useCallback: <T extends (...args: any[]) => any>(fn: T) => fn,
  useEffect: () => undefined,
  useRef: <T>(current: T) => ({ current })
}))

import { useSubmission } from '../app/useSubmission.js'
import { resetUiState } from '../app/uiStore.js'

const waitMicrotasks = () => new Promise(resolve => setTimeout(resolve, 0))

describe('useSubmission', () => {
  beforeEach(() => {
    resetUiState()
  })

  it('creates a session and submits instead of queueing the first prompt', async () => {
    const enqueue = vi.fn()
    const clearIn = vi.fn()
    const pushHistory = vi.fn()
    const ensureSession = vi.fn(async () => 'sid-1')
    const gw = {
      request: vi.fn((method: string) => {
        if (method === 'input.detect_drop') {
          return Promise.resolve({ matched: false })
        }

        return Promise.resolve({ status: 'streaming' })
      })
    }

    const { dispatchSubmission } = useSubmission({
      appendMessage: vi.fn(),
      composerActions: {
        clearIn,
        dequeue: vi.fn(),
        enqueue,
        pushHistory,
        replaceQueue: vi.fn(),
        setInput: vi.fn(),
        setInputBuf: vi.fn(),
        setQueueEdit: vi.fn(),
        syncQueue: vi.fn()
      } as any,
      composerRefs: {
        queueEditRef: { current: null },
        queueRef: { current: [] }
      } as any,
      composerState: {
        compIdx: 0,
        compReplace: 0,
        completions: [],
        inputBuf: [],
        pasteSnips: []
      } as any,
      ensureSession,
      gw: gw as any,
      maybeGoodVibes: vi.fn(),
      setLastUserMsg: vi.fn(),
      slashRef: { current: vi.fn() },
      submitRef: { current: vi.fn() },
      sys: vi.fn()
    })

    dispatchSubmission('hello')
    await waitMicrotasks()
    await waitMicrotasks()

    expect(clearIn).toHaveBeenCalled()
    expect(pushHistory).toHaveBeenCalledWith('hello')
    expect(enqueue).not.toHaveBeenCalled()
    expect(ensureSession).toHaveBeenCalledTimes(1)
    expect(gw.request).toHaveBeenCalledWith('input.detect_drop', { session_id: 'sid-1', text: 'hello' })
    expect(gw.request).toHaveBeenCalledWith('prompt.submit', { session_id: 'sid-1', text: 'hello' })
  })
})
