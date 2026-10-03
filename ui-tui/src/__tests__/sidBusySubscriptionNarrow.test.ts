import { afterEach, describe, expect, it } from 'vitest'

import { $uiBusy, $uiSessionId, patchUiState, resetUiState } from '../app/uiStore.js'

afterEach(resetUiState)

const unrelatedPatches = (): Array<Parameters<typeof patchUiState>[0]> => [
  { status: 'thinking…' },
  { status: 'summoning hermes…' },
  { compacting: true },
  { compacting: false },
  { streaming: false },
  { streaming: true },
  { timestamps: true },
  { timestamps: false },
  { compact: true },
  { compact: false },
  { focusView: true },
  { focusView: false },
  { showReasoning: true },
  { showReasoning: false },
  { liveSessionCount: 2 },
  { liveSessionCount: 0 },
  { sessionTitle: 't' },
  { sessionTitle: '' },
  { statusBar: 'bottom' },
  { statusBar: 'top' }
]

describe('sid/busy subscription narrow (#99773 deferred slice)', () => {
  it('20 unrelated $uiState writes notify $uiSessionId zero times when sid is stable', () => {
    patchUiState({ sid: 'sess-a', busy: false })
    let hits = 0
    const unsub = $uiSessionId.listen(() => {
      hits += 1
    })

    try {
      for (const patch of unrelatedPatches()) {
        patchUiState(patch)
      }
      expect(hits).toBe(0)
      expect($uiSessionId.get()).toBe('sess-a')
    } finally {
      unsub()
    }
  })

  it('20 unrelated $uiState writes notify $uiBusy zero times when busy is stable', () => {
    patchUiState({ sid: 'sess-a', busy: true })
    let hits = 0
    const unsub = $uiBusy.listen(() => {
      hits += 1
    })

    try {
      for (const patch of unrelatedPatches()) {
        patchUiState(patch)
      }
      expect(hits).toBe(0)
      expect($uiBusy.get()).toBe(true)
    } finally {
      unsub()
    }
  })

  it('sid change still notifies $uiSessionId exactly once', () => {
    patchUiState({ sid: 'sess-a' })
    let hits = 0
    const unsub = $uiSessionId.listen(() => {
      hits += 1
    })

    try {
      patchUiState({ sid: 'sess-b' })
      expect(hits).toBe(1)
      expect($uiSessionId.get()).toBe('sess-b')
    } finally {
      unsub()
    }
  })

  it('busy change still notifies $uiBusy exactly once', () => {
    patchUiState({ busy: false })
    let hits = 0
    const unsub = $uiBusy.listen(() => {
      hits += 1
    })

    try {
      patchUiState({ busy: true })
      expect(hits).toBe(1)
      expect($uiBusy.get()).toBe(true)
    } finally {
      unsub()
    }
  })
})
