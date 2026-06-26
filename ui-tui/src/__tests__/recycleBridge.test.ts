import { afterEach, describe, expect, it } from 'vitest'

import { canRecycle, registerRecycleHandler, triggerRecycle } from '../lib/recycleBridge.js'

describe('recycleBridge (Stage 1 recycle guard + dispatch)', () => {
  afterEach(() => {
    // Clear any registered handler between tests by registering+unregistering.
    const off = registerRecycleHandler(() => {})
    off()
  })

  describe('canRecycle', () => {
    it('is true in attach mode (HERMES_TUI_GATEWAY_URL set)', () => {
      expect(canRecycle({ HERMES_TUI_GATEWAY_URL: 'ws://127.0.0.1:9/api/ws' } as NodeJS.ProcessEnv)).toBe(true)
    })
    it('is false in spawned-gateway mode (no attach url)', () => {
      expect(canRecycle({} as NodeJS.ProcessEnv)).toBe(false)
    })
    it('is false for an empty/whitespace attach url', () => {
      expect(canRecycle({ HERMES_TUI_GATEWAY_URL: '   ' } as NodeJS.ProcessEnv)).toBe(false)
    })
  })

  describe('triggerRecycle', () => {
    it('fires the handler and returns true in attach mode', () => {
      let fired = 0
      registerRecycleHandler(() => {
        fired++
      })
      const ok = triggerRecycle({ HERMES_TUI_GATEWAY_URL: 'ws://x/api/ws' } as NodeJS.ProcessEnv)
      expect(ok).toBe(true)
      expect(fired).toBe(1)
    })

    it('does NOT fire (returns false) in spawned-gateway mode — exiting would kill the session', () => {
      let fired = 0
      registerRecycleHandler(() => {
        fired++
      })
      const ok = triggerRecycle({} as NodeJS.ProcessEnv)
      expect(ok).toBe(false)
      expect(fired).toBe(0)
    })

    it('returns false when no handler is registered', () => {
      const off = registerRecycleHandler(() => {})
      off() // unregister
      expect(triggerRecycle({ HERMES_TUI_GATEWAY_URL: 'ws://x/api/ws' } as NodeJS.ProcessEnv)).toBe(false)
    })

    it('unregister only clears the matching handler', () => {
      let aFired = 0
      const offA = registerRecycleHandler(() => {
        aFired++
      })
      let bFired = 0
      registerRecycleHandler(() => {
        bFired++
      })
      // offA should be a no-op now (b is the active handler).
      offA()
      triggerRecycle({ HERMES_TUI_GATEWAY_URL: 'ws://x/api/ws' } as NodeJS.ProcessEnv)
      expect(aFired).toBe(0)
      expect(bFired).toBe(1)
    })
  })
})
