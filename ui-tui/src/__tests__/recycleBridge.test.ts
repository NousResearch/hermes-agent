import { afterEach, describe, expect, it } from 'vitest'

import { canRecycle, registerRecycleHandler, triggerRecycle, RECYCLE_EXIT_CODE } from '../lib/recycleBridge.js'

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

  describe('RECYCLE_EXIT_CODE (recycle-vs-quit contract)', () => {
    it('is 97 — the exact code the orchestrator maps to a deliberate recycle', () => {
      // Single source of truth shared with tui_gateway/orchestrator.py's
      // RECYCLE_EXIT_CODE. A drift here silently reverts the disambiguation:
      // the supervisor would read a recycle as an unknown crash (or, if it
      // collided with 0, as a user /quit and tear the session down).
      expect(RECYCLE_EXIT_CODE).toBe(97)
    })

    it('is distinct from a user /quit (0) so recycle is never read as a quit', () => {
      expect(RECYCLE_EXIT_CODE).not.toBe(0)
    })

    it('sits clear of the shell (0-2) and signal (126-165) exit ranges', () => {
      expect(RECYCLE_EXIT_CODE).toBeGreaterThan(2)
      expect(RECYCLE_EXIT_CODE).toBeLessThan(126)
    })
  })
})
