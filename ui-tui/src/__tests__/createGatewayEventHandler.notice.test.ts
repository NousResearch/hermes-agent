import { beforeEach, describe, expect, it, vi } from 'vitest'

import { resetConnectionOperationsForTests } from '../app/connectionOperationStore.js'
import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { resetOverlayState } from '../app/overlayStore.js'
import { resetServerRequestsForTests } from '../app/serverRequestStore.js'
import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import type { Msg } from '../types.js'

// Split out of createGatewayEventHandler.test.ts (file-size budget); the mock,
// ctx builder and beforeEach are duplicated from there unchanged.
const openExternalUrlMock = vi.fn((_url: string) => true)
vi.mock('../lib/openExternalUrl.js', () => ({
  openExternalUrl: (url: string) => openExternalUrlMock(url)
}))

const ref = <T>(current: T) => ({ current })

const buildCtx = (appended: Msg[]) =>
  ({
    composer: {
      dequeue: () => undefined,
      queueEditRef: ref<null | number>(null),
      sendQueued: vi.fn(),
      setInput: vi.fn()
    },
    gateway: {
      gw: { request: vi.fn(async () => null) },
      rpc: vi.fn(async () => null)
    },
    session: {
      STARTUP_RESUME_ID: '',
      colsRef: ref(80),
      newSession: vi.fn(),
      resetSession: vi.fn(),
      resumeById: vi.fn(),
      setCatalog: vi.fn()
    },
    submission: {
      submitRef: { current: vi.fn() }
    },
    system: {
      bellOnComplete: false,
      sys: vi.fn()
    },
    transcript: {
      appendMessage: (msg: Msg) => appended.push(msg),
      panel: (title: string, sections: any[]) =>
        appended.push({ kind: 'panel', panelData: { sections, title }, role: 'system', text: '' }),
      setHistoryItems: vi.fn()
    },
    voice: {
      setProcessing: vi.fn(),
      setRecording: vi.fn(),
      setVoiceEnabled: vi.fn()
    }
  }) as any

describe('createGatewayEventHandler', () => {
  beforeEach(() => {
    resetOverlayState()
    resetUiState()
    resetTurnState()
    resetServerRequestsForTests()
    resetConnectionOperationsForTests()
    turnController.fullReset()
    patchUiState({ showReasoning: true })
  })

  // ── Credits notice (Strategy B) ──────────────────────────────────────
  describe('credits notice', () => {
    it('shows a notice immediately when idle (no turn in flight)', () => {
      const onEvent = createGatewayEventHandler(buildCtx([]))

      onEvent({
        payload: { key: 'credits.depleted', kind: 'sticky', level: 'error', text: '✕ credits exhausted' },
        type: 'notification.show'
      } as any)

      expect(getUiState().notice).toMatchObject({
        key: 'credits.depleted',
        kind: 'sticky',
        level: 'error',
        text: '✕ credits exhausted'
      })
    })

    it('holds a notice arriving mid-turn (busy) and flushes it at message.complete', () => {
      const onEvent = createGatewayEventHandler(buildCtx([]))

      onEvent({ payload: {}, type: 'message.start' } as any)
      expect(getUiState().busy).toBe(true)

      onEvent({
        payload: { key: 'credits.90', kind: 'sticky', level: 'warn', text: '⚠ 90% used' },
        type: 'notification.show'
      } as any)

      // Mid-turn: busy wins, notice is held, not visible yet.
      expect(getUiState().notice).toBeNull()

      onEvent({ payload: { text: 'done' }, type: 'message.complete' } as any)

      // Turn end flushes the held notice.
      expect(getUiState().notice).toMatchObject({ key: 'credits.90', text: '⚠ 90% used' })
    })

    it('flushes a held notice at interruptTurn (turn-end via ctrl-c)', () => {
      vi.useFakeTimers()

      try {
        const ctx = buildCtx([])
        ctx.gateway.gw.request = vi.fn(async () => ({ status: 'interrupted' }))
        const onEvent = createGatewayEventHandler(ctx)

        patchUiState({ sid: 'sess-1' })
        onEvent({ payload: {}, type: 'message.start' } as any)
        onEvent({
          payload: { key: 'credits.depleted', kind: 'sticky', level: 'error', text: '✕ out' },
          type: 'notification.show'
        } as any)
        expect(getUiState().notice).toBeNull()

        turnController.interruptTurn({
          appendMessage: vi.fn(),
          gw: ctx.gateway.gw,
          sid: 'sess-1',
          sys: ctx.system.sys
        })

        expect(getUiState().notice).toMatchObject({ key: 'credits.depleted', text: '✕ out' })
      } finally {
        vi.runAllTimers()
        vi.useRealTimers()
      }
    })

    it('flushes a held notice at recordError (turn-end via error)', () => {
      const onEvent = createGatewayEventHandler(buildCtx([]))

      onEvent({ payload: {}, type: 'message.start' } as any)
      onEvent({
        payload: { key: 'credits.90', kind: 'sticky', level: 'warn', text: '⚠ 90% used' },
        type: 'notification.show'
      } as any)
      expect(getUiState().notice).toBeNull()

      onEvent({ payload: { message: 'boom' }, type: 'error' } as any)

      expect(getUiState().notice).toMatchObject({ key: 'credits.90', text: '⚠ 90% used' })
    })

    it('latest-wins: a second mid-turn notice replaces the first held one', () => {
      const onEvent = createGatewayEventHandler(buildCtx([]))

      onEvent({ payload: {}, type: 'message.start' } as any)
      onEvent({
        payload: { key: 'credits.90', kind: 'sticky', level: 'warn', text: '⚠ 90% used' },
        type: 'notification.show'
      } as any)
      onEvent({
        payload: { key: 'credits.depleted', kind: 'sticky', level: 'error', text: '✕ exhausted' },
        type: 'notification.show'
      } as any)

      onEvent({ payload: { text: 'done' }, type: 'message.complete' } as any)

      // Only the latest held notice surfaces.
      expect(getUiState().notice).toMatchObject({ key: 'credits.depleted', text: '✕ exhausted' })
    })

    it('clears a visible notice only when the clear key matches (no-op otherwise)', () => {
      const onEvent = createGatewayEventHandler(buildCtx([]))

      onEvent({
        payload: { key: 'credits.grant_spent', kind: 'sticky', level: 'warn', text: '⚠ grant spent' },
        type: 'notification.show'
      } as any)
      expect(getUiState().notice).not.toBeNull()

      // Stale/late clear for a DIFFERENT key must not wipe the newer notice.
      onEvent({ payload: { key: 'credits.something_else' }, type: 'notification.clear' } as any)
      expect(getUiState().notice).toMatchObject({ key: 'credits.grant_spent' })

      // Matching key clears.
      onEvent({ payload: { key: 'credits.grant_spent' }, type: 'notification.clear' } as any)
      expect(getUiState().notice).toBeNull()
    })

    it('drops a held pending notice on a matching clear before it can surface', () => {
      const onEvent = createGatewayEventHandler(buildCtx([]))

      onEvent({ payload: {}, type: 'message.start' } as any)
      onEvent({
        payload: { key: 'credits.grant_spent', kind: 'sticky', level: 'warn', text: '⚠ grant spent' },
        type: 'notification.show'
      } as any)
      // Clear arrives mid-turn before the held notice flushes.
      onEvent({ payload: { key: 'credits.grant_spent' }, type: 'notification.clear' } as any)

      onEvent({ payload: { text: 'done' }, type: 'message.complete' } as any)

      // Nothing surfaces — the pending notice was dropped by the matching clear.
      expect(getUiState().notice).toBeNull()
    })

    it('R3-C2: a ttl notice self-expires even when statusTimer is also armed (timer isolation)', () => {
      // Regression guard for the whole reason `noticeTimer` is a separate
      // timer from `statusTimer`. A concurrent `status.update` (goal path)
      // arms `statusTimer` via restoreStatusAfter; if the two timers shared
      // a slot, clearing statusTimer would cancel the TTL and the notice
      // would never self-expire.
      vi.useFakeTimers()

      try {
        const ctx = buildCtx([])
        const onEvent = createGatewayEventHandler(ctx)

        // 1. While idle, show a ttl notice → applies immediately, arms noticeTimer.
        onEvent({
          payload: { key: 'credits.restored', kind: 'ttl', level: 'success', text: '✓ restored', ttl_ms: 8000 },
          type: 'notification.show'
        } as any)
        expect(getUiState().notice).toMatchObject({ key: 'credits.restored' })

        // 2. A goal status.update arms turnController.statusTimer (via restoreStatusAfter).
        onEvent({
          payload: { kind: 'goal', text: '✓ Goal achieved: some reason' },
          type: 'status.update'
        } as any)
        // statusTimer is now live; notice must still be visible.
        expect(getUiState().notice).toMatchObject({ key: 'credits.restored' })

        // 3. Advance past the TTL — the notice's own dedicated timer fires.
        vi.advanceTimersByTime(8001)

        // 4. Notice self-expired: statusTimer did NOT cancel noticeTimer.
        expect(getUiState().notice).toBeNull()
      } finally {
        vi.runAllTimers()
        vi.useRealTimers()
      }
    })

    it('starts the ttl clock when the notice becomes VISIBLE (at turn end), not on arrival', () => {
      vi.useFakeTimers()

      try {
        const onEvent = createGatewayEventHandler(buildCtx([]))

        onEvent({ payload: {}, type: 'message.start' } as any)
        onEvent({
          payload: { key: 'credits.restored', kind: 'ttl', level: 'success', text: '✓ restored', ttl_ms: 8000 },
          type: 'notification.show'
        } as any)

        // Long busy turn: the TTL must NOT have started while held.
        vi.advanceTimersByTime(10_000)
        expect(getUiState().notice).toBeNull()

        onEvent({ payload: { text: 'done' }, type: 'message.complete' } as any)
        expect(getUiState().notice).toMatchObject({ key: 'credits.restored' })

        // Full 8s starts now (on apply), so it survives nearly that long.
        vi.advanceTimersByTime(7999)
        expect(getUiState().notice).not.toBeNull()
        vi.advanceTimersByTime(2)
        expect(getUiState().notice).toBeNull()
      } finally {
        vi.useRealTimers()
      }
    })

    it('latest-wins cancels a prior ttl timer so it cannot wipe the newer notice', () => {
      vi.useFakeTimers()

      try {
        const onEvent = createGatewayEventHandler(buildCtx([]))

        onEvent({
          payload: { id: 'a', key: 'credits.restored', kind: 'ttl', level: 'success', text: '✓ a', ttl_ms: 5000 },
          type: 'notification.show'
        } as any)

        vi.advanceTimersByTime(4000)

        // A newer sticky arrives before the first's TTL fires.
        onEvent({
          payload: { id: 'b', key: 'credits.depleted', kind: 'sticky', level: 'error', text: '✕ b' },
          type: 'notification.show'
        } as any)
        expect(getUiState().notice).toMatchObject({ id: 'b' })

        // The first notice's stale TTL must NOT clear the newer one.
        vi.advanceTimersByTime(2000)
        expect(getUiState().notice).toMatchObject({ id: 'b', text: '✕ b' })
      } finally {
        vi.useRealTimers()
      }
    })

    it('sticky survives a turn: applied with no pending notice does not clear it', () => {
      const onEvent = createGatewayEventHandler(buildCtx([]))

      // A standing sticky notice from a prior turn.
      onEvent({
        payload: { key: 'credits.depleted', kind: 'sticky', level: 'error', text: '✕ exhausted' },
        type: 'notification.show'
      } as any)
      expect(getUiState().notice).toMatchObject({ key: 'credits.depleted' })

      // A new turn runs with NO new notice arriving.
      onEvent({ payload: {}, type: 'message.start' } as any)
      onEvent({ payload: { text: 'reply' }, type: 'message.complete' } as any)

      // The standing sticky must REappear untouched at turn end.
      expect(getUiState().notice).toMatchObject({ key: 'credits.depleted', text: '✕ exhausted' })
    })

    it('reset()/fullReset() clears pending + timer + visible notice (no cross-session leak)', () => {
      vi.useFakeTimers()

      try {
        const onEvent = createGatewayEventHandler(buildCtx([]))

        // Session A: a visible sticky + a held pending notice mid-turn.
        onEvent({
          payload: { key: 'credits.depleted', kind: 'sticky', level: 'error', text: '✕ A cut' },
          type: 'notification.show'
        } as any)
        onEvent({ payload: {}, type: 'message.start' } as any)
        onEvent({
          payload: { key: 'credits.90', kind: 'sticky', level: 'warn', text: '⚠ A 90%' },
          type: 'notification.show'
        } as any)
        expect(getUiState().notice).toMatchObject({ key: 'credits.depleted' })

        // Session boundary.
        turnController.fullReset()
        expect(getUiState().notice).toBeNull()

        // Session B: a turn ends with nothing held — A's notice must not bleed in.
        onEvent({ payload: {}, type: 'message.start' } as any)
        onEvent({ payload: { text: 'B reply' }, type: 'message.complete' } as any)
        expect(getUiState().notice).toBeNull()
      } finally {
        vi.runAllTimers()
        vi.useRealTimers()
      }
    })

    it('ignores a notification.show with no text', () => {
      const onEvent = createGatewayEventHandler(buildCtx([]))

      onEvent({ payload: { key: 'credits.90', level: 'warn' }, type: 'notification.show' } as any)
      expect(getUiState().notice).toBeNull()
    })
  })

  describe('billing.step_up.verification', () => {
    beforeEach(() => {
      openExternalUrlMock.mockClear()
    })

    it('renders the verification link + code and opens the browser', () => {
      const ctx = buildCtx([])
      const onEvent = createGatewayEventHandler(ctx)

      onEvent({
        payload: { user_code: 'WXYZ-9999', verification_url: 'https://portal.example/device?code=WXYZ' },
        type: 'billing.step_up.verification'
      } as any)

      const printed = (ctx.system.sys as ReturnType<typeof vi.fn>).mock.calls.map(c => c[0]).join('\n')
      expect(printed).toContain('https://portal.example/device?code=WXYZ')
      expect(printed).toContain('WXYZ-9999')
      expect(openExternalUrlMock).toHaveBeenCalledWith('https://portal.example/device?code=WXYZ')
    })

    it('no-ops on a missing verification_url (never opens a browser)', () => {
      const ctx = buildCtx([])
      const onEvent = createGatewayEventHandler(ctx)

      onEvent({ payload: { verification_url: '' }, type: 'billing.step_up.verification' } as any)

      expect(openExternalUrlMock).not.toHaveBeenCalled()
    })
  })
})
