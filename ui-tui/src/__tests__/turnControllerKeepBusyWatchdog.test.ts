import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import type { Msg } from '../types.js'

// Cross-fork follow-up to the desktop+gateway fix in #45001: the TUI has
// the same recoverability hole when `interruptTurn({ keepBusy: true })`
// re-asserts busy and waits for the gateway's real settle edge
// (`message.complete`). If that edge never arrives — orphaned session,
// transport dropped, agent thread crash — the user is stranded in a
// permanent busy state with no recovery path.
//
// The watchdog arms in `interruptTurn`, cancels in the natural settle
// sites (`recordMessageComplete`, `recordError`, `startMessage`,
// `reset`/`fullReset`), and fires after KEEP_BUSY_WATCHDOG_MS otherwise
// to force `idle()` and surface a ttl notice.
const buildDeps = () => ({
  appendMessage: vi.fn((_msg: Msg) => {}),
  // `interruptTurn` chains `.catch(() => {})` on the request — the mock
  // must return a thenable so the chained catch doesn't blow up. The
  // resolved value type is irrelevant; the call is fire-and-forget.
  gw: { request: vi.fn(async () => null) },
  sid: 'sess-watchdog',
  sys: vi.fn()
})

const startTurn = () => {
  patchUiState({ sid: 'sess-watchdog' })
  turnController.startMessage()
  // Simulate the first delta to mirror a real turn (revs bufRef etc.).
  turnController.recordMessageDelta({ text: 'thinking…' })
}

describe('turnController.interruptTurn — keepBusy settle watchdog', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    resetUiState()
    resetTurnState()
    turnController.fullReset()
  })

  afterEach(() => {
    // Always drain any pending fake timers and restore real timers so a
    // mid-test failure can't leak the watchdog across test files (the
    // same cross-file flake concern already covered for
    // interrupt-cooldown in createGatewayEventHandler.test.ts).
    try {
      vi.runOnlyPendingTimers()
    } catch {
      // ignore
    }

    vi.useRealTimers()
  })

  it('cancels the watchdog when the gateway settles naturally via recordMessageComplete', () => {
    // The natural settle path: message.complete lands well within the
    // watchdog window. The watchdog must NOT fire, busy must flip false,
    // and no ttl notice should be surfaced.
    startTurn()
    const deps = buildDeps()

    turnController.interruptTurn(deps, { keepBusy: true })
    expect(getUiState().busy).toBe(true)

    // Settle well before the watchdog (10s) fires.
    turnController.recordMessageComplete({ text: 'Operation interrupted' })

    expect(getUiState().busy).toBe(false)
    expect(getUiState().notice?.key).not.toBe('tui.busy_watchdog')

    // Advance past the watchdog — nothing should happen, it's already
    // cancelled. (We use `runOnlyPendingTimers` after advancing to prove
    // no scheduled timer is still alive.)
    vi.advanceTimersByTime(20_000)
    expect(getUiState().busy).toBe(false)
    expect(getUiState().notice?.key).not.toBe('tui.busy_watchdog')
  })

  it('forces idle() and emits a ttl notice when the watchdog fires with no settle', () => {
    // The orphan path: the interrupt was sent, keepBusy re-asserted, but
    // the gateway never delivers message.complete (session died, transport
    // dropped, agent thread crashed). After KEEP_BUSY_WATCHDOG_MS the
    // watchdog must force busy:false, surface a ttl notice, and the user
    // is no longer stranded.
    startTurn()
    const deps = buildDeps()

    turnController.interruptTurn(deps, { keepBusy: true })
    expect(getUiState().busy).toBe(true)
    expect(getUiState().notice).toBeNull()

    vi.advanceTimersByTime(10_000)

    // Watchdog fired: busy cleared, ttl notice surfaced, status ready.
    expect(getUiState().busy).toBe(false)
    expect(getUiState().status).toBe('ready')
    expect(getUiState().notice).toMatchObject({
      key: 'tui.busy_watchdog',
      kind: 'ttl',
      level: 'warn',
      text: '⚠ Live turn lost — UI settled automatically'
    })
    // Stream state was force-cleared by idle().
    expect(turnController.bufRef).toBe('')
  })

  it('does not fire the watchdog if a new genuine turn has already started (startMessage cancels it)', () => {
    // The "queued message drained a new turn" path: the interrupted turn
    // never settles (no message.complete), but the gateway sends
    // message.start for the next turn. The watchdog must be cancelled by
    // startMessage() — otherwise it would clobber the new, healthy turn.
    startTurn()
    const deps = buildDeps()

    turnController.interruptTurn(deps, { keepBusy: true })
    expect(getUiState().busy).toBe(true)

    // A new turn starts (e.g. queued message drain, or a fresh user
    // submit landed and the gateway's session.interrupt has resolved
    // into a new turn). startMessage() is the signal.
    turnController.startMessage()
    expect(getUiState().busy).toBe(true)

    vi.advanceTimersByTime(20_000)

    // Watchdog never fired: no ttl notice, busy still true (the new turn
    // is in flight, that's correct).
    expect(getUiState().notice?.key).not.toBe('tui.busy_watchdog')
    expect(getUiState().busy).toBe(true)
  })

  it('does not arm a watchdog when keepBusy is not requested', () => {
    // Plain interrupt (Esc, ctrl-c with no queue): the cooldown handles
    // status decay, no keepBusy hold. Advancing well past the watchdog
    // window must be a no-op.
    startTurn()
    const deps = buildDeps()

    turnController.interruptTurn(deps) // no keepBusy
    expect(getUiState().busy).toBe(false)

    vi.advanceTimersByTime(20_000)

    expect(getUiState().notice).toBeNull()
    expect(getUiState().busy).toBe(false)
  })

  it('cancels the watchdog on recordError (defensive settle for keepBusy hold)', () => {
    // message.error from the gateway also means the interrupted turn is
    // over. The watchdog must cancel here too so it can't fire later
    // and clobber an already-idle UI.
    startTurn()
    const deps = buildDeps()

    turnController.interruptTurn(deps, { keepBusy: true })
    expect(getUiState().busy).toBe(true)

    turnController.recordError()

    expect(getUiState().busy).toBe(false)
    vi.advanceTimersByTime(20_000)
    expect(getUiState().notice?.key).not.toBe('tui.busy_watchdog')
  })

  it('cancels the watchdog on reset()/fullReset() (session boundary)', () => {
    // Session boundary: a pending watchdog from session A must not fire
    // into session B (or after fullReset teardown).
    startTurn()
    const deps = buildDeps()

    turnController.interruptTurn(deps, { keepBusy: true })
    expect(getUiState().busy).toBe(true)

    turnController.reset()
    vi.advanceTimersByTime(20_000)
    expect(getUiState().notice?.key).not.toBe('tui.busy_watchdog')

    // Re-arm + fullReset path.
    startTurn()
    turnController.interruptTurn(buildDeps(), { keepBusy: true })
    turnController.fullReset()
    vi.advanceTimersByTime(20_000)
    expect(getUiState().notice?.key).not.toBe('tui.busy_watchdog')
  })

  it('re-arms cleanly across two back-to-back keepBusy interrupts (no leaked timer)', () => {
    // Re-arming must cancel the prior handle, otherwise the first
    // watchdog fires during the second keepBusy hold and clobbers a
    // turn we didn't mean to settle. Two interleaved interrupts:
    //   t=0:    1st interruptTurn({ keepBusy: true })  → wd#1 fires t=10s
    //   t=2s:   2nd interruptTurn({ keepBusy: true })  → wd#2 fires t=12s
    //                                                  (wd#1 cancelled)
    //   t=11s:  advance — wd#1 has NOT fired (cancelled),
    //           wd#2 has NOT yet fired (1s to go), busy=true, no notice
    //   t=13s:  advance — wd#2 fires, busy=false, notice set
    startTurn()
    turnController.interruptTurn(buildDeps(), { keepBusy: true })
    expect(getUiState().busy).toBe(true)

    // Re-arm at t=2s. The first watchdog is cancelled; the second is
    // scheduled for t=2s + 10s = 12s.
    vi.advanceTimersByTime(2_000)
    startTurn()
    turnController.interruptTurn(buildDeps(), { keepBusy: true })

    // At t=11s: wd#1 would have fired at t=10s, but the re-arm at t=2s
    // cancelled it. wd#2 fires at t=12s, so we should still be busy.
    vi.advanceTimersByTime(9_000)
    expect(getUiState().busy).toBe(true)
    expect(getUiState().notice?.key).not.toBe('tui.busy_watchdog')

    // At t=13s: wd#2 has fired (1s past its 12s deadline).
    vi.advanceTimersByTime(2_000)
    expect(getUiState().busy).toBe(false)
    expect(getUiState().notice?.key).toBe('tui.busy_watchdog')
  })
})
