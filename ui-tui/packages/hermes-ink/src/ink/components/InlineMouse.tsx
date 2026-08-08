import React, { type PropsWithChildren, useContext, useInsertionEffect } from 'react'

import instances from '../instances.js'
import { DISABLE_MOUSE_TRACKING, enableMouseTrackingFor, type MouseTrackingMode } from '../termio/dec.js'
import { TerminalWriteContext } from '../useTerminalNotification.js'

type Props = PropsWithChildren<{
  /**
   * Which SGR mouse-tracking preset to arm. Default `'all'`. `'off'` renders
   * children with no mouse side effects (identical to a bare Fragment).
   */
  mouseTracking?: MouseTrackingMode
}>

/**
 * Arm SGR mouse tracking for the INLINE (non-alt-screen) TUI without
 * entering the alternate screen buffer.
 *
 * `<AlternateScreen>` arms tracking as a side effect of entering the alt
 * screen, so the primary-buffer TUI (inline mode — Termux by default, or
 * `HERMES_TUI_INLINE=1`) never armed it and every composer click was
 * dropped (issue #30536). This is the missing mouse half: on mount it
 * resets every mouse mode, enables the requested preset, and tells the Ink
 * instance to route mouse events through inline hit-testing
 * (Ink.setInlineMouseActive → resolveMouseTarget, which translates the
 * bottom-anchored frame's coordinates). On unmount it disarms.
 *
 * Renders children directly (a Fragment) so inline flex-column flow — the
 * host terminal's native scrollback above a bottom-anchored composer — is
 * unchanged. The only difference from a bare Fragment is the terminal-mode
 * side effect. `useInsertionEffect` matches `<AlternateScreen>` so the DEC
 * writes are ordered relative to other terminal mutations, not batched with
 * ordinary passive effects.
 */
export function InlineMouse({ children, mouseTracking = 'all' }: Props) {
  const writeRaw = useContext(TerminalWriteContext)

  useInsertionEffect(() => {
    const ink = instances.get(process.stdout)

    if (!writeRaw || mouseTracking === 'off') {
      ink?.setInlineMouseActive(false)

      return
    }

    // DISABLE first so we land in the exact preset even if a previous
    // instance (crash, another app, lingering DECSET) left a different
    // mouse mode asserted — same rationale as <AlternateScreen>.
    writeRaw(DISABLE_MOUSE_TRACKING + enableMouseTrackingFor(mouseTracking))
    ink?.setInlineMouseActive(true, mouseTracking)

    return () => {
      ink?.setInlineMouseActive(false)
      ink?.clearTextSelection()
      // Unconditional reset — safe even if tracking was never armed — so a
      // crash mid-mount can't leak DEC mouse modes back to the host shell.
      writeRaw(DISABLE_MOUSE_TRACKING)
    }
  }, [writeRaw, mouseTracking])

  return <>{children}</>
}
