import React, { type PropsWithChildren, useContext, useInsertionEffect } from 'react'
import { c as _c } from 'react/compiler-runtime'

import instances from '../instances.js'
import { CURSOR_HOME, ERASE_SCREEN, ERASE_SCROLLBACK } from '../termio/csi.js'
import {
  DISABLE_MOUSE_TRACKING,
  enableMouseTrackingFor,
  ENTER_ALT_SCREEN,
  EXIT_ALT_SCREEN,
  type MouseTrackingMode
} from '../termio/dec.js'
import { TerminalWriteContext } from '../useTerminalNotification.js'

import Box from './Box.js'
import { TerminalSizeContext } from './TerminalSizeContext.js'
type Props = PropsWithChildren<{
  /**
   * Which SGR mouse-tracking preset to enable. Default `'all'` — wheel +
   * click + drag + hover (1000 + 1002 + 1003 + 1006). Set to `'wheel'`
   * (1000 + 1006) to silence the noisy hover events that tmux turns into
   * "No image in clipboard" spam over the prompt row, while keeping
   * scroll-wheel scrolling. `'off'` disables tracking entirely.
   */
  mouseTracking?: MouseTrackingMode
}>

/**
 * Run children in the terminal's alternate screen buffer, constrained to
 * the viewport height. While mounted:
 *
 * - Enters the alt screen (DEC 1049), clears it, homes the cursor
 * - Constrains its own height to the terminal row count, so overflow must
 *   be handled via `overflow: scroll` / flexbox (no native scrollback)
 * - Optionally enables a subset of SGR mouse tracking (wheel-only,
 *   wheel+drag, or wheel+drag+hover) — events surface as `ParsedKey`
 *   (wheel) and update the Ink instance's selection state (click/drag).
 *   See `MouseTrackingMode` for the available presets.
 *
 * On unmount, disables mouse tracking and exits the alt screen, restoring
 * the main screen's content. Safe for use in ctrl-o transcript overlays
 * and similar temporary fullscreen views — the main screen is preserved.
 *
 * Notifies the Ink instance via `setAltScreenActive()` so the renderer
 * keeps the cursor inside the viewport (preventing the cursor-restore LF
 * from scrolling content) and so signal-exit cleanup can exit the alt
 * screen if the component's own unmount doesn't run.
 */
export function AlternateScreen(t0: Props) {
  const $ = _c(7)

  const { children, mouseTracking: t1 } = t0

  const mouseTracking: MouseTrackingMode = t1 === undefined ? 'all' : t1
  const size = useContext(TerminalSizeContext)
  const writeRaw = useContext(TerminalWriteContext)
  let t2
  let t3

  if ($[0] !== mouseTracking || $[1] !== writeRaw) {
    t2 = () => {
      const ink = instances.get(process.stdout)

      if (!writeRaw) {
        return
      }

      const enableMouse = enableMouseTrackingFor(mouseTracking)

      // Always reset every mouse mode before enabling the requested preset
      // so the terminal lands in an exact state. If a previous instance
      // (crash, another app, lingering DECSET from a debugger) left DEC
      // 1003 hover events asserted, picking 'wheel' or 'buttons' without
      // an unconditional DISABLE would silently leave hover on and defeat
      // the point of the preset.
      writeRaw(ENTER_ALT_SCREEN + ERASE_SCROLLBACK + ERASE_SCREEN + CURSOR_HOME + DISABLE_MOUSE_TRACKING + enableMouse)
      ink?.setAltScreenActive(true, mouseTracking)
      // setAltScreenActive(true, mouseTracking) above stores the mode for
      // SIGCONT/resize/stdin-gap re-assertion. We don't also call
      // setAltScreenMouseTracking(mouseTracking) here: it would early-return
      // in the happy mode-change path (active flipped false→true with the
      // new mode), and on any path where setAltScreenActive saw active was
      // already true (so it didn't store mode), the writeRaw above has
      // already DISABLE'd + enabled the new mode. A second
      // setAltScreenMouseTracking would just duplicate the same DEC bytes.

      return () => {
        ink?.setAltScreenActive(false)
        ink?.clearTextSelection()
        // DISABLE_MOUSE_TRACKING is safe to send even when we never enabled
        // tracking (it unconditionally resets all four modes). Sending it
        // on every teardown means a crash mid-mount can't leak DEC modes
        // back to the host shell.
        writeRaw(DISABLE_MOUSE_TRACKING + EXIT_ALT_SCREEN)
      }
    }

    t3 = [writeRaw, mouseTracking]
    $[0] = mouseTracking
    $[1] = writeRaw
    $[2] = t2
    $[3] = t3
  } else {
    t2 = $[2]
    t3 = $[3]
  }

  useInsertionEffect(t2, t3)
  const t4 = size?.rows ?? 24
  let t5

  if ($[4] !== children || $[5] !== t4) {
    t5 = (
      <Box flexDirection="column" flexShrink={0} height={t4} width="100%">
        {children}
      </Box>
    )
    $[4] = children
    $[5] = t4
    $[6] = t5
  } else {
    t5 = $[6]
  }

  return t5
}
