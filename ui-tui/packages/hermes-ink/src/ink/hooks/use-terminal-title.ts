import { useContext, useEffect } from 'react'
import stripAnsi from 'strip-ansi'

import { OSC, osc } from '../termio/osc.js'
import { TerminalWriteContext } from '../useTerminalNotification.js'

/**
 * Declaratively set the terminal tab/window title.
 *
 * Pass a single string to set both the tab and window title (OSC 0).
 * Pass `{ tab, window }` to set them independently: the short `tab` string
 * goes to OSC 1 (icon/tab label) and the longer `window` string goes to
 * OSC 2 (window title bar). This matters for terminals like Apple
 * Terminal.app whose narrow background tabs truncate the title from the
 * left — a single long OSC 0 string leaves only the tail visible, while a
 * separate short OSC 1 keeps the session name readable.
 *
 * Pass `null` to opt out — the hook becomes a no-op and leaves the
 * terminal title untouched.
 *
 * On Windows, uses `process.title` (classic conhost doesn't support OSC).
 */
export function useTerminalTitle(title: string | TerminalTitlePair | null): void {
  const writeRaw = useContext(TerminalWriteContext)

  useEffect(() => {
    if (title === null || !writeRaw) {
      return
    }

    if (process.platform === 'win32') {
      const clean = stripAnsi(typeof title === 'string' ? title : (title.window ?? title.tab ?? ''))
      process.title = clean

      return
    }

    if (typeof title === 'string') {
      writeRaw(osc(OSC.SET_TITLE_AND_ICON, stripAnsi(title)))

      return
    }

    // Separate tab (OSC 1) and window (OSC 2) titles so narrow tab bars
    // show the short session name instead of a truncated tail.
    const tab = stripAnsi(title.tab ?? '')
    const window = stripAnsi(title.window ?? '')

    if (tab && window) {
      writeRaw(osc(OSC.SET_ICON, tab) + osc(OSC.SET_TITLE, window))
    } else if (window) {
      writeRaw(osc(OSC.SET_TITLE_AND_ICON, window))
    } else if (tab) {
      writeRaw(osc(OSC.SET_TITLE_AND_ICON, tab))
    }
  }, [title, writeRaw])
}

export interface TerminalTitlePair {
  /** Short title for the tab/icon label (OSC 1). */
  tab?: string
  /** Full title for the window title bar (OSC 2). */
  window?: string
}
