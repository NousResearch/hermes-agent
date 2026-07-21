import { useStdout } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'

import { $overlayState, patchOverlayState } from '../app/overlayStore.js'
import { $uiTheme } from '../app/uiStore.js'

import { getWidgetApp } from './registry.js'
import type { WidgetApp, WidgetInput } from './types.js'

/**
 * The widget-app host. Core integrates through exactly three touchpoints:
 * launch (slash commands), dispatch (the input pipeline), and the render
 * slot (appLayout). Everything else — state shape, keybindings, placement —
 * belongs to the app.
 */

/** Launch by id. Returns null on success, a printable error/usage line on
 *  refusal — the caller owns the transcript. */
export function launchWidget(id: string, arg = ''): null | string {
  const app = getWidgetApp(id)

  if (!app) {
    return `unknown widget app: ${id}`
  }

  const state = app.init(arg)

  if (state === null) {
    return app.usage ?? `usage: /${id}`
  }

  patchOverlayState({ widget: { appId: id, state } })

  return null
}

export const closeWidget = () => patchOverlayState({ widget: null })

/** Programmatic, TYPED launch — bypasses string parsing. Apps use this to
 *  stack each other (the host swaps the active app). */
export function openWidget<S>(app: WidgetApp<S>, state: S): void {
  patchOverlayState({ widget: { appId: app.id, state } })
}

/** Feed one keypress to the active app. Returns true when a widget is active
 *  (apps swallow every key while open — the overlay is modal). */
export function dispatchWidgetInput(input: WidgetInput): boolean {
  const active = $overlayState.get().widget

  if (!active) {
    return false
  }

  const app = getWidgetApp(active.appId)

  if (!app) {
    closeWidget()

    return true
  }

  const next = app.reduce(active.state as never, input)

  if (next === null) {
    closeWidget()
  } else if (next !== active.state) {
    patchOverlayState({ widget: { appId: active.appId, state: next } })
  }

  return true
}

/** Render slot for the active app — viewport-level, so apps can anchor
 *  `Overlay` zones and backdrops against the full terminal. */
export function ActiveWidgetSlot(): ReactNode {
  const overlay = useStore($overlayState)
  const t = useStore($uiTheme)
  const { stdout } = useStdout()

  if (!overlay.widget) {
    return null
  }

  const app = getWidgetApp(overlay.widget.appId)

  if (!app) {
    return null
  }

  return app.render({
    cols: stdout?.columns ?? 80,
    rows: stdout?.rows ?? 24,
    state: overlay.widget.state as never,
    t
  })
}
