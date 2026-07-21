import { useStdout } from '@hermes/ink'
import { Box } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'

import { $overlayState, patchOverlayState } from '../app/overlayStore.js'
import { $uiTheme } from '../app/uiStore.js'

import { getWidgetApp } from './registry.js'
import type { ActiveWidget, WidgetApp, WidgetInput } from './types.js'

/**
 * The widget-app host. Core integrates through exactly four touchpoints:
 * launch (slash commands), dispatch (the input pipeline), the MODAL render
 * slot (viewport-level), and the AMBIENT dock (in-flow, above the status
 * bar). Everything else — state shape, keybindings, presentation — belongs
 * to the app.
 */

const isAmbient = (app: WidgetApp<never>) => app.mode === 'ambient'

const withoutApp = (dock: ActiveWidget[], id: string) => dock.filter(active => active.appId !== id)

const dockWith = (dock: ActiveWidget[], entry: ActiveWidget) => [...withoutApp(dock, entry.appId), entry]

/** Launch by id. Returns null on success, a printable error/usage line on
 *  refusal — the caller owns the transcript. Relaunching a DOCKED ambient
 *  app (with no new argument) toggles it out of the dock — ambient apps
 *  capture no input, so the command is their only dismissal. */
export function launchWidget(id: string, arg = ''): null | string {
  const app = getWidgetApp(id)

  if (!app) {
    return `unknown widget app: ${id}`
  }

  if (isAmbient(app)) {
    const dock = $overlayState.get().ambient

    if (dock.some(active => active.appId === id) && !arg.trim()) {
      patchOverlayState({ ambient: withoutApp(dock, id) })

      return null
    }
  }

  const state = app.init(arg)

  if (state === null) {
    return app.usage ?? `usage: /${id}`
  }

  if (isAmbient(app)) {
    patchOverlayState({ ambient: dockWith($overlayState.get().ambient, { appId: id, state }) })
  } else {
    patchOverlayState({ widget: { appId: id, state } })
  }

  return null
}

/** Close the MODAL app. Ambient apps dismiss via their launch toggle, so a
 *  modal's Esc can't collaterally clear the dock. */
export const closeWidget = () => patchOverlayState({ widget: null })

/** Programmatic, TYPED launch — bypasses string parsing. Apps use this to
 *  stack each other (the host swaps the active modal app). */
export function openWidget<S>(app: WidgetApp<S>, state: S): void {
  if (isAmbient(app as WidgetApp<never>)) {
    patchOverlayState({ ambient: dockWith($overlayState.get().ambient, { appId: app.id, state }) })
  } else {
    patchOverlayState({ widget: { appId: app.id, state } })
  }
}

/** Async state delivery: patch the app's state ONLY while it is still active
 *  in its slot — a late fetch resolution can never resurrect a closed app or
 *  clobber a different one. This is how data-backed apps land results
 *  outside the input pipeline (see the weather reference app). */
export function updateWidget<S>(app: WidgetApp<S>, fn: (state: S) => S): void {
  if (isAmbient(app as WidgetApp<never>)) {
    const dock = $overlayState.get().ambient

    if (!dock.some(active => active.appId === app.id)) {
      return
    }

    patchOverlayState({
      ambient: dock.map(active => (active.appId === app.id ? { appId: app.id, state: fn(active.state as S) } : active))
    })

    return
  }

  const active = $overlayState.get().widget

  if (active?.appId !== app.id) {
    return
  }

  patchOverlayState({ widget: { appId: app.id, state: fn(active.state as S) } })
}

/** Feed one keypress to the active MODAL app (ambient apps capture no
 *  input). Returns true when a modal app is active — apps swallow every key
 *  while open. */
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

const renderApp = (active: ActiveWidget, ctx: { cols: number; rows: number; t: never }) => {
  const app = getWidgetApp(active.appId)

  return app ? app.render({ ...ctx, state: active.state as never }) : null
}

/** Render slot for the MODAL app — viewport-level, so it can anchor
 *  `Overlay` zones and backdrops against the full terminal. */
export function ActiveWidgetSlot(): ReactNode {
  const overlay = useStore($overlayState)
  const t = useStore($uiTheme)
  const { stdout } = useStdout()

  if (!overlay.widget) {
    return null
  }

  return renderApp(overlay.widget, { cols: stdout?.columns ?? 80, rows: stdout?.rows ?? 24, t: t as never })
}

/** The ambient dock: in-FLOW (never floats over the transcript),
 *  right-aligned, sitting directly above the status bar — GUI-style
 *  "widgets that just sit there" while the composer stays live. */
export function AmbientDock(): ReactNode {
  const overlay = useStore($overlayState)
  const t = useStore($uiTheme)
  const { stdout } = useStdout()

  if (!overlay.ambient.length) {
    return null
  }

  const ctx = { cols: stdout?.columns ?? 80, rows: stdout?.rows ?? 24, t: t as never }

  return (
    <Box columnGap={1} flexDirection="row" justifyContent="flex-end" width="100%">
      {overlay.ambient.map(active => (
        <Box key={active.appId}>{renderApp(active, ctx)}</Box>
      ))}
    </Box>
  )
}
