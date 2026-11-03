import { Box, Text, useStdout } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { Component, type ReactNode } from 'react'

import { $overlayState, patchOverlayState } from '../app/overlayStore.js'
import { $uiTheme } from '../app/uiStore.js'
import { recordParentLifecycle } from '../lib/parentLog.js'

import { getWidgetApp } from './registry.js'
import type { ActiveWidget, AmbientZone, WidgetApp, WidgetInput } from './types.js'

/**
 * The widget-app host. Core integrates through exactly four touchpoints:
 * launch (slash commands), dispatch (the input pipeline), the MODAL render
 * slot (viewport-level), and the AMBIENT surfaces (dock rows + side rails,
 * all reserving real space). Everything else — state shape, keybindings,
 * presentation — belongs to the app.
 */

// ── placement ────────────────────────────────────────────────────────

const isAmbient = (app: WidgetApp<never>) => app.mode === 'ambient'

const zoneOf = (active: ActiveWidget): AmbientZone => getWidgetApp(active.appId)?.zone ?? 'dock-bottom'

const withoutApp = (ambient: ActiveWidget[], id: string) => ambient.filter(active => active.appId !== id)

/** Route a launched app to its slot: ambient apps join the dock array
 *  (replacing any prior instance), modal apps take the single modal slot. */
function place(app: WidgetApp<never>, state: unknown): void {
  if (isAmbient(app)) {
    patchOverlayState({ ambient: [...withoutApp($overlayState.get().ambient, app.id), { appId: app.id, state }] })
  } else {
    patchOverlayState({ widget: { appId: app.id, state } })
  }
}

// ── launch / close / update ──────────────────────────────────────────

/** Launch by id. Returns null on success, a printable error/usage line on
 *  refusal — the caller owns the transcript. Relaunching an active ambient
 *  app (with no new argument) toggles it away — ambient apps capture no
 *  input, so the command is their only dismissal. */
export function launchWidget(id: string, arg = ''): null | string {
  const app = getWidgetApp(id)

  if (!app) {
    return `unknown widget app: ${id}`
  }

  if (isAmbient(app)) {
    const ambient = $overlayState.get().ambient

    if (ambient.some(active => active.appId === id) && !arg.trim()) {
      patchOverlayState({ ambient: withoutApp(ambient, id) })

      return null
    }
  }

  const state = app.init(arg)

  if (state === null) {
    return app.usage ?? `usage: /${id}`
  }

  place(app, state)

  return null
}

/** Close the MODAL app. Ambient apps dismiss via their launch toggle, so a
 *  modal's Esc can't collaterally clear the dock. */
export const closeWidget = () => patchOverlayState({ widget: null })

/** Programmatic, TYPED launch — bypasses string parsing. Apps use this to
 *  stack each other (the host swaps the active modal app). */
export const openWidget = <S,>(app: WidgetApp<S>, state: S): void => place(app as WidgetApp<never>, state)

/** Async state delivery: patch the app's state ONLY while it is still active
 *  in its slot — a late fetch resolution can never resurrect a closed app or
 *  clobber a different one. This is how data-backed apps land results
 *  outside the input pipeline (see the weather reference app). */
export function updateWidget<S>(app: WidgetApp<S>, fn: (state: S) => S): void {
  const overlay = $overlayState.get()

  if (isAmbient(app as WidgetApp<never>)) {
    if (overlay.ambient.some(active => active.appId === app.id)) {
      patchOverlayState({
        ambient: overlay.ambient.map(active =>
          active.appId === app.id ? { appId: app.id, state: fn(active.state as S) } : active
        )
      })
    }

    return
  }

  if (overlay.widget?.appId === app.id) {
    patchOverlayState({ widget: { appId: app.id, state: fn(overlay.widget.state as S) } })
  }
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

// ── render ───────────────────────────────────────────────────────────

/** Crash isolation: a widget throwing in render must NEVER take the TUI
 *  down (user widgets are agent-generated code). The boundary swaps the
 *  card for a compact error chip and logs; the app stays registered so a
 *  hot-reloaded fix re-renders on the next state change. */
class WidgetBoundary extends Component<
  { appId: string; children: ReactNode; errorColor: string },
  { message: null | string }
> {
  override state: { message: null | string } = { message: null }

  static getDerivedStateFromError(error: unknown) {
    return { message: error instanceof Error ? error.message : String(error) }
  }

  override componentDidCatch(error: unknown) {
    recordParentLifecycle(
      `widget /${this.props.appId} crashed in render: ${error instanceof Error ? error.message : String(error)}`
    )
  }

  override render() {
    if (this.state.message !== null) {
      return (
        <Text color={this.props.errorColor} wrap="truncate-end">
          ⚠ /{this.props.appId}: {this.state.message}
        </Text>
      )
    }

    return this.props.children
  }
}

interface RenderCtx {
  cols: number
  rows: number
  t: never
}

const useRenderCtx = (): RenderCtx => {
  const t = useStore($uiTheme)
  const { stdout } = useStdout()

  return { cols: stdout?.columns ?? 80, rows: stdout?.rows ?? 24, t: t as never }
}

const renderApp = (active: ActiveWidget, ctx: RenderCtx) => {
  const app = getWidgetApp(active.appId)

  if (!app) {
    return null
  }

  return (
    <WidgetBoundary
      appId={active.appId}
      errorColor={(ctx.t as { color: { error: string } }).color.error}
      key={active.appId}
    >
      {app.render({ ...ctx, state: active.state as never })}
    </WidgetBoundary>
  )
}

const CardStack = ({ apps, ctx }: { apps: ActiveWidget[]; ctx: RenderCtx }) => (
  <Box flexDirection="column" rowGap={1}>
    {apps.map(active => (
      <Box key={active.appId}>{renderApp(active, ctx)}</Box>
    ))}
  </Box>
)

/** Render slot for the MODAL app — viewport-level, so it can anchor
 *  `Overlay` zones and backdrops against the full terminal. */
export function ActiveWidgetSlot(): ReactNode {
  const overlay = useStore($overlayState)
  const ctx = useRenderCtx()

  return overlay.widget ? renderApp(overlay.widget, ctx) : null
}

/** An in-FLOW dock row: reserves real rows in the chrome (never covers
 *  content), right-aligned cards. `dock-top` renders under the top status
 *  bar, `dock-bottom` above the bottom one. */
export function AmbientDock({ placement }: { placement: 'dock-bottom' | 'dock-top' }): ReactNode {
  const overlay = useStore($overlayState)
  const ctx = useRenderCtx()
  const docked = overlay.ambient.filter(active => zoneOf(active) === placement)

  if (!docked.length) {
    return null
  }

  // paddingRight keeps card borders off the terminal's last column — an
  // exact-edge border char trips pending-wrap and reads as a clipped border.
  return (
    <Box columnGap={1} flexDirection="row" justifyContent="flex-end" paddingRight={2} width="100%">
      {docked.map(active => (
        <Box key={active.appId}>{renderApp(active, ctx)}</Box>
      ))}
    </Box>
  )
}

// ── rails ────────────────────────────────────────────────────────────

const DEFAULT_RAIL_WIDTH = 44

const railSide = (zone: AmbientZone): 'left' | 'right' | null =>
  zone.endsWith('-left') ? 'left' : zone.endsWith('-right') ? 'right' : null

const railApps = (ambient: ActiveWidget[], side: 'left' | 'right') =>
  ambient.filter(active => railSide(zoneOf(active)) === side)

/** Columns a rail RESERVES (0 when empty) — the transcript's width budget
 *  subtracts this, so widgets genuinely take up space and text reflows
 *  beside them instead of being painted over. */
export function ambientRailWidth(side: 'left' | 'right', ambient = $overlayState.get().ambient): number {
  const apps = railApps(ambient, side)

  return apps.length ? Math.max(...apps.map(active => getWidgetApp(active.appId)?.width ?? DEFAULT_RAIL_WIDTH)) : 0
}

/** Live rail width for layout math (re-renders on dock changes). */
export function useAmbientRailWidth(side: 'left' | 'right'): number {
  return ambientRailWidth(side, useStore($overlayState).ambient)
}

/** A side rail: a RESERVED column beside the transcript holding corner
 *  widgets — `top-*` zones stack from its top, `bottom-*` from its bottom. */
export function AmbientRail({ side }: { side: 'left' | 'right' }): ReactNode {
  const overlay = useStore($overlayState)
  const ctx = useRenderCtx()
  const apps = railApps(overlay.ambient, side)

  if (!apps.length) {
    return null
  }

  return (
    <Box
      flexDirection="column"
      flexShrink={0}
      justifyContent="space-between"
      paddingX={1}
      width={ambientRailWidth(side, overlay.ambient)}
    >
      <CardStack apps={apps.filter(active => zoneOf(active).startsWith('top'))} ctx={ctx} />
      <CardStack apps={apps.filter(active => zoneOf(active).startsWith('bottom'))} ctx={ctx} />
    </Box>
  )
}
