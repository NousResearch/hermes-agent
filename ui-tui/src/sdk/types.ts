import type { Key } from '@hermes/ink'
import type { ReactNode } from 'react'

import type { Theme } from '../theme.js'

/** One keypress, as the input pipeline delivers it. */
export interface WidgetInput {
  ch: string
  key: Key
}

export interface WidgetRenderCtx<S> {
  /** Terminal columns available to the app. */
  cols: number
  /** Terminal rows available to the app. */
  rows: number
  state: S
  t: Theme
}

/**
 * A widget app: a self-contained overlay surface with its own state, input
 * reducer, and render — the TUI equivalent of a desktop panel. The host owns
 * exactly one active app at a time; while active, the app receives every
 * keypress and the composer is blocked.
 *
 * Contract:
 * - `init(arg)` parses the launch argument (slash-command tail) into initial
 *   state; `null` refuses the launch and the launcher prints `usage`.
 * - `reduce(state, input)` returns the next state, the SAME reference to
 *   swallow the key unchanged, or `null` to close the app.
 * - `render(ctx)` returns the overlay node. Compose with the SDK primitives
 *   (`Overlay`, `Dialog`, `WidgetGrid`, `GridAreas`, `chipRowProps`, …) so
 *   placement and theming stay engine-derived.
 */
export interface WidgetApp<S = unknown> {
  id: string
  /** One-line description — surfaces in `/` completions and command help. */
  help: string
  /**
   * `modal` (default): owns every keypress, blocks the composer.
   * `ambient`: glanceable panel — no input capture, no blocking; launching
   * the same id again toggles it closed.
   */
  mode?: 'ambient' | 'modal'
  /** Ambient placement — see AmbientZone. Default `dock-bottom`. */
  zone?: AmbientZone
  /** Card width in cells (ambient). Floats RESERVE this as a transcript
   *  rail, so match your Dialog width. Default 44. */
  width?: number
  init(arg: string): null | S
  reduce(state: S, input: WidgetInput): null | S
  render(ctx: WidgetRenderCtx<S>): ReactNode
  usage?: string
}

/**
 * Where an ambient widget lives. Two placement families:
 *
 * DOCKS are in-FLOW chrome rows (they reserve real rows, never cover
 * content): `dock-top` under the top status bar, `dock-bottom` above the
 * bottom one. Each dock is a right-aligned row of cards.
 *
 * FLOATS overlay the transcript margins without reserving layout
 * (position:absolute against the viewport, GUI-corner style):
 * `top-left` | `top-right` | `bottom-left` | `bottom-right`. Floats in the
 * same corner stack vertically. Content under a float stays live — floats
 * suit sparse corners; prefer docks for anything tall.
 *
 * Users phrase placement loosely ("top right", "pin it above the status
 * bar") — map words to the nearest zone; corners mean floats.
 */
export type AmbientZone = 'bottom-left' | 'bottom-right' | 'dock-bottom' | 'dock-top' | 'top-left' | 'top-right'

/** The host's serializable record of the active app. */
export interface ActiveWidget {
  appId: string
  state: unknown
}

/** Ctrl+<letter> test, shared so app reducers match the core pipeline. */
export const isCtrl = (key: { ctrl: boolean }, ch: string, target: string): boolean =>
  key.ctrl && ch.toLowerCase() === target
