/**
 * The TUI widget SDK — the one import surface a widget app needs.
 *
 * An app is a `WidgetApp` (state + reducer + render) registered with
 * `defineWidgetApp` and launched by id (usually from a slash command via
 * `launchWidget`). While active it owns every keypress and renders in a
 * viewport-level slot, composing the same layout/theme primitives every
 * built-in surface uses — so apps inherit grid tracks, zoned overlays,
 * selection chips, and skin-derived color by construction.
 *
 * See `sdk/apps/` for the reference apps (`/grid-test`, `/dialog-test`).
 */

// Theme + chrome primitives
export { Accordion } from '../components/accordion.js'
export { Shimmer, ShimmerRows, shimmerSegments, useShimmerPhase } from '../components/loaders.js'
// Layout components + overlay primitives
export { Dialog, Overlay, type OverlayZone } from '../components/overlay.js'
export { OverlayHint, windowItems } from '../components/overlayControls.js'
export {
  ActionRow,
  chipRowProps,
  listRowStyle,
  MenuRow,
  scrollbarColors,
  useMenu
} from '../components/overlayPrimitives.js'

export { GridAreas, WidgetGrid } from '../components/widgetGrid.js'

export { gauge, hbars, sparkline, sparkRows } from '../lib/charts.js'
export { contrastRatio, liftForContrast, mix, relativeLuminance } from '../lib/color.js'
// Layout engine
export {
  type GridAreaItem,
  type GridAreasLayout,
  type GridAreasOptions,
  type GridTrackSize,
  layoutGridAreas,
  layoutWidgetGrid,
  resolveGridTracks,
  type WidgetGridItem,
  type WidgetGridLayout,
  type WidgetGridLayoutOptions
} from '../lib/widgetGrid.js'

export type { Theme, ThemeColors } from '../theme.js'
// App contract + host
export {
  ActiveWidgetSlot,
  AmbientDock,
  AmbientRail,
  ambientRailWidth,
  closeWidget,
  dispatchWidgetInput,
  launchWidget,
  openWidget,
  updateWidget
} from './host.js'
export { defineWidgetApp, getWidgetApp, listWidgetApps } from './registry.js'
export {
  type ActiveWidget,
  type AmbientZone,
  isCtrl,
  type WidgetApp,
  type WidgetInput,
  type WidgetRenderCtx
} from './types.js'
export { loadUserWidgets, type UserWidgetLoadResult, widgetSdk, type WidgetSdk } from './userWidgets.js'
