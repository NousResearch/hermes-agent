import { beforeEach, describe, expect, it, vi } from 'vitest'

import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { $uiState, resetUiState } from '../app/uiStore.js'
import { applyDisplay, normalizeWidgetRefreshMs } from '../app/useConfigSync.js'
import { launchWidget, softUpdateWidget, updateWidget } from '../sdk/host.js'
import { defineWidgetApp } from '../sdk/registry.js'
import { widgetSdk } from '../sdk/userWidgets.js'

describe('normalizeWidgetRefreshMs', () => {
  it('defaults missing/invalid to 30000 and clamps low positives to 1000', () => {
    expect(normalizeWidgetRefreshMs(undefined)).toBe(30_000)
    expect(normalizeWidgetRefreshMs('nope')).toBe(30_000)
    expect(normalizeWidgetRefreshMs(-5)).toBe(30_000)
    expect(normalizeWidgetRefreshMs(500)).toBe(1_000)
    expect(normalizeWidgetRefreshMs(12_000)).toBe(12_000)
    expect(normalizeWidgetRefreshMs(0)).toBe(0)
    expect(normalizeWidgetRefreshMs('45000')).toBe(45_000)
  })
})

describe('applyDisplay widgetRefreshMs', () => {
  beforeEach(() => {
    resetUiState()
  })

  it('fans display.tui_widget_refresh_ms into $uiState.widgetRefreshMs', () => {
    applyDisplay({ config: { display: { tui_widget_refresh_ms: 15_000 } } }, vi.fn())
    expect($uiState.get().widgetRefreshMs).toBe(15_000)

    applyDisplay({ config: { display: { tui_widget_refresh_ms: 0 } } }, vi.fn())
    expect($uiState.get().widgetRefreshMs).toBe(0)

    applyDisplay({ config: { display: {} } }, vi.fn())
    expect($uiState.get().widgetRefreshMs).toBe(30_000)
  })
})

describe('soft widget updates', () => {
  beforeEach(() => {
    resetOverlayState()
    resetUiState()
  })

  it('skips overlay writes when updateWidget returns the same state ref', () => {
    defineWidgetApp<{ n: number }>({
      help: 'noop',
      id: 'soft-noop',
      mode: 'ambient',
      init: () => ({ n: 1 }),
      reduce: s => s,
      render: () => null
    })

    expect(launchWidget('soft-noop', '')).toBeNull()
    const before = getOverlayState().ambient

    const app = defineWidgetApp<{ n: number }>({
      help: 'noop',
      id: 'soft-noop',
      mode: 'ambient',
      init: () => ({ n: 1 }),
      reduce: s => s,
      render: () => null
    })

    updateWidget(app, s => s)
    expect(getOverlayState().ambient).toBe(before)
  })

  it('softUpdateWidget only rewrites when a value actually changes', () => {
    const app = defineWidgetApp<{ label: string; pct: number }>({
      help: 'gauge',
      id: 'soft-gauge',
      mode: 'ambient',
      init: () => ({ label: 'ok', pct: 10 }),
      reduce: s => s,
      render: () => null
    })

    expect(launchWidget('soft-gauge', '')).toBeNull()
    const ambient0 = getOverlayState().ambient
    const state0 = ambient0[0]?.state

    softUpdateWidget(app, { pct: 10, label: 'ok' })
    expect(getOverlayState().ambient).toBe(ambient0)
    expect(getOverlayState().ambient[0]?.state).toBe(state0)

    softUpdateWidget(app, { pct: 11 })
    const ambient1 = getOverlayState().ambient
    expect(ambient1).not.toBe(ambient0)
    expect(ambient1[0]?.state).toEqual({ label: 'ok', pct: 11 })
    // Unrelated ambient entries keep identity when present
  })

  it('exposes refreshMs from widgetSdk from ui state', () => {
    applyDisplay({ config: { display: { tui_widget_refresh_ms: 9_000 } } }, vi.fn())
    expect(widgetSdk.refreshMs()).toBe(9_000)
  })
})
