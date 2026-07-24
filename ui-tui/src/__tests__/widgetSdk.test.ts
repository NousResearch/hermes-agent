import { beforeEach, describe, expect, it } from 'vitest'

import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { dialogTestApp, gridTestApp } from '../sdk/apps/index.js'
import { closeWidget, dispatchWidgetInput, launchWidget, openWidget } from '../sdk/host.js'
import { getWidgetApp, listWidgetApps } from '../sdk/registry.js'
import type { WidgetInput } from '../sdk/types.js'

const key = (overrides: Partial<WidgetInput['key']> = {}, ch = ''): WidgetInput =>
  ({
    ch,
    key: { ctrl: false, escape: false, leftArrow: false, return: false, rightArrow: false, ...overrides }
  }) as WidgetInput

beforeEach(() => resetOverlayState())

describe('widget SDK host', () => {
  it('registers the reference apps', () => {
    expect(listWidgetApps().map(app => app.id)).toEqual(
      expect.arrayContaining(['dialog-test', 'grid-test', 'ticker', 'weather'])
    )
    expect(getWidgetApp('grid-test')).toBe(gridTestApp)
  })

  it('launch → dispatch → close lifecycle drives the overlay slot', () => {
    expect(launchWidget('grid-test', '5x2')).toBeNull()
    expect(getOverlayState().widget).toMatchObject({ appId: 'grid-test' })
    expect(getOverlayState().widget?.state).toMatchObject({ cols: 5, rows: 2 })

    // Reducer output lands back in the slot.
    expect(dispatchWidgetInput(key({}, 'l'))).toBe(true)
    expect(getOverlayState().widget?.state).toMatchObject({ activeCol: 1 })

    // null from reduce closes.
    expect(dispatchWidgetInput(key({ escape: true }))).toBe(true)
    expect(getOverlayState().widget).toBeNull()

    // Nothing active → not handled.
    expect(dispatchWidgetInput(key({}, 'x'))).toBe(false)
  })

  it('refused launches return the usage line and leave the slot empty', () => {
    expect(launchWidget('grid-test', 'not-a-size !')).toBe(gridTestApp.usage)
    expect(launchWidget('nope', '')).toMatch(/unknown widget app/)
    expect(getOverlayState().widget).toBeNull()
  })

  it('apps stack each other via the typed programmatic launch', () => {
    expect(launchWidget('grid-test', '')).toBeNull()

    // `d` swaps the active app to the dialog demo.
    expect(dispatchWidgetInput(key({}, 'd'))).toBe(true)
    expect(getOverlayState().widget).toMatchObject({ appId: 'dialog-test' })

    // Enter closes the dialog app.
    expect(dispatchWidgetInput(key({ return: true }))).toBe(true)
    expect(getOverlayState().widget).toBeNull()
  })

  it('a widget that throws in render shows an error chip, not a dead TUI', async () => {
    const { defineWidgetApp } = await import('../sdk/registry.js')
    const { AmbientDock } = await import('../sdk/host.js')
    const { renderToScreen } = await import('../../packages/hermes-ink/src/ink/render-to-screen.js')
    const { createElement } = await import('react')

    defineWidgetApp({
      help: 'crash test',
      id: 'crash-test',
      mode: 'ambient',
      init: () => ({}),
      reduce: state => state,
      render: () => {
        throw new Error('boom')
      }
    })

    launchWidget('crash-test', 'x')

    // Renders the boundary chip instead of propagating the throw.
    expect(() => renderToScreen(createElement(AmbientDock, { placement: 'dock-bottom' }), 60)).not.toThrow()
  })

  it('openWidget is a typed direct launch', () => {
    openWidget(dialogTestApp, { body: 'hi', zone: 'top-right' })
    expect(getOverlayState().widget).toMatchObject({ appId: 'dialog-test', state: { zone: 'top-right' } })
    closeWidget()
    expect(getOverlayState().widget).toBeNull()
  })

  it('a MODAL widget blocks the composer; ambient never does', async () => {
    const { $isBlocked } = await import('../app/overlayStore.js')

    expect($isBlocked.get()).toBe(false)
    launchWidget('ticker', '')
    expect($isBlocked.get()).toBe(false)
    launchWidget('dialog-test', 'center')
    expect($isBlocked.get()).toBe(true)
  })

  it('ambient zones route by the app contract (docks + floats)', async () => {
    const { defineWidgetApp } = await import('../sdk/registry.js')
    const { Text } = await import('@hermes/ink')
    const { createElement } = await import('react')

    defineWidgetApp({
      help: 'corner test app',
      id: 'corner-test',
      mode: 'ambient',
      zone: 'top-right',
      init: () => ({}),
      reduce: state => state,
      render: () => createElement(Text, null, 'corner')
    })

    launchWidget('corner-test', 'x')
    launchWidget('ticker', 'x')

    const zoneOf = (id: string) => getWidgetApp(id)?.zone ?? 'dock-bottom'

    expect(getOverlayState().ambient.map(a => [a.appId, zoneOf(a.appId)])).toEqual([
      ['corner-test', 'top-right'],
      ['ticker', 'dock-bottom']
    ])
  })

  it('rails reserve the widest railed app; docks reserve nothing sideways', async () => {
    const { ambientRailWidth } = await import('../sdk/host.js')
    const { defineWidgetApp } = await import('../sdk/registry.js')
    const { Text } = await import('@hermes/ink')
    const { createElement } = await import('react')

    defineWidgetApp({
      help: 'wide rail app',
      id: 'rail-wide',
      mode: 'ambient',
      width: 52,
      zone: 'top-right',
      init: () => ({}),
      reduce: state => state,
      render: () => createElement(Text, null, 'wide')
    })

    expect(ambientRailWidth('right')).toBe(0)
    launchWidget('corner-test', 'x') // top-right, default width 44
    launchWidget('rail-wide', 'x')
    launchWidget('ticker', 'x') // dock-bottom — no rail contribution

    expect(ambientRailWidth('right')).toBe(52)
    expect(ambientRailWidth('left')).toBe(0)
  })

  it('ambient apps dock together and toggle independently', () => {
    expect(launchWidget('ticker', 'eurusd')).toBeNull()
    expect(launchWidget('weather', '')).toBeNull()
    expect(getOverlayState().ambient.map(a => a.appId)).toEqual(['ticker', 'weather'])

    // Relaunch with no arg toggles just that app out of the dock.
    expect(launchWidget('ticker', '')).toBeNull()
    expect(getOverlayState().ambient.map(a => a.appId)).toEqual(['weather'])
  })
})
