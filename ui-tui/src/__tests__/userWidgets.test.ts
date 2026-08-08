import { mkdir, mkdtemp, rm, writeFile } from 'fs/promises'
import { tmpdir } from 'os'
import { join } from 'path'

import { beforeEach, describe, expect, it } from 'vitest'

import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { closeWidget, launchWidget } from '../sdk/host.js'
import { getWidgetApp } from '../sdk/registry.js'
import {
  loadUserWidgets,
  loadWidgetPath,
  onWidgetRefresh,
  reloadWidgetFile,
  requestWidgetRefresh,
  unloadWidgetApp
} from '../sdk/userWidgets.js'

const widgetSource = (id: string, help = 'from disk') => `
export default function register(sdk) {
  sdk.defineWidgetApp({
    id: '${id}',
    help: '${help}',
    mode: 'ambient',
    init: arg => ({ arg }),
    reduce: state => state,
    render: ({ state, t }) => sdk.h(sdk.Text, { color: t.color.label }, state.arg)
  })
}
`

const widgetWithRefresh = (id: string) => `
export default function register(sdk) {
  sdk.onWidgetRefresh(() => {})
  sdk.defineWidgetApp({
    id: '${id}',
    help: 'refresh',
    mode: 'ambient',
    init: arg => ({ arg }),
    reduce: state => state,
    render: ({ state, t }) => sdk.h(sdk.Text, { color: t.color.label }, state.arg)
  })
}
`

const widgetWithSiblingImport = (id: string) => `
import { label } from './lib/helper.mjs'
export default function register(sdk) {
  sdk.defineWidgetApp({
    id: '${id}',
    help: label,
    mode: 'ambient',
    init: arg => ({ arg }),
    reduce: state => state,
    render: ({ state, t }) => sdk.h(sdk.Text, { color: t.color.label }, state.arg)
  })
}
`

beforeEach(() => resetOverlayState())

describe('user widget loading', () => {
  it('missing directory is a clean no-op', async () => {
    const result = await loadUserWidgets(join(tmpdir(), 'definitely-missing-widgets-dir'))

    expect(result).toEqual({ added: [], errors: [], loaded: [], removed: [] })
  })

  it('loads .mjs from disk, registers, dispatches, and reports broken files', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'tui-widgets-'))

    await writeFile(join(dir, 'good.mjs'), widgetSource('test-user-widget'))
    await writeFile(join(dir, 'broken.mjs'), 'export default 42')
    await writeFile(join(dir, 'ignored.txt'), 'not a widget')

    const result = await loadUserWidgets(dir)

    expect(result.loaded).toEqual(['good.mjs'])
    expect(result.added).toEqual(['test-user-widget'])
    expect(result.errors).toMatchObject([{ file: 'broken.mjs' }])

    // Registered like any built-in: catalog metadata + launchable.
    expect(getWidgetApp('test-user-widget')).toMatchObject({ help: 'from disk', mode: 'ambient' })
    expect(launchWidget('test-user-widget', 'hi')).toBeNull()
    expect(getOverlayState().ambient).toMatchObject([{ appId: 'test-user-widget', state: { arg: 'hi' } }])
  })

  it('a deleted file unregisters its apps on the next scan', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'tui-widgets-'))
    const file = join(dir, 'gone.mjs')

    await writeFile(file, widgetSource('soon-gone'))
    await loadUserWidgets(dir)
    expect(getWidgetApp('soon-gone')).toBeDefined()

    await rm(file)
    const result = await loadUserWidgets(dir)

    expect(result.removed).toEqual(['soon-gone'])
    expect(getWidgetApp('soon-gone')).toBeUndefined()
  })

  it('reloadWidgetFile re-imports one file by app id and keeps it docked', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'tui-widgets-'))

    await writeFile(join(dir, 'solo.mjs'), widgetSource('solo-app', 'v1'))
    await loadUserWidgets(dir)
    expect(getWidgetApp('solo-app')?.help).toBe('v1')
    expect(launchWidget('solo-app', 'docked')).toBeNull()
    expect(getOverlayState().ambient.some(a => a.appId === 'solo-app' && a.state.arg === 'docked')).toBe(true)

    await writeFile(join(dir, 'solo.mjs'), widgetSource('solo-app', 'v2'))
    const result = await reloadWidgetFile('solo-app', dir)

    expect(result.loaded).toEqual(['solo.mjs'])
    expect(result.removed).not.toContain('solo-app')
    expect(result.added).not.toContain('solo-app')
    expect(getWidgetApp('solo-app')?.help).toBe('v2')
    expect(getOverlayState().ambient.some(a => a.appId === 'solo-app' && a.state.arg === 'docked')).toBe(true)
  })

  it('failed reload keeps the prior registration and dock state', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'tui-widgets-'))

    await writeFile(join(dir, 'keep.mjs'), widgetSource('keep-app', 'good'))
    await loadUserWidgets(dir)
    expect(launchWidget('keep-app', 'stay')).toBeNull()

    await writeFile(join(dir, 'keep.mjs'), 'export default 42')
    const result = await reloadWidgetFile('keep-app', dir)

    expect(result.errors).toMatchObject([{ file: 'keep.mjs' }])
    expect(result.loaded).toEqual([])
    expect(getWidgetApp('keep-app')?.help).toBe('good')
    expect(getOverlayState().ambient.some(a => a.appId === 'keep-app')).toBe(true)
  })

  it('preserves sibling relative imports during reload', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'tui-widgets-'))
    const lib = join(dir, 'lib')

    await mkdir(lib, { recursive: true })
    await writeFile(join(lib, 'helper.mjs'), 'export const label = "sibling-help"\n')
    await writeFile(join(dir, 'main.mjs'), widgetWithSiblingImport('sibling-app'))

    const first = await loadUserWidgets(dir)

    expect(first.errors).toEqual([])
    expect(getWidgetApp('sibling-app')?.help).toBe('sibling-help')

    await writeFile(join(lib, 'helper.mjs'), 'export const label = "sibling-v2"\n')
    const result = await reloadWidgetFile('sibling-app', dir)

    expect(result.errors).toEqual([])
    expect(result.loaded).toEqual(['main.mjs'])
    expect(getWidgetApp('sibling-app')?.help).toBe('sibling-v2')
  })

  it('loadWidgetPath registers from an absolute path outside the widgets dir', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'tui-widgets-ext-'))
    const file = join(dir, 'external.mjs')

    await writeFile(file, widgetSource('external-app'))
    const result = await loadWidgetPath(file)

    expect(result.added).toEqual(['external-app'])
    expect(getWidgetApp('external-app')).toBeDefined()
    expect(launchWidget('external-app', 'x')).toBeNull()
  })

  it('unloadWidgetApp dismisses dock + stays unloaded across scans', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'tui-widgets-'))

    await writeFile(join(dir, 'bye.mjs'), widgetSource('bye-app'))
    await loadUserWidgets(dir)
    expect(launchWidget('bye-app', 'x')).toBeNull()
    expect(getOverlayState().ambient.some(a => a.appId === 'bye-app')).toBe(true)

    const r = unloadWidgetApp('bye-app')

    expect(r.ok).toBe(true)
    expect(getWidgetApp('bye-app')).toBeUndefined()
    expect(getOverlayState().ambient.some(a => a.appId === 'bye-app')).toBe(false)

    const scan = await loadUserWidgets(dir)

    expect(scan.added).not.toContain('bye-app')
    expect(getWidgetApp('bye-app')).toBeUndefined()

    const reenabled = await reloadWidgetFile('bye-app', dir)

    expect(reenabled.added).toContain('bye-app')
    expect(getWidgetApp('bye-app')).toBeDefined()
  })

  it('reload disposes prior onWidgetRefresh listeners from register()', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'tui-widgets-'))

    await writeFile(join(dir, 'bus.mjs'), widgetWithRefresh('bus-app'))
    await loadUserWidgets(dir)

    const afterFirst = requestWidgetRefresh()

    await reloadWidgetFile('bus-app', dir)
    const afterReload = requestWidgetRefresh()

    // One register() subscription should remain owned by the file, not stack.
    expect(afterReload).toBe(afterFirst)
  })

  it('requestWidgetRefresh notifies subscribers with optional id', () => {
    const seen: (null | string)[] = []
    const stop = onWidgetRefresh(id => seen.push(id))

    expect(requestWidgetRefresh()).toBeGreaterThanOrEqual(1)
    expect(requestWidgetRefresh('weather')).toBeGreaterThanOrEqual(1)
    stop()
    expect(seen).toEqual([null, 'weather'])
  })
})

describe('closeWidget by id', () => {
  it('dismisses an ambient dock entry without touching other apps', async () => {
    const { defineWidgetApp } = await import('../sdk/registry.js')

    defineWidgetApp({
      help: 'a',
      id: 'close-a',
      mode: 'ambient',
      init: () => ({}),
      reduce: s => s,
      render: () => null
    })
    defineWidgetApp({
      help: 'b',
      id: 'close-b',
      mode: 'ambient',
      init: () => ({}),
      reduce: s => s,
      render: () => null
    })

    launchWidget('close-a', 'x')
    launchWidget('close-b', 'x')
    closeWidget('close-a')

    expect(getOverlayState().ambient.map(a => a.appId)).toEqual(['close-b'])
  })
})
