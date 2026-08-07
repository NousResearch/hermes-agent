import { mkdtemp, rm, writeFile } from 'fs/promises'
import { tmpdir } from 'os'
import { join } from 'path'

import { beforeEach, describe, expect, it } from 'vitest'

import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { launchWidget } from '../sdk/host.js'
import { getWidgetApp } from '../sdk/registry.js'
import { loadUserWidgets } from '../sdk/userWidgets.js'

const WIDGET = `
export default function register(sdk) {
  sdk.defineWidgetApp({
    id: 'test-user-widget',
    help: 'from disk',
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

    await writeFile(join(dir, 'good.mjs'), WIDGET)
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

    await writeFile(file, WIDGET.replace('test-user-widget', 'soon-gone'))
    await loadUserWidgets(dir)
    expect(getWidgetApp('soon-gone')).toBeDefined()

    await rm(file)
    const result = await loadUserWidgets(dir)

    expect(result.removed).toEqual(['soon-gone'])
    expect(getWidgetApp('soon-gone')).toBeUndefined()
  })
})
