import type { ReactElement } from 'react'
import { createElement } from 'react'
import { beforeEach, describe, expect, it } from 'vitest'

import { resetOverlayState } from '../app/overlayStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { normalizeWidgetOrder } from '../app/useConfigSync.js'
import { AmbientDock, launchWidget } from '../sdk/host.js'
import { defineWidgetApp } from '../sdk/registry.js'

const card = (id: string) =>
  defineWidgetApp({
    help: `${id} card`,
    id,
    mode: 'ambient',
    init: () => ({ n: 0 }),
    reduce: state => state,
    render: () => createElement('ink-text', null, `[${id}]`)
  })

/**
 * The dock row as PAINTED: each launched card's id in left→right screen
 * order. The store's launch order is deliberately untouched — ordering is a
 * render concern, so the contract lives on the painted row, not the array.
 */
const paintedOrder = async (): Promise<string> => {
  const { renderToScreen } = await import('../../packages/hermes-ink/src/ink/render-to-screen.js')
  const { cellAtIndex } = await import('../../packages/hermes-ink/src/ink/screen.js')

  const { screen } = renderToScreen(createElement(AmbientDock, { placement: 'dock-bottom' }) as ReactElement, 120)
  const ids = ['order-a', 'order-b', 'order-c', 'order-unlisted', 'plain-x', 'plain-y']
  const found: { col: number; id: string }[] = []

  for (let row = 0; row < screen.height; row++) {
    let text = ''

    for (let col = 0; col < screen.width; col++) {
      text += cellAtIndex(screen, row * screen.width + col).char
    }

    for (const id of ids) {
      const at = text.indexOf(`[${id}]`)

      if (at >= 0 && !found.some(f => f.id === id)) {
        found.push({ col: at, id })
      }
    }
  }

  return found
    .sort((a, b) => a.col - b.col)
    .map(f => f.id)
    .join(' ')
}

beforeEach(() => {
  resetOverlayState()
  resetUiState()
})

describe('display.tui_widgets.order (#69269)', () => {
  it('normalizes raw YAML: only string entries survive, garbage = null', () => {
    expect(normalizeWidgetOrder(undefined)).toBeNull()
    expect(normalizeWidgetOrder('grok')).toBeNull()
    expect(normalizeWidgetOrder([])).toBeNull()
    expect(normalizeWidgetOrder([null, '  ', 3])).toBeNull()
    expect(normalizeWidgetOrder(['b', 'a'])).toEqual(['b', 'a'])
  })

  it('paints listed ids in list position; unlisted keep launch order after them', async () => {
    card('order-a')
    card('order-c')
    card('order-b')
    card('order-unlisted')

    // Launch order ≠ config order on purpose.
    launchWidget('order-unlisted', '')
    launchWidget('order-c', '')
    launchWidget('order-a', '')
    launchWidget('order-b', '')

    // No config: launch order paints.
    expect(await paintedOrder()).toBe('order-unlisted order-c order-a order-b')

    patchUiState({ widgetOrder: ['order-b', 'order-a'] })
    expect(await paintedOrder()).toBe('order-b order-a order-unlisted order-c')
  })

  it('null order keeps launch order; unknown ids pass through harmlessly', async () => {
    card('plain-x')
    card('plain-y')
    launchWidget('plain-y', '')
    launchWidget('plain-x', '')

    patchUiState({ widgetOrder: null })
    expect(await paintedOrder()).toBe('plain-y plain-x')

    // An id that isn't docked doesn't shift anything.
    patchUiState({ widgetOrder: ['no-such-widget', 'plain-x'] })
    expect(await paintedOrder()).toBe('plain-x plain-y')
  })
})
