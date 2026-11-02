import { createElement } from 'react'
import { describe, expect, it } from 'vitest'

import { renderToScreen } from '../../packages/hermes-ink/src/ink/render-to-screen.js'
import { cellAtIndex } from '../../packages/hermes-ink/src/ink/screen.js'
import { MessageLine } from '../components/messageLine.js'
import { DEFAULT_THEME } from '../theme.js'
import type { Msg } from '../types.js'

const SCREEN_WIDTH = 100
const BODY_COLS = 60

const paintedBeyond = (screen: Parameters<typeof cellAtIndex>[0], x1: number) => {
  for (let y = 0; y < screen.height; y++) {
    for (let x = x1; x < SCREEN_WIDTH; x++) {
      const cell = cellAtIndex(screen, y * SCREEN_WIDTH + x)

      if (cell && cell.char.trim() !== '') {
        return { x, y }
      }
    }
  }

  return null
}

describe('MessageLine width reservation', () => {
  it('keeps tool-trail rows within the pet/rail-aware body width', () => {
    const msg: Msg = {
      kind: 'trail',
      role: 'system',
      text: '',
      tools: [`terminal: ${'x'.repeat(160)}`]
    }

    const { screen } = renderToScreen(
      createElement(MessageLine, {
        cols: BODY_COLS,
        detailsMode: 'expanded',
        detailsModeCommandOverride: true,
        msg,
        t: DEFAULT_THEME
      }),
      SCREEN_WIDTH
    )

    expect(paintedBeyond(screen, BODY_COLS + 2)).toBeNull()
  })

  it('keeps assistant detail sections within the pet/rail-aware body width', () => {
    const msg: Msg = {
      role: 'assistant',
      text: 'short reply',
      tools: [`terminal: ${'y'.repeat(160)}`]
    }

    const { screen } = renderToScreen(
      createElement(MessageLine, {
        cols: BODY_COLS,
        detailsMode: 'expanded',
        detailsModeCommandOverride: true,
        msg,
        t: DEFAULT_THEME
      }),
      SCREEN_WIDTH
    )

    expect(paintedBeyond(screen, BODY_COLS + 2)).toBeNull()
  })
})
