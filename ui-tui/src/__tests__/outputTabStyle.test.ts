import { expect, it } from 'vitest'

import Output from '../../packages/hermes-ink/src/ink/output.js'
import { cellAt, CharPool, createScreen, HyperlinkPool, StylePool } from '../../packages/hermes-ink/src/ink/screen.js'

it('expands tabs like styled spaces at absolute tab stops, including cached and right-edge writes', () => {
  for (const width of [6, 20]) {
    for (const x of [0, 3]) {
      const stylePool = new StylePool()
      const screen = createScreen(width, 1, stylePool, new CharPool(), new HyperlinkPool())
      const output = new Output({ width, height: 1, stylePool, screen })
      const link = '\u001b]8;;https://example.invalid/fixture\u0007'
      const closeLink = '\u001b]8;;\u0007'

      for (const prefix of ['', '\u001b[41m', '\u001b[4m', link + '\u001b[44m']) {
        const text = prefix + 'x\t' + '\u001b[0m' + closeLink + 'y'

        // Reuse the same Output so the second pass also exercises charCache.
        for (let frame = 0; frame < 2; frame++) {
          output.reset(width, 1, screen)
          output.write(x, 0, text)
          output.get()
          const first = cellAt(screen, x, 0)!
          const stop = Math.min(8, width)

          for (let col = x + 1; col < stop; col++) {
            expect(cellAt(screen, col, 0)).toEqual({ ...first, char: ' ' })
          }

          if (width > 8) {
            expect(cellAt(screen, 8, 0)).toMatchObject({ char: 'y', styleId: stylePool.none, hyperlink: undefined })
          }

          for (let col = 0; col < x; col++) {
            expect(cellAt(screen, col, 0)?.styleId).toBe(stylePool.none)
          }
        }
      }
    }
  }
})
