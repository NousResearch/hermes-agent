import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

const APP_LAYOUT_PATH = join(dirname(fileURLToPath(import.meta.url)), '..', 'components', 'appLayout.tsx')
const source = readFileSync(APP_LAYOUT_PATH, 'utf8')

describe('AppLayout terminal width', () => {
  it('pins the root layout to the current terminal columns', () => {
    expect(source).toMatch(/<Box\s+flexDirection="column"\s+flexGrow=\{1\}\s+width=\{composer\.cols\}>/)
  })

  it('lets the main transcript row fill the root layout width', () => {
    expect(source).toMatch(/<Box\s+flexDirection="row"\s+flexGrow=\{1\}\s+width="100%">/)
  })
})
