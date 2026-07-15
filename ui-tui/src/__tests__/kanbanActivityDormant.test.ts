import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'
const srcRoot = fileURLToPath(new URL('../', import.meta.url))
const read = (relative: string) => readFileSync(new URL(relative, `file://${srcRoot}/`), 'utf8')
describe('dormant Kanban activity slice', () => {
  it('is not mounted, polled, or imported by live composition', () => {
    for (const path of ['app/useMainApp.ts', 'components/appLayout.tsx', 'app/interfaces.ts']) {expect(read(path)).not.toMatch(/kanbanActivity|KanbanActivity|kanban\.activity/)}
  })
  it('does not import gateway, transcript, message, or session stores', () => {
    expect([read('lib/kanbanActivity.ts'), read('components/kanbanActivity.tsx')].join('\n')).not.toMatch(/gatewayClient|historyItems|appendMessage|messageStore|sessionStore|uiStore/)
  })
})
