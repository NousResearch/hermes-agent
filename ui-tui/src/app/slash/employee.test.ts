import { expect, it } from 'vitest'

import { findSlashCommand } from './registry.js'

it('keeps employee exclusions out of local command dispatch', () => {
  for (const name of ['skills', 'reload-skills', 'cron', 'personality', 'kanban']) {
    expect(findSlashCommand(name)).toBeUndefined()
  }

  expect(findSlashCommand('stop')).toBeDefined()
})
