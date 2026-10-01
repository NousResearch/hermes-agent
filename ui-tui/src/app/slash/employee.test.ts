import { expect, it } from 'vitest'

import { visibleEmployeeCommand } from './employee.js'
import { findSlashCommand } from './registry.js'

it('hides controls while preserving native local dispatch', () => {
  for (const name of ['skills', 'reload-skills', 'cron', 'personality']) {
    expect(visibleEmployeeCommand(`/${name}`)).toBe(false)
    if (name !== 'cron') {
      expect(findSlashCommand(name)).toBeDefined()
    }
  }

  expect(visibleEmployeeCommand('/stop')).toBe(true)
})
