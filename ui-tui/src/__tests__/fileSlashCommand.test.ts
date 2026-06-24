// fileSlashCommand.test.ts
import { describe, expect, it } from 'vitest'

import { coreCommands } from '../app/slash/commands/core.js'

describe('coreCommands', () => {
  it('includes /file command', () => {
    const fileCmd = coreCommands.find(c => c.name === 'file')
    expect(fileCmd).toBeDefined()
    expect(fileCmd?.help).toBe('open local file in preview panel')
    expect(fileCmd?.usage).toBe('/file <path>')
  })
})