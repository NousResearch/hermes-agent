import { describe, expect, it } from 'vitest'

import { buildShellExecParams, shellExecResultLines } from '../app/useSubmission.js'

describe('TUI shell command cancellation', () => {
  it('binds shell.exec to the live session so Ctrl+C can terminate it', () => {
    expect(buildShellExecParams('sleep 30', 'sid-123')).toEqual({
      command: 'sleep 30',
      session_id: 'sid-123'
    })
  })

  it('suppresses output from a command completed by interruption', () => {
    expect(
      shellExecResultLines({
        code: 130,
        interrupted: true,
        stderr: '',
        stdout: 'SHOULD_NOT_PRINT\n'
      })
    ).toEqual([])
  })
})
