import { describe, expect, it } from 'vitest'

import { suspendOnCtrlZ } from './components/App.js'

describe('suspendOnCtrlZ', () => {
  it('is off unless HERMES_TUI_CTRL_Z_SUSPEND opts in', () => {
    expect(suspendOnCtrlZ({}, 'linux')).toBe(false)
    expect(suspendOnCtrlZ({ HERMES_TUI_CTRL_Z_SUSPEND: '0' }, 'linux')).toBe(false)
    expect(suspendOnCtrlZ({ HERMES_TUI_CTRL_Z_SUSPEND: '1' }, 'linux')).toBe(true)
    expect(suspendOnCtrlZ({ HERMES_TUI_CTRL_Z_SUSPEND: 'true' }, 'darwin')).toBe(true)
  })

  it('never suspends on Windows', () => {
    expect(suspendOnCtrlZ({ HERMES_TUI_CTRL_Z_SUSPEND: '1' }, 'win32')).toBe(false)
  })

  it('never suspends under the dashboard PTY, which has no shell to resume it', () => {
    expect(suspendOnCtrlZ({ HERMES_TUI_CTRL_Z_SUSPEND: '1', HERMES_PTY_HOST: 'dashboard' }, 'linux')).toBe(false)
  })
})
