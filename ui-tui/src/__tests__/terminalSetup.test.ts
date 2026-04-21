import { describe, expect, it, vi } from 'vitest'

import {
  configureDetectedTerminalKeybindings,
  configureTerminalKeybindings,
  detectVSCodeLikeTerminal,
  getVSCodeStyleConfigDir,
  shouldPromptForTerminalSetup,
  stripJsonComments
} from '../lib/terminalSetup.js'

describe('terminalSetup helpers', () => {
  it('detects VS Code family terminals from environment', () => {
    expect(detectVSCodeLikeTerminal({ CURSOR_TRACE_ID: 'x' } as NodeJS.ProcessEnv)).toBe('cursor')
    expect(detectVSCodeLikeTerminal({ VSCODE_GIT_ASKPASS_MAIN: '/tmp/windsurf' } as NodeJS.ProcessEnv)).toBe('windsurf')
    expect(detectVSCodeLikeTerminal({ TERM_PROGRAM: 'vscode' } as NodeJS.ProcessEnv)).toBe('vscode')
    expect(detectVSCodeLikeTerminal({} as NodeJS.ProcessEnv)).toBeNull()
  })

  it('computes VS Code style config dirs cross-platform', () => {
    expect(getVSCodeStyleConfigDir('Code', 'darwin', {} as NodeJS.ProcessEnv, '/home/me')).toBe(
      '/home/me/Library/Application Support/Code/User'
    )
    expect(getVSCodeStyleConfigDir('Code', 'linux', {} as NodeJS.ProcessEnv, '/home/me')).toBe('/home/me/.config/Code/User')
    expect(getVSCodeStyleConfigDir('Code', 'win32', { APPDATA: 'C:/Users/me/AppData/Roaming' } as NodeJS.ProcessEnv, '/home/me')).toBe(
      'C:/Users/me/AppData/Roaming/Code/User'
    )
  })

  it('strips line comments from keybindings JSON', () => {
    expect(stripJsonComments('// comment\n[{"key":"shift+enter"}]')).toBe('\n[{"key":"shift+enter"}]')
  })

  it('strips inline comments and block comments', () => {
    expect(stripJsonComments('[{"key":"a"} // inline\n]')).toBe('[{"key":"a"} \n]')
    expect(stripJsonComments('[/* block */{"key":"a"}]')).toBe('[{"key":"a"}]')
  })

  it('removes trailing commas before ] or }', () => {
    expect(JSON.parse(stripJsonComments('[{"key":"a"},]'))).toEqual([{ key: 'a' }])
    expect(JSON.parse(stripJsonComments('[{"key":"a",}]'))).toEqual([{ key: 'a' }])
  })

  it('preserves comment-like sequences inside strings', () => {
    const input = '[{"key":"a","args":{"text":"// not a comment"}}]'
    expect(JSON.parse(stripJsonComments(input))).toEqual([{ key: 'a', args: { text: '// not a comment' } }])
  })

  it('handles unterminated block comments gracefully', () => {
    const input = '[{"key":"a"} /* never closed'
    const stripped = stripJsonComments(input)
    // The unterminated comment is consumed to end-of-file; the remainder is parseable
    expect(stripped).toBe('[{"key":"a"} ')
  })
})

describe('configureTerminalKeybindings', () => {
  it('writes missing bindings into a VS Code style keybindings file', async () => {
    const mkdir = vi.fn().mockResolvedValue(undefined)
    const readFile = vi.fn().mockRejectedValue(Object.assign(new Error('missing'), { code: 'ENOENT' }))
    const writeFile = vi.fn().mockResolvedValue(undefined)
    const copyFile = vi.fn().mockResolvedValue(undefined)

    const result = await configureTerminalKeybindings('vscode', {
      fileOps: { copyFile, mkdir, readFile, writeFile },
      homeDir: '/Users/me',
      platform: 'darwin'
    })

    expect(result.success).toBe(true)
    expect(result.requiresRestart).toBe(true)
    expect(writeFile).toHaveBeenCalledTimes(1)
    expect(copyFile).not.toHaveBeenCalled() // no existing file to back up
    const written = writeFile.mock.calls[0]?.[1] as string
    expect(written).toContain('shift+enter')
    expect(written).toContain('cmd+enter')
    expect(written).toContain('cmd+z')
  })

  it('reports conflicts without overwriting existing bindings', async () => {
    const mkdir = vi.fn().mockResolvedValue(undefined)
    const readFile = vi.fn().mockResolvedValue(
      JSON.stringify([
        {
          key: 'cmd+z',
          command: 'something.else',
          when: 'terminalFocus',
          args: { text: 'noop' }
        }
      ])
    )
    const writeFile = vi.fn().mockResolvedValue(undefined)
    const copyFile = vi.fn().mockResolvedValue(undefined)

    const result = await configureTerminalKeybindings('cursor', {
      fileOps: { copyFile, mkdir, readFile, writeFile },
      homeDir: '/Users/me',
      platform: 'darwin'
    })

    expect(result.success).toBe(false)
    expect(result.message).toContain('cmd+z')
    expect(writeFile).not.toHaveBeenCalled()
    expect(copyFile).not.toHaveBeenCalled() // no backup when not writing
  })

  it('backs up existing keybindings.json only when writing changes', async () => {
    const mkdir = vi.fn().mockResolvedValue(undefined)
    const readFile = vi.fn().mockResolvedValue(JSON.stringify([]))
    const writeFile = vi.fn().mockResolvedValue(undefined)
    const copyFile = vi.fn().mockResolvedValue(undefined)

    const result = await configureTerminalKeybindings('vscode', {
      fileOps: { copyFile, mkdir, readFile, writeFile },
      homeDir: '/Users/me',
      platform: 'darwin'
    })

    expect(result.success).toBe(true)
    expect(writeFile).toHaveBeenCalledTimes(1)
    expect(copyFile).toHaveBeenCalledTimes(1) // backup created before writing
  })

  it('reports error when keybindings.json is not readable (EACCES)', async () => {
    const mkdir = vi.fn().mockResolvedValue(undefined)
    const readFile = vi.fn().mockRejectedValue(Object.assign(new Error('permission denied'), { code: 'EACCES' }))
    const writeFile = vi.fn().mockResolvedValue(undefined)
    const copyFile = vi.fn().mockResolvedValue(undefined)

    const result = await configureTerminalKeybindings('vscode', {
      fileOps: { copyFile, mkdir, readFile, writeFile },
      homeDir: '/Users/me',
      platform: 'darwin'
    })

    expect(result.success).toBe(false)
    expect(result.message).toContain('Failed to read')
    expect(writeFile).not.toHaveBeenCalled()
  })

  it('auto-detects the current IDE terminal', async () => {
    const mkdir = vi.fn().mockResolvedValue(undefined)
    const readFile = vi.fn().mockRejectedValue(Object.assign(new Error('missing'), { code: 'ENOENT' }))
    const writeFile = vi.fn().mockResolvedValue(undefined)
    const copyFile = vi.fn().mockResolvedValue(undefined)

    const result = await configureDetectedTerminalKeybindings({
      env: { CURSOR_TRACE_ID: 'trace' } as NodeJS.ProcessEnv,
      fileOps: { copyFile, mkdir, readFile, writeFile },
      homeDir: '/Users/me',
      platform: 'darwin'
    })

    expect(result.success).toBe(true)
    expect(writeFile).toHaveBeenCalled()
  })

  it('refuses to configure IDE bindings from an SSH session', async () => {
    const result = await configureDetectedTerminalKeybindings({
      env: { SSH_CONNECTION: '1 2 3 4', TERM_PROGRAM: 'vscode' } as NodeJS.ProcessEnv,
      homeDir: '/Users/me',
      platform: 'darwin'
    })

    expect(result.success).toBe(false)
    expect(result.message).toContain('local machine')
  })

  it('prompts for setup when bindings are missing and suppresses prompt when complete', async () => {
    const readMissing = vi.fn().mockRejectedValue(Object.assign(new Error('missing'), { code: 'ENOENT' }))
    await expect(
      shouldPromptForTerminalSetup({
        env: { TERM_PROGRAM: 'vscode' } as NodeJS.ProcessEnv,
        fileOps: { readFile: readMissing }
      })
    ).resolves.toBe(true)

    const readComplete = vi.fn().mockResolvedValue(
      JSON.stringify([
        {
          key: 'shift+enter',
          command: 'workbench.action.terminal.sendSequence',
          when: 'terminalFocus',
          args: { text: '\\\r\n' }
        },
        {
          key: 'ctrl+enter',
          command: 'workbench.action.terminal.sendSequence',
          when: 'terminalFocus',
          args: { text: '\\\r\n' }
        },
        {
          key: 'cmd+enter',
          command: 'workbench.action.terminal.sendSequence',
          when: 'terminalFocus',
          args: { text: '\\\r\n' }
        },
        {
          key: 'cmd+z',
          command: 'workbench.action.terminal.sendSequence',
          when: 'terminalFocus',
          args: { text: '\u001b[122;9u' }
        },
        {
          key: 'shift+cmd+z',
          command: 'workbench.action.terminal.sendSequence',
          when: 'terminalFocus',
          args: { text: '\u001b[122;10u' }
        }
      ])
    )
    await expect(
      shouldPromptForTerminalSetup({
        env: { TERM_PROGRAM: 'vscode' } as NodeJS.ProcessEnv,
        fileOps: { readFile: readComplete }
      })
    ).resolves.toBe(false)
  })

  it('suppresses terminal setup prompts inside SSH sessions', async () => {
    await expect(
      shouldPromptForTerminalSetup({
        env: { SSH_CONNECTION: '1 2 3 4', TERM_PROGRAM: 'vscode' } as NodeJS.ProcessEnv
      })
    ).resolves.toBe(false)
  })
})
