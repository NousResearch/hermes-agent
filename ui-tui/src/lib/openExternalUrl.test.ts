import type { ChildProcess } from 'node:child_process'

import { describe, expect, it, vi } from 'vitest'

import { openCommand, openExternalUrl, parseSafeUrl } from './openExternalUrl.js'

describe('parseSafeUrl', () => {
  it('accepts http and https URLs', () => {
    expect(parseSafeUrl('https://example.com')?.href).toBe('https://example.com/')
    expect(parseSafeUrl('http://example.com/path?q=1')?.href).toBe('http://example.com/path?q=1')
  })

  it('rejects file: URLs (would let a hostile model trigger arbitrary local handlers)', () => {
    expect(parseSafeUrl('file:///etc/passwd')).toBeNull()
  })

  it('rejects javascript:, data:, and vbscript: URLs', () => {
    expect(parseSafeUrl('javascript:alert(1)')).toBeNull()
    expect(parseSafeUrl('data:text/html,<script>alert(1)</script>')).toBeNull()
    expect(parseSafeUrl('vbscript:msgbox')).toBeNull()
  })

  it('rejects mailto:, ftp:, and other non-web protocols', () => {
    expect(parseSafeUrl('mailto:test@example.com')).toBeNull()
    expect(parseSafeUrl('ftp://example.com')).toBeNull()
    expect(parseSafeUrl('ssh://example.com')).toBeNull()
  })

  it('rejects unparseable strings', () => {
    expect(parseSafeUrl('not a url')).toBeNull()
    expect(parseSafeUrl('')).toBeNull()
  })

  it('rejects non-string inputs defensively', () => {
    expect(parseSafeUrl(undefined as unknown as string)).toBeNull()
    expect(parseSafeUrl(null as unknown as string)).toBeNull()
    expect(parseSafeUrl(123 as unknown as string)).toBeNull()
  })
})

describe('openCommand', () => {
  it('returns macOS open(1) on darwin', () => {
    expect(openCommand('darwin')).toEqual({ command: 'open', args: [] })
  })

  it('routes through explorer.exe on win32 — not cmd.exe — so URLs with & | ^ < > stay safe', () => {
    // win32 must not route through cmd.exe — see comment in openCommand.
    // Test pins the contract that we use explorer.exe (non-shell) so URLs
    // with `&`/`|`/`^`/`<`/`>` aren't reparsed by cmd's tokenizer.
    const cmd = openCommand('win32')
    expect(cmd?.command).toBe('explorer.exe')
    expect(cmd?.args).toEqual([])
  })

  it('falls back to xdg-open on linux/bsd', () => {
    expect(openCommand('linux')).toEqual({ command: 'xdg-open', args: [] })
    expect(openCommand('freebsd')).toEqual({ command: 'xdg-open', args: [] })
    expect(openCommand('openbsd')).toEqual({ command: 'xdg-open', args: [] })
  })
})

describe('openExternalUrl', () => {
  function mockSpawn(): {
    spawn: typeof import('node:child_process').spawn
    calls: Array<{ command: string; args: readonly string[] }>
  } {
    const calls: Array<{ command: string; args: readonly string[] }> = []
    const spawn = vi.fn((command: string, args: readonly string[]) => {
      calls.push({ command, args })

      return {
        unref: vi.fn(),
        on: vi.fn(),
        once: vi.fn()
      } as unknown as ChildProcess
    }) as unknown as typeof import('node:child_process').spawn

    return { spawn, calls }
  }

  it('opens a normal https URL via the platform command', () => {
    const { spawn, calls } = mockSpawn()

    expect(openExternalUrl('https://example.com/foo', { spawn, platform: () => 'darwin' })).toBe(true)
    expect(calls).toHaveLength(1)
    expect(calls[0]!.command).toBe('open')
    expect(calls[0]!.args).toEqual(['https://example.com/foo'])
  })

  it('uses xdg-open on linux', () => {
    const { spawn, calls } = mockSpawn()

    openExternalUrl('https://example.com/', { spawn, platform: () => 'linux' })
    expect(calls[0]!.command).toBe('xdg-open')
  })

  it('refuses to open file: URLs and does not spawn', () => {
    const { spawn, calls } = mockSpawn()

    expect(openExternalUrl('file:///etc/passwd', { spawn, platform: () => 'darwin' })).toBe(false)
    expect(calls).toHaveLength(0)
  })

  it('refuses to open javascript: URLs and does not spawn', () => {
    const { spawn, calls } = mockSpawn()

    expect(openExternalUrl('javascript:alert(1)', { spawn, platform: () => 'darwin' })).toBe(false)
    expect(calls).toHaveLength(0)
  })

  it('passes URLs containing shell metacharacters as plain args (no shell interpolation)', () => {
    const { spawn, calls } = mockSpawn()

    // A URL with `; & ` plus URL-encoded backticks. spawn(..., args) without
    // shell:true means the OS receives these as a single argv element.
    const hostile = 'https://example.com/path%3Bevil%20%26%20rm%20-rf'

    openExternalUrl(hostile, { spawn, platform: () => 'darwin' })
    expect(calls).toHaveLength(1)
    expect(calls[0]!.args[calls[0]!.args.length - 1]).toBe(hostile)
  })

  it('on win32, a URL with & | ^ < > is forwarded as a single argv element via explorer.exe', () => {
    const { spawn, calls } = mockSpawn()

    // Plain http URL with & in query (very common, e.g. analytics params)
    // plus other cmd metacharacters that would split or reinterpret the
    // command if win32 routed through cmd.exe /c start. Note that the URL
    // parser percent-encodes `<` and `>` (which is fine — encoded forms
    // can't be reinterpreted by any shell), but `&`, `|`, `^` survive
    // and would tokenize cmd.exe if we ever regressed back to it.
    const meta = 'https://example.com/q?a=1&b=2|c^d<e>f'

    expect(openExternalUrl(meta, { spawn, platform: () => 'win32' })).toBe(true)
    expect(calls).toHaveLength(1)
    expect(calls[0]!.command).toBe('explorer.exe')
    // The URL must arrive as exactly one argv element — not split on &/|/^/etc.
    const forwarded = calls[0]!.args[0]!
    expect(calls[0]!.args).toHaveLength(1)
    expect(forwarded).toContain('a=1&b=2')
    expect(forwarded).toContain('|c^d')
  })

  it('on win32, common http URLs with & query params are forwarded intact', () => {
    const { spawn, calls } = mockSpawn()
    const url = 'https://example.com/search?q=foo&page=2&utm_source=hermes'

    openExternalUrl(url, { spawn, platform: () => 'win32' })
    expect(calls[0]!.args).toEqual([url])
  })

  it('returns false on synchronous spawn failure', () => {
    const spawn = vi.fn(() => {
      throw new Error('ENOENT')
    }) as unknown as typeof import('node:child_process').spawn

    expect(openExternalUrl('https://example.com/', { spawn, platform: () => 'linux' })).toBe(false)
  })
})
