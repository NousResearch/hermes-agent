import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import type * as Ink from '@hermes/ink'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('@hermes/ink', async importOriginal => ({
  ...(await importOriginal<typeof Ink>()),
  useInput: () => {}
}))

import { ApprovalPrompt, ConfirmPrompt } from '../components/prompts.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

import { en } from './en.js'
import { sv } from './sv.js'

import { getTranslations, setTuiLanguage } from './index.js'

afterEach(() => setTuiLanguage('en'))

function render(element: React.ReactElement): string {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  Object.assign(stdout, { columns: 100, isTTY: false, rows: 40 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  let output = ''
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(element, {
    patchConsole: false,
    stdout: stdout as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stderr: stderr as unknown as NodeJS.WriteStream
  })

  instance.unmount()
  instance.cleanup()

  return stripAnsi(output)
}

describe('Swedish TUI', () => {
  it.each(['sv', 'sv-SE', 'sv-FI', 'sv_SE', 'sv_FI', 'svenska', 'Swedish'])('resolves %s', value => {
    setTuiLanguage(value)
    expect(getTranslations()).toBe(sv)
  })

  it.each([undefined, '', 'unknown', 'en', 'de'])('falls back for %s', value => {
    setTuiLanguage('sv')
    setTuiLanguage(value)
    expect(getTranslations()).toBe(en)
  })

  it('covers the English keys and preserves interpolation values', () => {
    expect(Object.keys(sv).sort()).toEqual(Object.keys(en).sort())

    for (const section of Object.keys(en) as (keyof typeof en)[]) {
      expect(Object.keys(sv[section]).sort()).toEqual(Object.keys(en[section]).sort())

      for (const [key, value] of Object.entries(en[section])) {
        const translated = (sv[section] as Record<string, unknown>)[key]
        expect(typeof translated).toBe(typeof value)

        if (typeof value === 'function') {
          const cases: unknown[][] = [
            ...[0, 1, 2].map(count => Array.from({ length: value.length }, (_, index) => count + index * 7)),
            Array.from({ length: value.length }, (_, index) => `ARG_${index}_END`)
          ]

          for (const args of cases) {
            const source = (value as (...args: unknown[]) => string)(...args)
            const target = (translated as (...args: unknown[]) => string)(...args)

            for (const argument of args) {
              if (source.includes(String(argument))) {
                expect(target).toContain(String(argument))
              }
            }
          }
        } else {
          expect(String(translated).trim()).not.toBe('')
        }
      }
    }
  })

  it('renders Swedish approval labels without changing command text or choice ordering', () => {
    setTuiLanguage('sv')

    const output = render(
      <ApprovalPrompt
        onChoice={() => {}}
        req={{ command: 'example --unchanged', description: 'External description' }}
        t={DEFAULT_THEME}
      />
    )

    expect(output).toContain('godkännande krävs')
    expect(output).toContain('example --unchanged')
    expect(output).toContain('External description')
    expect(output).toContain('1. Tillåt en gång')
    expect(output).toContain('2. Tillåt under sessionen')
    expect(output).toContain('3. Tillåt alltid')
    expect(output).toContain('4. Neka')
  })

  it('renders Swedish confirmation defaults and preserves caller-provided labels', () => {
    setTuiLanguage('sv')

    const defaults = render(
      <ConfirmPrompt
        onCancel={() => {}}
        onConfirm={() => {}}
        req={{ onConfirm: () => {}, title: 'Example' }}
        t={DEFAULT_THEME}
      />
    )

    expect(defaults).toContain('Nej')
    expect(defaults).toContain('Ja')
    expect(defaults).toContain('Y/N')

    const custom = render(
      <ConfirmPrompt
        onCancel={() => {}}
        onConfirm={() => {}}
        req={{ onConfirm: () => {}, title: 'Example', confirmLabel: 'Custom action' }}
        t={DEFAULT_THEME}
      />
    )

    expect(custom).toContain('Custom action')
  })
})
