import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { CompletionItem } from '../app/interfaces.js'
import { SLASH_COMMANDS } from '../app/slash/registry.js'
import { resetUiState } from '../app/uiStore.js'
import type { GatewayClient } from '../gatewayClient.js'
import { mergeTuiCommandItems, useCompletion } from '../hooks/useCompletion.js'

// Reproduce the screenshot: the gateway knows these skill matches, but not
// the client-only /tab controls.
const skills: CompletionItem[] = [
  {
    text: 'airtable ',
    display: '/airtable',
    meta: 'Airtable REST API. Records CRUD, filters, upserts.',
    kind: 'skill'
  },
  { text: 'notion ', display: '/notion', meta: 'Notion API: databases and tables.', kind: 'skill' }
]

const teardown: (() => void)[] = []

function mount(input: string, request: ReturnType<typeof vi.fn>, blocked = false) {
  let state!: ReturnType<typeof useCompletion>
  const gateway = { request } as unknown as GatewayClient

  function Harness({ value, disabled }: { value: string; disabled: boolean }) {
    state = useCompletion(value, disabled, gateway)

    return <Text>{state.completions.map(item => `${item.display} ${item.meta ?? ''}`).join('\n')}</Text>
  }

  const instance = renderSync(<Harness disabled={blocked} value={input} />, {
    stdin: new PassThrough() as never,
    stdout: Object.assign(new PassThrough(), { columns: 120, rows: 30, isTTY: false }) as never,
    stderr: new PassThrough() as never,
    patchConsole: false
  })

  teardown.push(() => {
    instance.unmount()
    instance.cleanup()
  })

  return {
    current: () => state,
    update: (value: string, disabled = false) => instance.rerender(<Harness disabled={disabled} value={value} />)
  }
}

afterEach(() => {
  while (teardown.length) {
    teardown.pop()!()
  }

  resetUiState()
})

describe('TUI slash suggestion candidates', () => {
  it('ranks the exact /tab and its prefixes before unrelated skill matches', () => {
    const items = mergeTuiCommandItems('/tab', skills)
    expect(items[0]?.text).toBe('/tab')

    for (const name of ['/tabs', '/tabnew', '/tabnext', '/tabprev', '/tabclose']) {
      expect(items.findIndex(item => item.text === name)).toBeGreaterThan(0)
      expect(items.findIndex(item => item.text === name)).toBeLessThan(
        items.findIndex(item => item.display === '/airtable')
      )
    }

    expect(items).toContain(skills[0])
    expect(items).toContain(skills[1])
  })

  it('discovers every local command and alias directly from the dispatch registry', () => {
    const names = mergeTuiCommandItems('/', []).map(item => item.text)

    for (const command of SLASH_COMMANDS) {
      for (const name of [command.name, ...(command.aliases ?? [])]) {
        expect(names).toContain(`/${name}`)
      }
    }

    expect(new Set(names).size).toBe(names.length)
  })

  it.each(['/tab', '/tabs'])('suggests the existing tab command for %s without changing its handler', query => {
    const items = mergeTuiCommandItems(query, [])
    expect(items[0]).toMatchObject({ text: query, meta: expect.stringContaining('list or select tabs') })
  })

  it('suggests tab aliases', () => {
    for (const name of ['/tabnext', '/tabnew', '/next', '/prev', '/previous']) {
      expect(mergeTuiCommandItems(name, [])[0]?.text).toBe(name)
    }
  })

  it('keeps the local /new help and does not duplicate backend names', () => {
    const items = mergeTuiCommandItems('/new', [
      { text: '/new', display: '/new', meta: 'clear the current session', kind: 'command' }
    ])

    expect(items.filter(item => item.text === '/new')).toEqual([
      expect.objectContaining({ meta: expect.stringContaining('without closing or interrupting') })
    ])
  })

  it('preserves plugin completion metadata', () => {
    const plugin = { text: '/plugin-action', display: '/plugin-action', meta: 'A plugin command', kind: 'plugin' }
    expect(mergeTuiCommandItems('/plugin-action', [plugin])[0]).toBe(plugin)
  })

  it('deduplicates bare gateway names and preserves their insertion text and trailing space', () => {
    const items = mergeTuiCommandItems('/message', [
      { text: 'steer ', display: '/steer', meta: 'Inject a message without interrupting', kind: 'command' },
      { text: 's ', display: '/s', meta: 'Inject a message', kind: 'command' }
    ])

    expect(items.filter(item => item.display === '/steer')).toHaveLength(1)
    expect(items.filter(item => item.display === '/s')).toHaveLength(1)
    expect(items.find(item => item.display === '/steer')?.text).toBe('steer ')
    expect(items.find(item => item.display === '/s')?.text).toBe('s ')
  })

  it.each(['/tab next', 'use /tab', '/help /message'])(
    'does not merge command names into arguments or inline references: %s',
    input => {
      expect(mergeTuiCommandItems(input, skills)).toBe(skills)
    }
  )
})

describe('live completion hook', () => {
  it('shows local suggestions even while the backend completion request remains pending', async () => {
    const request = vi.fn(() => new Promise<never>(() => {}))
    const view = mount('/tab', request)
    await vi.waitFor(() => expect(view.current().completions[0]?.text).toBe('/tab'))
    await vi.waitFor(() => expect(request).toHaveBeenCalledWith('complete.slash', { text: '/tab' }))
    expect(view.current().completions[0]?.text).toBe('/tab')
    expect(view.current().compReplace).toBe(1)
  })

  it('keeps the local commands after the real-shaped backend reply arrives', async () => {
    const request = vi.fn(async (_method: string) => ({ items: skills, replace_from: 1 }))
    const view = mount('/tab', request)
    await vi.waitFor(() => expect(view.current().completions).toContain(skills[0]))
    expect(view.current().completions[0]?.text).toBe('/tab')
    expect(request.mock.calls.every(([method]) => method === 'complete.slash')).toBe(true)
  })

  it('retains local tab suggestions when backend completion fails', async () => {
    const request = vi.fn(async () => {
      throw new Error('test backend is unavailable')
    })

    const view = mount('/tab', request)
    await vi.waitFor(() => expect(request).toHaveBeenCalled())
    await new Promise(resolve => setImmediate(resolve))
    expect(view.current().completions[0]?.text).toBe('/tab')
  })

  it('keeps mid-message completion skills-only', async () => {
    const request = vi.fn(async () => ({
      items: [...skills, { text: '/tab', display: '/tab', kind: 'command' }],
      replace_from: 1
    }))

    const view = mount('use /tab', request)
    await vi.waitFor(() => expect(view.current().completions).toHaveLength(2))
    expect(view.current().completions).toEqual(skills)
    expect(view.current().compReplace).toBe(5)
  })

  it('leaves argument completion and its replacement offset unchanged', async () => {
    const args = [{ text: 'next', display: 'next', meta: 'next tab' }]
    const request = vi.fn(async () => ({ items: args, replace_from: 5 }))
    const view = mount('/tab ne', request)
    await vi.waitFor(() => expect(view.current().completions).toEqual(args))
    expect(view.current().compReplace).toBe(5)
  })

  it('does not let a stale backend reply replace the next command suggestions', async () => {
    let reply!: (value: unknown) => void

    const request = vi.fn(
      () =>
        new Promise(resolve => {
          reply = resolve
        })
    )

    const view = mount('/tab', request)
    await vi.waitFor(() => expect(request).toHaveBeenCalled())
    const oldReply = reply
    view.update('/tabclose')
    await vi.waitFor(() => expect(view.current().completions[0]?.text).toBe('/tabclose'))
    oldReply({ items: skills, replace_from: 1 })
    await new Promise(resolve => setImmediate(resolve))
    expect(view.current().completions[0]?.text).toBe('/tabclose')
    expect(view.current().completions).not.toContain(skills[0])
  })

  it('clears the menu without requesting completions while an overlay blocks input', async () => {
    const request = vi.fn(async () => ({ items: skills }))
    const view = mount('/tab', request, true)
    await new Promise(resolve => setTimeout(resolve, 90))
    expect(view.current().completions).toEqual([])
    expect(request).not.toHaveBeenCalled()
  })
})
