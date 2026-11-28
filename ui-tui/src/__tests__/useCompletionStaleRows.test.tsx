import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { resetUiState } from '../app/uiStore.js'
import { useCompletion } from '../hooks/useCompletion.js'

const wait = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

afterEach(() => resetUiState())

// Enter/Tab act on the held rows only when they answer the text being acted on. A key burst
// (`/` opens the menu, then `zzqx` + Enter land before the refresh) used to apply row 0 of the
// `/` menu, so typing an unknown command ran `/new`.
it('held rows name the input they answer until the refresh for the new text lands', async () => {
  const pending: Array<() => void> = []

  const request = vi.fn((_method: string, params: { text: string }) =>
    params.text === '/'
      ? Promise.resolve({ items: [{ display: '/new', text: '/new' }], replace_from: 1 })
      : new Promise(resolve => pending.push(() => resolve({ items: [], replace_from: 1 })))
  )

  let input = '/'
  let rerender!: () => void
  let seen!: ReturnType<typeof useCompletion>

  function Harness() {
    seen = useCompletion(input, false, { request } as never)

    return <Text>{seen.completions.length}</Text>
  }

  const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 10 })

  const instance = renderSync(<Harness />, {
    patchConsole: false,
    stderr: new PassThrough() as never,
    stdout: stdout as never
  })

  rerender = () => instance.rerender(<Harness />)

  try {
    await wait(120)
    expect(seen.completions.map(c => c.text)).toEqual(['/new'])
    expect(seen.compFor).toBe('/')

    input = '/zzqx'
    rerender()
    await wait(120)
    // The `/` rows are still on screen (no blink), but they do not answer `/zzqx`.
    expect(seen.completions.map(c => c.text)).toEqual(['/new'])
    expect(seen.compFor).not.toBe(input)

    pending.splice(0).forEach(release => release())
    await wait(40)
    expect(seen.completions).toEqual([])
  } finally {
    instance.unmount()
    instance.cleanup()
  }
})
