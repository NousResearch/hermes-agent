// @vitest-environment jsdom
import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { applyProcessSnapshot, useProcessRows } from '../app/processRoster.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { applyLocale, resetLocale } from '../i18n/runtime.js'

let root: Root
let container: HTMLDivElement

function Probe() {
  const row = useProcessRows(1000)[0]

  return (
    <span>
      {row?.command} / {row?.detail}
    </span>
  )
}

beforeEach(() => {
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
  resetUiState()
  patchUiState({ sid: 'current' })
  applyProcessSnapshot('current', [{ session_id: 'background', status: 'running' }])
  container = document.createElement('div')
  root = createRoot(container)
})

afterEach(async () => {
  await act(async () => root.unmount())
  resetLocale()
  resetUiState()
  applyProcessSnapshot(null)
  vi.unstubAllGlobals()
})

it('refreshes process copy when the same language receives a new overlay', async () => {
  applyLocale('pack', { lang: 'pack', surface: 'tui', messages: { 'process.background': 'First overlay' } })
  await act(async () => root.render(<Probe />))
  expect(container.textContent).toContain('First overlay')

  await act(async () => {
    applyLocale('pack', { lang: 'pack', surface: 'tui', messages: { 'process.background': 'Updated overlay' } })
  })
  expect(container.textContent).toContain('Updated overlay')
  expect(container.textContent).not.toContain('First overlay')
})
