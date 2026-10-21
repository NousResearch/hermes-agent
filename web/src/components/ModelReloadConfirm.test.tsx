// @vitest-environment jsdom
//
// ModelReloadConfirm: the copy is the user-visible contract from #49732. By
// the time this dialog renders the model is already saved, so the body must
// describe the RELOAD as the thing that starts a fresh chat — the old
// "Switching to X starts a fresh chat" wording read as if Cancel could undo
// the save. Harness: createRoot + act, as in the rest of `web/`.

import { act, type ReactNode } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ModelReloadConfirm } from './ModelReloadConfirm'

vi.mock('@nous-research/ui/ui/components/button', () => ({
  Button: ({
    children,
    onClick
  }: {
    children?: ReactNode
    onClick?: () => void
  }) => <button onClick={onClick}>{children}</button>
}))

let container: HTMLDivElement
let root: Root

async function render(ui: ReactNode) {
  container = document.createElement('div')
  document.body.append(container)
  root = createRoot(container)
  await act(async () => root.render(ui))
}

function bodyText() {
  return document.body.textContent ?? ''
}

function buttonByLabel(label: string) {
  const el = [...document.querySelectorAll('button')].find(
    b => (b.textContent ?? '').trim() === label
  )
  if (!el) throw new Error(`no button labelled ${label}`)
  return el as HTMLButtonElement
}

let reloadSpy: ReturnType<typeof vi.fn>

beforeEach(() => {
  reloadSpy = vi.fn()
  Object.defineProperty(window, 'location', {
    configurable: true,
    value: { ...window.location, reload: reloadSpy },
    writable: true
  })
})

afterEach(async () => {
  await act(async () => root?.unmount())
  container?.remove()
})

describe('ModelReloadConfirm', () => {
  it('says the model is saved, names it, and drops the misleading switch wording', async () => {
    await render(<ModelReloadConfirm model="sonnet" onCancel={() => {}} />)

    expect(bodyText()).toContain('Model saved.')
    expect(bodyText()).toContain('sonnet')
    expect(bodyText()).not.toContain('Switching to')
  })

  it('lets a caller replace the body copy (Models page path)', async () => {
    await render(
      <ModelReloadConfirm
        model="sonnet"
        onCancel={() => {}}
        description="Custom copy for the Models page."
      />
    )

    expect(bodyText()).toContain('Custom copy for the Models page.')
    expect(bodyText()).not.toContain('Model saved.')
  })

  it('renders nothing while no model is pending', async () => {
    await render(<ModelReloadConfirm model={null} onCancel={() => {}} />)

    expect(bodyText()).toBe('')
  })

  it('reloads the page on Reload and hands Cancel to the caller', async () => {
    const onCancel = vi.fn()
    await render(<ModelReloadConfirm model="sonnet" onCancel={onCancel} />)

    await act(async () => buttonByLabel('Reload').click())
    expect(reloadSpy).toHaveBeenCalledTimes(1)
    expect(onCancel).not.toHaveBeenCalled()

    await act(async () => buttonByLabel('Cancel').click())
    expect(onCancel).toHaveBeenCalledTimes(1)
  })
})
