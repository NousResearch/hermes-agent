// @vitest-environment jsdom
//
// ChatSidebar model-switch flow (#49732; the sidebar-flow test the review
// asked for). The honoured invariants:
//
//   1. With a chat live, saving a model does not move the badge until the
//      user declines the reload dialog — Cancel is what refreshes it. The
//      badge must never flip before the user has been asked.
//   2. With no live chat there is nothing to interrupt: no dialog, the badge
//      refreshes inline, and the notice says the next chat picks it up.
//   3. Closing the picker without saving still refreshes the badge, so the
//      deferral cannot swallow the ordinary close path.
//
// Harness: `web/` mounts with createRoot + act (see ChatSidebar.test.tsx);
// @testing-library/react is a desktop-app dependency, not a `web/` one.

import { act, type ReactNode } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const OLD_MODEL = 'vendor/old-model'
const NEW_MODEL = 'deepseek/deepseek-chat'

const apiMocks = vi.hoisted(() => ({
  buildWsUrl: vi.fn(async () => 'ws://localhost/api/events?channel=chat-1'),
  getModelInfo: vi.fn(),
  getModelOptions: vi.fn(),
  // Stands in for config.yaml: a save writes it, a read reflects it.
  savedModel: { current: 'vendor/old-model' },
  setModelAssignment: vi.fn()
}))

vi.mock('react-router', () => ({ useNavigate: () => vi.fn() }))

vi.mock('@/lib/api', () => ({
  HERMES_BASE_PATH: '',
  api: {
    getModelInfo: apiMocks.getModelInfo,
    getModelOptions: apiMocks.getModelOptions,
    setModelAssignment: apiMocks.setModelAssignment
  },
  buildWsUrl: apiMocks.buildWsUrl
}))

vi.mock('@/lib/gatewayClient', () => ({
  GatewayClient: class {
    close = vi.fn()
    connect = vi.fn(async () => undefined)
    on = vi.fn(() => () => undefined)
    onState = vi.fn(() => () => undefined)
    request = vi.fn(async () => ({ session_id: 'sidecar-1' }))
  }
}))

// The picker is the save surface. Its buttons stand in for a selection
// followed by the picker's own close — ModelPickerDialog calls onClose right
// after a successful onApply, which is the path the deferral has to survive.
vi.mock('@/components/ModelPickerDialog', () => ({
  ModelPickerDialog: ({
    onApply,
    onClose
  }: {
    onApply?: (args: {
      confirmExpensiveModel: boolean
      model: string
      provider: string
    }) => Promise<unknown>
    onClose?: () => void
  }) => (
    <div data-testid="picker">
      <button
        data-testid="picker-save"
        onClick={async () => {
          await onApply?.({
            confirmExpensiveModel: false,
            model: 'deepseek/deepseek-chat',
            provider: 'deepseek'
          })
          onClose?.()
        }}
      >
        save
      </button>

      <button data-testid="picker-close" onClick={onClose}>
        close
      </button>
    </div>
  )
}))

vi.mock('@/components/ReasoningPicker', () => ({
  ReasoningPicker: () => null
}))

vi.mock('@nous-research/ui/ui/components/button', () => ({
  Button: ({
    children,
    onClick,
    title
  }: {
    children?: ReactNode
    onClick?: () => void
    title?: string
  }) => (
    <button onClick={onClick} title={title}>
      {children}
    </button>
  )
}))

vi.mock('@nous-research/ui/ui/components/badge', () => ({
  Badge: ({ children }: { children?: ReactNode }) => <span>{children}</span>
}))

vi.mock('@nous-research/ui/ui/components/card', () => ({
  Card: ({ children }: { children?: ReactNode }) => <div>{children}</div>
}))

// The sidebar's event feed opens a WebSocket; jsdom has none.
class FakeWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3

  readyState = FakeWebSocket.CONNECTING
  url: string

  constructor(url: string) {
    this.url = url
  }

  addEventListener() {}

  removeEventListener() {}

  send() {}

  close() {}
}

let container: HTMLDivElement
let root: Root

async function render(ui: ReactNode) {
  container = document.createElement('div')
  document.body.append(container)
  root = createRoot(container)
  await act(async () => root.render(ui))
}

/** Retry `check` until it stops throwing (async REST reads land in effects). */
async function waitFor(check: () => void, timeoutMs = 2_000) {
  const deadline = Date.now() + timeoutMs
  for (;;) {
    try {
      check()
      return
    } catch (err) {
      if (Date.now() > deadline) throw err
      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 10))
      })
    }
  }
}

/** The sidebar's own text. The confirm dialog renders through a portal into
 *  document.body, so it is deliberately NOT part of this. */
function sidebarText() {
  return container.textContent ?? ''
}

function dialogText() {
  return document.body.textContent ?? ''
}

async function click(testId: string) {
  const el = container.querySelector<HTMLButtonElement>(`[data-testid="${testId}"]`)
  if (!el) throw new Error(`no [data-testid="${testId}"] mounted`)
  await act(async () => el.click())
}

async function openPicker() {
  const trigger = container.querySelector<HTMLButtonElement>('button[title]')
  if (!trigger) throw new Error('no model trigger button')
  await act(async () => trigger.click())
}

function buttonByLabel(label: string) {
  const el = [...document.querySelectorAll('button')].find(
    b => (b.textContent ?? '').trim() === label
  )
  if (!el) throw new Error(`no button labelled ${label}`)
  return el as HTMLButtonElement
}

/** The model badge is the picker trigger; its title carries the full model
 *  name. Asserting on this (not on textContent) separates "the badge moved"
 *  from the notice/dialog copy, which also names the model. */
function badgeModel() {
  const trigger = container.querySelector<HTMLButtonElement>('button[title]')
  if (!trigger) throw new Error('no model badge button')
  return trigger.getAttribute('title') ?? ''
}

beforeEach(() => {
  apiMocks.savedModel.current = OLD_MODEL
  apiMocks.getModelInfo.mockImplementation(async () => ({
    capabilities: { supports_reasoning: false },
    model: apiMocks.savedModel.current
  }))
  apiMocks.getModelOptions.mockResolvedValue({ providers: [] })
  apiMocks.setModelAssignment.mockImplementation(async () => {
    apiMocks.savedModel.current = NEW_MODEL
    return { confirm_required: false }
  })
  vi.stubGlobal('WebSocket', FakeWebSocket)
})

afterEach(async () => {
  await act(async () => root?.unmount())
  container?.remove()
  vi.unstubAllGlobals()
})

describe('ChatSidebar model-switch flow', () => {
  it('holds the saved model off the badge until the reload dialog is declined', async () => {
    const { ChatSidebar } = await import('./ChatSidebar')

    await render(<ChatSidebar channel="chat-1" ptyActive />)
    await waitFor(() => expect(badgeModel()).toBe(OLD_MODEL))

    await openPicker()
    await click('picker-save')

    // The save landed and the dialog is asking about the reload...
    await waitFor(() => expect(dialogText()).toContain('Switch model?'))
    expect(dialogText()).toContain('Model saved.')
    expect(apiMocks.savedModel.current).toBe(NEW_MODEL)

    // ...but the badge still names the model the running chat is using, so
    // Cancel cannot look like a no-op.
    expect(badgeModel()).toBe(OLD_MODEL)
    expect(sidebarText()).not.toContain('deepseek-chat')

    await act(async () => buttonByLabel('Cancel').click())

    // Declining the reload is what re-reads the model.
    await waitFor(() => expect(badgeModel()).toBe(NEW_MODEL))
    expect(sidebarText()).toContain('Model set to')
    expect(dialogText()).not.toContain('Switch model?')
  })

  it('refreshes inline with no dialog when no chat is live', async () => {
    const { ChatSidebar } = await import('./ChatSidebar')

    await render(<ChatSidebar channel="chat-1" ptyActive={false} />)
    await waitFor(() => expect(badgeModel()).toBe(OLD_MODEL))

    await openPicker()
    await click('picker-save')

    await waitFor(() => expect(badgeModel()).toBe(NEW_MODEL))
    expect(sidebarText()).toContain('Model set to')
    expect(dialogText()).not.toContain('Switch model?')
  })

  it('still re-reads the model when the picker is closed without saving', async () => {
    const { ChatSidebar } = await import('./ChatSidebar')

    await render(<ChatSidebar channel="chat-1" ptyActive />)
    await waitFor(() => expect(badgeModel()).toBe(OLD_MODEL))

    await openPicker()
    // The model changed elsewhere while the picker was open (the Models page,
    // another tab). No save here, so no dialog was deferred: the plain close
    // path has to pick the change up.
    apiMocks.savedModel.current = 'other/newer-model'
    await click('picker-close')

    await waitFor(() => expect(badgeModel()).toBe('other/newer-model'))
    expect(dialogText()).not.toContain('Switch model?')
  })
})
