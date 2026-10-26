// @vitest-environment jsdom

import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import ComputerPage from './ComputerPage'

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true

const { getSessions, getComputerStatus, authedFetch } = vi.hoisted(() => ({
  getSessions: vi.fn(),
  getComputerStatus: vi.fn(),
  authedFetch: vi.fn(),
}))

vi.mock('@nous-research/ui/ui/components/button', () => ({
  Button: ({
    children,
    onClick,
    disabled,
    ...props
  }: React.ButtonHTMLAttributes<HTMLButtonElement>) => (
    <button type="button" onClick={onClick} disabled={disabled} {...props}>
      {children}
    </button>
  )
}))

vi.mock('@nous-research/ui/ui/components/segmented', () => ({
  Segmented: ({
    value,
    onChange
  }: {
    value: string
    onChange: (v: string) => void
    options: Array<{ value: string; label: string }>
  }) => (
    <span data-testid="segmented" data-value={value} onClick={() => onChange('10s')}>
      {value}
    </span>
  )
}))

vi.mock('@nous-research/ui/ui/components/card', () => ({
  Card: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>,
  CardContent: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>,
  CardHeader: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>,
  CardTitle: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>
}))

vi.mock('@nous-research/ui/ui/components/badge', () => ({
  Badge: ({ children }: { children?: React.ReactNode }) => <span>{children}</span>
}))

vi.mock('@nous-research/ui/ui/components/spinner', () => ({
  Spinner: () => <span>Loading</span>
}))

vi.mock('@/contexts/usePageHeader', () => ({
  usePageHeader: () => ({ setAfterTitle: vi.fn(), setEnd: vi.fn() })
}))

vi.mock('@/lib/api', () => ({
  authedFetch,
  fetchJSON: async (url: string) => {
    if (url.startsWith('/api/computer/status')) return getComputerStatus()
    if (url.startsWith('/api/sessions')) return getSessions()
    throw new Error(`unmocked ${url}`)
  },
  getManagementProfile: () => 'default'
}))

vi.mock('@/lib/utils', () => ({
  timeAgo: () => 'just now'
}))

function session(id: string) {
  return {
    id,
    source: 'desktop',
    model: null,
    title: `Session ${id}`,
    started_at: 1,
    ended_at: null,
    last_active: Date.now() / 1000,
    is_active: id === 'live-one',
    message_count: 3,
    tool_call_count: 0,
    input_tokens: 0,
    output_tokens: 0,
    preview: null
  }
}

const urlStub = vi.fn(() => 'blob:mock')
const revokeStub = vi.fn()

beforeEach(() => {
  vi.stubGlobal('URL', Object.assign(URL, { createObjectURL: urlStub, revokeObjectURL: revokeStub }))
  urlStub.mockClear()
  revokeStub.mockClear()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('ComputerPage', () => {
  let container: HTMLDivElement
  let root: Root

  beforeEach(() => {
    container = document.createElement('div')
    document.body.appendChild(container)
    root = createRoot(container)
    getSessions.mockResolvedValue({ sessions: [session('live-one'), session('idle-one')], total: 2, limit: 24, offset: 0 })
  })

  afterEach(async () => {
    await act(async () => {
      root.unmount()
    })
    container.remove()
  })

  it('renders the live-session grid with active dots and the opt-in hint when disabled', async () => {
    getComputerStatus.mockResolvedValue({ enabled: false, last_capture_at: null, count: 0 })
    await act(async () => {
      root.render(<MemoryRouter><ComputerPage /></MemoryRouter>)
    })
    const cells = document.querySelectorAll('button[title^="Session "]')
    expect(cells.length).toBe(2)
    expect(document.body.textContent).toContain('Session live-one')
    expect(document.body.textContent).toContain('dashboard.computer_view')
    expect(authedFetch).not.toHaveBeenCalled()
  })

  it('captures automatically once when opted in and exposes capture controls', async () => {
    getComputerStatus.mockResolvedValue({ enabled: true, last_capture_at: 1_000, count: 1 })
    authedFetch.mockResolvedValue(new Response(new Blob(['png'])))
    await act(async () => {
      root.render(<MemoryRouter><ComputerPage /></MemoryRouter>)
    })
    // second act + waitFor: the capture rides the status->shot effect wave
    await act(async () => {
      await Promise.resolve()
    })
    await vi.waitFor(() => expect(authedFetch).toHaveBeenCalledTimes(1))
    await act(async () => {
      await Promise.resolve()
    })
    expect(authedFetch).toHaveBeenCalledTimes(1)
    expect(document.body.textContent).toContain('Capture now')
    expect(document.body.textContent).not.toContain('dashboard.computer_view')
  })

  it('does not crash when the status probe fails', async () => {
    getComputerStatus.mockRejectedValue(new Error('boom'))
    await act(async () => {
      root.render(<MemoryRouter><ComputerPage /></MemoryRouter>)
    })
    expect(document.body.textContent).toContain('Session live-one')
  })
})