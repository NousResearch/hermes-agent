// @vitest-environment jsdom

import { act, useState } from 'react'
import { flushSync } from 'react-dom'
import { createRoot, type Root } from 'react-dom/client'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ChatSessionList } from './ChatSessionList'

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true

const { deleteSession, getSessions } = vi.hoisted(() => ({
  deleteSession: vi.fn(),
  getSessions: vi.fn()
}))

vi.mock('@nous-research/ui/ui/components/button', () => ({
  Button: ({
    children,
    ghost,
    onClick,
    outlined,
    prefix,
    size,
    ...props
  }: React.ButtonHTMLAttributes<HTMLButtonElement> & { ghost?: boolean; outlined?: boolean; size?: string }) => {
    void ghost
    void outlined
    void prefix
    void size
    return (
      <button type="button" onClick={onClick} {...props}>
        {children}
      </button>
    )
  }
}))

vi.mock('@nous-research/ui/ui/components/list-item', () => ({
  ListItem: ({ children, onClick, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) => (
    <button type="button" onClick={onClick} {...props}>
      {children}
    </button>
  )
}))

vi.mock('@nous-research/ui/ui/components/spinner', () => ({
  Spinner: () => <span>Loading</span>
}))

vi.mock('./DeleteConfirmDialog', () => ({
  DeleteConfirmDialog: ({
    onCancel,
    onConfirm,
    open,
    title
  }: {
    onCancel: () => void
    onConfirm: () => void
    open: boolean
    title: string
  }) =>
    open ? (
      <div role="dialog" aria-label={title}>
        <button type="button" onClick={onCancel}>Cancel delete</button>
        <button type="button" onClick={onConfirm}>Confirm delete</button>
      </div>
    ) : null
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { loading: 'Loading', refresh: 'Refresh', retry: 'Retry' },
      sessions: {
        confirmDeleteMessage: 'This permanently deletes the conversation.',
        confirmDeleteTitle: 'Delete session?',
        deleteSession: 'Delete session',
        failedToDelete: 'Failed to delete session',
        newChat: 'New chat',
        noSessions: 'No sessions',
        sessionDeleted: 'Session deleted',
        title: 'Sessions',
        untitledSession: 'Untitled session'
      }
    }
  })
}))

vi.mock('@/lib/api', () => ({ api: { deleteSession, getSessions } }))

vi.mock('@/lib/utils', () => ({
  cn: (...classes: Array<string | false | null | undefined>) => classes.filter(Boolean).join(' '),
  timeAgo: () => 'just now'
}))

function session(id: string, title: string) {
  return {
    id,
    source: 'cli',
    model: null,
    title,
    started_at: 0,
    ended_at: null,
    last_active: 0,
    is_active: false,
    message_count: 0,
    tool_call_count: 0,
    input_tokens: 0,
    output_tokens: 0,
    preview: null
  }
}

function response(id: string, title: string) {
  return { sessions: [session(id, title)], total: 1, limit: 30, offset: 0 }
}

describe('ChatSessionList', () => {
  let container: HTMLDivElement
  let root: Root

  beforeEach(() => {
    vi.useFakeTimers()
    deleteSession.mockReset()
    getSessions.mockReset()
    Object.defineProperty(document, 'visibilityState', {
      configurable: true,
      value: 'visible'
    })
    container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
  })

  afterEach(async () => {
    await act(async () => root.unmount())
    container.remove()
    vi.useRealTimers()
  })

  it('renders sessions returned by an automatic refresh', async () => {
    getSessions
      .mockResolvedValueOnce(response('first', 'First session'))
      .mockResolvedValueOnce(response('second', 'Newly created session'))

    await act(async () => {
      root.render(
        <MemoryRouter>
          <ChatSessionList activeSessionId={null} />
        </MemoryRouter>
      )
    })

    expect(container.textContent).toContain('First session')
    expect(getSessions).toHaveBeenCalledTimes(1)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(20_000)
    })

    expect(getSessions).toHaveBeenCalledTimes(2)
    expect(container.textContent).toContain('Newly created session')
  })

  it('refreshes immediately when the tab becomes visible', async () => {
    getSessions
      .mockResolvedValueOnce(response('first', 'First session'))
      .mockResolvedValueOnce(response('second', 'Newly created session'))

    await act(async () => {
      root.render(
        <MemoryRouter>
          <ChatSessionList activeSessionId={null} />
        </MemoryRouter>
      )
    })

    Object.defineProperty(document, 'visibilityState', {
      configurable: true,
      value: 'visible'
    })
    await act(async () => {
      document.dispatchEvent(new Event('visibilitychange'))
    })

    expect(getSessions).toHaveBeenCalledTimes(2)
    expect(container.textContent).toContain('Newly created session')
  })

  it('waits for the Chat tab to become active before loading or polling', async () => {
    getSessions
      .mockResolvedValueOnce(response('first', 'First session'))
      .mockResolvedValueOnce(response('second', 'Newly created session'))

    await act(async () => {
      root.render(
        <MemoryRouter>
          <ChatSessionList activeSessionId={null} isActive={false} />
        </MemoryRouter>
      )
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(20_000)
    })
    expect(getSessions).not.toHaveBeenCalled()

    await act(async () => {
      root.render(
        <MemoryRouter>
          <ChatSessionList activeSessionId={null} isActive />
        </MemoryRouter>
      )
    })

    expect(getSessions).toHaveBeenCalledTimes(1)
    expect(container.textContent).toContain('First session')

    await act(async () => {
      await vi.advanceTimersByTimeAsync(20_000)
    })

    expect(getSessions).toHaveBeenCalledTimes(2)
    expect(container.textContent).toContain('Newly created session')
  })

  it('shows a delete button and removes the confirmed session without opening it', async () => {
    getSessions.mockResolvedValueOnce({
      sessions: [{ ...session('first', 'First session'), profile: 'row-owner' }],
      total: 1,
      limit: 30,
      offset: 0
    })
    deleteSession.mockResolvedValueOnce({ ok: true })

    await act(async () => {
      root.render(
        <MemoryRouter initialEntries={['/chat']}>
          <ChatSessionList activeSessionId={null} profile="worker" />
        </MemoryRouter>
      )
    })

    const deleteButton = container.querySelector<HTMLButtonElement>('[aria-label="Delete session: First session"]')
    expect(deleteButton).not.toBeNull()

    await act(async () => deleteButton!.click())
    expect(container.querySelector('[role="dialog"]')).not.toBeNull()

    const confirm = Array.from(container.querySelectorAll('button')).find(
      button => button.textContent === 'Confirm delete'
    )
    await act(async () => confirm!.click())

    expect(deleteSession).toHaveBeenCalledWith('first', 'row-owner')
    expect(container.textContent).not.toContain('First session')
  })

  it('starts a fresh chat after successfully deleting the session still being viewed', async () => {
    getSessions.mockResolvedValueOnce(response('first', 'First session'))
    deleteSession.mockResolvedValueOnce({ ok: true })
    const onNewChat = vi.fn()

    await act(async () => {
      root.render(
        <MemoryRouter initialEntries={['/chat?resume=first']}>
          <ChatSessionList activeSessionId="first" onNewChat={onNewChat} />
        </MemoryRouter>
      )
    })

    const deleteButton = container.querySelector<HTMLButtonElement>('[aria-label="Delete session: First session"]')
    await act(async () => deleteButton!.click())
    const confirm = Array.from(container.querySelectorAll('button')).find(
      button => button.textContent === 'Confirm delete'
    )
    await act(async () => confirm!.click())

    expect(onNewChat).toHaveBeenCalledTimes(1)
  })

  it('keeps the row and active chat when deletion fails', async () => {
    getSessions.mockResolvedValueOnce(response('first', 'First session'))
    deleteSession.mockRejectedValueOnce(new Error('session_busy'))
    const onNewChat = vi.fn()

    await act(async () => {
      root.render(
        <MemoryRouter initialEntries={['/chat?resume=first']}>
          <ChatSessionList activeSessionId="first" onNewChat={onNewChat} />
        </MemoryRouter>
      )
    })

    const deleteButton = container.querySelector<HTMLButtonElement>('[aria-label="Delete session: First session"]')
    await act(async () => deleteButton!.click())
    const confirm = Array.from(container.querySelectorAll('button')).find(
      button => button.textContent === 'Confirm delete'
    )
    await act(async () => confirm!.click())

    expect(container.textContent).toContain('First session')
    expect(container.textContent).toContain('session_busy')
    expect(onNewChat).not.toHaveBeenCalled()
  })

  it('does not start a fresh chat when navigation changed during an active-session delete', async () => {
    getSessions.mockResolvedValueOnce(response('first', 'First session'))
    let resolveDelete!: (value: { ok: boolean }) => void
    deleteSession.mockReturnValueOnce(new Promise(resolve => {
      resolveDelete = resolve
    }))
    const onNewChat = vi.fn()
    let setActiveSessionId!: (id: string | null) => void

    function Harness() {
      const [activeSessionId, setActive] = useState<string | null>('first')
      setActiveSessionId = setActive

      return <ChatSessionList activeSessionId={activeSessionId} onNewChat={onNewChat} />
    }

    await act(async () => {
      root.render(
        <MemoryRouter initialEntries={['/chat']}>
          <Harness />
        </MemoryRouter>
      )
    })
    const deleteButton = container.querySelector<HTMLButtonElement>('[aria-label="Delete session: First session"]')
    await act(async () => deleteButton!.click())
    const confirm = Array.from(container.querySelectorAll('button')).find(
      button => button.textContent === 'Confirm delete'
    )
    await act(async () => confirm!.click())

    // Update the SAME mounted component, then resolve before passive effects
    // get a chance to synchronize stale refs.
    flushSync(() => setActiveSessionId('second'))
    await act(async () => resolveDelete({ ok: true }))

    expect(onNewChat).not.toHaveBeenCalled()
  })

  it('disables every delete action while one deletion is in flight', async () => {
    getSessions.mockResolvedValueOnce({
      sessions: [session('first', 'First session'), session('second', 'Second session')],
      total: 2,
      limit: 30,
      offset: 0
    })
    let resolveDelete!: (value: { ok: boolean }) => void
    deleteSession.mockReturnValueOnce(new Promise(resolve => {
      resolveDelete = resolve
    }))

    await act(async () => {
      root.render(
        <MemoryRouter>
          <ChatSessionList activeSessionId={null} />
        </MemoryRouter>
      )
    })

    const firstDelete = container.querySelector<HTMLButtonElement>('[aria-label="Delete session: First session"]')
    await act(async () => firstDelete!.click())
    const confirm = Array.from(container.querySelectorAll('button')).find(
      button => button.textContent === 'Confirm delete'
    )
    await act(async () => confirm!.click())

    const deleteButtons = container.querySelectorAll<HTMLButtonElement>('[aria-label^="Delete session:"]')
    expect(Array.from(deleteButtons).every(button => button.disabled)).toBe(true)

    await act(async () => resolveDelete({ ok: true }))
  })
})
