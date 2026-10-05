// @vitest-environment jsdom
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { expect, it, vi } from 'vitest'
import { api } from '@/lib/api'
import { BackgroundReviewToggle } from './BackgroundReviewToggle'

vi.mock('@/lib/api', () => ({ api: { saveConfig: vi.fn() } }))

it('persists only review consent and rolls the switch back when saving fails', async () => {
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
  const container = document.createElement('div')
  document.body.append(container)
  const root = createRoot(container)
  const onSaved = vi.fn()
  const save = vi.mocked(api.saveConfig)
  try {
    await act(async () => root.render(<BackgroundReviewToggle enabled={false} onSaved={onSaved} />))
    const control = container.querySelector<HTMLButtonElement>('[role="switch"]')!
    expect(control.getAttribute('aria-checked')).toBe('false')
    save.mockRejectedValueOnce(new Error('cannot save review setting'))
    await act(async () => control.click())
    expect(save).toHaveBeenCalledWith({ auxiliary: { background_review: { enabled: true } } })
    expect(control.getAttribute('aria-checked')).toBe('false')
    expect(container.querySelector('[role="alert"]')?.textContent).toContain('cannot save review setting')
    expect(onSaved).not.toHaveBeenCalled()
    save.mockResolvedValueOnce({ ok: true })
    await act(async () => control.click())
    expect(control.getAttribute('aria-checked')).toBe('true')
    expect(container.querySelector('[role="alert"]')).toBeNull()
    expect(onSaved).toHaveBeenCalledOnce()
    await act(async () => root.render(<BackgroundReviewToggle enabled={true} onSaved={onSaved} />))
    await act(async () => root.render(<BackgroundReviewToggle enabled={false} onSaved={onSaved} />))
    expect(control.getAttribute('aria-checked')).toBe('false')
  } finally {
    await act(async () => root.unmount())
    container.remove()
    vi.unstubAllGlobals()
  }
})
