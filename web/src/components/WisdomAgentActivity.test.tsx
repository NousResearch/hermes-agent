// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { WisdomAgentActivity } from './WisdomAgentActivity'

const { read } = vi.hoisted(() => ({ read: vi.fn() }))
vi.mock('@/lib/api', () => ({ api: { getWisdomMediation: read } }))
afterEach(cleanup)

it('shows shared advice and canonical warnings without an apply control', async () => {
  read.mockResolvedValue({ mode: 'agent', assessments: [{ id: 'one', state: 'delivered', advice: {
    title: 'A useful skill', explanation: 'May overlap with your existing runbook.'
  } }], interactions: [{ id: 'control', assessment_id: 'one', state: 'pending', operation: 'update', facts: {
    slug: 'Runbook', version: 2, sensitive_expansion: ['Additional network access'],
    security_check: { status: 'pass', checks: [] }, professionalism_check: { status: 'advisory', checks: [] }
  } }] })
  render(<WisdomAgentActivity profile="research" />)
  await screen.findByText('A useful skill')
  expect(screen.getByText('Additional network access')).toBeTruthy()
  expect(screen.queryByRole('button')).toBeNull()
  expect(read).toHaveBeenCalledWith('research')
})

it('leaves the fixed notification surface unchanged', async () => {
  read.mockResolvedValue({ mode: 'fixed', assessments: [], interactions: [] })
  const { container } = render(<WisdomAgentActivity />)
  await vi.waitFor(() => expect(read).toHaveBeenCalled())
  expect(container.textContent).toBe('')
})
