// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const {
  getWisdomStatus,
  getWisdomDiscovery,
  getWisdomCandidates,
  getWisdomDrafts,
  getWisdomSkill,
  getWisdomInstallations,
  getWisdomVersionContent,
  planWisdomInstall,
  applyWisdomInstall,
  checkWisdom,
  suggestWisdomSkill,
  reviewWisdomDraft,
  reviseWisdomDraft,
  decideWisdomDraft,
  setupWisdom,
  getActionStatus
} = vi.hoisted(() => ({
  getWisdomStatus: vi.fn(),
  getWisdomDiscovery: vi.fn(),
  getWisdomCandidates: vi.fn(),
  getWisdomDrafts: vi.fn(),
  getWisdomSkill: vi.fn(),
  getWisdomInstallations: vi.fn(),
  getWisdomVersionContent: vi.fn(),
  planWisdomInstall: vi.fn(),
  applyWisdomInstall: vi.fn(),
  checkWisdom: vi.fn(),
  suggestWisdomSkill: vi.fn(),
  reviewWisdomDraft: vi.fn(),
  reviseWisdomDraft: vi.fn(),
  decideWisdomDraft: vi.fn(),
  setupWisdom: vi.fn(),
  getActionStatus: vi.fn()
}))

vi.mock('@/lib/api', () => ({
  api: {
    decideWisdomDraft,
    setupWisdom,
    getActionStatus,
    getWisdomCandidates,
    getWisdomDiscovery,
    getWisdomDrafts,
    getWisdomSkill,
    getWisdomInstallations,
    getWisdomVersionContent,
    getWisdomStatus,
    reviewWisdomDraft,
    reviseWisdomDraft,
    suggestWisdomSkill,
    planWisdomInstall,
    applyWisdomInstall,
    checkWisdom
  }
}))

import { CollectiveWisdomPanel } from './CollectiveWisdomPanel'

function systemSpecification() {
  return {
    hermes: { minimum_version: '0.17.0' },
    platforms: [],
    architectures: [],
    model: { capabilities: [], minimum_context_window: null },
    tools: [],
    plugins: [],
    credentials: [],
    connections: [],
    filesystem: { read: [], write: [] },
    network: { destinations: [] },
    runtime: { shell: false, browser: false, code: false, sandbox: true },
    hardware: [],
    known_limitations: []
  }
}

function manifestJson() {
  return JSON.stringify({
    schema_version: 1,
    name: 'deployment-checklist',
    requirements: systemSpecification()
  })
}

beforeEach(() => {
  getWisdomStatus.mockResolvedValue({ configured: true, verified_org_id: 'org-1' })
  getWisdomCandidates.mockResolvedValue({ candidates: [] })
  getWisdomDrafts.mockResolvedValue({ drafts: [] })
  getWisdomDiscovery.mockResolvedValue({ next_cursor: null, skills: [] })
  getWisdomInstallations.mockResolvedValue({ installations: [], notifications: [] })
  checkWisdom.mockResolvedValue({ installations: [] })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('CollectiveWisdomPanel', () => {
  it('requires explicit disclosure setup before loading collective data', async () => {
    getWisdomStatus
      .mockResolvedValueOnce({ configured: false, verified_org_id: null })
      .mockResolvedValueOnce({ configured: true, verified_org_id: 'org-1' })
    setupWisdom.mockResolvedValue({ ok: true, name: 'wisdom-setup', pid: 1 })
    getActionStatus.mockResolvedValue({
      name: 'wisdom-setup',
      running: false,
      exit_code: 0,
      pid: 1,
      lines: []
    })

    render(<CollectiveWisdomPanel profile="research" />)
    expect(await screen.findByText(/Candidate qualification stays on this profile/)).toBeTruthy()
    expect(getWisdomDiscovery).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: /set up this profile/ }))

    await waitFor(() => expect(setupWisdom).toHaveBeenCalledWith('research'))
    expect(getActionStatus).toHaveBeenCalledWith('wisdom-setup', 80)
    await waitFor(() => expect(getWisdomDiscovery).toHaveBeenCalledWith('research'))
  })

  it('shows the structured setup failure instead of trailing log punctuation', async () => {
    getWisdomStatus.mockResolvedValue({ configured: false, verified_org_id: null })
    setupWisdom.mockResolvedValue({ ok: true, name: 'wisdom-setup', pid: 1 })
    getActionStatus.mockResolvedValue({
      name: 'wisdom-setup',
      running: false,
      exit_code: 3,
      pid: 1,
      lines: [
        '=== wisdom-setup started ===',
        '{',
        '  "category": 3,',
        '  "error": "Collective Wisdom is unavailable for this account",',
        '  "ok": false',
        '}'
      ]
    })

    render(<CollectiveWisdomPanel profile="research" />)
    fireEvent.click(await screen.findByRole('button', { name: /set up this profile/ }))

    expect(await screen.findByText('Collective Wisdom is unavailable for this account')).toBeTruthy()
    expect(screen.queryByText('}')).toBeNull()
  })

  it('escapes server-controlled text and labels server scan state explicitly', async () => {
    getWisdomDiscovery.mockResolvedValue({
      next_cursor: null,
      skills: [
        {
          id: 'skill-1',
          slug: '<img src=x onerror=alert(1)>',
          author_description: '<script>window.pwned=true</script>',
          latest_version: 1,
          install_count: 0,
          state: 'active',
          security_check: { status: 'pass' }
        }
      ]
    })
    render(<CollectiveWisdomPanel profile="research" />)
    expect(await screen.findByText('<img src=x onerror=alert(1)>')).toBeTruthy()
    expect(screen.getByText('Security: Pass')).toBeTruthy()
    expect(document.querySelector('script')).toBeNull()
  })

  it('keeps the first and returning qualification notice attached to each candidate', async () => {
    getWisdomCandidates.mockResolvedValue({
      candidates: [
        {
          local_skill_id: 'local-first',
          name: 'first-skill',
          editorial_name: 'First Skill',
          editorial_description: 'A friendly description for people.',
          path: '/skills/first-skill',
          content_hash: 'sha256:first',
          eligibility: 'eligible',
          reason: null,
          qualification: 'meaningful_refinements',
          qualification_sequence: 1,
          notice_variant: 'first',
          organization_name: 'Nous Research',
          contribution_state: 'new'
        },
        {
          local_skill_id: 'local-returning',
          name: 'returning-skill',
          path: '/skills/returning-skill',
          content_hash: 'sha256:returning',
          eligibility: 'eligible',
          reason: null,
          qualification: 'consecutive_business_days',
          qualification_sequence: 2,
          notice_variant: 'returning',
          organization_name: 'Nous Research',
          contribution_state: 'new'
        }
      ]
    })

    render(<CollectiveWisdomPanel profile="research" />)

    expect(await screen.findByText(/Your organisation \(Nous Research\) has enabled Collective Wisdom/)).toBeTruthy()
    expect(screen.getByText('First Skill')).toBeTruthy()
    expect(screen.getByText('A friendly description for people.')).toBeTruthy()
    expect(screen.getByText(/Hermes detected another skill that could be useful to your team/)).toBeTruthy()
  })

  it('keeps preparation local until explicit owner copy and System Specification submission', async () => {
    getWisdomCandidates.mockResolvedValue({
      candidates: [
        {
          local_skill_id: 'local-1',
          name: 'candidate-skill',
          eligibility: 'eligible',
          reason: null,
          qualification: 'manual_selection',
          contribution_state: 'new'
        }
      ]
    })
    suggestWisdomSkill
      .mockResolvedValueOnce({
        network_submission: false,
        local_draft_id: 'local-1',
        overlay_path: '/private/overlay',
        drafted_description: 'Drafted copy',
        system_specification: systemSpecification(),
        next_step: 'review'
      })
      .mockResolvedValueOnce({ draft: { id: 'draft-1' } })

    render(<CollectiveWisdomPanel profile="research" />)
    fireEvent.click(await screen.findByText('View all local skills (1)'))
    fireEvent.click(await screen.findByRole('button', { name: 'Start contribution' }))
    fireEvent.change(await screen.findByLabelText('Owner-authored description'), {
      target: { value: 'Owner approved' }
    })
    expect(screen.queryByLabelText(/System Specification \(declarative metadata/)).toBeNull()
    fireEvent.click(screen.getByRole('checkbox', { name: 'Linux' }))
    fireEvent.click(screen.getByRole('button', { name: 'Submit draft' }))

    await waitFor(() => expect(suggestWisdomSkill).toHaveBeenCalledTimes(2))
    expect(suggestWisdomSkill.mock.calls[1]).toEqual([
      'candidate-skill',
      'research',
      'Owner approved',
      { ...systemSpecification(), platforms: ['Linux'] },
      'local-1'
    ])
    expect(JSON.stringify(suggestWisdomSkill.mock.calls[1])).not.toMatch(/usage|refinement|ranking|stability/)
  })

  it('uses user-facing activity copy and hides completed drafts from the action queue', async () => {
    getWisdomDrafts.mockResolvedValue({
      drafts: [
        { id: 'draft-ready', slug: 'deployment-checklist', state: 'ready' },
        { id: 'draft-pending', slug: 'policy-canary', state: 'pending_moderation' },
        { id: 'draft-published', slug: 'incident-handoff', state: 'published' }
      ]
    })
    getWisdomInstallations.mockResolvedValue({
      installations: [],
      notifications: [
        {
          event_id: 'decision-1',
          kind: 'owner_decision',
          skill_id: 'local-internal-id',
          payload: { slug: 'incident-handoff', state: 'published' }
        }
      ]
    })

    render(<CollectiveWisdomPanel profile="research" />)

    expect(await screen.findByText('incident-handoff was approved and is now shared with your team.')).toBeTruthy()
    expect(screen.queryByText('owner_decision')).toBeNull()
    expect(screen.queryByText('local-internal-id')).toBeNull()
    expect(screen.getByText('deployment-checklist')).toBeTruthy()
    expect(screen.getByText('policy-canary')).toBeTruthy()
    expect(screen.getByText('Waiting for collective administrator approval')).toBeTruthy()
    expect(
      screen.getByText('Drafts awaiting your review and submissions waiting for collective approval.')
    ).toBeTruthy()
    expect(screen.queryByText('incident-handoff', { selector: 'span.font-mono' })).toBeNull()
  })

  it('edits owner copy and Markdown through a rescanned successor before approval', async () => {
    getWisdomDrafts.mockResolvedValue({
      drafts: [{ id: 'draft-1', slug: 'deployment-checklist', state: 'ready' }]
    })
    const initialReview = {
      draft: {
        id: 'draft-1',
        slug: 'deployment-checklist',
        state: 'ready',
        authorDescription: 'Original description',
        scanVerdict: 'pass',
        scan: { verdict: 'pass' },
        explanation: 'Two inert files.',
        systemSpec: { hermes: { minimum_version: '0.17.0' } }
      },
      files: [
        { path: 'SKILL.md', mode: 'file', hash: 'sha256:old-skill', content_utf8: '# Original\n' },
        {
          path: 'skill.manifest.json',
          mode: 'file',
          hash: 'sha256:old-manifest',
          content_utf8: manifestJson()
        }
      ],
      hashes: {
        content: 'sha256:old-content',
        author_description: 'sha256:old-description',
        package_manifest: 'sha256:old-manifest'
      },
      receipt: null
    }
    const revisedReview = {
      ...initialReview,
      draft: {
        ...initialReview.draft,
        id: 'draft-2',
        authorDescription: 'Updated description'
      },
      files: [
        { path: 'SKILL.md', mode: 'file', hash: 'sha256:new-skill', content_utf8: '# Updated\n' },
        initialReview.files[1]
      ],
      hashes: {
        content: 'sha256:new-content',
        author_description: 'sha256:new-description',
        package_manifest: 'sha256:old-manifest'
      }
    }
    reviewWisdomDraft.mockResolvedValueOnce(initialReview).mockResolvedValueOnce(revisedReview)
    reviseWisdomDraft.mockResolvedValue({
      draft: { id: 'draft-2' },
      local_scan: {},
      notice: 'rescanned'
    })

    render(<CollectiveWisdomPanel profile="research" />)
    fireEvent.click(await screen.findByRole('button', { name: /View details/ }))
    fireEvent.change(await screen.findByLabelText('Edit owner-authored description'), {
      target: { value: 'Updated description' }
    })
    fireEvent.change(screen.getByLabelText('Edit SKILL.md'), {
      target: { value: '# Updated\n' }
    })
    expect(screen.queryByLabelText('Edit skill.manifest.json')).toBeNull()
    const minimumVersion = screen.getByLabelText('Minimum Hermes version')
    expect(minimumVersion).toHaveProperty('value', '0.17.0')
    expect(screen.getByText(/Hermes pre-fills new drafts from this authoring device/)).toBeTruthy()
    expect(screen.getByText(/Checked systems are the allowed install targets/)).toBeTruthy()
    fireEvent.change(minimumVersion, { target: { value: '' } })
    expect(screen.getByText('Minimum Hermes version is required.')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Save changes & rescan' })).toHaveProperty('disabled', true)
    fireEvent.change(minimumVersion, { target: { value: '0.17.0' } })
    fireEvent.click(screen.getByRole('checkbox', { name: 'Shell commands' }))

    expect(screen.getByText(/These changes have not been scanned/)).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Approve exact content & publish' })).toHaveProperty('disabled', true)
    fireEvent.click(screen.getByRole('button', { name: 'Preview' }))
    expect(screen.getByRole('heading', { name: 'Updated' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Save changes & rescan' }))

    await waitFor(() => expect(reviseWisdomDraft).toHaveBeenCalledTimes(1))
    expect(reviseWisdomDraft).toHaveBeenCalledWith(
      'draft-1',
      'Updated description',
      [
        { path: 'SKILL.md', content_utf8: '# Updated\n' },
        {
          path: 'skill.manifest.json',
          content_utf8: JSON.stringify({
            schema_version: 1,
            name: 'deployment-checklist',
            requirements: {
              ...systemSpecification(),
              runtime: { shell: true, browser: false, code: false, sandbox: true }
            }
          })
        }
      ],
      initialReview.hashes,
      'research'
    )
    expect(await screen.findByText('content sha256:new-content')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Approve exact content & publish' })).toHaveProperty('disabled', false)
  })

  it('shows exact version bytes and applies only a verified install receipt', async () => {
    getWisdomDiscovery.mockResolvedValue({
      next_cursor: null,
      skills: [
        {
          id: 'skill-1',
          slug: 'managed-skill',
          author_description: 'Does work',
          latest_version: 2,
          install_count: 0,
          state: 'active'
        }
      ]
    })
    getWisdomSkill.mockResolvedValue({ skill: { id: 'skill-1', slug: 'managed-skill' }, versions: [{ version: 2 }] })
    getWisdomVersionContent.mockResolvedValue({
      commit: 'sha256:commit',
      content_hash: 'sha256:content',
      files: [{ path: 'SKILL.md', mode: 'file', hash: 'sha256:file', content_utf8: '# Exact bytes' }]
    })
    planWisdomInstall.mockResolvedValue({
      receipt: 'wip_1',
      skill_id: 'skill-1',
      version: 2,
      compatibility: { outcome: 'compatible' }
    })
    applyWisdomInstall.mockResolvedValue({ installed: true })

    render(<CollectiveWisdomPanel profile="research" />)
    fireEvent.click(await screen.findByRole('button', { name: /managed-skill/ }))
    expect(await screen.findByText('# Exact bytes')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Install…' }))
    expect(await screen.findByText(/wip_1/)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Confirm install' }))
    await waitFor(() => expect(applyWisdomInstall).toHaveBeenCalledWith('wip_1', false, 'research'))
  })

  it('refreshes shared discovery explicitly and after checking for updates', async () => {
    const initialSkill = {
      id: 'skill-1',
      slug: 'initial-skill',
      author_description: 'Already visible',
      latest_version: 1,
      install_count: 0,
      state: 'active'
    }
    const refreshedSkill = {
      id: 'skill-2',
      slug: 'refreshed-skill',
      author_description: 'Found by refresh',
      latest_version: 1,
      install_count: 0,
      state: 'active'
    }
    const checkedSkill = {
      id: 'skill-3',
      slug: 'found-during-update-check',
      author_description: 'Found while checking updates',
      latest_version: 1,
      install_count: 0,
      state: 'active'
    }
    getWisdomDiscovery
      .mockResolvedValueOnce({ next_cursor: null, skills: [initialSkill] })
      .mockResolvedValueOnce({ next_cursor: null, skills: [initialSkill, refreshedSkill] })
      .mockResolvedValueOnce({ next_cursor: null, skills: [initialSkill, refreshedSkill, checkedSkill] })
    checkWisdom.mockResolvedValue({ installations: [] })

    render(<CollectiveWisdomPanel profile="research" />)
    expect(await screen.findByText('initial-skill')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Refresh shared skills' }))
    expect(await screen.findByText('refreshed-skill')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Check updates' }))
    await waitFor(() => expect(checkWisdom).toHaveBeenCalledWith('research'))
    expect(await screen.findByText('found-during-update-check')).toBeTruthy()
  })

  it('checks on load and shows the target version on installed skills with pending manual updates', async () => {
    getWisdomDiscovery.mockResolvedValue({
      next_cursor: null,
      skills: [
        {
          id: 'skill-1',
          slug: 'gateway-pull-canary',
          author_description: 'Managed canary',
          latest_version: 2,
          install_count: 1,
          state: 'active'
        }
      ]
    })
    getWisdomInstallations.mockResolvedValue({
      installations: [
        {
          skill_id: 'skill-1',
          slug: 'gateway-pull-canary',
          version: 1,
          update_mode: 'MANUAL',
          state: 'active',
          target_path: '/managed/gateway-pull-canary'
        }
      ],
      notifications: []
    })
    checkWisdom.mockResolvedValue({
      installations: [
        {
          skill_id: 'skill-1',
          state: 'update_available',
          plan: { skill_id: 'skill-1', version: 2, receipt: 'wup_1' }
        }
      ]
    })
    getWisdomSkill.mockResolvedValue({
      skill: { id: 'skill-1', slug: 'gateway-pull-canary' },
      versions: [{ version: 2 }]
    })
    getWisdomVersionContent.mockResolvedValue({ commit: 'sha256:commit', content_hash: 'sha256:content', files: [] })

    render(<CollectiveWisdomPanel profile="research" />)

    await waitFor(() => expect(checkWisdom).toHaveBeenCalledWith('research'))
    expect(await screen.findByText('v2 update available')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Check updates (1)' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /gateway-pull-canary/ }))
    expect(await screen.findByRole('button', { name: 'Review update' })).toBeTruthy()
  })

  it('plans a pasted Portal link and requires explicit confirmation before install', async () => {
    const portalLink =
      'http://127.0.0.1:3111/orgs/wisdom-local/wisdom/skills/0a192cc7-486e-426d-a6b4-493119c1c011?version=1'
    planWisdomInstall
      .mockResolvedValueOnce({
        receipt: 'wip_link_default',
        skill_id: '0a192cc7-486e-426d-a6b4-493119c1c011',
        version: 1,
        compatibility: { outcome: 'compatible' }
      })
      .mockResolvedValueOnce({
        receipt: 'wip_link_auto',
        skill_id: '0a192cc7-486e-426d-a6b4-493119c1c011',
        version: 1,
        update_mode: 'AUTO_WITH_NOTICE',
        compatibility: { outcome: 'compatible' }
      })
    applyWisdomInstall.mockResolvedValue({ installed: true })

    render(<CollectiveWisdomPanel profile="research" />)
    const input = await screen.findByLabelText('Install from link or skill ID')
    fireEvent.change(input, { target: { value: portalLink } })
    fireEvent.click(screen.getByRole('button', { name: 'Review install' }))

    await waitFor(() => expect(planWisdomInstall).toHaveBeenCalledWith(portalLink, 'research', undefined))
    expect(applyWisdomInstall).not.toHaveBeenCalled()
    const dialog = await screen.findByRole('dialog', { name: 'Verified managed action plan' })
    expect(within(dialog).getByText(/wip_link_default/)).toBeTruthy()
    fireEvent.change(within(dialog).getByLabelText('Future updates'), { target: { value: 'AUTO_WITH_NOTICE' } })
    await waitFor(() => expect(planWisdomInstall).toHaveBeenLastCalledWith(portalLink, 'research', 'AUTO_WITH_NOTICE'))
    expect(within(dialog).getByText(/wip_link_auto/)).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Confirm install' }))
    await waitFor(() => expect(applyWisdomInstall).toHaveBeenCalledWith('wip_link_auto', false, 'research'))
    await waitFor(() => expect(input).toHaveProperty('value', ''))
  })
})
