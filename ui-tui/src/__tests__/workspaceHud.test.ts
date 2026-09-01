import { stringWidth } from '@hermes/ink'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { StatusRule } from '../components/appChrome.js'
import {
  EMPTY_GIT_WORKSPACE,
  fitWorkspaceHudParts,
  formatWorkspaceHud,
  normalizeGithubOrigin,
  parseWorkspaceInfo,
  workspaceHudParts,
  type WorkspaceHudSnapshot
} from '../domain/workspaceHud.js'
import { DEFAULT_THEME } from '../theme.js'

const snapshot: WorkspaceHudSnapshot = {
  branch: 'feature/footer-hud',
  dirty: true,
  gitRoot: '/Users/example/work/hermes-agent',
  github: { fullName: 'NousResearch/hermes-agent', owner: 'NousResearch', repo: 'hermes-agent' },
  projectName: 'Hermes Agent',
  pullRequest: { number: 42, state: 'open', title: 'Show workspace https://user:hidden@github.com/a/b token=hidden' },
  upstream: { ahead: 2, behind: 1 }
}

const textContent = (node: React.ReactNode): string => {
  if (node === null || node === undefined || typeof node === 'boolean') {
    return ''
  }

  if (typeof node === 'string' || typeof node === 'number') {
    return String(node)
  }

  if (Array.isArray(node)) {
    return node.map(textContent).join('')
  }

  if (React.isValidElement(node)) {
    return textContent(node.props.children)
  }

  return ''
}

describe('workspace HUD data normalization', () => {
  it('returns only owner/repo from credential-bearing GitHub remotes', () => {
    expect(normalizeGithubOrigin('https://user:hidden@github.com/NousResearch/hermes-agent.git?x=hidden')).toEqual({
      fullName: 'NousResearch/hermes-agent',
      owner: 'NousResearch',
      repo: 'hermes-agent'
    })
    expect(normalizeGithubOrigin('git@github.com:NousResearch/hermes-agent.git')?.fullName).toBe(
      'NousResearch/hermes-agent'
    )
    expect(normalizeGithubOrigin('https://gitlab.com/NousResearch/hermes-agent.git')).toBeNull()
  })

  it('sanitizes PR titles and tolerates absent remote/PR data', () => {
    const parsed = parseWorkspaceInfo({
      branch: 'main',
      dirty: false,
      git_root: '/repo',
      github: { owner: 'NousResearch', repo: 'hermes-agent' },
      pull_request: {
        number: 7,
        state: 'OPEN',
        title: 'Review https://user:hidden@github.com/a/b token=hidden'
      },
      upstream: { ahead: 0, behind: 0 }
    })

    expect(parsed.pullRequest?.title).not.toContain('hidden')
    expect(parsed.pullRequest?.state).toBe('open')
    expect(
      parseWorkspaceInfo({ pull_request: { number: 8, state: 'https://user:hidden@example.com' } }).pullRequest?.state
    ).toBe('')
    expect(parseWorkspaceInfo({ git_root: '/repo', branch: 'main' }).pullRequest).toBeNull()
    expect(formatWorkspaceHud({ ...EMPTY_GIT_WORKSPACE, projectName: null }, 100)).toBe('')
  })
})

describe('workspace HUD layout', () => {
  it('keeps identity ahead of sync and PR details as width shrinks', () => {
    const wide = formatWorkspaceHud(snapshot, 180)
    const narrow = formatWorkspaceHud(snapshot, 52)

    expect(wide).toContain('◆ Hermes Agent')
    expect(wide).toContain('⌂')
    expect(wide).toContain('⎇ feature/footer-hud')
    expect(wide).toContain('GH NousResearch/hermes-agent')
    expect(wide).toContain('↑2 ↓1')
    expect(wide).toContain('PR #42 open')
    expect(wide).not.toContain('hidden')
    expect(stringWidth(narrow)).toBeLessThanOrEqual(52)
    expect(narrow).toContain('◆')
    expect(narrow).toContain('⎇')
    expect(narrow).not.toContain('PR #42')
    expect(narrow).not.toContain('↑2')

    const tiny = formatWorkspaceHud(snapshot, 1)
    expect(stringWidth(tiny)).toBeLessThanOrEqual(1)
    expect(tiny).not.toContain('└')

    const fitted = fitWorkspaceHudParts(workspaceHudParts(snapshot), 8)
    expect(stringWidth(`└ ${fitted.map(part => part.text).join(' · ')}`)).toBeLessThanOrEqual(8)
  })

  it('honors the existing status field filter, including aggregate workspace selection', () => {
    expect(formatWorkspaceHud(snapshot, 160, new Set(['model']))).toBe('')
    expect(formatWorkspaceHud(snapshot, 160, new Set(['project', 'git_branch']))).toContain('◆ Hermes Agent')
    expect(formatWorkspaceHud(snapshot, 160, new Set(['project', 'git_branch']))).not.toContain('GH ')
    expect(formatWorkspaceHud(snapshot, 160, new Set(['workspace']))).toContain('GH NousResearch/hermes-agent')
  })

  it('reserves a second row only for visible workspace data and keeps provider/tier in the model segment', () => {
    const base = {
      bgCount: 0,
      busy: false,
      cols: 140,
      cwdLabel: '~/repo',
      liveSessionCount: 0,
      model: 'zai/glm-5.2',
      modelProvider: 'zai',
      modelReasoningEffort: 'high',
      modelServiceTier: 'flex',
      sessionStartedAt: null,
      status: 'ready',
      statusColor: DEFAULT_THEME.color.ok,
      t: DEFAULT_THEME,
      turnStartedAt: null,
      usage: { context_max: 200_000, context_percent: 25, context_used: 50_000, total: 50_000 },
      voiceLabel: ''
    }

    const withHud = StatusRule({ ...base, workspace: snapshot })
    const hiddenHud = StatusRule({ ...base, statusBarFields: new Set(['model']), workspace: snapshot })

    expect(withHud.props.height).toBe(2)
    expect(textContent(withHud)).toContain('zai/glm 5.2 high flex')
    expect(hiddenHud.props.height).toBe(1)
    expect(textContent(hiddenHud)).not.toContain('Hermes Agent')
  })
})
