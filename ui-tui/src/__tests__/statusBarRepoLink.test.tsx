import React from 'react'
import { describe, expect, it } from 'vitest'

import { StatusRule } from '../components/appChrome.js'
import { forgeUrlFromRemote, gitRemoteNameForBranch } from '../domain/paths.js'
import { DEFAULT_THEME } from '../theme.js'

type ReactNodeLike = React.ReactNode

// Find the nearest element carrying a `url` prop (hermes-ink's <Link/>) whose
// rendered text contains the needle. Asserting on the prop — rather than on
// raw OSC 8 bytes — is what proves the segment is actually a hyperlink.
const findLinkWithText = (node: ReactNodeLike, needle: string): null | React.ReactElement => {
  if (node === null || node === undefined || typeof node === 'boolean') {
    return null
  }

  if (Array.isArray(node)) {
    for (const child of node) {
      const found = findLinkWithText(child, needle)

      if (found) {
        return found
      }
    }

    return null
  }

  if (!React.isValidElement(node)) {
    return null
  }

  const text = textContent(node)

  if (typeof (node.props as { url?: unknown }).url === 'string' && text.includes(needle)) {
    return node
  }

  return findLinkWithText((node.props as { children?: ReactNodeLike }).children, needle)
}

const textContent = (node: ReactNodeLike): string => {
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
    return textContent((node.props as { children?: ReactNodeLike }).children)
  }

  return ''
}

const baseProps = {
  bgCount: 0,
  busy: false,
  cols: 160,
  cwdLabel: '~/repo',
  liveSessionCount: 0,
  model: 'opus-4.8',
  sessionStartedAt: null,
  status: 'ready',
  statusColor: DEFAULT_THEME.color.ok,
  t: DEFAULT_THEME,
  turnStartedAt: null,
  usage: { context_max: 200_000, context_percent: 25, context_used: 50_000, total: 50_000 },
  voiceLabel: ''
}

// Mirrors ~/.config/zsh/bin/starship-git-url, the resolver the zsh prompt
// already uses: scp-style, ssh://, git://, https:// and http:// all collapse
// to the same https browse URL, and only known forges are linked.
describe('forgeUrlFromRemote', () => {
  it('normalizes every supported GitHub remote form to one https repo URL', () => {
    for (const remote of [
      'git@github.com:saforem2/ezpz.git',
      'ssh://git@github.com/saforem2/ezpz.git',
      'git://github.com/saforem2/ezpz.git',
      'https://github.com/saforem2/ezpz.git',
      'https://github.com/saforem2/ezpz',
      'https://user@github.com:443/saforem2/ezpz.git'
    ]) {
      expect(forgeUrlFromRemote(remote), remote).toEqual({
        repo: 'https://github.com/saforem2/ezpz',
        tree: '/tree/'
      })
    }
  })

  it('uses GitLab browse paths and refuses unknown or unparseable remotes', () => {
    expect(forgeUrlFromRemote('git@gitlab.com:group/proj.git')).toEqual({
      repo: 'https://gitlab.com/group/proj',
      tree: '/-/tree/'
    })

    for (const remote of ['', '/srv/git/bare.git', 'git@example.com:team/internal.git', 'not a url']) {
      expect(forgeUrlFromRemote(remote), remote).toBeNull()
    }
  })

  it('builds a branch URL only when a branch is known', () => {
    const forge = forgeUrlFromRemote('git@github.com:saforem2/ezpz.git')

    expect(`${forge!.repo}${forge!.tree}feat/x`).toBe('https://github.com/saforem2/ezpz/tree/feat/x')
  })
})

// A branch pushed to a fork does not exist on `origin`, so linking it against
// origin yields a 404. The link must follow the branch's configured upstream.
describe('gitRemoteNameForBranch', () => {
  it('prefers the branch upstream remote over origin', () => {
    expect(gitRemoteNameForBranch('fork')).toBe('fork')
  })

  it('falls back to origin when the branch has no upstream', () => {
    expect(gitRemoteNameForBranch('')).toBe('origin')
    expect(gitRemoteNameForBranch(null)).toBe('origin')
  })

  it('ignores a detached-head placeholder upstream', () => {
    expect(gitRemoteNameForBranch('.')).toBe('origin')
  })
})

describe('StatusRule workspace repository link', () => {
  it('links the working-directory segment to the branch on GitHub', () => {
    const element = StatusRule({
      ...baseProps,
      gitStatus: {
        ahead: 0,
        behind: 0,
        branch: 'main',
        conflicted: 0,
        modified: 0,
        staged: 0,
        untracked: 0
      },
      pathLabel: '~/p/ezpz',
      repoUrl: 'https://github.com/saforem2/ezpz/tree/main'
    })

    const link = findLinkWithText(element, '~/p/ezpz')

    expect(link).not.toBeNull()
    expect((link!.props as { url: string }).url).toBe('https://github.com/saforem2/ezpz/tree/main')
  })

  it('renders the path as plain text when the repo has no forge remote', () => {
    const element = StatusRule({
      ...baseProps,
      pathLabel: '~/p/local-only'
    })

    expect(textContent(element)).toContain('~/p/local-only')
    expect(findLinkWithText(element, '~/p/local-only')).toBeNull()
  })
})
