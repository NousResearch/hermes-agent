import { describe, expect, it } from 'vitest'

import { contextRail, contextRailParts } from '../components/appChrome.js'
import { formatGitStatus, formatStatusPath } from '../domain/paths.js'
import { EMPTY_GIT_STATUS, parseGitStatus } from '../hooks/useGitBranch.js'

const HOME = process.env.HOME ?? ''

describe('parseGitStatus', () => {
  it('reads branch, divergence, and every dirty class from porcelain v2', () => {
    const status = parseGitStatus(
      [
        '# branch.oid abc123',
        '# branch.head feature/login',
        '# branch.ab +3 -2',
        '1 M. N... 100644 100644 100644 aaa bbb staged.txt',
        '1 .M N... 100644 100644 100644 ccc ddd modified.txt',
        '1 MM N... 100644 100644 100644 eee fff both.txt',
        '2 R. N... 100644 100644 100644 ggg hhh R100 new.txt\told.txt',
        'u UU N... 100644 100644 100644 100644 iii jjj kkk conflict.txt',
        '? untracked.txt'
      ].join('\n')
    )

    expect(status.branch).toBe('feature/login')
    expect(status.ahead).toBe(3)
    expect(status.behind).toBe(2)
    // `MM` counts once on each side; the rename record is staged-only.
    expect(status.staged).toBe(3)
    expect(status.modified).toBe(2)
    expect(status.conflicted).toBe(1)
    expect(status.untracked).toBe(1)
  })

  it('maps a detached head and yields an empty snapshot for no output', () => {
    expect(parseGitStatus('# branch.head (detached)').branch).toBe('detached')
    expect(parseGitStatus('')).toEqual(EMPTY_GIT_STATUS)
  })
})

describe('formatGitStatus', () => {
  it('orders counters = + ! ? ⇡ ⇣ and hides the zeros', () => {
    expect(
      formatGitStatus({
        ahead: 5,
        behind: 6,
        branch: 'main',
        conflicted: 1,
        modified: 3,
        staged: 2,
        untracked: 4
      })
    ).toBe(' main =1 +2 !3 ?4 ⇡5 ⇣6')

    expect(formatGitStatus({ ...EMPTY_GIT_STATUS, branch: 'main' })).toBe(' main')
    expect(formatGitStatus(null)).toBe('')
    expect(formatGitStatus(EMPTY_GIT_STATUS)).toBe('')
  })
})

describe('formatStatusPath', () => {
  it('abbreviates $HOME and contracts every parent except the basename', () => {
    expect(formatStatusPath(`${HOME}/projects/hermes`)).toBe('~/p/hermes')
    expect(formatStatusPath('/private/tmp/hermes-tui-vim')).toBe('/p/t/hermes-tui-vim')
    expect(formatStatusPath(HOME)).toBe('~')
    expect(formatStatusPath('/')).toBe('/')
  })

  it('contracts every parent, with dot-directories keeping the dot', () => {
    const deep = `${HOME}/${['alpha', '.config', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'tail'].join('/')}`

    expect(formatStatusPath(deep)).toBe('~/a/.c/c/c/c/c/c/c/c/tail')
    // No component is ever sliced mid-name (the old tail-truncation behaviour).
    expect(formatStatusPath(deep)).not.toContain('…')
  })
})

describe('OMP context rail', () => {
  it('has exact width and embeds percent, compaction boundary, and capacity', () => {
    const rail = contextRail(40, 4, 1_100_000)
    const parts = contextRailParts(40, 4, 1_100_000)

    expect(rail).toHaveLength(40)
    expect(rail).toContain('4%')
    expect(rail).toContain('┃')
    expect(rail).toContain('1.1M')
    expect(parts.active).toMatch(/4%$/)
    expect(parts.boundary).toBe('┃')
    expect(parts.capacity).toBe('1.1M')
    expect(Object.values(parts).join('')).toBe(rail)
  })

  it('stays bounded at tiny widths and unknown usage', () => {
    expect(contextRail(5)).toHaveLength(5)
    expect(contextRail(0, 50, 200_000)).toBe('')
  })
})
