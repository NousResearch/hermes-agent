import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  composeTabTitle,
  fmtCwdBranch,
  fmtProjectCwdBranch,
  renderTitleTemplate,
  resolveTerminalTitle,
  shortCwd,
  shortProject,
  shortSessionName,
  type TitleTokens
} from '../domain/paths.js'

describe('shortCwd', () => {
  const origHome = process.env.HOME

  beforeEach(() => {
    process.env.HOME = '/Users/bb'
  })

  afterEach(() => {
    process.env.HOME = origHome
  })

  it('collapses HOME to ~', () => {
    expect(shortCwd('/Users/bb/proj/repo')).toBe('~/proj/repo')
  })

  it('leaves non-HOME paths alone', () => {
    expect(shortCwd('/tmp/work')).toBe('/tmp/work')
  })

  it('truncates long paths from the left with ellipsis', () => {
    const out = shortCwd('/var/long/deeply/nested/workspace/here', 10)
    expect(out.startsWith('…')).toBe(true)
    expect(out.length).toBe(10)
    expect('/var/long/deeply/nested/workspace/here'.endsWith(out.slice(1))).toBe(true)
  })

  it('keeps paths shorter than max intact', () => {
    expect(shortCwd('/a/b', 10)).toBe('/a/b')
  })
})

describe('fmtCwdBranch', () => {
  const origHome = process.env.HOME

  beforeEach(() => {
    process.env.HOME = '/Users/bb'
  })

  afterEach(() => {
    process.env.HOME = origHome
  })

  it('returns bare cwd when branch is null', () => {
    expect(fmtCwdBranch('/Users/bb/proj', null)).toBe('~/proj')
  })

  it('returns bare cwd when branch is empty', () => {
    expect(fmtCwdBranch('/Users/bb/proj', '')).toBe('~/proj')
  })

  it('appends branch in parens', () => {
    expect(fmtCwdBranch('/Users/bb/proj', 'main')).toBe('~/proj (main)')
  })

  it('truncates the path to keep the branch tag readable', () => {
    const out = fmtCwdBranch('/Users/bb/very/deeply/nested/project/folder', 'feature-branch', 30)
    expect(out).toMatch(/ \(feature-branch\)$/)
    expect(out.length).toBeLessThanOrEqual(30)
  })

  it('truncates very long branch names from the right', () => {
    const out = fmtCwdBranch('/Users/bb/p', 'a-very-long-feature-branch-name')
    expect(out).toMatch(/^~\/p \(…/)
    expect(out).toContain(')')
  })
})

describe('shortProject', () => {
  it('trims whitespace', () => {
    expect(shortProject('  website  ')).toBe('website')
  })

  it('truncates long project names from the right', () => {
    expect(shortProject('a-very-long-project-name', 10)).toBe('a-very-lo…')
  })
})

describe('fmtProjectCwdBranch', () => {
  const origHome = process.env.HOME

  beforeEach(() => {
    process.env.HOME = '/Users/bb'
  })

  afterEach(() => {
    process.env.HOME = origHome
  })

  it('prefixes the cwd/branch label with the project name', () => {
    expect(fmtProjectCwdBranch('/Users/bb/proj', 'main', 'website', 28)).toBe('website · ~/proj (main)')
  })

  it('falls back to the cwd/branch label when no project is known', () => {
    expect(fmtProjectCwdBranch('/Users/bb/proj', 'main', null, 28)).toBe('~/proj (main)')
  })

  it('keeps the project visible when space is tight', () => {
    expect(fmtProjectCwdBranch('/Users/bb/proj', 'main', 'hermes-agent', 12)).toBe('hermes-agent')
  })
})

describe('composeTabTitle', () => {
  it('joins marker, name, model, and cwd in order', () => {
    expect(composeTabTitle('✓', 'auth refactor', 'opus-4', '~/proj')).toBe('✓ auth refactor · opus-4 · ~/proj')
  })

  it('glues the marker to the first segment with a space, not a separator', () => {
    expect(composeTabTitle('⏳', 'my session', 'opus-4', '~/proj').startsWith('⏳ my session')).toBe(true)
  })

  it('omits the session name when empty (matches the pre-name format)', () => {
    expect(composeTabTitle('✓', '', 'opus-4', '~/proj')).toBe('✓ opus-4 · ~/proj')
  })

  it('treats a whitespace-only name as absent', () => {
    expect(composeTabTitle('✓', '   ', 'opus-4', '~/proj')).toBe('✓ opus-4 · ~/proj')
  })

  it('omits the cwd when empty', () => {
    expect(composeTabTitle('✓', 'my session', 'opus-4', '')).toBe('✓ my session · opus-4')
  })

  it('falls back to just the marker when only the marker is present', () => {
    expect(composeTabTitle('✓', '', '', '')).toBe('✓')
  })

  it('truncates an over-long session name with an ellipsis', () => {
    const long = 'a'.repeat(40)
    const out = composeTabTitle('✓', long, 'opus-4', '', 28)
    const namePart = out.slice('✓ '.length).split(' · ')[0]
    expect(namePart.endsWith('…')).toBe(true)
    expect(namePart.length).toBe(28)
  })

  it('keeps a name at the boundary length intact', () => {
    const name = 'b'.repeat(28)
    const out = composeTabTitle('✓', name, 'opus-4', '', 28)
    expect(out).toBe(`✓ ${name} · opus-4`)
  })
})

describe('shortSessionName', () => {
  it('leaves a short name untouched and trims it', () => {
    expect(shortSessionName('  auth refactor  ')).toBe('auth refactor')
  })

  it('truncates past the cap with an ellipsis', () => {
    const out = shortSessionName('a'.repeat(40), 28)
    expect(out.endsWith('…')).toBe(true)
    expect(out.length).toBe(28)
  })
})

describe('renderTitleTemplate', () => {
  const tokens: TitleTokens = {
    cwd: '~/proj',
    cwdFull: '/Users/bb/very/long/path/to/proj',
    marker: '✓',
    model: 'opus-4',
    modelFull: 'anthropic/opus-4',
    session: 'a very long session name th…',
    sessionFull: 'a very long session name that is not clipped'
  }

  it('returns null for an empty template so the caller keeps its default', () => {
    expect(renderTitleTemplate('', tokens)).toBeNull()
    expect(renderTitleTemplate('   ', tokens)).toBeNull()
  })

  it('substitutes every known token', () => {
    expect(renderTitleTemplate('{marker} {session} · {model} · {cwd}', tokens)).toBe(
      '✓ a very long session name th… · opus-4 · ~/proj'
    )
  })

  it('renders the full variants verbatim, uncapped', () => {
    expect(renderTitleTemplate('{session_full}', tokens)).toBe('a very long session name that is not clipped')
    expect(renderTitleTemplate('{model_full}', tokens)).toBe('anthropic/opus-4')
    expect(renderTitleTemplate('{cwd_full}', tokens)).toBe('/Users/bb/very/long/path/to/proj')
  })

  it('reproduces the default window composition when given the default template', () => {
    const dflt = { ...tokens, session: 'auth refactor' }
    expect(renderTitleTemplate('{marker} {session} · {model} · {cwd}', dflt)).toBe(
      composeTabTitle('✓', 'auth refactor', 'opus-4', '~/proj')
    )
  })

  it('reproduces the default tab composition when given the default tab template', () => {
    const dflt = { ...tokens, session: 'auth refactor' }
    expect(renderTitleTemplate('{marker} {session}', dflt)).toBe(composeTabTitle('✓', 'auth refactor', '', ''))
  })

  it('preserves whitespace inside literal text (no global collapse)', () => {
    expect(renderTitleTemplate('prefix  literal {marker}', tokens)).toBe('prefix  literal ✓')
  })

  it('keeps *_full values byte-verbatim, including runs of spaces', () => {
    const spaced: TitleTokens = { ...tokens, cwdFull: '/tmp/a  b', sessionFull: 'alpha  beta' }
    expect(renderTitleTemplate('{session_full}', spaced)).toBe('alpha  beta')
    expect(renderTitleTemplate('{cwd_full}', spaced)).toBe('/tmp/a  b')
  })

  it('leaves a literal separator run alone when no token was empty', () => {
    expect(renderTitleTemplate('A · · B {marker}', tokens)).toBe('A · · B ✓')
  })

  it('collapses the separator orphaned by an empty token', () => {
    const noModel: TitleTokens = { ...tokens, model: '', session: 'auth refactor' }
    expect(renderTitleTemplate('{marker} {session} · {model} · {cwd}', noModel)).toBe('✓ auth refactor · ~/proj')
  })

  it('trims separators left dangling at either end', () => {
    const bare: TitleTokens = { ...tokens, cwd: '', marker: '', model: '', session: 'auth refactor' }
    expect(renderTitleTemplate('{marker} {session} · {model} · {cwd}', bare)).toBe('auth refactor')
  })

  it('keeps an unknown token verbatim so a typo stays visible', () => {
    expect(renderTitleTemplate('{marker} {sesion}', tokens)).toBe('✓ {sesion}')
  })

  it('drops only the space around an empty token, not the neighbours', () => {
    const noMarker: TitleTokens = { ...tokens, marker: '', session: 'auth refactor' }
    expect(renderTitleTemplate('{marker} {session}', noMarker)).toBe('auth refactor')
  })

  it('passes literal text through unchanged', () => {
    expect(renderTitleTemplate('hermes: {session_full}', tokens)).toBe(
      'hermes: a very long session name that is not clipped'
    )
  })
})

describe('resolveTerminalTitle', () => {
  const tokens: TitleTokens = {
    cwd: '~/proj',
    cwdFull: '/Users/bb/proj',
    marker: '✓',
    model: 'opus-4',
    modelFull: 'anthropic/opus-4',
    session: 'auth refactor',
    sessionFull: 'auth refactor'
  }

  const noModel: TitleTokens = { ...tokens, model: '', modelFull: '' }

  it('falls back to the brand string when nothing is configured and no model is known', () => {
    expect(resolveTerminalTitle({ tab: '', window: '' }, noModel, 'auth refactor')).toBe('Hermes')
  })

  it('honors a configured template BEFORE the model arrives', () => {
    expect(resolveTerminalTitle({ tab: '{session_full}', window: '' }, noModel, 'auth refactor')).toEqual({
      tab: 'auth refactor',
      window: '✓ auth refactor · ~/proj'
    })
  })

  it('honors a window-only template with no model', () => {
    const out = resolveTerminalTitle({ tab: '', window: 'hermes: {session_full}' }, noModel, 'auth refactor')
    expect(out).toEqual({ tab: '✓ auth refactor', window: 'hermes: auth refactor' })
  })

  it('reproduces the built-in split when no template is set', () => {
    expect(resolveTerminalTitle({ tab: '', window: '' }, tokens, 'auth refactor')).toEqual({
      tab: '✓ auth refactor',
      window: '✓ auth refactor · opus-4 · ~/proj'
    })
  })
})
