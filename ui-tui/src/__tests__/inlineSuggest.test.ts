import { describe, expect, it } from 'vitest'

import { inlineGhostText } from '../components/textInput.js'
import { completionGhost, historyGhost, inlineSuggestion } from '../domain/inlineSuggest.js'

const rows = (...texts: string[]) => texts.map(text => ({ display: text, text }))

describe('completionGhost', () => {
  it('ghosts the remainder of the shortest matching command (CLI tie-break)', () => {
    // `/he` must ghost "lp" (→ /help), not "artbeat" (→ /heartbeat): the CLI
    // walks its registry shortest-name-first for exactly this reason.
    expect(completionGhost('/he', rows('heartbeat', 'help'), 1)).toBe('lp')
  })

  it('ghosts argument completions from their own replace point', () => {
    expect(completionGhost('/cron ad', rows('add', 'list'), 6)).toBe('d')
  })

  it('drops the row that no longer extends what is typed', () => {
    expect(completionGhost('/help', rows('exit'), 1)).toBe('')
  })

  it('ignores an exact match completing to itself plus the menu-keeping space', () => {
    // The gateway appends a trailing space to an exact hit so prompt_toolkit
    // keeps the dropdown open. There is no text to ghost there.
    expect(completionGhost('/help', rows('help '), 1)).toBe('')
  })

  it('handles client-side widget rows that carry their leading slash', () => {
    // mergeWidgetAppItems emits `/timer`; applyCompletion drops the duplicate
    // slash, and the ghost must follow the same rule rather than show "/timer".
    expect(completionGhost('/tim', rows('/timer'), 1)).toBe('er')
  })

  it('refuses a replace point that belongs to a longer, older input', () => {
    // Rows lag the input by one debounce. A stale `replace_from` past the end
    // of the text would append the row wholesale: `/he` + `add` → `/headd`.
    expect(completionGhost('/he', rows('add'), 7)).toBe('')
  })

  it('ghosts an inline /skill reference typed mid-prose', () => {
    expect(completionGhost('tidy this with /cle', rows('clean-up'), 16)).toBe('an-up')
  })
})

describe('historyGhost', () => {
  it('suggests the most recent entry that extends the input', () => {
    expect(historyGhost('git ', ['git status', 'git push origin main'])).toBe('push origin main')
  })

  it('ghosts only the first line of a multi-line recall', () => {
    // A `\n` in the ghost would reflow the composer under the hint.
    expect(historyGhost('fix ', ['fix the parser\nthen run tests'])).toBe('the parser')
  })

  it('has nothing to add for an exact recall', () => {
    expect(historyGhost('deploy', ['deploy'])).toBe('')
  })

  it('stays quiet for empty or multi-line input', () => {
    expect(historyGhost('   ', ['deploy now'])).toBe('')
    expect(historyGhost('deploy\n', ['deploy now'])).toBe('')
  })
})

describe('inlineSuggestion', () => {
  const history = ['/cron add nightly backup', 'ship the release notes']

  it('prefers a completion row over history', () => {
    expect(inlineSuggestion({ compReplace: 1, completions: rows('help'), history, value: '/he' })).toBe('lp')
  })

  it('never falls back to history while the command NAME is being typed', () => {
    // `/c` must not ghost yesterday's `/cron add nightly backup` over the
    // command it is in the middle of naming. The CLI returns None here.
    expect(inlineSuggestion({ compReplace: 1, completions: [], history, value: '/c' })).toBe('')
  })

  it('falls back to history once the command has an argument', () => {
    expect(inlineSuggestion({ compReplace: 6, completions: [], history, value: '/cron add n' })).toBe('ightly backup')
  })

  it('falls back to history for ordinary prose', () => {
    expect(inlineSuggestion({ compReplace: 0, completions: [], history, value: 'ship the' })).toBe(' release notes')
  })

  it('stays quiet on empty and multi-line input', () => {
    expect(inlineSuggestion({ compReplace: 0, completions: rows('help'), history, value: '' })).toBe('')
    expect(inlineSuggestion({ compReplace: 0, completions: [], history, value: 'ship the\nrest' })).toBe('')
  })
})

describe('inlineGhostText — when the cells after the caret are free', () => {
  const base = {
    columns: 40,
    cursor: 3,
    focus: true,
    masked: false,
    selected: false,
    suggestion: 'lp',
    value: '/he'
  }

  it('paints at the end of a focused, unmasked, unselected line', () => {
    expect(inlineGhostText(base)).toBe('lp')
  })

  it('stays hidden when the caret is not at the end', () => {
    expect(inlineGhostText({ ...base, cursor: 1 })).toBe('')
  })

  it('stays hidden over a mask, a selection, or an unfocused input', () => {
    expect(inlineGhostText({ ...base, masked: true })).toBe('')
    expect(inlineGhostText({ ...base, selected: true })).toBe('')
    expect(inlineGhostText({ ...base, focus: false })).toBe('')
  })

  it('stays hidden when it would not fit the rest of the row', () => {
    // A wrapping ghost would grow the composer by a line the real value does
    // not occupy — the input box must never resize under a hint.
    expect(inlineGhostText({ ...base, columns: 5 })).toBe('')
    expect(inlineGhostText({ ...base, columns: 6 })).toBe('lp')
  })

  it('measures only the last visual line of a multi-line buffer', () => {
    expect(inlineGhostText({ ...base, cursor: 9, value: 'first\n/he' })).toBe('lp')
  })
})
