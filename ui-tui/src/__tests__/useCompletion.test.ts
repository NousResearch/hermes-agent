import { describe, expect, it } from 'vitest'

import { completionRequestForInput, mergeLocalSlashItems } from '../hooks/useCompletion.js'

describe('mergeLocalSlashItems', () => {
  it('adds TUI-local commands from the dispatch registry', () => {
    expect(mergeLocalSlashItems('/wid', []).map(item => item.text)).toContain('/widgets-reload')
  })

  it('preserves gateway entries and does not suggest commands after arguments', () => {
    const gatewayItem = {
      display: '/widgets-reload',
      meta: 'gateway metadata',
      text: '/widgets-reload'
    }

    const widgetAppItem = {
      display: '/widget-usage',
      meta: 'widget app metadata',
      text: '/widget-usage'
    }

    expect(mergeLocalSlashItems('/wid', [gatewayItem, widgetAppItem])).toEqual([gatewayItem, widgetAppItem])
    expect(mergeLocalSlashItems('/wid reload', [])).toEqual([])
  })
})

describe('completionRequestForInput', () => {
  it('routes real slash commands to slash completion', () => {
    expect(completionRequestForInput('/help')).toMatchObject({
      method: 'complete.slash',
      params: { text: '/help' },
      replaceFrom: 1
    })
  })

  it('does not route absolute paths through slash completion', () => {
    expect(
      completionRequestForInput('/home/d/Desktop/agenda/CrimsonRed/.hermes/plans/2026-05-04-HANDOFF-NEXT.md')
    ).toMatchObject({
      method: 'complete.path',
      params: { word: '/home/d/Desktop/agenda/CrimsonRed/.hermes/plans/2026-05-04-HANDOFF-NEXT.md' },
      replaceFrom: 0
    })
  })

  it('keeps path completion for trailing absolute path tokens', () => {
    expect(completionRequestForInput('read /home/d/Desktop/file.md')).toMatchObject({
      method: 'complete.path',
      params: { word: '/home/d/Desktop/file.md' },
      replaceFrom: 5
    })
  })

  it('leaves plain text alone', () => {
    expect(completionRequestForInput('hello there')).toBeNull()
  })
})
