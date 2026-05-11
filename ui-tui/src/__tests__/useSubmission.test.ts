import { describe, expect, it } from 'vitest'

import { completionSubmitReplacement } from '../app/useSubmission.js'

describe('completionSubmitReplacement', () => {
  it('does not apply slash completion when the only difference is trailing whitespace', () => {
    expect(
      completionSubmitReplacement('/reset', { display: '/reset', text: 'reset ' }, 1)
    ).toBeNull()

    expect(
      completionSubmitReplacement('/new', { display: '/new', text: 'new ' }, 1)
    ).toBeNull()
  })

  it('still applies slash completion for partial command input', () => {
    expect(
      completionSubmitReplacement('/re', { display: '/reset', text: 'reset' }, 1)
    ).toBe('/reset')
  })

  it('still applies @-style completion on Enter', () => {
    expect(
      completionSubmitReplacement('@rea', { display: '@README.md', text: '@README.md' }, 0)
    ).toBe('@README.md')
  })

  it('still applies path completion on Enter', () => {
    expect(
      completionSubmitReplacement('./sr', { display: './src/', text: './src/' }, 0)
    ).toBe('./src/')
  })
})
