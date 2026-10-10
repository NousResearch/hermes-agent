import { setClipboard } from '@hermes/ink'
import { describe, expect, it } from 'vitest'

describe('@hermes/ink public exports', () => {
  it('exports setClipboard from the runtime package entry point', () => {
    expect(setClipboard).toBeTypeOf('function')
  })
})
