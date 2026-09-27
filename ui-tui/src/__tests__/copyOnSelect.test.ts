import { describe, expect, it } from 'vitest'

import { shouldCopyStableSelection, type StableSelectionInput } from '../lib/copyOnSelect.js'

const gated = (over: Partial<Parameters<typeof shouldCopyStableSelection>[0]>) =>
  shouldCopyStableSelection({
    hasSelection: true,
    isDragging: false,
    version: 0,
    lastCopiedVersion: null,
    ...over
  })

describe('shouldCopyStableSelection', () => {
  it('copies a stable selection with a new version', () => {
    expect(gated({ hasSelection: true, isDragging: false, version: 3, lastCopiedVersion: 2 })).toBe(true)
  })

  it('does not copy when there is no selection', () => {
    expect(gated({ hasSelection: false, version: 3 })).toBe(false)
  })

  it('does not copy while the user is still dragging', () => {
    expect(gated({ isDragging: true, version: 3, lastCopiedVersion: 2 })).toBe(false)
  })

  it('treats undefined isDragging as stable (matches original `state?.isDragging` semantics)', () => {
    expect(gated({ isDragging: undefined, version: 3, lastCopiedVersion: 2 })).toBe(true)
  })

  it('does not copy an unchanged version (de-dupe, avoids clipboard spam)', () => {
    expect(gated({ version: 3, lastCopiedVersion: 3 })).toBe(false)
  })

  it('de-dupes across repeated releases of the same selection', () => {
    const input: StableSelectionInput = { hasSelection: true, isDragging: false, version: 5, lastCopiedVersion: null }
    expect(shouldCopyStableSelection(input)).toBe(true)
    // After the caller records the version it copied, the same selection must not re-copy.
    input.lastCopiedVersion = 5
    expect(shouldCopyStableSelection(input)).toBe(false)
  })

  it('copies again after the selection version advances', () => {
    const input: StableSelectionInput = { hasSelection: true, isDragging: false, version: 5, lastCopiedVersion: 5 }
    expect(shouldCopyStableSelection(input)).toBe(false)
    input.version = 6
    expect(shouldCopyStableSelection(input)).toBe(true)
  })
})
