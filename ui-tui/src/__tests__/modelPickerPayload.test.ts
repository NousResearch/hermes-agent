import { describe, expect, it } from 'vitest'

import { modelPayloadFromArg } from '../app/slash/canonicalSessionControls.js'
import { modelPickerCommand } from '../components/modelPicker.js'

describe('picker command round-trips through the canonical model payload', () => {
  it('carries the picked effort into the mutation payload', () => {
    const arg = modelPickerCommand('claude-opus-5-5', 'anthropic', false, 'high')

    expect(arg).toBe('claude-opus-5-5 --provider anthropic --reasoning high --tui-session')
    expect(modelPayloadFromArg(arg)).toEqual({
      model: 'claude-opus-5-5',
      provider: 'anthropic',
      reasoning: 'high'
    })
  })

  it('omits the effort key when "keep current" was picked', () => {
    const arg = modelPickerCommand('claude-opus-5-5', 'anthropic', false, '')

    expect(modelPayloadFromArg(arg)).toEqual({ model: 'claude-opus-5-5', provider: 'anthropic' })
  })

  it('still refuses a genuinely unknown flag', () => {
    expect(() => modelPayloadFromArg('claude-opus-5-5 --provider anthropic --bogus x')).toThrow(
      /unsupported canonical model option/
    )
  })

  it('rejects a bare --reasoning with no level rather than eating the next flag', () => {
    expect(() => modelPayloadFromArg('claude-opus-5-5 --reasoning --tui-session')).toThrow(
      /unsupported canonical model option/
    )
  })

  it('documents the known gap: the picker global toggle has no canonical route', () => {
    // Pre-existing, NOT introduced with the --reasoning fix: `prepare_model` hardcodes
    // `is_global=False`, so global persistence has no route through session.mutate. Changing this
    // is a scope decision, not a bug fix. Pin the current behaviour so a future change is deliberate.
    expect(() => modelPayloadFromArg('claude-opus-5-5 --provider anthropic --reasoning high --global')).toThrow(
      /unsupported canonical model option: --global/
    )
  })
})
