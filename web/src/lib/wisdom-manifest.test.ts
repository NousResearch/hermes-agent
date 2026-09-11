import { describe, expect, it } from 'vitest'

import {
  parseWisdomManifest,
  wisdomManifestValidationError,
  wisdomSystemSpecificationValidationError
} from './wisdom-manifest'

function validManifest() {
  return {
    schema_version: 1,
    name: 'deployment-checklist',
    requirements: {
      hermes: { minimum_version: '0.20.0' },
      platforms: [],
      architectures: [],
      model: { capabilities: [], minimum_context_window: null },
      tools: [],
      plugins: [],
      credentials: [],
      connections: [],
      filesystem: { read: [], write: [] },
      network: { destinations: [] },
      runtime: { shell: false, browser: false, code: false, sandbox: true },
      hardware: [],
      known_limitations: []
    }
  }
}

describe('Wisdom V1 manifest form contract', () => {
  it('parses the complete schema used by the structured form', () => {
    const manifest = parseWisdomManifest(JSON.stringify(validManifest()))
    expect(manifest.name).toBe('deployment-checklist')
    expect(manifest.requirements.runtime.sandbox).toBe(true)
  })

  it('rejects unknown fields instead of dropping them from a form rewrite', () => {
    const manifest = validManifest()
    const value = { ...manifest, unsupported: 'must not disappear' }
    expect(() => parseWisdomManifest(JSON.stringify(value))).toThrow(/unsupported fields/)
  })

  it('never accepts automatic tool installation', () => {
    const manifest = {
      ...validManifest(),
      requirements: {
        ...validManifest().requirements,
        tools: [
          {
            name: 'terminal',
            minimum_version: null,
            auto_install: true,
            requires_admin: false
          }
        ]
      }
    }
    expect(() => parseWisdomManifest(JSON.stringify(manifest))).toThrow(/cannot request automatic installation/)
  })

  it('reports form-editable semantic errors without losing the form shape', () => {
    const manifest = parseWisdomManifest(JSON.stringify(validManifest()))
    manifest.requirements.hermes.minimum_version = ''
    expect(wisdomSystemSpecificationValidationError(manifest.requirements)).toBe('Minimum Hermes version is required.')
    expect(wisdomManifestValidationError(JSON.stringify(manifest))).toBe('Minimum Hermes version is required.')
  })
})
