import { delimiter } from 'node:path'

import { describe, expect, it } from 'vitest'

import { editorChildEnv } from '../lib/editor-env.js'

describe('editorChildEnv', () => {
  it('drops Hermes bundled Python from PATH and the interpreter variables', () => {
    const home = '/home/user'
    const tools = `${home}/.hermes/tools/python-3.14.7+20260101-linux-x64/bin`
    const venv = `${home}/.hermes/installs/abc/environments/def/venv/bin`
    const env = editorChildEnv({
      EDITOR: 'vim',
      PATH: `${tools}${delimiter}/usr/bin${delimiter}${venv}`,
      PYTHONHOME: tools,
      PYTHONPATH: `${home}/hermes`,
      VIRTUAL_ENV: `${home}/.hermes/installs/abc/environments/def/venv`
    })

    expect(env.PATH).toBe('/usr/bin')
    expect(env.PYTHONHOME).toBeUndefined()
    expect(env.PYTHONPATH).toBeUndefined()
    expect(env.VIRTUAL_ENV).toBeUndefined()
    expect(env.EDITOR).toBe('vim')
  })
})
