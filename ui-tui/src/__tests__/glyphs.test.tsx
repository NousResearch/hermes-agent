import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import { stripAnsi } from '@hermes/shared/ansi'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { ToolTrail } from '../components/thinking.js'
import { GlyphProvider, glyphsForPreset } from '../lib/glyphs.js'
import { DEFAULT_THEME } from '../theme.js'

describe('glyph presets', () => {
  it('guarantees ASCII chrome contains no non-ASCII code points', () => {
    const chrome = Object.values(glyphsForPreset('ascii')).join('')

    expect([...chrome].every(ch => ch.codePointAt(0)! <= 0x7f)).toBe(true)
  })

  it('keeps unicode as the portable box-drawing baseline', () => {
    const g = glyphsForPreset('unicode')

    expect(g.disclosureClosed).toBe('▸ ')
    expect(g.treeMid).toBe('├─ ')
    expect(g.toolBullet).toBe('● ')
  })

  it('re-renders core tool-tree chrome under the ASCII provider', () => {
    const stdout = new PassThrough()
    const stdin = new PassThrough()
    const stderr = new PassThrough()
    let output = ''

    Object.assign(stdout, { columns: 80, isTTY: false, rows: 20 })
    Object.assign(stdin, { isTTY: false })
    Object.assign(stderr, { isTTY: false })
    stdout.on('data', chunk => {
      output += chunk.toString()
    })

    const instance = renderSync(
      <GlyphProvider preset="ascii">
        <ToolTrail
          detailsMode="expanded"
          sections={{ tools: 'expanded' }}
          t={DEFAULT_THEME}
          trail={['Read File("src/a.ts") ✓']}
        />
      </GlyphProvider>,
      {
        patchConsole: false,
        stderr: stderr as NodeJS.WriteStream,
        stdin: stdin as NodeJS.ReadStream,
        stdout: stdout as NodeJS.WriteStream
      }
    )

    const printable = stripAnsi(output)

    expect(printable).toContain('v Tool calls')
    expect(printable).toContain('* Read File("src/a.ts")')
    expect(printable).not.toMatch(/[▸▾├└│●✗✓]/)

    instance.unmount()
    instance.cleanup()
  })
})
