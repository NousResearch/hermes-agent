import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, describe, expect, it } from 'vitest'

import { discoverRailInputsFile, loadRailInputs, parseRailInputs } from '../domain/railInputs.js'

const validBlock = `## Context rail inputs

- product: Test app
- flow: .flow
- checks: npm test
- evidence: docs/ops-log.md
- decisions:
  - 2026-09-05: keep Flow authoritative
- ignored: safe to ignore

## Other context

- unrelated: value
`

describe('rail input discovery and parsing', () => {
  const roots: string[] = []

  afterEach(async () => {
    await Promise.all(roots.splice(0).map(root => rm(root, { force: true, recursive: true })))
  })

  it('discovers the repo context file and returns only the strict block', async () => {
    const root = await mkdtemp(join(tmpdir(), 'hermes-rail-'))
    const nested = join(root, 'src')
    roots.push(root)
    await mkdir(join(root, '.git'))
    await mkdir(nested)
    await writeFile(join(root, 'AGENTS.md'), `# Project\n\n${validBlock}`)

    expect(await discoverRailInputsFile(nested)).toBe(join(root, 'AGENTS.md'))

    const loaded = await loadRailInputs(nested)

    expect(loaded).toMatchObject({
      checks: 'npm test',
      decisions: ['2026-09-05: keep Flow authoritative'],
      evidence: 'docs/ops-log.md',
      flow: '.flow',
      product: 'Test app',
      projectRoot: root,
      sourceFile: join(root, 'AGENTS.md')
    })
    expect(loaded?.mtimeMs).toBeGreaterThan(0)
  })

  it('gives a repo .hermes.md file precedence over AGENTS.md', async () => {
    const root = await mkdtemp(join(tmpdir(), 'hermes-rail-'))
    roots.push(root)
    await mkdir(join(root, '.git'))
    await writeFile(join(root, 'AGENTS.md'), validBlock)
    await writeFile(join(root, '.hermes.md'), validBlock.replace('Test app', 'Hermes app'))

    expect(await discoverRailInputsFile(root)).toBe(join(root, '.hermes.md'))
    expect((await loadRailInputs(root))?.product).toBe('Hermes app')
  })

  it('loads a valid block from a bounded guide larger than 64 KiB', async () => {
    const root = await mkdtemp(join(tmpdir(), 'hermes-rail-'))
    roots.push(root)
    await mkdir(join(root, '.git'))
    await writeFile(join(root, 'AGENTS.md'), `${'guide text '.repeat(7_200)}\n${validBlock}`)

    expect((await loadRailInputs(root))?.product).toBe('Test app')
  })

  it('hides missing and malformed blocks instead of partially guessing', () => {
    expect(parseRailInputs('# no rail here')).toBeNull()
    expect(parseRailInputs(validBlock.replace('- product: Test app', 'product: missing dash'))).toBeNull()
    expect(parseRailInputs(validBlock.replace('- checks: npm test', '- checks:'))).toBeNull()
  })
})
