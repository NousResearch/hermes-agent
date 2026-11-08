import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

import { afterEach, expect, it } from 'vitest'

import { canonicalRequest } from '../canonicalGateway.js'
import { applyLocale, resetLocale } from '../i18n/runtime.js'
import { stageImagePath } from '../lib/imageAttachments.js'

// pastels §3: this PR's shared-gateway client text (image staging, launch-contract refusals) is
// resolved through the catalog at call time, so an installed pack translates it.

afterEach(() => resetLocale())

it('shared-gateway refusals render from the active locale pack, English otherwise', async () => {
  const contract = { sources: ['tui'], parameters: ['request_id', 'source'] }

  const refusal = () => { try { canonicalRequest('session.create', { yolo: true }, contract);

 return '' } catch (e) { return (e as Error).message } }

  const dir = await mkdtemp(join(process.env.TMPDIR ?? '.', 'i18n-canonical-'))
  const bogus = join(dir, 'not-an-image.png')

  await writeFile(bogus, 'plain text')
  const imageError = () => stageImagePath(bogus, {} as never, { sid: 's', profileHome: dir } as never).catch((e: Error) => e.message)

  try {
    expect(refusal()).toBe('gateway does not support TUI launch options: yolo')
    expect(await imageError()).toBe('Unsupported image: expected PNG, JPEG, GIF, or WebP bytes')

    applyLocale('xx', { lang: 'xx', surface: 'tui', messages: {
      'canonical.launch.unsupportedOptions': 'XX-launch {0}',
      'canonical.images.unsupported': 'XX-image'
    } })

    expect(refusal()).toBe('XX-launch yolo')
    expect(await imageError()).toBe('XX-image')
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})
