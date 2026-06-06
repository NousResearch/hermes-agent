import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { encodeKitty } from '../../lib/imageProtocol/kitty.js'

const here = dirname(fileURLToPath(import.meta.url))
const FIXTURE = join(here, 'fixtures', 'tiny.png')

describe('encodeKitty', () => {
  it('produces the canonical Kitty escape sequence for a 1x1 PNG', () => {
    const data = readFileSync(FIXTURE)
    const result = encodeKitty(data, { width: 1, height: 1 })
    // Kitty Graphics Protocol: APC + 'G' + key=value pairs + ; + base64 payload + ST
    expect(result).toMatch(/^\x1b_G/)
    expect(result).toMatch(/a=T/) // action=transmit
    expect(result).toMatch(/f=100/) // format=PNG
    expect(result).toMatch(/s=1/) // width=1
    expect(result).toMatch(/v=1/) // height=1
    expect(result).toMatch(/\x1b\\$/) // String Terminator
  })

  it('handles large payloads by chunking (m=1 then m=0)', () => {
    // 5KB of payload, max chunk 4096 bytes — expect at least 2 chunks.
    const data = Buffer.alloc(5000, 0x42)
    const result = encodeKitty(data, { width: 10, height: 10, maxChunkBytes: 4096 })
    const chunkMarkers = (result.match(/m=\d/g) ?? []).length
    expect(chunkMarkers).toBeGreaterThan(1)
  })
})
