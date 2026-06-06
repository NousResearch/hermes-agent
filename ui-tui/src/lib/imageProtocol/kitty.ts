// Encode a PNG into the Kitty Graphics Protocol escape sequence.
// See: https://sw.kovidgoyal.net/kitty/graphics-protocol/

export interface KittyOptions {
  width: number
  height: number
  maxChunkBytes?: number
}

const APC = '\x1b_G'
const ST = '\x1b\\'
const DEFAULT_CHUNK = 4096

export function encodeKitty(png: Buffer, opts: KittyOptions): string {
  const chunkSize = opts.maxChunkBytes ?? DEFAULT_CHUNK
  const b64 = png.toString('base64')
  const totalChunks = Math.ceil(b64.length / chunkSize)
  const parts: string[] = []

  for (let i = 0; i < totalChunks; i++) {
    const start = i * chunkSize
    const end = Math.min(start + chunkSize, b64.length)
    const chunk = b64.slice(start, end)
    const isFirst = i === 0
    const isLast = i === totalChunks - 1
    const m = isFirst ? 1 : isLast ? 0 : 1

    const keys: string[] = []
    if (isFirst) {
      keys.push('a=T') // transmit + show
      keys.push('f=100') // format: PNG
      keys.push(`s=${opts.width}`)
      keys.push(`v=${opts.height}`)
    }
    keys.push(`m=${m}`) // chunked transmission marker
    parts.push(`${APC}${keys.join(',')};${chunk}${ST}`)
  }

  return parts.join('')
}
