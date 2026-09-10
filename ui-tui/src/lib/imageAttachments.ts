import { randomUUID } from 'node:crypto'
import { mkdir, open, readFile, stat } from 'node:fs/promises'
import { homedir } from 'node:os'
import { basename, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import type { SubmissionDestination } from '../app/submissionDestination.js'
import type { GatewayClient } from '../gatewayClient.js'

export interface ImageAttachment { path: string; mime: string }
export interface StagedImage extends ImageAttachment { name: string; remainder?: string }

const MAX_IMAGE_BYTES = 20 * 1024 * 1024

function imageFormat(bytes: Buffer): { mime: string; ext: string } {
  if (bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) {
    return { mime: 'image/png', ext: '.png' }
  }

  if (bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255) {
    return { mime: 'image/jpeg', ext: '.jpg' }
  }

  if (/^GIF8[79]a$/.test(bytes.subarray(0, 6).toString('ascii'))) {
    return { mime: 'image/gif', ext: '.gif' }
  }

  if (bytes.subarray(0, 4).toString('ascii') === 'RIFF' && bytes.subarray(8, 12).toString('ascii') === 'WEBP') {
    return { mime: 'image/webp', ext: '.webp' }
  }

  throw new Error('Unsupported image: expected PNG, JPEG, GIF, or WebP bytes')
}

// Match an existing whole path first, so unquoted filenames with spaces survive.
async function resolveImagePath(raw: string): Promise<{ path: string; remainder: string }> {
  const value = raw.trim()
  const quoted = /^(["'])(.*?)\1(?:\s+([\s\S]*))?$/.exec(value)

  const candidates = quoted ? [[quoted[2]!, quoted[3] ?? '']] : [
    [value, ''],
    ...[...value.matchAll(/\s+/g)].reverse().map(match => [value.slice(0, match.index), value.slice(match.index).trim()])
  ]

  for (const [candidate, remainder] of candidates) {
    let path = candidate!.replace(/\\([ ()'"[\]])/g, '$1')

    if (path.startsWith('file://')) { path = fileURLToPath(path) }

    if (path.startsWith('~/')) { path = join(homedir(), path.slice(2)) }
    path = resolve(path)

    try {
      const info = await stat(path)

      if (!info.isFile()) { continue }

      if (info.size > MAX_IMAGE_BYTES) { throw new Error('Image exceeds 20 MiB upload limit') }

      return { path, remainder: remainder! }
    } catch (error) {
      if (!['ENOENT', 'ENOTDIR'].includes((error as NodeJS.ErrnoException).code ?? '')) { throw error }
    }
  }

  throw new Error(`Image not found: ${value}`)
}

export async function stageImagePath(raw: string, gw: GatewayClient, destination: SubmissionDestination): Promise<StagedImage> {
  const source = await resolveImagePath(raw)
  const bytes = await readFile(source.path)

  if (bytes.length > MAX_IMAGE_BYTES) { throw new Error('Image exceeds 20 MiB upload limit') }
  const { mime, ext } = imageFormat(bytes)
  const name = basename(source.path)

  // Explicit URL attachments may target another machine, even a loopback SSH
  // tunnel. Only discovery-owned local connections share our profile filesystem.
  if (process.env.HERMES_TUI_GATEWAY_URL?.trim()) {
    const result = await gw.request<{ path?: string }>('image.attach_bytes', {
      session_id: destination.sid, filename: `${randomUUID()}${ext}`, content_base64: bytes.toString('base64')
    })

    if (!result?.path) { throw new Error('Image upload did not return an owner path') }

    return { path: result.path, mime, name, remainder: source.remainder }
  }

  const directory = join(destination.profileHome, 'cache', 'images')
  await mkdir(directory, { recursive: true, mode: 0o700 })
  const path = join(directory, `${randomUUID()}${ext}`)
  const file = await open(path, 'wx', 0o600)

  try { await file.writeFile(bytes); await file.sync() } finally { await file.close() }

  return { path, mime, name, remainder: source.remainder }
}
