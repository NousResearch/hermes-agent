import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const entry = path.resolve(__dirname, '..', 'dist', 'entry.js')

if (!fs.existsSync(entry)) {
  throw new Error(`Missing built TUI entry: ${entry}`)
}

if (process.platform !== 'win32') {
  fs.chmodSync(entry, 0o755)
}
