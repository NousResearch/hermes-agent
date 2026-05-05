import { chmodSync, statSync } from 'node:fs'

const target = process.argv[2]

if (!target) {
  console.error('usage: node scripts/mark-executable.mjs <path>')
  process.exit(2)
}

try {
  const current = statSync(target).mode
  chmodSync(target, current | 0o111)
} catch (error) {
  const message = error instanceof Error ? error.message : String(error)
  console.error(`failed to mark ${target} executable: ${message}`)
  process.exit(1)
}
