import { existsSync } from 'node:fs'
import { resolve } from 'node:path'

export function pythonHelperPath(name: string, pythonRoot: string): string {
  // esbuild keeps import.meta.dirname relative to the shipped entry.js.
  const bundled = resolve(import.meta.dirname, 'scripts', name)

  return existsSync(bundled) ? bundled : resolve(pythonRoot, 'ui-tui/scripts', name)
}
