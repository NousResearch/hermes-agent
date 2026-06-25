import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

const USE_MAIN_APP_PATH = join(dirname(fileURLToPath(import.meta.url)), '..', 'app', 'useMainApp.ts')
const source = readFileSync(USE_MAIN_APP_PATH, 'utf8')

describe('useMainApp resize columns', () => {
  it('reads live stdout columns during render', () => {
    expect(source).toContain('const [colsSnapshot, setColsSnapshot] = useState(stdout?.columns ?? 80)')
    expect(source).toContain('const cols = stdout?.columns ?? colsSnapshot')
  })
})
