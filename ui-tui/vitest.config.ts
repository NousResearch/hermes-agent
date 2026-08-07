import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    exclude: ['dist/**', 'node_modules/**'],
    // The cursor-layout regressions intentionally exercise thousands of
    // wrap/segment combinations. Parallel files can oversubscribe CPU badly
    // enough to turn a ~12s case into >60s, so keep file execution serial.
    fileParallelism: false
  }
})
