// Zero-dependency launcher for the visual harness (`npm run visual`).
//
// - Sets FORCE_COLOR/COLORTERM itself instead of POSIX `VAR=x cmd` shell
//   assignments (which break under the Windows npm command shell) — no
//   cross-env needed.
// - Resolves electron from the install tree (the desktop workspace already
//   ships it; a root `npm install` hoists it) instead of declaring a second
//   copy as a ui-tui dependency. ELECTRON_BIN overrides for exotic setups.
import { spawnSync } from 'child_process'
import { createRequire } from 'module'
import { dirname, join } from 'path'
import { fileURLToPath } from 'url'

const here = dirname(fileURLToPath(import.meta.url))
const require = createRequire(import.meta.url)

const run = (bin, args, env = {}) => {
  const { status } = spawnSync(bin, args, { env: { ...process.env, ...env }, stdio: 'inherit' })

  if (status !== 0) {
    process.exit(status ?? 1)
  }
}

// 1. Render the scene sheet (tsx is a ui-tui devDependency).
run(process.execPath, [require.resolve('tsx/cli'), join(here, 'render.tsx')], {
  COLORTERM: 'truecolor',
  FORCE_COLOR: '3'
})

// 2. Screenshot it with electron, borrowed from the workspace that owns it.
let electronBin = process.env.ELECTRON_BIN

if (!electronBin) {
  try {
    // In plain Node, `require('electron')` evaluates to the binary path.
    electronBin = require('electron')
  } catch {
    console.error(
      'electron is not installed in this tree — the visual harness borrows it from the\n' +
        'desktop workspace. Run `npm install` at the repo root, or point ELECTRON_BIN at a binary.'
    )
    process.exit(1)
  }
}

run(electronBin, [join(here, 'shot.mjs')])
