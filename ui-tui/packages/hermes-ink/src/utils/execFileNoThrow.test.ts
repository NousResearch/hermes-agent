import { chmodSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { execFileNoThrow } from './execFileNoThrow.js'

// These tests shell out to /bin/sh, use chmodSync(0o755), and rely on
// POSIX sleep/job control. They will not work on Windows.
const onWindows = process.platform === 'win32'

// We simulate `wl-copy`'s daemonization behavior with a tiny shell script:
//   1. Fork a short-lived background sleeper that inherits stdio (so the
//      parent process's pipes can never close).
//   2. Record the sleeper PID to a file so afterEach can clean it up.
//   3. Exit immediately with status 0.
//
// Without resolveOnExit, the await on `'close'` hangs until SIGTERM at
// timeout — exactly the production wl-copy bug. With resolveOnExit, the
// promise settles on `'exit'` regardless of the inherited pipes.

let scriptDir: string
let daemonScript: string
let sleeperPids: number[]

/** Read the PID file the daemon script writes, and track it for afterEach cleanup. */
function trackSleeperPid(pidFile: string): void {
  try {
    const pid = parseInt(readFileSync(pidFile, 'utf8').trim(), 10)
    if (pid > 0) {
      sleeperPids.push(pid)
    }
  } catch {
    // PID file not written or unreadable — sleeper may have already exited.
  }
}

beforeEach(() => {
  sleeperPids = []
  scriptDir = join(tmpdir(), `hermes-execfile-test-${process.pid}-${Date.now()}`)
  mkdirSync(scriptDir, { recursive: true })
  daemonScript = join(scriptDir, 'fake-daemonizer.sh')
  // Posix sh: the `sleep 3 &` child inherits stdin/stdout/stderr from the
  // shell, which inherited them from `spawn(stdio: 'pipe')`. The shell
  // exits but its child (the sleeper) keeps the pipes open. Mirrors how
  // wl-copy double-forks then exits while the daemon holds the selection.
  // The sleeper writes its PID to $1 so we can clean it up reliably.
  writeFileSync(daemonScript, '#!/bin/sh\nsleep 3 &\necho $! > "$1"\nexit 0\n')
  chmodSync(daemonScript, 0o755)
})

afterEach(() => {
  // Kill orphaned sleepers so they don't accumulate across watch runs.
  for (const pid of sleeperPids) {
    try {
      process.kill(pid, 'SIGKILL')
    } catch {
      // Already exited — fine.
    }
  }
  rmSync(scriptDir, { recursive: true, force: true })
})

describe.skipIf(onWindows)('execFileNoThrow with daemon-style children', () => {
  // Skipped because the bug it documents is a forever-hang. Without
  // resolveOnExit, the 'close' event doesn't fire when the immediate
  // child has exited but a forked daemon still holds stdio open. Even
  // SIGTERM at the timeout doesn't help — the daemon survives it. To
  // verify by hand: remove `it.skip` and watch the test timeout. This
  // test is here so a reviewer reading the resolveOnExit option knows
  // *why* every clipboard-tool spawn in osc.ts wires it on.
  it.skip("(documented hang) without resolveOnExit, await never resolves when daemon inherits stdio", async () => {
    const pidFile = join(scriptDir, 'sleeper-skip.pid')
    const result = await execFileNoThrow(daemonScript, [pidFile], { timeout: 300 })
    trackSleeperPid(pidFile)

    expect(result.code).toBe(124)
  })

  it("settles immediately on 'exit' when resolveOnExit is true, regardless of daemon stdio", async () => {
    const pidFile = join(scriptDir, 'sleeper-exit.pid')
    const start = Date.now()

    const result = await execFileNoThrow(daemonScript, [pidFile], {
      timeout: 2000,
      resolveOnExit: true
    })
    trackSleeperPid(pidFile)

    const elapsed = Date.now() - start

    // The shell exits in a few ms. resolveOnExit lets us return on exit
    // (code 0) instead of waiting for the orphaned sleeper to release
    // stdio. Should be well under 200ms even on slow CI.
    expect(result.code).toBe(0)
    expect(elapsed).toBeLessThan(500)
  })

  it("still surfaces the right code when resolveOnExit'd child exits non-zero", async () => {
    const pidFile = join(scriptDir, 'sleeper-fail.pid')
    const failScript = join(scriptDir, 'fail.sh')
    writeFileSync(failScript, `#!/bin/sh\nsleep 3 &\necho $! > "${pidFile}"\nexit 7\n`)
    chmodSync(failScript, 0o755)

    const result = await execFileNoThrow(failScript, [], {
      timeout: 2000,
      resolveOnExit: true
    })
    trackSleeperPid(pidFile)

    expect(result.code).toBe(7)
  })

  it('settles on timeout=124 when the child itself never exits, even with resolveOnExit', async () => {
    const slowScript = join(scriptDir, 'slow.sh')
    writeFileSync(slowScript, '#!/bin/sh\nsleep 30\n')
    chmodSync(slowScript, 0o755)

    const result = await execFileNoThrow(slowScript, [], {
      timeout: 200,
      resolveOnExit: true
    })

    // Child process never exits on its own → timer fires → SIGTERM →
    // child exits → 'exit' fires with non-null signal. The settle()
    // call from the timer registers code=124 first. Either way: 124.
    expect(result.code).toBe(124)
  })

  it('does not double-resolve when both timer and exit fire', async () => {
    const pidFile = join(scriptDir, 'sleeper-race.pid')
    // Race: child happens to exit right around the timeout. The settled
    // guard ensures only the first resolution wins.
    const result = await execFileNoThrow(daemonScript, [pidFile], {
      timeout: 50, // very tight
      resolveOnExit: true
    })
    trackSleeperPid(pidFile)

    // Either code=0 (exit beat timer) or code=124 (timer beat exit).
    // Both are valid outcomes; the contract is that the promise settles
    // exactly once and doesn't throw.
    expect([0, 124]).toContain(result.code)
  })
})
