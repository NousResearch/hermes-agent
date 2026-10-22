import { ignoredSignalsForTuiMode, setupGracefulExit } from '../../lib/gracefulExit.js'

const mode = process.argv[2]

if (mode !== 'dashboard' && mode !== 'terminal') {
  throw new Error('expected a dashboard or terminal mode')
}

setupGracefulExit({
  cleanups: [() => process.stdout.write('cleanup\n')],
  failsafeMs: 2_000,
  ignoredSignals: ignoredSignalsForTuiMode(mode === 'dashboard'),
  onSignal: signal => process.stdout.write(`signal:${signal}\n`)
})

process.stdout.write('ready\n')
setInterval(() => {}, 1_000)
