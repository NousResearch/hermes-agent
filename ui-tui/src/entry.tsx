#!/usr/bin/env node
// Import order matters for cold start: `GatewayClient` + `bootBanner` have
// only node-builtin deps (<20ms), so we can paint the banner and spawn the
// python gateway before loading @hermes/ink + App (~170ms combined).
// `<AlternateScreen>` wipes the normal-screen buffer on Ink mount, so the
// boot banner is replaced seamlessly by the real React render.
import { bootBanner } from './bootBanner.js'
import { GatewayClient } from './gatewayClient.js'

if (!process.stdin.isTTY) {
  console.log('hermes-tui: no TTY')
  process.exit(0)
}

process.stdout.write(bootBanner())

const gw = new GatewayClient()
gw.start()

const [{ render }, { App }] = await Promise.all([import('@hermes/ink'), import('./app.js')])

render(<App gw={gw} />, { exitOnCtrlC: false })
