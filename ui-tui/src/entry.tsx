#!/usr/bin/env node
// Import order matters for cold start: `GatewayClient` has only node-builtin
// deps (<20ms), so spawning the python gateway before loading @hermes/ink
// + App (~200ms combined) gives python ~200ms of free parallel time to run
// its own module imports instead of starting those after node is done.
import { GatewayClient } from './gatewayClient.js'

if (!process.stdin.isTTY) {
  console.log('hermes-tui: no TTY')
  process.exit(0)
}

const gw = new GatewayClient()
gw.start()

const [{ render }, { App }] = await Promise.all([import('@hermes/ink'), import('./app.js')])

render(<App gw={gw} />, { exitOnCtrlC: false })
