#!/usr/bin/env node
import { render } from 'ink'
import React from 'react'

import { App } from './app.js'
import { GatewayClient } from './gatewayClient.js'

if (!process.stdin.isTTY) {
  console.log('hermes-tui: no TTY')
  process.exit(0)
}

const gw = new GatewayClient()
gw.start()
render(<App gw={gw} />, { exitOnCtrlC: false, maxFps: 60 })
