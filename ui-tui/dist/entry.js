#!/usr/bin/env node
import { jsx as _jsx } from "react/jsx-runtime";
// Order matters: paint banner + spawn python before loading @hermes/ink.
import { bootBanner } from './bootBanner.js';
import { GatewayClient } from './gatewayClient.js';
if (!process.stdin.isTTY) {
    console.log('hermes-tui: no TTY');
    process.exit(0);
}
process.stdout.write(bootBanner());
const gw = new GatewayClient();
gw.start();
const [{ render }, { App }] = await Promise.all([import('@hermes/ink'), import('./app.js')]);
render(_jsx(App, { gw: gw }), { exitOnCtrlC: false });
