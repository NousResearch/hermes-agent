#!/usr/bin/env node
/**
 * Lightpanda browser adapter
 *
 * Implements the same JSON command protocol as agent-browser but connects to
 * Lightpanda's CDP server with the correct initialization sequence:
 *   chromium.connectOverCDP() → newContext() → newPage()
 *
 * Lightpanda starts with an empty browser (no pages), so the standard
 * agent-browser CDP path fails.  This adapter handles that difference and is
 * invoked automatically by browser_tool.py whenever a Lightpanda session is
 * detected (session_info.features.lightpanda === true).
 *
 * Architecture: daemon + thin client (mirrors agent-browser's pattern)
 *   - First call for a session name: forks a background daemon, waits for its
 *     Unix socket, sends the command, exits.
 *   - Subsequent calls: connect directly to the existing socket.
 *
 * Usage (internal, called by browser_tool.py):
 *   node lightpanda_adapter.js --cdp <ws_url> --session <name> --json <cmd> [args…]
 *
 * Supported commands (matching agent-browser output format exactly):
 *   open <url>          navigate to URL
 *   snapshot [-c]       aria snapshot with [ref=eN] annotations
 *   click <ref>         click element by ref
 *   fill <ref> <text>   fill input by ref
 *   scroll <dir> [px]   scroll up/down
 *   back                go back in history
 *   press <key>         keyboard press
 *   console             browser console messages (stub — Lightpanda limitation)
 *   errors              JS errors (stub)
 *   eval <expr>         evaluate JavaScript
 *   close               terminate daemon
 *   record …            stub (no-op)
 */

'use strict';

const net    = require('net');
const fs     = require('fs');
const path   = require('path');
const os     = require('os');
const crypto = require('crypto');
const { spawn } = require('child_process');

// ─── playwright-core resolution ──────────────────────────────────────────────
// playwright-core may live in the hermes-agent install or a local node_modules.

function loadPlaywright() {
  const candidates = [
    // repo-local node_modules (npm install in project root)
    path.join(__dirname, '..', 'node_modules', 'playwright-core'),
    path.join(__dirname, 'node_modules', 'playwright-core'),
    // hermes-agent installed copy (default hermes install path)
    path.join(os.homedir(), '.hermes', 'hermes-agent', 'node_modules', 'playwright-core'),
    // global npm
    path.join(os.homedir(), '.nvm', 'versions', 'node',
      (process.version || 'vX'), 'lib', 'node_modules', 'playwright-core'),
  ];
  for (const p of candidates) {
    try { return require(p); } catch {}
  }
  // Last resort: hope it is in NODE_PATH / global
  try { return require('playwright-core'); } catch {}
  throw new Error(
    'playwright-core not found. Install it: npm install -g playwright-core\n' +
    'Or run: npm install   in the hermes-agent repo root.'
  );
}

// ─── constants ────────────────────────────────────────────────────────────────

const NAV_TIMEOUT     = 30_000;
const CMD_TIMEOUT     = 15_000;
const SOCKET_WAIT_MS  = 10_000;
const SOCKET_POLL_MS  = 100;
// Daemon exits after this many ms of no commands (prevents orphan accumulation
// when the parent process dies without sending an explicit close).
const IDLE_TIMEOUT_MS = 5 * 60_000;

const INTERACTIVE_ROLES = new Set([
  'link', 'button', 'textbox', 'searchbox', 'checkbox', 'radio',
  'combobox', 'listbox', 'menuitem', 'menuitemcheckbox', 'menuitemradio',
  'option', 'tab', 'treeitem', 'spinbutton',
]);

// ─── argument parsing ─────────────────────────────────────────────────────────

function parseArgs(argv) {
  const o = { cdpUrl: null, session: 'default', cmd: null, cmdArgs: [], daemon: false };
  for (let i = 0; i < argv.length; i++) {
    switch (argv[i]) {
      case '--cdp':     o.cdpUrl  = argv[++i]; break;
      case '--session': o.session = argv[++i]; break;
      case '--daemon':  o.daemon  = true;      break;
      case '--json':
        o.cmd     = argv[++i];
        o.cmdArgs = argv.slice(i + 1);
        i = argv.length; // stop parsing
        break;
    }
  }
  return o;
}

// ─── socket helpers ───────────────────────────────────────────────────────────

function socketPath(session) {
  const dir = process.env.AGENT_BROWSER_SOCKET_DIR
    || path.join(os.homedir(), '.lightpanda-sessions');
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  // Linux sockaddr_un caps path at 108 bytes (macOS: 104).  Long session
  // names — especially task IDs that are full UUIDs — push the full socket
  // path over the limit, causing Node.js to silently truncate.  The daemon
  // then listens on a truncated path while the client polls for the full
  // path and never finds it.  Collapse the filename to a short hash so the
  // full path stays well under the limit.
  const shortName = 'lp_' + crypto.createHash('sha1').update(session).digest('hex').slice(0, 12) + '.sock';
  return path.join(dir, shortName);
}

function waitForSocket(sockPath) {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + SOCKET_WAIT_MS;
    const poll = () => {
      if (fs.existsSync(sockPath)) return resolve();
      if (Date.now() >= deadline)  return reject(new Error('Daemon socket did not appear in time'));
      setTimeout(poll, SOCKET_POLL_MS);
    };
    poll();
  });
}

function sendToSocket(sockPath, cmd, args) {
  return new Promise((resolve, reject) => {
    const conn = net.createConnection(sockPath);
    let buf = '';
    conn.setTimeout(CMD_TIMEOUT);
    conn.on('connect', () => {
      conn.write(JSON.stringify({ cmd, args }) + '\n');
      conn.end();  // half-close: done writing, still reading
    });
    conn.on('data',    c => { buf += c; });
    conn.on('end',     () => {
      try { resolve(JSON.parse(buf.trim())); }
      catch (e) { reject(new Error('Bad daemon response: ' + buf)); }
    });
    conn.on('error',   reject);
    conn.on('timeout', () => { conn.destroy(); reject(new Error('Socket command timed out')); });
  });
}

// ─── ARIA snapshot → agent-browser format ────────────────────────────────────

function buildSnapshotWithRefs(rawSnap) {
  // NOTE: Only elements with a quoted name (e.g. `button "Submit"`) get a
  // [ref=eN] annotation.  Unnamed elements like bare `- textbox` or
  // `- button` are left as-is and cannot be targeted by click/fill.
  // This matches Lightpanda's ARIA snapshot format where some form inputs
  // lack accessible names.
  const refs = {};
  let counter = 0;
  const lines = rawSnap.split('\n');
  const out   = [];

  for (const line of lines) {
    // Match:  <indent>- <role> "<name>"<rest>
    const m = line.match(/^(\s*- )(\w+)\s+"([^"]+)"(.*)/);
    if (m) {
      const [, indent, role, name, rest] = m;
      if (INTERACTIVE_ROLES.has(role.toLowerCase())) {
        counter++;
        const ref = `e${counter}`;
        refs[ref] = { name, role: role.toLowerCase() };
        out.push(`${indent}${role} "${name}" [ref=${ref}]${rest}`);
        continue;
      }
    }
    out.push(line);
  }

  // Wrap in "- document:" to match agent-browser format
  const snapshot = '- document:\n' + out.map(l => '  ' + l).join('\n');
  return { snapshot, refs };
}

// ─── command handlers ─────────────────────────────────────────────────────────

async function handleCommand(cmd, args, page, refMap) {
  switch (cmd) {

    case 'open': {
      const url = args[0];
      if (!url) return fail('open requires a URL');
      await page.goto(url, { waitUntil: 'domcontentloaded', timeout: NAV_TIMEOUT });
      return ok({ title: await page.title(), url: page.url() });
    }

    case 'snapshot': {
      const rawSnap = await page.ariaSnapshot({ timeout: CMD_TIMEOUT });
      const { snapshot, refs } = buildSnapshotWithRefs(rawSnap);
      Object.assign(refMap, refs);  // persist for click/fill
      return ok({ snapshot, refs });
    }

    case 'click': {
      const ref = args[0];
      if (!ref) return fail('click requires a ref (e.g. e1)');
      const info = refMap[ref];
      if (!info) return fail(`Unknown ref ${ref} — call snapshot first`);
      await page.getByRole(info.role, { name: info.name, exact: false })
                .first()
                .click({ timeout: CMD_TIMEOUT });
      return ok({ clicked: true });
    }

    case 'fill': {
      const ref  = args[0];
      const text = args.slice(1).join(' ');
      if (!ref) return fail('fill requires ref and text');
      const info = refMap[ref];
      const loc  = info
        ? page.getByRole(info.role, { name: info.name, exact: false }).first()
        : page.locator('input:visible, textarea:visible').first();
      await loc.fill(text, { timeout: CMD_TIMEOUT });
      return ok({ filled: true });
    }

    case 'scroll': {
      const dir    = args[0] || 'down';
      const pixels = parseInt(args[1], 10) || 300;
      const delta  = dir === 'up' ? -pixels : pixels;
      await page.evaluate(d => window.scrollBy(0, d), delta);
      return ok({ scrolled: true });
    }

    case 'back':
      // Lightpanda does not yet implement Page.getNavigationHistory (CDP).
      // Return an informative error rather than sending an unsupported protocol
      // command, which would cause Lightpanda to close the CDP connection.
      return fail('back is not supported by Lightpanda');

    case 'press': {
      const key = args[0];
      if (!key) return fail('press requires a key');
      await page.keyboard.press(key, { timeout: CMD_TIMEOUT });
      return ok({ pressed: true });
    }

    case 'console':
      // Lightpanda does not yet expose console capture via CDP
      return ok({ messages: [] });

    case 'errors':
      return ok({ errors: [] });

    case 'eval': {
      const expr = args[0];
      if (!expr) return fail('eval requires an expression');
      const result = await page.evaluate(expr);
      return ok({ result: String(result ?? '') });
    }

    case 'close':
      // Signal the daemon to shut down after writing the response.
      // We cannot call process.exit(0) here directly — the response would
      // never be written.  Use a sentinel object the server checks for.
      return { _close: true, success: true, data: { closed: true }, error: null };

    case 'record':
      return ok({});   // stub — Lightpanda does not support WebM recording

    default:
      return fail(`Unknown command: ${cmd}`);
  }
}

function ok(data)   { return { success: true,  data,  error: null }; }
function fail(msg)  { return { success: false,         error: msg  }; }

// ─── DAEMON ───────────────────────────────────────────────────────────────────

async function runDaemon(opts) {
  const { chromium } = loadPlaywright();
  const refMap = {};
  let browser, page;

  async function connectAndCreatePage() {
    browser = await chromium.connectOverCDP(opts.cdpUrl,
      { timeout: SOCKET_WAIT_MS });
    // Always create a fresh context + page.
    // Lightpanda's initial default context/page cannot be navigated — it gets
    // closed on the first page.goto() call.  A new context avoids this.
    const ctx = await browser.newContext();
    page = await ctx.newPage();
    // Clear stale refs — they pointed at elements on the old page.
    for (const k of Object.keys(refMap)) delete refMap[k];
  }

  try {
    await connectAndCreatePage();
  } catch (e) {
    // Write error marker so the client knows the daemon failed
    const errPath = socketPath(opts.session) + '.err';
    try { fs.writeFileSync(errPath, e.message); } catch {}
    process.exit(1);
  }

  const sock = socketPath(opts.session);
  if (fs.existsSync(sock)) fs.unlinkSync(sock);

  // Write a PID file so the parent process can cleanly shut us down.
  const pidFile = sock + '.pid';
  try { fs.writeFileSync(pidFile, String(process.pid)); } catch {}

  // Idle watchdog — exit if no commands received within IDLE_TIMEOUT_MS.
  // This prevents orphan daemons when the parent dies without sending close.
  let idleTimer = setTimeout(() => {
    process.stderr.write('lightpanda_adapter: idle timeout, exiting\n');
    process.exit(0);
  }, IDLE_TIMEOUT_MS);
  const resetIdleTimer = () => {
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      process.stderr.write('lightpanda_adapter: idle timeout, exiting\n');
      process.exit(0);
    }, IDLE_TIMEOUT_MS);
  };

  const server = net.createServer({ allowHalfOpen: true }, conn => {
    resetIdleTimer();
    let buf = '';
    conn.on('data', chunk => { buf += chunk; });
    conn.on('end', async () => {
      let result;
      try {
        const req = JSON.parse(buf.trim());
        result = await handleCommand(req.cmd, req.args || [], page, refMap);
      } catch (e) {
        // If the page/context/browser was closed (e.g. after a server error
        // response that triggers a Lightpanda disconnect), attempt to
        // reconnect and retry the command once.
        if (/closed|disposed|crashed/i.test(e.message)) {
          try {
            await connectAndCreatePage();
            const req = JSON.parse(buf.trim());
            result = await handleCommand(req.cmd, req.args || [], page, refMap);
          } catch (retryErr) {
            result = fail(retryErr.message);
          }
        } else {
          result = fail(e.message);
        }
      }
      const shouldClose = result && result._close;
      if (shouldClose) delete result._close;
      conn.write(JSON.stringify(result) + '\n');
      conn.end();
      if (shouldClose) setImmediate(() => { server.close(); cleanup(); process.exit(0); });
    });
    conn.on('error', () => {});
  });

  server.listen(sock, () => {
    try { fs.chmodSync(sock, 0o600); } catch {}
    // signal ready
  });

  function cleanup() {
    try { fs.unlinkSync(pidFile); } catch {}
    try { fs.unlinkSync(sock); } catch {}
  }

  process.on('SIGTERM', () => { server.close(); cleanup(); process.exit(0); });
  process.on('SIGINT',  () => { server.close(); cleanup(); process.exit(0); });
  process.on('exit', cleanup);
}

// ─── CLIENT (entry point) ─────────────────────────────────────────────────────

async function main() {
  const opts = parseArgs(process.argv.slice(2));

  if (opts.daemon) return runDaemon(opts);

  if (!opts.cmd) {
    process.stderr.write(
      'Usage: lightpanda_adapter.js --cdp <ws_url> --session <name> --json <cmd> [args…]\n'
    );
    process.exit(1);
  }

  const sock    = socketPath(opts.session);
  const errPath = sock + '.err';

  if (!fs.existsSync(sock)) {
    // Remove stale error marker
    try { fs.unlinkSync(errPath); } catch {}

    // Start daemon
    const child = spawn(
      process.execPath,
      [__filename, '--daemon',
        ...(opts.cdpUrl ? ['--cdp', opts.cdpUrl] : []),
        '--session', opts.session,
      ],
      { detached: true, stdio: 'ignore' }
    );
    child.unref();

    // Wait for socket or error marker
    const deadline = Date.now() + SOCKET_WAIT_MS;
    await new Promise((resolve, reject) => {
      const poll = () => {
        if (fs.existsSync(sock))    return resolve();
        if (fs.existsSync(errPath)) {
          const msg = fs.readFileSync(errPath, 'utf8');
          return reject(new Error('Daemon failed to start: ' + msg));
        }
        if (Date.now() >= deadline) return reject(new Error('Daemon did not start in time'));
        setTimeout(poll, SOCKET_POLL_MS);
      };
      poll();
    });
  }

  const result = await sendToSocket(sock, opts.cmd, opts.cmdArgs);
  process.stdout.write(JSON.stringify(result) + '\n');
  process.exit(result.success ? 0 : 1);
}

main().catch(e => {
  process.stdout.write(JSON.stringify(fail(e.message)) + '\n');
  process.exit(1);
});
