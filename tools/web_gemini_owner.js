#!/usr/bin/env node
'use strict';

const path = require('path');
const fs = require('fs');
const readline = require('readline');

const moduleRoots = [
  process.env.PATCHRIGHT_NODE_MODULES,
  'C:/Users/815da/OneDrive/Desktop/BOT Projects/APEX BOT/notebooklm-http/node_modules',
  'C:/Users/815da/AppData/Local/hermes/patchright/node_modules',
].filter(Boolean);
for (const root of moduleRoots) {
  if (!module.paths.includes(root)) module.paths.unshift(root);
}

const { chromium } = require('patchright');
const PROFILE_DIR = process.env.WEB_GEMINI_PROFILE_DIR ||
  'C:/Users/815da/AppData/Local/hermes/web-gemini-feasibility/profile';
const START_URL = process.env.WEB_GEMINI_START_URL || 'https://gemini.google.com/';
const TIMEOUT = 20000;
const NAV_TIMEOUT = 60000;
const RESPONSE_TIMEOUT = Math.min(120000, Math.max(5000, Number(process.env.WEB_GEMINI_RESPONSE_TIMEOUT_MS || 90000)));
const HOST_ALLOWLIST = new Set(['gemini.google.com']);
const AUTO_NONCE = String(process.env.WEB_GEMINI_FEASIBILITY_NONCE || '').trim();
const AUTO_PROMPT = String(process.env.WEB_GEMINI_AUTO_PROMPT || '').trim();
const AUTO_MARKER = String(process.env.WEB_GEMINI_AUTO_MARKER || '').trim();
const RESUME_URL = String(process.env.WEB_GEMINI_RESUME_URL || '').trim();
const LOCK_PATH = path.join(PROFILE_DIR, 'web_gemini_owner.lock');

let context;
let page;
let shuttingDown = false;
let lockHeld = false;

function acquireLock() {
  fs.mkdirSync(PROFILE_DIR, { recursive: true });
  try {
    const fd = fs.openSync(LOCK_PATH, 'wx');
    fs.writeFileSync(fd, JSON.stringify({ pid: process.pid, started_at: Date.now() }), 'utf8');
    fs.closeSync(fd);
    lockHeld = true;
  } catch (error) {
    if (error && error.code === 'EEXIST') {
      throw new Error('dedicated web_gemini profile is already owned; refusing duplicate owner');
    }
    throw error;
  }
}

function releaseLock() {
  if (!lockHeld) return;
  try { fs.unlinkSync(LOCK_PATH); } catch (_) {}
  lockHeld = false;
}

function emit(event, payload = {}) {
  process.stdout.write(`${JSON.stringify({ event, ...payload })}\n`);
}

function safeUrl(raw) {
  const url = new URL(String(raw || ''));
  if (url.protocol !== 'https:' || !HOST_ALLOWLIST.has(url.hostname)) {
    throw new Error('only https://gemini.google.com/ URLs are allowed');
  }
  return url.toString();
}

async function snapshot() {
  if (!page || page.isClosed()) throw new Error('page unavailable');
  const result = await page.evaluate(() => {
    const nodes = Array.from(document.querySelectorAll(
      'a,button,input,textarea,select,[role="button"],[role="textbox"],[contenteditable="true"]',
    )).slice(0, 160);
    const interactive = nodes.map((node, index) => {
      const ref = `e${index + 1}`;
      node.setAttribute('data-web-gemini-ref', ref);
      return {
        ref,
        tag: node.tagName.toLowerCase(),
        role: node.getAttribute('role') || '',
        text: String(node.innerText || node.getAttribute('aria-label') ||
          node.getAttribute('placeholder') || '').replace(/\s+/g, ' ').trim().slice(0, 240),
        aria_label: node.getAttribute('aria-label') || '',
        title: node.getAttribute('title') || '',
        test_id: node.getAttribute('data-testid') || node.getAttribute('data-test-id') || '',
        type: node.getAttribute('type') || '',
        disabled: Boolean(node.disabled || node.getAttribute('aria-disabled') === 'true'),
      };
    });
    const bodyText = String(document.body?.innerText || '').replace(/\s+/g, ' ').trim();
    const boundedText = bodyText.length <= 30000
      ? bodyText
      : `${bodyText.slice(0, 10000)} [...middle omitted...] ${bodyText.slice(-20000)}`;
    return {
      url: location.href,
      title: document.title,
      text: boundedText,
      interactive,
    };
  });
  return result;
}

function refLocator(ref) {
  if (!/^e\d+$/.test(String(ref || ''))) throw new Error('invalid page ref');
  return page.locator(`[data-web-gemini-ref="${String(ref)}"]`).first();
}

async function command(input) {
  const op = String(input.op || '');
  if (op === 'status') {
    return {
      pid: process.pid,
      profile_dir: PROFILE_DIR,
      page_count: context.pages().length,
      url: page && !page.isClosed() ? page.url() : null,
      title: page && !page.isClosed() ? await page.title().catch(() => '') : null,
      owner: 'web_gemini_feasibility',
    };
  }
  if (op === 'snapshot') return snapshot();
  if (op === 'navigate') {
    const url = safeUrl(input.url);
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: NAV_TIMEOUT });
    return snapshot();
  }
  if (op === 'click_ref') {
    const locator = refLocator(input.ref);
    await locator.click({ timeout: TIMEOUT });
    return snapshot();
  }
  if (op === 'type_ref') {
    const locator = refLocator(input.ref);
    await locator.fill(String(input.text || ''), { timeout: TIMEOUT });
    return snapshot();
  }
  if (op === 'press_ref') {
    const locator = refLocator(input.ref);
    await locator.press(String(input.key || 'Enter'));
    return snapshot();
  }
  if (op === 'wait') {
    const ms = Math.max(100, Math.min(30000, Number(input.ms || 1000)));
    await new Promise(resolve => setTimeout(resolve, ms));
    return snapshot();
  }
  if (op === 'close') {
    await shutdown();
    return { closed: true };
  }
  throw new Error(`unsupported operation: ${op}`);
}

async function runResumeCheck() {
  if (!RESUME_URL) return;
  const url = safeUrl(RESUME_URL);
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: NAV_TIMEOUT });
  await new Promise(resolve => setTimeout(resolve, 2000));
  const snap = await snapshot();
  const marker = (AUTO_MARKER || (AUTO_NONCE ? `WEB_GEMINI_FEASIBILITY_OK ${AUTO_NONCE}` : '')).trim();
  let occurrences = 0;
  if (marker) {
    let from = 0;
    while (true) {
      const index = snap.text.indexOf(marker, from);
      if (index < 0) break;
      occurrences += 1;
      from = index + marker.length;
    }
  }
  emit('resumed', {
    url: snap.url,
    title: snap.title,
    nonce: AUTO_NONCE || null,
    marker_occurrences: occurrences,
    response_present: Boolean(marker && occurrences >= 2),
  });
}

function countOccurrences(text, marker) {
  let count = 0;
  let from = 0;
  while (marker) {
    const index = text.indexOf(marker, from);
    if (index < 0) break;
    count += 1;
    from = index + marker.length;
  }
  return count;
}

function conversationIdFromUrl(url) {
  const match = String(url || '').match(/\/app\/([^/?#]+)/);
  return match ? match[1] : null;
}

async function runAutoProbe() {
  if (!AUTO_PROMPT && !AUTO_NONCE) return;
  const marker = AUTO_MARKER || `WEB_GEMINI_FEASIBILITY_OK ${AUTO_NONCE}`;
  const prompt = AUTO_PROMPT || `This is a harmless read-only browser-owner feasibility probe. Reply with exactly one line: ${marker}. Do not call tools, change files, or request credentials.`;
  if (prompt.length > 12000) throw new Error('prompt exceeds owner size limit');
  const composerCandidates = [
    page.locator('textarea[placeholder="Ask Gemini"]').first(),
    page.locator('textarea').first(),
    page.locator('[role="textbox"]').first(),
    page.locator('[contenteditable="true"]:not(.ql-clipboard)').last(),
  ];
  let filled = false;
  let lastFillError = null;
  for (const candidate of composerCandidates) {
    try {
      if (await candidate.count() === 0) continue;
      await candidate.waitFor({ state: 'visible', timeout: TIMEOUT });
      await candidate.fill(prompt, { timeout: TIMEOUT });
      filled = true;
      break;
    } catch (error) {
      lastFillError = error;
    }
  }
  if (!filled) throw lastFillError || new Error('Gemini composer was not found');
  await new Promise(resolve => setTimeout(resolve, 800));
  const composerSnapshot = await snapshot();
  emit('composer_filled', {
    url: composerSnapshot.url,
    buttons: composerSnapshot.interactive
      .filter(item => item.tag === 'button')
      .map(item => ({ ref: item.ref, text: item.text, aria_label: item.aria_label, title: item.title, test_id: item.test_id, disabled: item.disabled }))
      .slice(-20),
  });
  const sendCandidates = [
    page.getByRole('button', { name: /send/i }).last(),
    page.locator('button[aria-label*="Send" i]').last(),
    page.locator('button').filter({ hasText: /^Send message$/i }).last(),
  ];
  let sent = false;
  for (const candidate of sendCandidates) {
    try {
      if (await candidate.count() === 0) continue;
      await candidate.waitFor({ state: 'visible', timeout: TIMEOUT });
      await candidate.click({ timeout: TIMEOUT });
      sent = true;
      break;
    } catch (_) {
      // Try the next explicitly allowlisted send selector; never fall back to coordinates or arbitrary DOM execution.
    }
  }
  if (!sent) throw new Error('Gemini send button was not found after filling the composer');
  emit('submitted', { nonce: AUTO_NONCE || null, marker, url: page.url(), conversation_id: conversationIdFromUrl(page.url()) });
  const deadline = Date.now() + RESPONSE_TIMEOUT;
  while (Date.now() < deadline) {
    await new Promise(resolve => setTimeout(resolve, 1500));
    const snap = await snapshot();
    const first = snap.text.indexOf(marker);
    const second = first >= 0 ? snap.text.indexOf(marker, first + marker.length) : -1;
    if (second >= 0) {
      emit('matched', {
        nonce: AUTO_NONCE || null,
        marker,
        url: snap.url,
        title: snap.title,
        conversation_id: conversationIdFromUrl(snap.url),
        marker_occurrences: countOccurrences(snap.text, marker),
        response_text: snap.text.slice(second + marker.length, second + marker.length + 12000).trim(),
      });
      return;
    }
  }
  emit('timeout', { nonce: AUTO_NONCE || null, marker, url: page.url(), title: await page.title().catch(() => '') });
}

async function shutdown() {
  if (shuttingDown) return;
  shuttingDown = true;
  try {
    if (context) await context.close();
  } finally {
    releaseLock();
    process.exit(0);
  }
}

(async () => {
  acquireLock();
  context = await chromium.launchPersistentContext(PROFILE_DIR, {
    headless: false,
    viewport: { width: 1400, height: 1000 },
    args: ['--no-first-run', '--no-default-browser-check', '--disable-blink-features=AutomationControlled'],
  });
  context.setDefaultTimeout(TIMEOUT);
  context.setDefaultNavigationTimeout(NAV_TIMEOUT);
  page = context.pages()[0] || await context.newPage();
  await page.goto(safeUrl(START_URL), { waitUntil: 'domcontentloaded', timeout: NAV_TIMEOUT });
  emit('ready', await command({ op: 'status' }));
  emit('snapshot', await snapshot());

  // Install stdin control before any auto-turn. The worker may send close as
  // soon as matched/fatal is emitted; dropping that line would strand the
  // persistent profile lock and browser process.
  const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  rl.on('line', async line => {
    if (shuttingDown || !line.trim()) return;
    try {
      const result = await command(JSON.parse(line));
      emit('result', { ok: true, result });
    } catch (error) {
      emit('result', { ok: false, error: String(error && error.message || error) });
    }
  });
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);

  await runResumeCheck();
  await runAutoProbe();
})().catch(error => {
  releaseLock();
  emit('fatal', { error: String(error && error.stack || error) });
  process.exit(1);
});
