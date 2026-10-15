#!/usr/bin/env node
// Probe search on the LIVE docs site the way a visitor uses it.
//
// Search broke in production twice with a green build: once when the Algolia
// search key was deleted, once when Vercel access controls got in the way.
// Neither shows up in CI because CI never talks to the deployed site, so this
// script does: it reads the Algolia credentials out of the bundle the site is
// actually serving, queries Algolia with the site's Origin, opens the top
// hits, and then drives the real search modal in a headless browser.
//
// Usage: node website/scripts/check-live-search.mjs [--no-browser]
//   DOCS_URL       site to probe (default https://hermes-agent.nousresearch.com/docs)
//   PLAYWRIGHT_DIR directory holding node_modules/playwright (browser check)
// Exit 0 = search works; exit 1 = broken. Reasons go to stderr and, in
// Actions, to the step summary and `::error::` annotations (alert receivers
// read the annotations off the failed run).

import { appendFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';

const DOCS_URL = (process.env.DOCS_URL || 'https://hermes-agent.nousresearch.com/docs').replace(/\/$/, '');
const ORIGIN = new URL(DOCS_URL).origin;
// Terms that must always match many pages; one per major area of the docs.
const QUERIES = ['telegram', 'install', 'skills'];
const MIN_HITS = 5;
// Every request (body included) gets a deadline, and the whole probe gets one,
// so a slow-drip server fails the check with time left to retry and alert
// instead of running into the job's timeout.
const REQUEST_TIMEOUT_MS = 20_000;
const PROBE_DEADLINE_MS = 240_000;
const UA = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36 hermes-docs-search-monitor';

const failures = [];
const fail = (msg) => {
  failures.push(msg);
  console.error(`FAIL ${msg}`);
  if (process.env.GITHUB_ACTIONS) console.log(`::error title=Docs search::${msg.replace(/%/g, '%25').replace(/\r/g, '%0D').replace(/\n/g, '%0A')}`);
};
const ok = (msg) => console.log(`ok   ${msg}`);

async function get(url, headers = {}) {
  const res = await fetch(url, {
    headers: { 'user-agent': UA, ...headers },
    redirect: 'follow',
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
  });
  return { res, body: await res.text() };
}

// Vercel's firewall/challenge and password protection answer 403/401/429
// with an HTML page of their own; name that instead of "bad status".
function blockedBy(res) {
  if (res.headers.get('x-vercel-mitigated')) return `Vercel firewall (${res.headers.get('x-vercel-mitigated')})`;
  if (res.status === 401) return 'auth wall (401)';
  if (res.status === 403 || res.status === 429) return `HTTP ${res.status}`;
  return null;
}

async function readDeployedConfig() {
  const { res, body } = await get(`${DOCS_URL}/`);
  const block = blockedBy(res);
  if (block || !res.ok) {
    fail(`docs page ${DOCS_URL}/ is not reachable for visitors: ${block || `HTTP ${res.status}`}`);
    return null;
  }
  const bundle = body.match(/src="([^"]*\/main\.[0-9a-f]+\.js)"/)?.[1];
  if (!bundle) {
    fail('docs page has no main.<hash>.js bundle; cannot read the deployed search config');
    return null;
  }
  const js = (await get(new URL(bundle, ORIGIN).href)).body;
  // Theme code destructures `algolia:{...}` too; take the block that carries
  // all three fields, each read on its own so key order and quoting in the
  // compiled config do not matter.
  const field = (cfg, name) => cfg.match(new RegExp(`["']?${name}["']?\\s*:\\s*(["'])([^"']+)\\1`))?.[2];
  const [appId, apiKey, indexName] =
    js
      .split('algolia:{')
      .slice(1)
      .map((cfg) => ['appId', 'apiKey', 'indexName'].map((name) => field(cfg.slice(0, 2000), name)))
      .find((fields) => fields.every(Boolean)) || [];
  if (!appId) {
    fail(`deployed bundle ${bundle} has no complete Algolia config; search is not configured on the live site`);
    return null;
  }
  ok(`deployed config: app ${appId}, index "${indexName}" (from ${bundle})`);
  return { appId, apiKey, indexName };
}

async function checkAlgolia({ appId, apiKey, indexName }) {
  const res = await fetch(`https://${appId}-dsn.algolia.net/1/indexes/*/queries`, {
    method: 'POST',
    headers: {
      'x-algolia-application-id': appId,
      'x-algolia-api-key': apiKey,
      origin: ORIGIN,
      referer: `${DOCS_URL}/`,
      'content-type': 'application/json',
    },
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    body: JSON.stringify({
      requests: QUERIES.map((query) => ({ indexName, query, hitsPerPage: 3, attributesToRetrieve: ['url'] })),
    }),
  });
  const body = await res.text();
  if (!res.ok) {
    // 403 "Invalid Application-ID or API key" is the deleted-key outage.
    fail(`Algolia rejected the deployed search key: HTTP ${res.status} ${body.slice(0, 200)}`);
    return [];
  }
  const urls = [];
  for (const [i, result] of JSON.parse(body).results.entries()) {
    const hosts = result.hits.map((h) => new URL(h.url).origin);
    if (result.nbHits < MIN_HITS) fail(`query "${QUERIES[i]}" returned ${result.nbHits} hits (< ${MIN_HITS}); index looks empty`);
    else if (!hosts.every((h) => h === ORIGIN)) fail(`query "${QUERIES[i]}" hits point off-site: ${hosts.join(', ')}`);
    else ok(`query "${QUERIES[i]}": ${result.nbHits} hits`);
    if (result.hits[0]) urls.push(result.hits[0].url);
  }
  return urls;
}

async function checkHitPagesOpen(urls) {
  for (const url of urls) {
    const { res } = await get(url);
    const block = blockedBy(res);
    if (block || !res.ok) fail(`search result ${url} does not open: ${block || `HTTP ${res.status}`}`);
    else ok(`result opens: ${url}`);
  }
}

async function checkBrowser() {
  const require = createRequire(path.join(process.env.PLAYWRIGHT_DIR || process.cwd(), 'noop.js'));
  const { chromium } = require('playwright');
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({ userAgent: UA });
    const algoliaErrors = [];
    page.on('response', (r) => {
      if (r.url().includes('algolia') && r.status() >= 400) algoliaErrors.push(`${r.status()} ${r.url().split('?')[0]}`);
    });
    page.on('requestfailed', (r) => {
      if (r.url().includes('algolia')) algoliaErrors.push(`${r.failure()?.errorText} ${r.url().split('?')[0]}`);
    });
    await page.goto(`${DOCS_URL}/`, { waitUntil: 'domcontentloaded', timeout: 30_000 });
    await page.locator('.DocSearch-Button').first().click({ timeout: 20_000 });
    await page.locator('.DocSearch-Input').fill(QUERIES[0], { timeout: 10_000 });
    try {
      await page.locator('.DocSearch-Hit').first().waitFor({ timeout: 15_000 });
    } catch {
      const shown = (await page.locator('.DocSearch-Modal').innerText().catch(() => '')).replace(/\s+/g, ' ').slice(0, 200);
      fail(`search modal showed no results for "${QUERIES[0]}" (${algoliaErrors.join('; ') || 'no Algolia error seen'}; modal: "${shown}")`);
      return;
    }
    ok(`browser: search modal lists ${await page.locator('.DocSearch-Hit').count()} results for "${QUERIES[0]}"`);
  } finally {
    await browser.close();
  }
}

async function probe() {
  const config = await readDeployedConfig();
  if (config) await checkHitPagesOpen(await checkAlgolia(config));
  if (!process.argv.includes('--no-browser')) {
    await checkBrowser().catch((e) => fail(`browser check crashed: ${e.message.split('\n')[0]}`));
  }
}

// One exit path for clean failures, thrown errors and the deadline, so the
// alert always carries the real reason.
function finish() {
  if (process.env.GITHUB_STEP_SUMMARY) {
    const lines = failures.length ? failures.map((f) => `- ${f}`) : ['- all checks passed'];
    appendFileSync(process.env.GITHUB_STEP_SUMMARY, `### Docs search (${DOCS_URL})\n${lines.join('\n')}\n`);
  }
  process.exit(failures.length ? 1 : 0);
}

setTimeout(() => {
  fail(`probe did not finish within ${PROBE_DEADLINE_MS / 1000}s`);
  finish();
}, PROBE_DEADLINE_MS).unref();
probe()
  .catch((e) => fail(`probe crashed: ${e.cause?.message || e.message}`))
  .finally(finish);
