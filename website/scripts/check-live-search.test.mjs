import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

const monitor = fileURLToPath(new URL('./check-live-search.mjs', import.meta.url));

// The rendered button exists before the client can handle its clicks, as on Docusaurus.
const html = `<!doctype html><html data-has-hydrated="false"><head>
<script src="/docs/assets/js/main.123abc.js"></script></head><body>
<button class="DocSearch-Button">Search</button>
<script>
setTimeout(() => {
  document.querySelector('.DocSearch-Button').addEventListener('click', () => {
    const modal = document.createElement('div');
    modal.className = 'DocSearch-Modal';
    const input = document.createElement('input');
    input.className = 'DocSearch-Input';
    const results = document.createElement('ul');
    input.addEventListener('input', () => {
      results.replaceChildren(...Array.from({length: 5}, (_, index) => {
        const hit = document.createElement('li');
        hit.className = 'DocSearch-Hit';
        hit.textContent = 'telegram result ' + index;
        return hit;
      }));
    });
    modal.append(input, results);
    document.body.append(modal);
  });
  document.documentElement.dataset.hasHydrated = 'true';
}, 800);
</script></body></html>`;

const offlineFetch = `
const originalFetch = globalThis.fetch;
const docs = new URL(process.env.DOCS_URL);
globalThis.fetch = async (input, options) => {
  const url = new URL(typeof input === 'string' ? input : input.url);
  if (url.origin === docs.origin) return originalFetch(input, options);
  if (url.origin === 'https://fixture-dsn.algolia.net' && url.pathname === '/1/indexes/*/queries') {
    const {requests} = JSON.parse(options.body);
    return Response.json({results: requests.map(({query}) => ({
      nbHits: 10, hits: [{url: docs.origin + '/docs/result/' + encodeURIComponent(query)}],
    }))});
  }
  throw new Error('Offline regression blocked unexpected request: ' + url.origin);
};
`;

test('search probe waits for hydration before clicking the server-rendered button', { timeout: 30_000 }, async (t) => {
  const directory = await mkdtemp(path.join(tmpdir(), 'hermes-docs-hydration-test-'));
  let child;
  const server = createServer((request, response) => {
    if (request.url === '/docs/assets/js/main.123abc.js') {
      response.writeHead(200, { 'content-type': 'application/javascript' });
      response.end("window.fixtureConfig={algolia:{appId:'fixture',apiKey:'fixture',indexName:'fixture'}};");
      return;
    }
    response.writeHead(200, { 'content-type': 'text/html' });
    response.end(html);
  });
  t.after(async () => {
    child?.kill();
    server.closeAllConnections();
    if (server.listening) await new Promise((resolve) => server.close(resolve));
    await rm(directory, { recursive: true, force: true });
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  assert.ok(address && typeof address === 'object');
  const preload = path.join(directory, 'offline-fetch.mjs');
  await writeFile(preload, offlineFetch);
  const env = { ...process.env };
  for (const name of Object.keys(env)) {
    if (/API.?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL/i.test(name)) delete env[name];
  }
  delete env.GITHUB_ACTIONS;
  delete env.GITHUB_STEP_SUMMARY;
  delete env.GITHUB_OUTPUT;
  env.DOCS_URL = `http://127.0.0.1:${address.port}/docs`;
  env.NODE_OPTIONS = `--import=${pathToFileURL(preload).href}`;
  child = spawn(process.execPath, [monitor], { cwd: directory, env, stdio: ['ignore', 'pipe', 'pipe'] });
  let stdout = '';
  let stderr = '';
  child.stdout.setEncoding('utf8').on('data', (chunk) => { stdout += chunk; });
  child.stderr.setEncoding('utf8').on('data', (chunk) => { stderr += chunk; });
  const code = await new Promise((resolve, reject) => {
    child.once('error', reject);
    child.once('close', resolve);
  });
  assert.equal(code, 0, `Probe failed against the delayed-hydration fixture:\n${stdout}\n${stderr}`);
  assert.match(stdout, /browser: search modal lists 5 results for "telegram"/);
});
