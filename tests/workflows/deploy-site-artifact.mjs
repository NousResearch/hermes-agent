import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { networkInterfaces } from 'node:os';
import { performance } from 'node:perf_hooks';
import assert from 'node:assert/strict';

const [, , file, ...requested] = process.argv;
const workflow = readFileSync(file, 'utf8');
const block = workflow.split('      - name: Wait for the artifact to be listed\n')[1]?.split('\n      - name:')[0];
assert.ok(block);
const source = block.split('          script: |\n')[1].replace(/^ {12}/gm, '');
const pages = { name: 'github-pages' };
const loopback = Object.values(networkInterfaces()).flat().find(address => address.internal && address.family === "IPv4").address;
const cases = {
  immediate: [ [pages] ],
  delayed: [ [], [], [pages] ],
  never: [ [] ],
  transient: [ 503, [pages] ],
  rate: [ 429, [pages] ],
  permanent: [ 404 ],
  'hang-once': [ 'hang', [pages] ],
  hanging: [ 'hang' ],
  duplicate: [ [pages, pages], [pages] ],
};
await Promise.all((requested.length ? requested : Object.keys(cases)).map(async name => {
  let calls = 0;
  const times = [];
  const infos = [], warnings = [];
  const replies = cases[name];
  const server = createServer((req, res) => {
    req.resume();
    times.push(performance.now() - start);
    const reply = replies[Math.min(calls++, replies.length - 1)];
    if (reply === 'hang') return;
    if (typeof reply === 'number') { res.writeHead(reply).end(); return; }
    res.writeHead(200, {'content-type': 'application/json'}).end(JSON.stringify({ artifacts: reply }));
  });
  await new Promise(resolve => server.listen(0, loopback, resolve));
  const env = { ...process.env, ACTIONS_RUNTIME_TOKEN: 'test.' + Buffer.from(JSON.stringify({scp:'Actions.Results:1:2'})).toString('base64url') + '.test',
    ACTIONS_RESULTS_URL: `http://${loopback}:${server.address().port}` };
  const test = new (Object.getPrototypeOf(async function () {}).constructor)('core','process',source);
  const start = performance.now();
  try {
    await test({info: msg => infos.push(msg),warning: msg => warnings.push(msg)}, { env });
    const elapsed = performance.now() - start;
    console.log(JSON.stringify({ name, calls, elapsedMs: Math.round(elapsed), callMs: times.map(Math.round), lastInfo: infos.at(-1), firstInfo: infos[0], warnings }));
    if (['never','hanging'].includes(name)) {
      assert.match(warnings[0], /still not listed/);
      assert.ok(elapsed <= 120500, '120 s budget plus 500 ms scheduler tolerance');
    } else if (name === 'duplicate') {
      assert.equal(calls, 2, 'must wait until exactly one Pages artifact is listed');
    } else if (name === 'permanent') {
      assert.equal(calls, 1);
      assert.match(warnings[0], /HTTP 404/);
    } else {
      assert.equal(calls, replies.length);
      assert.match(infos.at(-1), /artifact is listed/);
      if (name === 'hang-once') assert.ok(times[1] >= 12900 && times[1] <= 13500);
    }
  } catch (e) { console.error(name + ': ' + e.message); process.exitCode = 1; }
  finally { server.closeAllConnections(); server.close(); }
}));
