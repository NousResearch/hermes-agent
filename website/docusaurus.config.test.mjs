import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {test} from 'node:test';

function loadSearchConfig(apiKey) {
  const env = {...process.env};
  delete env.ALGOLIA_SEARCH_API_KEY;
  if (apiKey !== undefined) {
    env.ALGOLIA_SEARCH_API_KEY = apiKey;
  }
  const result = spawnSync(process.execPath, ['--input-type=module', '-e', `
    const {default: config} = await import('./docusaurus.config.ts');
    console.log(JSON.stringify(config.themeConfig.algolia ?? null));
  `], {cwd: new URL('.', import.meta.url), env, encoding: 'utf8'});
  assert.equal(result.status, 0, result.stderr);
  return JSON.parse(result.stdout);
}

test('source builds omit DocSearch when no key is supplied', () => {
  for (const key of [undefined, '', '   ']) {
    assert.equal(loadSearchConfig(key) === null, true, 'DocSearch must be absent without a key');
  }
});

test('hosted builds use the supplied key and preserve contextual search', () => {
  const key = 'synthetic-docsearch-test-key';
  const config = loadSearchConfig(` ${key} `);
  assert.equal(config.apiKey === key, true, 'DocSearch must use only the supplied key');
  assert.equal(config.contextualSearch, true);
  assert.ok(config.appId);
  assert.ok(config.indexName);
});
