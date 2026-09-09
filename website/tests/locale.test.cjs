const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {test} = require('node:test');
const vm = require('node:vm');
const ts = require('typescript');

// Compile the real component, not a copy of its URL or preference logic.
// Only React/Docusaurus and browser APIs are replaced by boundary adapters.
const source = readFileSync(require.resolve('../src/theme/Root.tsx'), 'utf8');
const {outputText} = ts.transpileModule(source, {
  compilerOptions: {module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.React},
});
const LOCALE = 'hermes-agent-docs-locale';
const PENDING = 'hermes-agent-docs-pending-locale';

function visit({baseUrl = '/docs/', path = '/docs/', languages = ['en'],
  language = languages[0], storage = new Map()} = {}) {
  const listeners = new Set();
  const redirects = [];
  let cleanup;
  class Element {
    constructor(anchor) { this.anchor = anchor; }
    closest(selector) {
      assert.equal(selector, 'a[href]');
      return this.anchor;
    }
  }
  const location = new URL(path, 'https://example.com');
  location.replace = (url) => redirects.push(url);
  const document = {
    addEventListener(type, callback, capture) {
      assert.equal(type, 'click');
      assert.equal(capture, true);
      listeners.add(callback);
    },
    removeEventListener(type, callback, capture) {
      assert.equal(type, 'click');
      assert.equal(capture, true);
      listeners.delete(callback);
    },
  };
  const exports = {};
  vm.runInNewContext(outputText, {
    exports, URL, Element, document,
    navigator: {languages, language},
    window: {location, localStorage: {
      getItem: (key) => storage.get(key) ?? null,
      setItem: (key, value) => storage.set(key, value),
      removeItem: (key) => storage.delete(key),
    }},
    require(name) {
      if (name === 'react') return {
        default: {createElement: () => null},
        useEffect: (effect) => { cleanup = effect(); },
      };
      if (name === '@docusaurus/useDocusaurusContext') return {
        default: () => ({siteConfig: {baseUrl}}),
      };
      throw new Error(`Unexpected import: ${name}`);
    },
  }, {filename: 'Root.js'});
  exports.default({children: null});
  return {
    redirects, storage, listeners, cleanup,
    click(label, href) {
      const target = new Element({textContent: label, href});
      for (const listener of listeners) listener({target});
    },
  };
}

for (const baseUrl of ['/docs/', '/hermes-agent/docs/', '/']) {
  for (const [languages, locale] of [
    [['en'], 'en'], [['en-US'], 'en'], [['EN-gb'], 'en'],
    [['zh'], 'zh-Hans'], [['zh-CN'], 'zh-Hans'],
    [['zh-TW'], 'zh-Hans'], [['zh-Hant'], 'zh-Hans'],
    [['ZH-hans'], 'zh-Hans'], [['ko', 'zh-CN'], 'zh-Hans'],
    [['en', 'zh-CN'], 'en'], [['fr'], 'en'],
  ]) {
    test(`${baseUrl}: browser ${languages.join(',')} selects ${locale}`, () => {
      const page = visit({baseUrl, path: `${baseUrl}guide?q=a%2Fb#install`, languages});
      assert.deepEqual(page.redirects, locale === 'en'
        ? [] : [`${baseUrl}zh-Hans/guide?q=a%2Fb#install`]);
      if (locale === 'en') assert.equal(page.storage.get(LOCALE), 'en');
      page.cleanup();
      assert.equal(page.listeners.size, 0);
    });
  }
  test(`${baseUrl}: strip current locale and preserve query/hash`, () => {
    const page = visit({baseUrl, path: `${baseUrl}zh-Hans/guide?q=a%2Fb#install`});
    assert.deepEqual(page.redirects, [`${baseUrl}guide?q=a%2Fb#install`]);
  });
  test(`${baseUrl}: keep current Chinese locale without doubling prefix`, () => {
    const page = visit({baseUrl, path: `${baseUrl}zh-Hans/guide`, languages: ['zh-TW']});
    assert.deepEqual(page.redirects, []);
    assert.equal(page.storage.get(LOCALE), 'zh-Hans');
  });
  for (const [choice, label, opposite] of [
    ['en', 'English', 'zh-Hans'], ['zh-Hans', '简体中文', 'en'],
  ]) {
    test(`${baseUrl}: explicit ${choice} choice survives navigation and later visits`, () => {
      const storage = new Map([[LOCALE, opposite]]);
      const initial = visit({baseUrl, storage, languages: [opposite],
        path: `${baseUrl}${opposite === 'en' ? '' : 'zh-Hans/'}guide`});
      const destination = `${baseUrl}${choice === 'en' ? '' : 'zh-Hans/'}guide?q=1#title`;
      initial.click(` ${label} `, `https://example.com${destination}`);
      assert.equal(storage.get(PENDING), choice);
      assert.equal(storage.get(LOCALE), opposite);
      initial.cleanup();
      assert.equal(initial.listeners.size, 0);
      const arrival = visit({baseUrl, storage, path: destination, languages: [opposite]});
      assert.deepEqual(arrival.redirects, []);
      assert.equal(storage.get(LOCALE), choice);
      assert.equal(storage.has(PENDING), false);
      arrival.cleanup();
      const later = visit({baseUrl, storage, path: `${baseUrl}guide?q=1#title`, languages: [opposite]});
      assert.deepEqual(later.redirects, choice === 'en' ? [] : [destination]);
    });
  }
}

for (const [baseUrl, path, expected] of [
  ['/docs/', '/docs/', '/docs/zh-Hans'],
  ['/docs/', '/docs', '/docs/zh-Hans'],
  ['/hermes-agent/docs/', '/hermes-agent/docs/', '/hermes-agent/docs/zh-Hans'],
  ['docs', '/docs/guide', '/docs/zh-Hans/guide'],
]) {
  test(`base boundary ${baseUrl} at ${path}`, () => {
    assert.deepEqual(visit({baseUrl, path, languages: ['zh-CN']}).redirects, [expected]);
  });
}

test('navigator.language fallback when languages is unavailable', () => {
  assert.deepEqual(visit({languages: null, language: 'zh-CN'}).redirects, ['/docs/zh-Hans']);
  assert.deepEqual(visit({languages: [], language: 'zh-CN'}).redirects, ['/docs/zh-Hans']);
});

test('ordinary document links do not persist a locale choice', () => {
  const page = visit();
  page.click('Getting started', 'https://example.com/docs/guide');
  assert.equal(page.storage.has(PENDING), false);
});

test('pending choice is not committed on an unrelated locale page', () => {
  const storage = new Map([[LOCALE, 'en'], [PENDING, 'zh-Hans']]);
  visit({storage});
  assert.equal(storage.get(LOCALE), 'en');
  assert.equal(storage.get(PENDING), 'zh-Hans');
});
