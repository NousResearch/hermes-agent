// Pin the UI locale for the test run.
//
// Copy assertions are written against the English dictionary (`toContain("running
// low on memory")`). The app now resolves its locale from the server preference,
// then localStorage, then the browser language (`resolveLocale` in
// `src/i18n/resolve-locale.ts`), so pinning localStorage here keeps every suite
// deterministic regardless of the host's `navigator.language`. jsdom suites read
// this pin; the node-env suites see no localStorage and fall through to `en`
// anyway (Node's `navigator.language` is `en-US`).
try {
  globalThis.localStorage?.setItem("hermes-locale", "en");
} catch {
  // No localStorage in this test environment — nothing to pin.
}
