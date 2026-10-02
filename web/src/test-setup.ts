// Pin the UI locale for the test run.
//
// Copy assertions are written against the English dictionary (`toContain("running
// low on memory")`), so the suite must not depend on the app's default locale —
// production now defaults to Simplified Chinese, and a locale default change
// would otherwise fail ~20 unrelated tests.
try {
  globalThis.localStorage?.setItem("hermes-locale", "en");
} catch {
  // No localStorage in this test environment — nothing to pin.
}
