// Test-environment shims for the dashboard suite.
//
// 1) A usable `localStorage`.
//    Node 22+ exposes an experimental global `localStorage` that is inert unless the
//    process is started with `--localstorage-file` (it warns "localStorage is not
//    available because --localstorage-file was not provided" and reads as
//    `undefined`). jsdom does NOT override a global that already exists, so DOM
//    suites silently lost their storage and every localStorage-backed assertion
//    failed. Install a minimal in-memory Storage whenever the ambient one is
//    missing or unusable, so the suite behaves identically on every Node version.
class MemoryStorage implements Storage {
  #entries = new Map<string, string>();

  get length(): number {
    return this.#entries.size;
  }

  clear(): void {
    this.#entries.clear();
  }

  getItem(key: string): string | null {
    return this.#entries.has(key) ? (this.#entries.get(key) as string) : null;
  }

  key(index: number): string | null {
    return Array.from(this.#entries.keys())[index] ?? null;
  }

  removeItem(key: string): void {
    this.#entries.delete(key);
  }

  setItem(key: string, value: string): void {
    this.#entries.set(key, String(value));
  }
}

function ambientStorageUsable(): boolean {
  try {
    const storage = globalThis.localStorage;
    if (!storage || typeof storage.setItem !== "function") return false;
    storage.setItem("__probe__", "1");
    storage.removeItem("__probe__");
    return true;
  } catch {
    return false;
  }
}

if (!ambientStorageUsable()) {
  Object.defineProperty(globalThis, "localStorage", {
    value: new MemoryStorage(),
    configurable: true,
    writable: true,
  });
}

// 2) Pin the UI locale for the test run.
//
// Copy assertions are written against the English dictionary (`toContain("running
// low on memory")`). The app resolves its locale from the server preference, then
// localStorage, then the browser language (`resolveLocale` in
// `src/i18n/resolve-locale.ts`), so pinning localStorage here keeps every suite
// deterministic regardless of the host's `navigator.language`.
try {
  globalThis.localStorage?.setItem("hermes-locale", "en");
} catch {
  // No localStorage in this test environment — nothing to pin.
}
