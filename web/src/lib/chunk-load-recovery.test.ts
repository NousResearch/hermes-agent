import { describe, expect, it, vi } from "vitest";
import {
  type ChunkLoadRecoveryDeps,
  entryFromIndexHtml,
  reloadForChunkLoadError,
  startChunkLoadRecovery,
} from "./chunk-load-recovery";

function memoryStorage() {
  const data = new Map<string, string>();
  return {
    getItem: (k: string) => data.get(k) ?? null,
    setItem: (k: string, v: string) => void data.set(k, v),
  };
}

const staleChunk = () =>
  new Error("Failed to fetch dynamically imported module: http://127.0.0.1:9119/assets/ConfigPage-D09a41f-.js");

const entry = (build: string) => `http://127.0.0.1:9119/assets/index-${build}.js`;

/** A tab running build `running` while the server publishes build `published`. */
function builds(running: string, published: string | null, extra: ChunkLoadRecoveryDeps = {}) {
  const reload = vi.fn();
  const deps = {
    storage: memoryStorage(),
    reload,
    online: () => true,
    runningEntry: () => entry(running),
    publishedEntry: vi.fn(async () => (published ? entry(published) : null)),
    ...extra,
  };
  return { deps, reload };
}

describe("reloadForChunkLoadError", () => {
  it("reloads once per running build: a repeat on the same build is suppressed", async () => {
    const { deps, reload } = builds("A", "B");
    expect(await reloadForChunkLoadError(staleChunk(), deps)).toBe(true);
    expect(await reloadForChunkLoadError(staleChunk(), deps)).toBe(false);
    expect(reload).toHaveBeenCalledTimes(1);
  });

  it("ignores plugin bundles, ordinary errors, and tabs without session storage", async () => {
    const { deps, reload } = builds("A", "B");
    const plugin = new Error(
      "Failed to fetch dynamically imported module: http://127.0.0.1:9119/dashboard-plugins/kanban/dist/index.js",
    );
    expect(await reloadForChunkLoadError(plugin, { ...deps, storage: memoryStorage() })).toBe(false);
    expect(await reloadForChunkLoadError(new Error("boom"), { ...deps, storage: memoryStorage() })).toBe(false);
    expect(await reloadForChunkLoadError(staleChunk(), { ...deps, storage: null })).toBe(false);
    expect(reload).not.toHaveBeenCalled();
  });

  it("does not reload for a network failure when the build has not changed", async () => {
    const { deps, reload } = builds("A", "A");
    expect(await reloadForChunkLoadError(staleChunk(), deps)).toBe(false);
    expect(deps.publishedEntry).toHaveBeenCalledTimes(1);
    expect(reload).not.toHaveBeenCalled();
  });

  it("does not reload when the fresh index cannot be fetched", async () => {
    const { deps, reload } = builds("A", null);
    expect(await reloadForChunkLoadError(staleChunk(), deps)).toBe(false);
    expect(reload).not.toHaveBeenCalled();
  });

  it("does not reload, or even ask the server, while offline", async () => {
    const { deps, reload } = builds("A", "B", { online: () => false });
    expect(await reloadForChunkLoadError(staleChunk(), deps)).toBe(false);
    expect(deps.publishedEntry).not.toHaveBeenCalled();
    expect(reload).not.toHaveBeenCalled();
  });

  it("recovers again when a later build lands inside the same session (A -> B -> C)", async () => {
    const storage = memoryStorage();
    const onA = builds("A", "B", { storage });
    expect(await reloadForChunkLoadError(staleChunk(), onA.deps)).toBe(true);
    // The reload landed on B; C is published before B's lazy page is opened.
    const onB = builds("B", "C", { storage });
    expect(await reloadForChunkLoadError(staleChunk(), onB.deps)).toBe(true);
    expect(await reloadForChunkLoadError(staleChunk(), onB.deps)).toBe(false);
    expect(onA.reload).toHaveBeenCalledTimes(1);
    expect(onB.reload).toHaveBeenCalledTimes(1);
  });

  it("does not loop when the reload lands on the same build again", async () => {
    const storage = memoryStorage();
    expect(await reloadForChunkLoadError(staleChunk(), builds("A", "B", { storage }).deps)).toBe(true);
    // A stale index came back after the reload: still A, server still says B.
    const again = builds("A", "B", { storage });
    expect(await reloadForChunkLoadError(staleChunk(), again.deps)).toBe(false);
    expect(again.reload).not.toHaveBeenCalled();
  });

  it("starts one reload for concurrent failures on the same build", async () => {
    const { deps, reload } = builds("A", "B");
    const results = await Promise.all([
      reloadForChunkLoadError(staleChunk(), deps),
      reloadForChunkLoadError(staleChunk(), deps),
    ]);
    expect(results.filter(Boolean)).toHaveLength(1);
    expect(reload).toHaveBeenCalledTimes(1);
  });
});

describe("entryFromIndexHtml", () => {
  it("resolves the built module entry, base path included", () => {
    const html =
      '<script>window.__HERMES_SESSION_TOKEN__="t";</script>' +
      '<script type="module" crossorigin src="/hermes/assets/index-Bx1.js"></script>';
    expect(entryFromIndexHtml(html, "https://host/hermes/sessions")).toBe("https://host/hermes/assets/index-Bx1.js");
    expect(entryFromIndexHtml("<p>gateway error</p>", "https://host/")).toBeNull();
  });
});

describe("startChunkLoadRecovery", () => {
  function preloadError(payload: unknown) {
    return Object.assign(new Event("vite:preloadError"), { payload });
  }

  it("turns vite:preloadError into one reload, then suppresses the next one", async () => {
    const target = new EventTarget();
    const { deps, reload } = builds("A", "B");
    startChunkLoadRecovery(target as unknown as Window, deps);
    target.dispatchEvent(preloadError(staleChunk()));
    await vi.waitFor(() => expect(reload).toHaveBeenCalledTimes(1));
    target.dispatchEvent(preloadError(staleChunk()));
    await new Promise((r) => setTimeout(r, 0));
    expect(reload).toHaveBeenCalledTimes(1);
  });

  it.each([
    ["chunk CSS", "Unable to preload CSS for /assets/ConfigPage-D09a41f-.css"],
    ["Safari, no URL", "Importing a module script failed."],
    ["Firefox", "error loading dynamically imported module: http://127.0.0.1:9119/assets/ConfigPage-D09a41f-.js"],
  ])("reloads for a %s failure", async (_name, message) => {
    const target = new EventTarget();
    const { deps, reload } = builds("A", "B");
    startChunkLoadRecovery(target as unknown as Window, deps);
    target.dispatchEvent(preloadError(new Error(message)));
    await vi.waitFor(() => expect(reload).toHaveBeenCalledTimes(1));
  });

  it("never reloads when session storage throws", async () => {
    const target = new EventTarget();
    const throwing = {
      getItem: () => {
        throw new DOMException("denied", "SecurityError");
      },
      setItem: () => {
        throw new DOMException("denied", "SecurityError");
      },
    };
    const { deps, reload } = builds("A", "B", { storage: throwing });
    startChunkLoadRecovery(target as unknown as Window, deps);
    target.dispatchEvent(preloadError(staleChunk()));
    await new Promise((r) => setTimeout(r, 0));
    expect(reload).not.toHaveBeenCalled();
  });
});
