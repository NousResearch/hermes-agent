import { describe, expect, it, vi } from "vitest";
import { reloadForChunkLoadError } from "./chunk-load-recovery";

function memoryStorage() {
  const data = new Map<string, string>();
  return {
    getItem: (k: string) => data.get(k) ?? null,
    setItem: (k: string, v: string) => void data.set(k, v),
  };
}

const staleChunk = () =>
  new Error("Failed to fetch dynamically imported module: http://127.0.0.1:9119/assets/ConfigPage-D09a41f-.js");

describe("reloadForChunkLoadError", () => {
  it("reloads once for a missing build chunk, then not again within the guard window", () => {
    const storage = memoryStorage();
    const reload = vi.fn();
    const t0 = 1_000_000;
    expect(reloadForChunkLoadError(staleChunk(), storage, reload, t0)).toBe(true);
    expect(reloadForChunkLoadError(staleChunk(), storage, reload, t0 + 60_000)).toBe(false);
    expect(reload).toHaveBeenCalledTimes(1);
    expect(reloadForChunkLoadError(staleChunk(), storage, reload, t0 + 6 * 60_000)).toBe(true);
    expect(reload).toHaveBeenCalledTimes(2);
  });

  it("ignores plugin bundles, ordinary errors, and tabs without session storage", () => {
    const reload = vi.fn();
    const plugin = new Error(
      "Failed to fetch dynamically imported module: http://127.0.0.1:9119/dashboard-plugins/kanban/dist/index.js",
    );
    expect(reloadForChunkLoadError(plugin, memoryStorage(), reload)).toBe(false);
    expect(reloadForChunkLoadError(new Error("boom"), memoryStorage(), reload)).toBe(false);
    expect(reloadForChunkLoadError(staleChunk(), null, reload)).toBe(false);
    expect(reload).not.toHaveBeenCalled();
  });
});
