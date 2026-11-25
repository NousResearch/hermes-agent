import { describe, expect, it, vi } from "vitest";

import {
  refitWhenTerminalFontLoads,
  TERMINAL_FONT_FAMILY,
} from "./terminal-font-refit";

/** Mirrors xterm: the cell is re-measured only when fontFamily changes. */
function fakeTerminal() {
  let family = TERMINAL_FONT_FAMILY;
  const term = {
    measuredWith: [] as string[],
    options: {
      get fontFamily() {
        return family;
      },
      set fontFamily(next: string) {
        if (next === family) return;
        family = next;
        term.measuredWith.push(next);
      },
    },
    rows: 24,
    clearTextureAtlas: vi.fn(),
    refresh: vi.fn(),
  };
  return term;
}

function deferredFontSet(opts: { loaded?: boolean; faces?: unknown[] } = {}) {
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  return {
    release,
    fontSet: {
      check: vi.fn(() => opts.loaded ?? false),
      load: vi.fn(async () => {
        await gate;
        return (opts.faces ?? [{}]) as FontFace[];
      }),
    },
  };
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

describe("refitWhenTerminalFontLoads", () => {
  it("re-measures with the loaded face, then refits and redraws", async () => {
    const term = fakeTerminal();
    const fit = vi.fn();
    const { fontSet, release } = deferredFontSet();

    refitWhenTerminalFontLoads(term, fit, fontSet);
    await settle();
    expect(fit).not.toHaveBeenCalled();

    release();
    await settle();

    expect(term.measuredWith.at(-1)).toBe(TERMINAL_FONT_FAMILY);
    expect(term.options.fontFamily).toBe(TERMINAL_FONT_FAMILY);
    expect(fit).toHaveBeenCalledTimes(1);
    expect(term.clearTextureAtlas).toHaveBeenCalledTimes(1);
    expect(term.refresh).toHaveBeenCalledWith(0, 23);
  });

  it("does nothing after cleanup runs before the font arrives", async () => {
    const term = fakeTerminal();
    const fit = vi.fn();
    const { fontSet, release } = deferredFontSet();

    const cleanup = refitWhenTerminalFontLoads(term, fit, fontSet);
    cleanup();
    release();
    await settle();

    expect(term.measuredWith).toEqual([]);
    expect(fit).not.toHaveBeenCalled();
  });

  it("skips the refit when the faces were already loaded at open()", async () => {
    const term = fakeTerminal();
    const fit = vi.fn();
    const { fontSet } = deferredFontSet({ loaded: true });

    refitWhenTerminalFontLoads(term, fit, fontSet);
    await settle();

    expect(fontSet.load).not.toHaveBeenCalled();
    expect(fit).not.toHaveBeenCalled();
  });

  it("leaves fallback metrics alone when no bundled face could load", async () => {
    const term = fakeTerminal();
    const fit = vi.fn();
    const { fontSet, release } = deferredFontSet({ faces: [] });

    refitWhenTerminalFontLoads(term, fit, fontSet);
    release();
    await settle();

    expect(term.measuredWith).toEqual([]);
    expect(fit).not.toHaveBeenCalled();
  });

  it("is a no-op without a FontFaceSet", () => {
    const fit = vi.fn();
    expect(() => refitWhenTerminalFontLoads(fakeTerminal(), fit, undefined)()).not.toThrow();
    expect(fit).not.toHaveBeenCalled();
  });
});
