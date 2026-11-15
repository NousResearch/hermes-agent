// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest";

import {
  isSafari,
  probeWebglSupport,
  shouldUseWebglRenderer,
  textNeedsDomShaping,
} from "./xterm-webgl-gating";

const SUPPORTED = { contextAvailable: true, softwareRenderer: false };

describe("isSafari", () => {
  it("accepts desktop and iOS Safari", () => {
    expect(
      isSafari(
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
      ),
    ).toBe(true);
    expect(
      isSafari(
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
      ),
    ).toBe(true);
  });

  it("rejects Chromium derivatives and iOS wrappers that carry Safari/", () => {
    const uas = [
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
      "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36",
      "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/124.0.0.0 Mobile/15E148 Safari/604.1",
      "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) FxiOS/124.0 Mobile/15E148 Safari/605.1.15",
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    ];
    for (const ua of uas) expect(isSafari(ua)).toBe(false);
  });

  it("rejects non-Safari UAs", () => {
    expect(isSafari("Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0")).toBe(false);
    expect(isSafari("")).toBe(false);
  });
});

describe("shouldUseWebglRenderer", () => {
  const chromeUa =
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36";
  const safariUa =
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15";

  it("uses WebGL for wide Chrome hosts with hardware GL (#18773 fixed for others)", () => {
    expect(
      shouldUseWebglRenderer({ layoutWidthPx: 1024, userAgent: chromeUa, support: SUPPORTED }),
    ).toBe(true);
    expect(
      shouldUseWebglRenderer({ layoutWidthPx: 767, userAgent: chromeUa, support: SUPPORTED }),
    ).toBe(false);
  });

  it("never uses WebGL on Safari regardless of width (#18773)", () => {
    expect(
      shouldUseWebglRenderer({ layoutWidthPx: 1280, userAgent: safariUa, support: SUPPORTED }),
    ).toBe(false);
  });

  it("never uses WebGL without a GL context (#45520)", () => {
    expect(
      shouldUseWebglRenderer({
        layoutWidthPx: 1280,
        userAgent: chromeUa,
        support: { contextAvailable: false, softwareRenderer: false },
      }),
    ).toBe(false);
  });

  it("never uses WebGL on software rasterizers like llvmpipe (#45520)", () => {
    expect(
      shouldUseWebglRenderer({
        layoutWidthPx: 1280,
        userAgent: chromeUa,
        support: { contextAvailable: true, softwareRenderer: true },
      }),
    ).toBe(false);
  });
});

describe("probeWebglSupport", () => {
  it("reports unavailable when context creation fails", () => {
    const doc = {
      createElement: () => ({
        getContext: () => null,
      }),
    } as unknown as Document;
    expect(probeWebglSupport(doc)).toEqual({
      contextAvailable: false,
      softwareRenderer: false,
    });
  });

  it("flags llvmpipe as a software renderer and releases the probe context", () => {
    const loseContext = vi.fn();
    const getParameter = vi.fn(() => "llvmpipe (LLVM 21.1.8, 256 bits)");
    const getExtension = vi.fn((name: string) => {
      if (name === "WEBGL_debug_renderer_info") {
        return { UNMASKED_RENDERER_WEBGL: 0x9246 };
      }
      if (name === "WEBGL_lose_context") {
        return { loseContext };
      }
      return null;
    });
    const doc = {
      createElement: () => ({
        getContext: () => ({ getExtension, getParameter }),
      }),
    } as unknown as Document;

    const support = probeWebglSupport(doc);
    expect(support).toEqual({ contextAvailable: true, softwareRenderer: true });
    expect(loseContext).toHaveBeenCalledTimes(1);
  });

  it("treats a hardware renderer as usable and releases the probe context", () => {
    const loseContext = vi.fn();
    const getParameter = vi.fn(() => "Apple M2");
    const getExtension = vi.fn((name: string) => {
      if (name === "WEBGL_debug_renderer_info") {
        return { UNMASKED_RENDERER_WEBGL: 0x9246 };
      }
      if (name === "WEBGL_lose_context") {
        return { loseContext };
      }
      return null;
    });
    const doc = {
      createElement: () => ({
        getContext: () => ({ getExtension, getParameter }),
      }),
    } as unknown as Document;

    expect(probeWebglSupport(doc)).toEqual({
      contextAvailable: true,
      softwareRenderer: false,
    });
    expect(loseContext).toHaveBeenCalledTimes(1);
  });
});

describe("textNeedsDomShaping", () => {
  it("flags Bengali conjunct text but not plain ASCII or box-drawing", () => {
    expect(textNeedsDomShaping("প্রযুক্তি ক্লিপবোর্ড যুক্তাক্ষর")).toBe(true);
    expect(textNeedsDomShaping("Devanagari: क्रिया")).toBe(true);
    expect(textNeedsDomShaping("Khmer: សួស្តី")).toBe(true);

    expect(textNeedsDomShaping("plain ascii output")).toBe(false);
    expect(textNeedsDomShaping("box drawing: ╔═╗║╚╝")).toBe(false);
    expect(textNeedsDomShaping("emoji 👍 and surrogate pairs")).toBe(false);
    expect(textNeedsDomShaping("")).toBe(false);
  });

  it("handles code points above the BMP", () => {
    expect(textNeedsDomShaping("😀 ক")).toBe(true);
  });
});
