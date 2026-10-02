import { describe, expect, it } from "vitest";

import {
  ROUTING_KEYS,
  addFallbackRoute,
  hasMainFallbackSupport,
  hasSchemaKey,
  isRouteComplete,
  parseFallbackRoutes,
  removeFallbackRoute,
  serializeFallbackRoutes,
  updateFallbackRoute,
} from "./model-routing";

describe("parseFallbackRoutes", () => {
  it("accepts a single dict (legacy fallback_model) as a one-entry chain", () => {
    expect(parseFallbackRoutes({ provider: "openrouter", model: "x/y" })).toEqual([
      { provider: "openrouter", model: "x/y" },
    ]);
  });

  it("accepts a list of dicts and keeps order", () => {
    expect(
      parseFallbackRoutes([
        { provider: "openrouter", model: "a/b" },
        { provider: "anthropic", model: "c/d", base_url: "http://localhost:8000/v1" },
      ]),
    ).toEqual([
      { provider: "openrouter", model: "a/b" },
      { provider: "anthropic", model: "c/d", base_url: "http://localhost:8000/v1" },
    ]);
  });

  it("keeps optional base_url/api_mode only when non-blank", () => {
    expect(
      parseFallbackRoutes([{ provider: "p", model: "m", base_url: "  ", api_mode: "" }]),
    ).toEqual([{ provider: "p", model: "m" }]);
    expect(
      parseFallbackRoutes([{ provider: "p", model: "m", base_url: " http://x/v1 ", api_mode: "chat_completions" }]),
    ).toEqual([{ provider: "p", model: "m", base_url: "http://x/v1", api_mode: "chat_completions" }]);
  });

  it("preserves partial rows so an in-progress edit survives a re-render", () => {
    expect(parseFallbackRoutes([{ provider: "openrouter" }])).toEqual([
      { provider: "openrouter", model: "" },
    ]);
  });

  it("collapses null/undefined/strings/non-object entries to []", () => {
    expect(parseFallbackRoutes(null)).toEqual([]);
    expect(parseFallbackRoutes(undefined)).toEqual([]);
    expect(parseFallbackRoutes("")).toEqual([]);
    expect(parseFallbackRoutes(42)).toEqual([]);
    expect(parseFallbackRoutes(["nope", 7, null])).toEqual([]);
  });
});

describe("serializeFallbackRoutes", () => {
  it("drops all-blank rows and trims the rest", () => {
    expect(
      serializeFallbackRoutes([
        { provider: "  ", model: "" },
        { provider: " openrouter ", model: " a/b " },
      ]),
    ).toEqual([{ provider: "openrouter", model: "a/b" }]);
  });

  it("keeps a row with only one half filled (backend decides usability)", () => {
    expect(serializeFallbackRoutes([{ provider: "openrouter", model: "" }])).toEqual([
      { provider: "openrouter", model: "" },
    ]);
  });

  it("round-trips a parsed chain without the optional keys when unused", () => {
    const chain = [
      { provider: "openrouter", model: "a/b" },
      { provider: "anthropic", model: "c/d", base_url: "http://x/v1", api_mode: "anthropic_messages" },
    ];
    expect(serializeFallbackRoutes(parseFallbackRoutes(chain))).toEqual(chain);
  });
});

describe("route reducers", () => {
  const base = [{ provider: "a", model: "1" }, { provider: "b", model: "2" }];

  it("adds a blank row", () => {
    expect(addFallbackRoute(base)).toEqual([
      { provider: "a", model: "1" },
      { provider: "b", model: "2" },
      { provider: "", model: "" },
    ]);
  });

  it("patches one row without touching the others or the input array", () => {
    const next = updateFallbackRoute(base, 1, { model: "9" });
    expect(next).toEqual([
      { provider: "a", model: "1" },
      { provider: "b", model: "9" },
    ]);
    // Input is untouched (pure reducer).
    expect(base[1]).toEqual({ provider: "b", model: "2" });
  });

  it("removes one row and is a no-op for an out-of-range index", () => {
    expect(removeFallbackRoute(base, 0)).toEqual([{ provider: "b", model: "2" }]);
    expect(removeFallbackRoute(base, 5)).toBe(base);
    expect(updateFallbackRoute(base, -1, { model: "x" })).toBe(base);
  });

  it("reports completeness only when both halves are present", () => {
    expect(isRouteComplete({ provider: "p", model: "m" })).toBe(true);
    expect(isRouteComplete({ provider: "p", model: " " })).toBe(false);
    expect(isRouteComplete({ provider: "", model: "m" })).toBe(false);
  });
});

describe("schema gating", () => {
  it("hasSchemaKey answers presence, not truthiness", () => {
    expect(hasSchemaKey({ [ROUTING_KEYS.hotReload]: {} }, ROUTING_KEYS.hotReload)).toBe(true);
    expect(hasSchemaKey({}, ROUTING_KEYS.hotReload)).toBe(false);
    expect(hasSchemaKey(null, ROUTING_KEYS.hotReload)).toBe(false);
  });

  it("shows the main fallback control for either fallback key", () => {
    expect(hasMainFallbackSupport({ fallback_model: {} })).toBe(true);
    expect(hasMainFallbackSupport({ fallback_providers: {} })).toBe(true);
    expect(hasMainFallbackSupport({ model: {} })).toBe(false);
    expect(hasMainFallbackSupport(null)).toBe(false);
  });
});
