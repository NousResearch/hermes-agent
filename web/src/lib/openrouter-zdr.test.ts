import { describe, expect, it } from "vitest";
import { readOpenRouterZdr, withOpenRouterZdr } from "./openrouter-zdr";

describe("OpenRouter ZDR config helpers", () => {
  it("reads only an explicit boolean true", () => {
    expect(readOpenRouterZdr({ openrouter: { zdr: true } })).toBe(true);
    expect(readOpenRouterZdr({ openrouter: { zdr: false } })).toBe(false);
    expect(readOpenRouterZdr({ openrouter: { zdr: "true" } })).toBe(false);
    expect(readOpenRouterZdr({})).toBe(false);
  });

  it("updates ZDR without dropping neighboring OpenRouter settings", () => {
    const config = {
      model: { provider: "openrouter" },
      openrouter: { response_cache: true, min_coding_score: 0.65 },
    };
    const next = withOpenRouterZdr(config, true);
    expect(next).toEqual({
      model: { provider: "openrouter" },
      openrouter: {
        response_cache: true,
        min_coding_score: 0.65,
        zdr: true,
      },
    });
    expect(config.openrouter).not.toHaveProperty("zdr");
  });

  it("repairs a malformed OpenRouter section", () => {
    expect(withOpenRouterZdr({ openrouter: "bad" }, false)).toEqual({
      openrouter: { zdr: false },
    });
  });
});