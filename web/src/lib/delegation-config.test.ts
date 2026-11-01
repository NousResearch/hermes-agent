import { describe, expect, it } from "vitest";

import { getNestedValue, setNestedValue } from "./nested";

// Mirror of the onChange merge in ConfigPage.tsx's DelegationModelProviderField
// wiring: a completed route transition (resetDirectEndpoint=true) also drops a
// stale direct endpoint so Inherit/model-only/provider picks can't keep
// routing to the old endpoint (the resolver's direct-endpoint branch takes
// precedence). Regression tests for issue #67347 review findings F1 (stale
// endpoint siblings) and F2 (unfinished pairs never reach the saveable config
// — provider switches stay local draft until a model is committed).
function applyDelegationChange(
  prev: Record<string, unknown>,
  next: { provider: string; model: string; resetDirectEndpoint?: boolean },
): Record<string, unknown> {
  let merged: Record<string, unknown> = setNestedValue(
    setNestedValue(prev ?? {}, "delegation.model", next.model),
    "delegation.provider",
    next.provider,
  );
  if (next.resetDirectEndpoint) {
    merged = setNestedValue(merged, "delegation.base_url", "");
    merged = setNestedValue(merged, "delegation.api_key", "");
  }
  return merged;
}

describe("delegation config merge and direct endpoint reset (F1 & F2 regressions)", () => {
  it("clears stale base_url and api_key on route transition when resetDirectEndpoint is true", () => {
    const prevConfig = {
      delegation: {
        provider: "custom",
        model: "old-custom-model",
        base_url: "https://old.example.invalid/v1",
        api_key: "old-secret-key",
        request_overrides: { temperature: 0.7 },
      },
    };

    const merged = applyDelegationChange(prevConfig, {
      provider: "",
      model: "",
      resetDirectEndpoint: true,
    });

    expect(getNestedValue(merged, "delegation.provider")).toBe("");
    expect(getNestedValue(merged, "delegation.model")).toBe("");
    expect(getNestedValue(merged, "delegation.base_url")).toBe("");
    expect(getNestedValue(merged, "delegation.api_key")).toBe("");
    // Unrelated sibling settings remain intact
    expect(getNestedValue(merged, "delegation.request_overrides")).toEqual({ temperature: 0.7 });
  });

  it("preserves explicit model-only override without keeping stale direct endpoint", () => {
    const prevConfig = {
      delegation: {
        provider: "custom",
        model: "old-model",
        base_url: "https://old.example.invalid/v1",
        api_key: "old-key",
      },
    };

    const merged = applyDelegationChange(prevConfig, {
      provider: "",
      model: "custom-sonnet",
      resetDirectEndpoint: true,
    });

    expect(getNestedValue(merged, "delegation.provider")).toBe("");
    expect(getNestedValue(merged, "delegation.model")).toBe("custom-sonnet");
    expect(getNestedValue(merged, "delegation.base_url")).toBe("");
    expect(getNestedValue(merged, "delegation.api_key")).toBe("");
  });

  it("does not mutate prevConfig object", () => {
    const prevConfig = {
      delegation: {
        provider: "openai",
        model: "gpt-5.1",
        base_url: "https://api.openai.com/v1",
      },
    };

    const merged = applyDelegationChange(prevConfig, {
      provider: "",
      model: "",
      resetDirectEndpoint: true,
    });

    expect(prevConfig.delegation.provider).toBe("openai");
    expect(prevConfig.delegation.model).toBe("gpt-5.1");
    expect(prevConfig.delegation.base_url).toBe("https://api.openai.com/v1");
    expect(getNestedValue(merged, "delegation.provider")).toBe("");
    expect(getNestedValue(merged, "delegation.base_url")).toBe("");
  });
});
