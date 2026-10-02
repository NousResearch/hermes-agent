import { describe, expect, it } from "vitest";

import {
  configFieldCopy,
  configFieldDescription,
  configFieldLabel,
  configFieldSearchHaystack,
  lookupConfigFieldDescription,
  lookupConfigFieldLabel,
  synthesizedConfigFieldLabel,
  type ConfigCopyCarrier,
} from "./config-labels";

// Minimal carrier shaped like the real `t` object.
const t: ConfigCopyCarrier = {
  config: {
    fieldCopy: {
      model: "模型",
      model_desc: "主代理的默认模型。",
      delegation_provider: "should-not-be-used",
      "delegation.provider": "子代理服务商",
    },
  },
};

describe("synthesizedConfigFieldLabel", () => {
  it("title-cases the last path segment, keeping the legacy behaviour", () => {
    expect(synthesizedConfigFieldLabel("model")).toBe("Model");
    expect(synthesizedConfigFieldLabel("model_context_length")).toBe("Model Context Length");
    expect(synthesizedConfigFieldLabel("delegation.provider")).toBe("Provider");
    expect(synthesizedConfigFieldLabel("fallback_model")).toBe("Fallback Model");
  });
});

describe("configFieldCopy", () => {
  it("returns the map, or an empty object when absent/undefined", () => {
    expect(configFieldCopy(t).model).toBe("模型");
    expect(configFieldCopy({ config: {} })).toEqual({});
    expect(configFieldCopy(undefined)).toEqual({});
    expect(configFieldCopy({})).toEqual({});
  });
});

describe("label precedence", () => {
  it("an authored i18n label WINS over the synthesized English one", () => {
    expect(configFieldLabel(t, "model")).toBe("模型");
    expect(configFieldLabel(t, "delegation.provider")).toBe("子代理服务商");
  });

  it("falls back to the synthesized English label on a miss", () => {
    expect(lookupConfigFieldLabel(t, "toolsets")).toBeUndefined();
    expect(configFieldLabel(t, "toolsets")).toBe("Toolsets");
    expect(configFieldLabel(undefined, "max_live_sessions")).toBe("Max Live Sessions");
  });
});

describe("description precedence", () => {
  it("an authored `_desc` entry WINS over the backend schema prose", () => {
    expect(
      configFieldDescription(t, "model", "Default model (e.g. anthropic/claude-sonnet-4.6)"),
    ).toBe("主代理的默认模型。");
  });

  it("falls back to the backend schema prose on a miss", () => {
    expect(
      configFieldDescription(t, "toolsets", "Tool groups available to the agent"),
    ).toBe("Tool groups available to the agent");
    expect(lookupConfigFieldDescription(t, "toolsets")).toBeUndefined();
  });

  it("is empty when neither an authored copy nor backend prose exists", () => {
    expect(configFieldDescription(t, "toolsets", undefined)).toBe("");
    expect(configFieldDescription(t, "toolsets", null)).toBe("");
  });

  it("never treats the bare key as a description (the `_desc` suffix is required)", () => {
    // `model` exists in the map but `model_desc` is the description entry.
    expect(lookupConfigFieldDescription({ config: { fieldCopy: { model: "x" } } }, "model")).toBeUndefined();
  });
});

describe("configFieldSearchHaystack", () => {
  it("matches on the raw key, synthesized + authored labels, category and descriptions", () => {
    const schema = { category: "general", description: "Default model to use" };
    expect(configFieldSearchHaystack(t, "model", schema)).toContain("model");
    expect(configFieldSearchHaystack(t, "model", schema)).toContain("模型");
    expect(configFieldSearchHaystack(t, "model", schema)).toContain("general");
    expect(configFieldSearchHaystack(t, "model", schema)).toContain("default model to use");
  });

  it("lets a translated label make a field findable in the UI language", () => {
    const zh = { config: { fieldCopy: { max_live_sessions: "最大活动会话数" } } };
    expect(configFieldSearchHaystack(zh, "max_live_sessions", {})).toContain("最大活动会话数");
    // The synthesized English label still matches too.
    expect(configFieldSearchHaystack(zh, "max_live_sessions", {})).toContain("max live sessions");
  });

  it("tolerates a missing schema entry", () => {
    expect(configFieldSearchHaystack(t, "model", null)).toContain("model");
    expect(configFieldSearchHaystack(t, "model", undefined)).toContain("模型");
  });
});
