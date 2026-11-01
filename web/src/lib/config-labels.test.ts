import { describe, expect, it } from "vitest";

import {
  configFieldCopy,
  configFieldDescription,
  configFieldLabel,
  configFieldSearchHaystack,
  configFieldSectionLabel,
  lookupConfigFieldDescription,
  lookupConfigFieldLabel,
  synthesizedConfigFieldLabel,
  translateConfigText,
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

describe("whole-token matching (bug A regression: `... SAVED` → `... 保存D`)", () => {
  // A `save` copy entry must NEVER rewrite the English token `SAVE` inside a
  // larger word. A partial/word-boundary-less match left the trailing `D` and an
  // uppercase style rendered it as `... 保存D`.
  const partial: ConfigCopyCarrier = {
    config: { fieldCopy: { save: "保存", saving: "保存中…" } },
  };

  it("never partial-matches a copy key inside a longer token", () => {
    for (const word of ["SAVED", "Saved", "saved", "UNSAVED", "AUTOSAVED", "SAVES"]) {
      expect(lookupConfigFieldLabel(partial, word)).toBeUndefined();
    }
  });

  it("leaves a `... SAVED` label untouched on a miss (falls back byte-for-byte)", () => {
    // The exact screenshot shape: an uppercase English label ending in SAVED.
    // The whole token is preserved — `SAVE` is never rewritten and no stray
    // trailing `D` is left behind.
    expect(configFieldLabel(partial, "SETTINGS SAVED")).toBe("SETTINGS SAVED");
    expect(configFieldLabel(partial, "SETTINGS SAVED")).not.toContain("保存");
    expect(configFieldLabel(partial, "saved")).toBe("Saved");
    // A whole-value translate on a phrase keeps the original text verbatim.
    expect(translateConfigText(partial, "Q-version SAVED")).toBe("Q-version SAVED");
  });

  it("still matches the WHOLE key when it is the whole token", () => {
    expect(lookupConfigFieldLabel(partial, "save")).toBe("保存");
    expect(configFieldLabel(partial, "save")).toBe("保存");
  });

  it("never resolves inherited Object.prototype members", () => {
    expect(lookupConfigFieldLabel(partial, "constructor")).toBeUndefined();
    expect(lookupConfigFieldLabel(partial, "toString")).toBeUndefined();
    expect(lookupConfigFieldLabel(partial, "__proto__")).toBeUndefined();
    // …and the synthesized fallback still renders a readable label.
    expect(configFieldLabel(partial, "toString")).toBe("ToString");
  });

  it("is case-sensitive: an uppercased/lowercased key is a miss, never a fuzzy hit", () => {
    expect(lookupConfigFieldLabel(partial, "SAVE")).toBeUndefined();
    expect(lookupConfigFieldLabel(partial, "Save")).toBeUndefined();
  });
});

describe("configFieldSectionLabel / translateConfigText", () => {
  const t: ConfigCopyCarrier = { config: { fieldCopy: { delegation: "委托" } } };

  it("translates a section only on a whole-key hit", () => {
    expect(configFieldSectionLabel(t, "delegation")).toBe("委托");
    // Whole-key only: a different section is untouched.
    expect(configFieldSectionLabel(t, "delegations")).toBe("delegations");
  });

  it("falls back to the de-underscored section name on a miss", () => {
    expect(configFieldSectionLabel(t, "tool_output")).toBe("tool output");
    expect(configFieldSectionLabel(undefined, "tool_output")).toBe("tool output");
  });

  it("translateConfigText returns the original text on a miss, verbatim", () => {
    expect(translateConfigText(t, "  delegation  ")).toBe("委托");
    expect(translateConfigText(t, "NOT TRANSLATED")).toBe("NOT TRANSLATED");
    expect(translateConfigText(undefined, "SAVED")).toBe("SAVED");
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
