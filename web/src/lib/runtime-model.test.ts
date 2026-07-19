import { describe, expect, it } from "vitest";

import { runtimeModelFromSessionInfo, selectChatModel } from "./runtime-model";

describe("runtimeModelFromSessionInfo", () => {
  it("keeps the active fallback and its original primary identity", () => {
    expect(
      runtimeModelFromSessionInfo({
        model: "gpt-5.5",
        provider: "openai-codex",
        fallback_activated: true,
        primary_model: "kimi-k3",
        primary_provider: "opencode-go",
      }),
    ).toEqual({
      model: "gpt-5.5",
      provider: "openai-codex",
      fallbackActivated: true,
      primaryModel: "kimi-k3",
      primaryProvider: "opencode-go",
    });
  });

  it("rejects malformed event payloads", () => {
    expect(runtimeModelFromSessionInfo(null)).toBeNull();
    expect(runtimeModelFromSessionInfo({ model: 42 })).toBeNull();
  });
});

describe("selectChatModel", () => {
  it("prefers the live PTY runtime over the configured model", () => {
    expect(
      selectChatModel("kimi-k3", {
        model: "gpt-5.5",
        provider: "openai-codex",
        fallbackActivated: true,
        primaryModel: "kimi-k3",
        primaryProvider: "opencode-go",
      }),
    ).toBe("gpt-5.5");
  });

  it("uses the configured model until the PTY emits runtime identity", () => {
    expect(selectChatModel("kimi-k3", null)).toBe("kimi-k3");
  });
});
